# According to trained model,
# Use it to generate the title based on the article content

import torch
import os
import argparse
from model import GPT2LMHeadModel
from transformers import BertTokenizer
import torch.nn.functional as F
import copy


def set_args():
    # the parameters for generating the title
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str,
                        help='the graphics card used when predicting, if using CPU, then set -1')
    parser.add_argument('--model_path', default='output_dir_pre/best_model', type=str, help='location path of model')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, help='location path of vocabulary')
    parser.add_argument('--batch_size', default=3, type=int, help='the number of generating title')
    parser.add_argument('--generate_max_len', default=120, type=int, help='the maximum length of generated title')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, help='penalty of generating repeated words')
    parser.add_argument('--top_k', default=3, type=float,
                        help='number of top words with highest probability that we keep when decoding')
    parser.add_argument('--top_p', default=0.4, type=float, help='the probability threshold when decoding')
    parser.add_argument('--max_len', type=int, default=1020,
                        help='the maximum length of model input, it should be smaller than n_ctx of config')
    return parser.parse_args()


def top_k_top_p_filter(logits, top_k, top_p, filter_value=-float("Inf")):
    """
    top_k or top_p decoding strategy. Just keep the token which is top_k or its probability is over top_p,
    the other tokens are set as filter_value whose values are infinite small.
    :param logits: outcome of prediction, which is probability of words in the vocabulary.
    :param top_k: number of top words with highest probability that we keep when decoding
    :param top_p: the probability threshold when decoding
    :param filter_value: the value of filtered token
    :return:
    """
    # dimension of logits must be 2, that is size: [batch_size, vocab_size]
    assert logits.dim() == 2
    # get the smaller one between top_k and size of vocabulary. Namely, if top_k is bigger than size of vocabulary, we choose size of vocabulary.
    top_k = min(top_k, logits[0].size(-1))
    # if top_k is not 0, we keep top_k token in the logits
    if top_k > 0:
        # Because there are number of batch_size prediction outcomes, we get all the top_k tokens in every batch data
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value
    # if top_p is not 0, we keep token whose probability is smaller than top_p in the logits
    if top_p > 0.0:
        # sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # use softmax toward sorted result, then gain the cumulative probability
        # For example, [0.1, 0.2, 0.3, 0.4] turn into [0.1, 0.3, 0.6, 1.0]
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # delete the token whose probability is higher than top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


def predict_one_sample(model, tokenizer, device, args, content):
    """
    predict on one sample
    :param model: model
    :param tokenizer: tokenizer
    :param device: device
    :param args: configuration
    :param content: news content
    :return:
    """
    # do the preprocessing on the news content, if the length is too long, then truncate
    content_tokens = tokenizer.tokenize(content)
    if len(content_tokens) > args.max_len - 3 - args.generate_max_len:
        content_tokens = content_tokens[:args.max_len - 3 - args.generate_max_len]
    # gain the value of content_id、title_id、unk_id、sep_id
    content_id = tokenizer.convert_tokens_to_ids("[Content]")
    title_id = tokenizer.convert_tokens_to_ids("[Title]")
    unk_id = tokenizer.convert_tokens_to_ids("[UNK]")
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    # turn tokens into index, and make them become the format of model input
    content_tokens = ["[CLS]"] + content_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(content_tokens)
    # expand input_ids and token_type_ids to the number of generated title, namely batch_size
    input_ids = [copy.deepcopy(input_ids) for _ in range(args.batch_size)]
    token_type_ids = [[content_id] * len(content_tokens) for _ in range(args.batch_size)]
    # make input_ids and token_type_ids become tensor
    input_tensors = torch.tensor(input_ids).long().to(device)
    token_type_tensors = torch.tensor(token_type_ids).long().to(device)
    next_token_type = torch.tensor([[title_id] for _ in range(args.batch_size)]).long().to(device)
    # store the result of decoding
    generated = []
    # store the index of finished batch_size
    finish_set = set()
    with torch.no_grad():
        # iterate through the maximum length of generating title
        for _ in range(args.generate_max_len):
            outputs = model(input_ids=input_tensors, token_type_ids=token_type_tensors)
            # gain the last hidden layer, next_token_logits size：[batch_size, vocab_size]
            next_token_logits = outputs[0][:, -1, :]
            # iterate through batch_size, do the repetition penalty
            for index in range(args.batch_size):
                for token_id in set([token_ids[index] for token_ids in generated]):
                    next_token_logits[index][token_id] /= args.repetition_penalty
            # iterate through batch_size, set UNK as infinite small
            for next_token_logit in next_token_logits:
                next_token_logit[unk_id] = -float("Inf")
            # use top_k_top_p_filtering function to filter the result
            filter_logits = top_k_top_p_filter(next_token_logits, top_k=args.top_k, top_p=args.top_p)
            # use probability to choose token from filter_logits
            next_tokens = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1)
            # if prediction equals SEP, which means it is finished, then add this index to  finish_set
            for index, token_id in enumerate(next_tokens[:, 0]):
                if token_id == sep_id:
                    finish_set.add(index)
            # if finish_set contains all the index then stop the prediction
            finish_flag = True
            for index in range(args.batch_size):
                if index not in finish_set:
                    finish_flag = False
                    break
            if finish_flag:
                break
            # add the predicted token to generated
            generated.append([token.item() for token in next_tokens[:, 0]])
            # concat the last prediction to input_tensors and token_type_tensors, then use them all to continue to predict
            input_tensors = torch.cat((input_tensors, next_tokens), dim=-1)
            token_type_tensors = torch.cat((token_type_tensors, next_token_type), dim=-1)
        # store the result of generating
        candidate_responses = []
        # iterate through batch_size, turn token_id into Chinese
        for index in range(args.batch_size):
            responses = []
            for token_index in range(len(generated)):
                # if sep_id show up, then stop to add new generated token
                if generated[token_index][index] != sep_id:
                    responses.append(generated[token_index][index])
                else:
                    break
            # turn token_id into Chinese, remove "##" and turn [Space] into space
            candidate_responses.append(
                "".join(tokenizer.convert_ids_to_tokens(responses)).replace("##", "").replace("[space]", " "))
    return candidate_responses


def main():
    args = set_args()
    # set the device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    # set up tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    print('開始對新聞生成摘要，輸入 CTRL + Z 則退出')
    try:
        while True:
            content = input("輸入的新聞正文為:")
            titles = predict_one_sample(model, tokenizer, device, args, content)
            for i, title in enumerate(titles):
                print("生成的第{}個摘要為：{}".format(i + 1, title))
    except:
        pass


if __name__ == '__main__':
    main()
