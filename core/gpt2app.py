import copy
from typing import List

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast

from core.model import GPT2LMHeadModel


class GPT2App:
    def __init__(self, device, vocab_path, model_path, **kwargs):
        """
        :param vocab_path: vocab_path
        :param model: model
        :param tokenizer: tokenizer
        :param device: device
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and int(device) >= 0 else "cpu")
        self.tokenizer = BertTokenizerFast(vocab_file=vocab_path, do_lower_case=True)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, content, output_sent_num=3, max_len: int = 1020,
                 generate_max_len: int = 120, repetition_penalty: float = 1.2,
                 top_k: int = 3, top_p: float = 0.4, **kwargs) -> List[str]:
        """
        predict on one sample
        :param content: news content
        :param output_sent_num: the number of generating title
        :param max_len: the maximum length of model input, it should be smaller than n_ctx of config
        :param generate_max_len: the maximum length of generated title
        :param repetition_penalty: penalty of generating repeated words
        :param top_k: number of top words with highest probability that we keep when decoding
        :param top_p: the probability threshold when decoding
        :return: list of generated sentences
        """
        # do the preprocessing on the news content, if the length is too long, then truncate
        content_token_max_len = max_len - 3 - generate_max_len
        content_tokens = self.tokenizer.tokenize(content)
        if len(content_tokens) > content_token_max_len:
            content_tokens = content_tokens[:content_token_max_len]
        # gain the value of content_id、title_id、unk_id、sep_id
        content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        title_id = self.tokenizer.convert_tokens_to_ids("[Title]")
        unk_id = self.tokenizer.convert_tokens_to_ids("[UNK]")
        sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        # turn tokens into index, and make them become the format of model input
        content_tokens = ["[CLS]"] + content_tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(content_tokens)
        # expand input_ids and token_type_ids to the number of generated title, namely batch_size
        input_ids = [copy.deepcopy(input_ids) for _ in range(output_sent_num)]
        token_type_ids = [[content_id] * len(content_tokens) for _ in range(output_sent_num)]
        # make input_ids and token_type_ids become tensor
        input_tensors = torch.tensor(input_ids).long().to(self.device)
        token_type_tensors = torch.tensor(token_type_ids).long().to(self.device)
        next_token_type = torch.tensor([[title_id] for _ in range(output_sent_num)]).long().to(self.device)
        # store the result of decoding
        generated = []
        # store the index of finished batch_size
        finish_set = set()
        with torch.no_grad():
            # iterate through the maximum length of generating title
            for _ in range(generate_max_len):
                outputs = self.model(input_ids=input_tensors, token_type_ids=token_type_tensors)
                # gain the last hidden layer, next_token_logits size：[batch_size, vocab_size]
                next_token_logits = outputs[0][:, -1, :]
                # iterate through batch_size, do the repetition penalty
                for index, prediction in enumerate(next_token_logits):
                    token_id_set = set([token_ids[index] for token_ids in generated])
                    for token_id in token_id_set:
                        prediction[token_id] /= repetition_penalty
                # iterate through batch_size, set UNK as infinite small
                for next_token_logit in next_token_logits:
                    next_token_logit[unk_id] = -float("Inf")
                # use top_k_top_p_filtering function to filter the result
                filter_logits = top_k_top_p_filter(next_token_logits, top_k=top_k, top_p=top_p)
                # use probability to choose token from filter_logits
                next_tokens = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1)
                # if prediction equals SEP, which means it is finished, then add this index to  finish_set
                for index, token_id in enumerate(next_tokens[:, 0]):
                    if token_id == sep_id:
                        finish_set.add(index)
                # if finish_set contains all the index then stop the prediction
                finish_flag = True
                for index in range(output_sent_num):
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
            for index in range(output_sent_num):
                responses = []
                for token_index in range(len(generated)):
                    # if sep_id show up, then stop to add new generated token
                    if generated[token_index][index] != sep_id:
                        responses.append(generated[token_index][index])
                    else:
                        break
                # turn token_id into Chinese, remove "##" and turn [Space] into space
                candidate_responses.append(
                    "".join(self.tokenizer.convert_ids_to_tokens(responses)).replace("##", "").replace("[space]", " "))
        return candidate_responses


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
