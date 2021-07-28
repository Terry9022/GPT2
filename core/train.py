# GPT2 model training file using article content to generate summarization

import torch
import os
import random
import numpy as np
import argparse
import logging
from transformers.models.gpt2.modeling_gpt2 import GPT2Config
from model import GPT2LMHeadModel
from transformers import BertTokenizer
from data_set import GPT2NewsTitleDataSet, collate_func
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def set_args():
    # the parameters for training the model
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='the device used when training or testing')
    parser.add_argument('--config_path', default='config/config.json', type=str, help='parameters of model')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, help='vocabulary list location')
    parser.add_argument('--train_file_path', default='./data_dir/train_data.json', type=str,
                        help='training dataset location')
    parser.add_argument('--test_file_path', default='./data_dir/test_data.json', type=str,
                        help='testing dataset location')
    parser.add_argument('--pretrained_model_path', default='./pretrain_model', type=str,
                        help='pretrained model location ')
    parser.add_argument('--data_dir', default='./data_dir', type=str, help='data folder location')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of training epochs')
    parser.add_argument('--train_batch_size', default=8, type=int, help='batch size of training')
    parser.add_argument('--test_batch_size', default=8, type=int, help='batch size of testing')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate of training')
    parser.add_argument('--warmup_proportion', default=0.1, type=float,
                        help='proportion of warm up which means that how much percentage of training using warmup ')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='epsilon of Adam optimizer')
    parser.add_argument('--logging_steps', default=20, type=int, help='how many steps do we save a logger')
    parser.add_argument('--eval_steps', default=1000, type=int,
                        help='how many step do we evaluate the model when training')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help='gradient accumulation')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir_pre/', type=str, help='model output location')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--max_len', type=int, default=1020,
                        help='the maximum length of model input, it should be smaller than n_ctx of config')
    parser.add_argument('--title_max_len', type=int, default=120,
                        help='the maximum length of generating title, it should be smaller than max_len')
    return parser.parse_args()


def train(model, device, train_data, test_data, args):
    """
    train the model
    :param model: model
    :param device: device using
    :param train_data: training data
    :param test_data: testing data
    :param args: training parameter configuration
    :return:
    """
    tb_write = SummaryWriter()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps parameter is invalid, which should be greater than 1")
    # calculate the true size of training batch due to gradient accumulation
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler,
                                   batch_size=train_batch_size, collate_fn=collate_func)
    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    logger.info('total training steps is {}'.format(total_steps))
    model.to(device)
    # get all the parameters of the model
    param_optimizer = list(model.named_parameters())
    no_dacay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_dacay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_dacay)], 'weight_dacay': 0.0}
    ]
    # set the optimizer
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    # clear cuda's cache
    torch.cuda.empty_cache()
    # set the model as training mode
    model.train()
    title_id = train_data.title_id
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0
    min_loss = 1000
    early_stop_cnt = 0
    # start to train the model
    for iepoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(tqdm(train_data_loader)):
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            # get the training result
            outputs = model.forward(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids,
                                    title_id=title_id)
            loss = outputs[0]
            tr_loss += loss.item()

            # judge whether conduct gradient accumulation, if yes, then let the loss divided by accumulation steps
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # backward the loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # if training steps are divisible by accumulation steps, upgrade the parameters
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                # if steps are divisible by logging_steps, record loss and learning rate.
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_write.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_write.add_scalar("train_loss", (tr_loss - logging_loss) /
                                        (args.logging_steps * args.gradient_accumulation_steps), global_step)
                    logging_loss = tr_loss
                # if steps are divisible by eval_steps, evaluate the model
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, device, test_data, args)
                    tb_write.add_scalar("test_loss", eval_loss, global_step)
                    if eval_loss < min_loss:
                        # Save model if your model improved
                        min_loss = eval_loss
                        print(
                            'Saving model (epoch = {:4d}, step = {:4d}, loss = {:.4f})'.format(iepoch + 1, global_step,
                                                                                               min_loss))
                        # after each epoch, save the model
                        output_dir = os.path.join(args.output_dir, "../output_dir_pre/best_model")
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        early_stop_cnt = 0
                    else:
                        early_stop_cnt += 1

                # Stop training if your model stops improving for 30 args.eval_steps
                if early_stop_cnt > 30:
                    print("early stop at epoch", iepoch + 1)
                    break

                model.train()

        # empty the cuda cache
        torch.cuda.empty_cache()


def evaluate(model, device, test_data, args):
    """
    model's evaluation on testing data
    :param model: model
    :param device: device
    :param test_data: testing data
    :param args: training parameter configuration
    :return:
    """
    # construct DataLoader of testing data
    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func)

    title_id = test_data.title_id
    total_loss, total = 0.0, 0.0
    for step, batch in enumerate(tqdm(test_data_loader)):
        # set model mode as eval()
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            # get the result of evaluation
            outputs = model.forward(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids,
                                    title_id=title_id)
            loss = outputs[0]
            loss = loss.item()
            # accumulate the loss
            total_loss += loss * len(batch["input_ids"])
            total += len(batch["input_ids"])
    # calculate the final result of testing data
    test_loss = total_loss / total
    return test_loss


def main():
    # set model's configuration
    args = set_args()
    # set up the device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    print(f"using {device}")

    # set the random seed
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    # load the configuration of model
    model_config = GPT2Config.from_json_file(args.config_path)
    # whether use pre-trained model
    # If yes, load the pre-trained model. If not, initialize the model by the configuration we defined
    if args.pretrained_model_path:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path)
        print("using pretrained model")
    else:
        model = GPT2LMHeadModel(config=model_config)
        print("do not using pretrained model")
    # load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    # We add [Space] as a token. For example,"這個[Space]厲害", originally the result of tokenizing is "['這', '個', '[', 'Space', ']', '厲', '害', '。']";
    # After adding, the result of tokenizing is "['這', '個', '[Space]', '厲', '害', '。']";
    tokenizer.add_tokens("[Space]", special_tokens=True)
    # build the directory of output
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # load the training data and testing data
    train_data = GPT2NewsTitleDataSet(tokenizer, args.max_len, args.title_max_len, args.data_dir, "train",
                                      args.train_file_path)
    test_data = GPT2NewsTitleDataSet(tokenizer, args.max_len, args.title_max_len, args.data_dir, "test",
                                     args.test_file_path)
    # start training
    train(model, device, train_data, test_data, args)


if __name__ == "__main__":
    main()
