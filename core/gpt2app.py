import torch
from transformers import BertTokenizerFast

from generate_title import predict_one_sample, set_args
from model import GPT2LMHeadModel


class GPT2App:
    def __init__(self):
        self.args = set_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() and int(self.args.device) >= 0 else "cpu")
        self.tokenizer = BertTokenizerFast(vocab_file=self.args.vocab_path, do_lower_case=True)
        self.model = GPT2LMHeadModel.from_pretrained(self.args.model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, content):
        return predict_one_sample(self.model, self.tokenizer, self.device, self.args, content)
