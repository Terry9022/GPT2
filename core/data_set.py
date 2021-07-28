# define the data format according to model

import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import logging
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

class GPT2NewsTitleDataSet(Dataset):
    # the data format used by model
    def __init__(self, tokenizer, max_len,title_max_len,data_dir,data_set_name , path_file=None,is_overwrite=False) :
        """
        :param tokenizer: tokenizer
        :param max_len: the max length of data
        :param title_max_len: the max length of output title
        :param data_dir: the directory of data
        :param data_set_name: dataset name
        :param path_file: raw data
        :param is_overwrite: whether to generate the cached data
        """
        self.tokenizer=tokenizer
        # content_id and title_id are according to news content and title for each part, for model can distinguish them.
        self.content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        self.title_id = self.tokenizer.convert_tokens_to_ids("[Title]")
        # Space_id means the mark of space. Some title contains space, if we use tokenize it will disappear.
        self.space_id = self.tokenizer.convert_tokens_to_ids(["Space"])
        self.max_len=max_len
        self.title_max_len = title_max_len
        cached_feature_file = os.path.join(data_dir,"cached_{}_{}".format(data_set_name,max_len))
        # judge whether the cached file exists or not. If yes, directly load.
        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info("cached file exists{}, directly load.".format(cached_feature_file))
            self.data_set = torch.load(cached_feature_file)["data_set"]
        # judge whether the cached file exists or not. If not, save the cached file.
        else:
            logger.info("cached file doesn't exists, it need to be saved")
            self.data_set=self.load_data(path_file)
            logger.info("saving the cached file")
            torch.save({"data_set":self.data_set},cached_feature_file)

    def load_data(self,path_file):
        # load the raw data and do some preprocessing
        self.data_set = []
        with open(path_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            for idx, sample in enumerate(tqdm(data,desc="iter", disable = False)):
                # use convert_feature function and transform the title and content into embedding
                input_ids,token_type_ids = self.convert_feature(sample)
                self.data_set.append({"input_ids":input_ids,"token_type_ids":token_type_ids})
        return self.data_set

    def convert_feature(self,sample):
        """
        processing the data function
        :param sample: a dictionary, which contains new's content and title, format is {"content":content, "title":title}
        :return:
        """
        input_ids = []
        token_type_ids = []
        # tokenizer.tokenize toward news content
        content_tokens = self.tokenizer.tokenize(sample["content"])
        # tokenizer.tokenize toward news title
        title_tokens = self.tokenizer.tokenize(sample["title"].replace(" ","[Space]"))
        # if length of title is too long then truncate
        if len(title_tokens) > self.title_max_len:
            title_tokens = title_tokens[:self.title_max_len]
        # if length of content is too long then truncate
        if len(content_tokens) > self.max_len - len(title_tokens) - 3:
            content_tokens = content_tokens[:self.max_len - len(title_tokens) - 3]
        # generate the input_ids and token_type_ids for model
        # Format: CLS content SEP title SEP
        # CLS content
        input_ids.append(self.tokenizer.cls_token_id)
        token_type_ids.append(self.content_id)
        # content
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(content_tokens))
        token_type_ids.extend([self.content_id]*len(content_tokens))
        # SEP content
        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(self.content_id)
        # title
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(title_tokens))
        token_type_ids.extend([self.title_id] * len(title_tokens))
        # SEP title
        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(self.title_id)
        # judge whether the length of input_ids equals to token_type_ids
        assert len(input_ids) == len(token_type_ids)
        # judge whether the length of input_ids is less than or equal to the maximum length
        assert len(input_ids) <= self.max_len

        return input_ids, token_type_ids

    def __len__(self):
        return len(self.data_set)
    def __getitem__(self, item):
        instance = self.data_set[item]
        return instance



def collate_func(batch_data):
    """
    pytorch DataLoader's collate_fun function, make data become tensor format
    :return:
    """
    batch_size = len(batch_data)
    # if batch_size is 0 then return empty dictionary
    if batch_size==0:
        return {}
    input_ids_list, token_type_ids_list = [],[]
    for instance in batch_data:
        # according to the maximum length of batch, do the padding to the batch data
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
    # put input_ids_temp and token_type_ids_temp to the according list
    input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
    token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
    # use pad_sequence function which can do the padding to all the tensor in the list
    # make up till the maximum length of batch data, the padding element is padding_value
    return {
        "input_ids":pad_sequence(input_ids_list,batch_first=True,padding_value=0),
        "token_type_ids":pad_sequence(token_type_ids_list,batch_first = True,padding_value=0)
    }




