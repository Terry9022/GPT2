# 對資料預處理，進行資料清洗，並生成訓練資料集和測試資料集
# Do the data preprocessing, data cleaning, generating the training and testing data

import re
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
import random

def clean_title(title: str):
    """
    data cleaning toward title
    :param title:
    :return:
    """
    title = re.sub(r"#","",title) # remove #
    title = re.sub(r"\s+", " ", title) # remove duplicated space
    return title

def clean_content(content: str):
    """
    data cleaning toward content
    :param content:
    :return:
    """
    content = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b","",content)
    content = re.sub(r"\s+"," ",content)
    content = content.replace("\u200b","")
    return content

def clean_data(sample):
    (content,title) = sample
    sample = dict()
    sample['title']=clean_title(title.strip())
    sample['content']=clean_content(content.strip())
    return sample

def build_news_data(content_path, title_path,train_save_path,test_save_path):
    """
    Data cleaning, generating the training and testing data
    :param data_path: raw data path
    :param train_save_path: training data path
    :param test_save_path: testing data path
    :return:
    """
    # open raw data and zip them as a dict
    content_data = open(content_path, "r", encoding="utf-8")
    title_data = open(title_path, "r", encoding="utf-8")
    data = zip(content_data.readlines(), title_data.readlines())
    threads = min(8,cpu_count())
    with Pool(threads) as p:
        annoate_ = partial(clean_data)
        data=list(tqdm(p.imap(annoate_,data,chunksize=8), desc="build data"))

    # filter the data
    data_set=set()
    data_new = []
    for d in data:
        if d['content'] in data_set or len(d["content"])<100 or len(d["title"]) <2 :
            continue
        else:
            data_set.add(d["content"])
            data_new.append(d)
    # split the data
    random.shuffle(data_new)
    train_data= data_new[:-3000]
    test_data = data_new[-3000:] # use the last 3000 data as test data
    f = open(train_save_path,"w",encoding="utf-8")
    f.write(json.dumps(train_data, indent=4, ensure_ascii=False))
    f.close()
    f = open(test_save_path, "w", encoding="utf-8")
    f.write(json.dumps(test_data,indent=4, ensure_ascii=False))
    f.close()

if __name__ == '__main__':
    #data_path_dir = "data_dir/train_with_summary.txt"
    #train_save_path_dir = "data_dir/train_data.json"
    #test_save_path_dir = "data_dir/test_data.json"

    content_path_dir = "data_dir/train_text.txt"
    title_path_dir = "data_dir/train_label.txt"
    train_save_path_dir = "data_dir/train_data.json"
    test_save_path_dir = "../data_dir/test_data.json"
    build_news_data(content_path_dir, title_path_dir, train_save_path_dir, test_save_path_dir)



