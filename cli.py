# According to trained model,
# Use it to generate the title based on the article content

import os
import argparse

from core.gpt2app import GPT2App


def set_args():
    # the parameters for generating the title
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str,
                        help='the graphics card used when predicting, if using CPU, then set -1')
    parser.add_argument('--model_path', default='output_dir_pre/best_model', type=str, help='location path of model')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, help='location path of vocabulary')
    parser.add_argument('--batch_size', default=1, type=int, help='the number of generating title')
    parser.add_argument('--generate_max_len', default=120, type=int, help='the maximum length of generated title')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, help='penalty of generating repeated words')
    parser.add_argument('--top_k', default=3, type=float,
                        help='number of top words with highest probability that we keep when decoding')
    parser.add_argument('--top_p', default=0.4, type=float, help='the probability threshold when decoding')
    parser.add_argument('--max_len', type=int, default=1020,
                        help='the maximum length of model input, it should be smaller than n_ctx of config')
    return parser.parse_args()


def main():
    args = set_args()
    # set the device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    gpt2_app = GPT2App(**args.__dict__)
    print('開始對新聞生成摘要，輸入 CTRL + Z 則退出')
    try:
        while True:
            content = input("輸入的新聞正文為:")
            titles = gpt2_app.generate(content=content, **args.__dict__)
            for i, title in enumerate(titles):
                print("生成的第{}個摘要為：{}".format(i + 1, title))
    except:
        pass


if __name__ == '__main__':
    main()
