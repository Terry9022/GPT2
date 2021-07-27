from pydantic import BaseSettings


class Settings(BaseSettings):
    device: str = '0'  # the graphics card used when predicting, if using CPU, then set -1
    model_path: str = 'output_dir_pre/best_model'  # location path of model
    vocab_path: str = 'vocab/vocab.txt'  # location path of vocabulary
    batch_size: int = 1  # the number of generating title
    generate_max_len: int = 120  # the maximum length of generated title
    repetition_penalty: float = 1.2  # penalty of generating repeated words
    top_k: float = 3  # number of top words with highest probability that we keep when decoding
    top_p: float = 0.4  # the probability threshold when decoding
    max_len: int = 1020  # the maximum length of model input, it should be smaller than n_ctx of config


settings = Settings()
