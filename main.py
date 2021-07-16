# FastAPI Demo

from pydantic import BaseModel
from fastapi import FastAPI
from generate_title import predict_one_sample, set_args, top_k_top_p_filter
import torch
from model import GPT2LMHeadModel
from transformers import BertTokenizer
import os


args = set_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
model = GPT2LMHeadModel.from_pretrained(args.model_path)
model.to(device)
model.eval()
print("load model ending!")


app = FastAPI()

class SummaryRequest(BaseModel):
    text: str


@app.post("/summary_generate")
async def response_request(request: SummaryRequest):
    content = request.text
    content = content.strip()
    titles = predict_one_sample(model, tokenizer, device, args, content)  #目前在generate_title 中設定產生三個摘要
    summary = {"第1個標題為" : titles[0],
               "第2個標題為" : titles[1],
               "第3個標題為" : titles[2]}

    return summary

@app.get("/")
async def root():
    return {"message": "go to /summary_generate to generate summary"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7000)


