# FastAPI Demo
import logging
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.logger import logger
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.gpt2app import GPT2App
from generate_title import predict_one_sample, set_args

args = set_args()
logger.setLevel(logging.DEBUG)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = args.device
logger.info("Model loaded successful!")
gpt2_app = GPT2App()

tags_metadata = [
    {
        "name": "summary",
        "description": "自動摘要",
    },
    {
        "name": "pages",
        "description": "站台頁面",
    },
]

app = FastAPI(openapi_tags=tags_metadata)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class SummaryRequest(BaseModel):
    content: str


@app.get("/generate_summary", tags=['pages'], name='generate summary demo site')
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "生成式摘要"})


@app.post("/generate_summary", tags=['summary'], name="generate summary api")
async def generate_summary(data: SummaryRequest):
    content = data.content
    content = content.strip()
    logger.info("generating summary...")
    titles = gpt2_app.predict(content)  # 目前在generate_title 中設定產生三個摘要
    summary = [
        titles[0],
        titles[1],
        titles[2]
    ]

    logger.info("done.")
    return summary


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=7000)
