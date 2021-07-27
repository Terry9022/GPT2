# FastAPI Demo
import logging
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.logger import logger
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, BaseSettings

from core.gpt2app import GPT2App
from starlette.concurrency import run_in_threadpool
from api_config import settings

logger.setLevel(logging.DEBUG)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = settings.device
logger.info("Model loaded successful!")
gpt2_app = GPT2App(**settings.__dict__)

tags_metadata = [
    {
        "name": "apis",
        "description": "API們",
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
    content: str = Field(None, title="輸入內文", max_length=settings.max_len, description="欲進行文字生成的文章內文")
    output_sent_num: int = Field(1, title="輸出數量", description="欲輸出的候選句子數量")


@app.get("/generate_summary", tags=['pages'], name='generate-summary-site')
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "生成式摘要"})


@app.post("/generate_summary", tags=['apis'], name="generate-summary-api")
async def generate_summary(data: SummaryRequest):
    content = data.content.strip()
    output_sent_num = data.output_sent_num
    return await run_in_threadpool(gpt2_app.generate, content=content, output_sent_num=output_sent_num)


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=7000)
