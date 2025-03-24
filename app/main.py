from fastapi import FastAPI
from .models import CrawlRequest, QuestionRequest


app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/crawl")
def crawl_endpoint(payload: CrawlRequest):
    # TODO: 나중에 크롤러 함수를 호출하고 결과 반환
    return {"status": "crawl completed"}


@app.post("/ask")
def ask_endpoint(question: QuestionRequest):
    # TODO: 나중에 QA 함수 호출
    return {"answer": "this is a mock answer"}
