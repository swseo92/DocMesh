from pydantic import BaseModel


class CrawlRequest(BaseModel):
    url: str


class QuestionRequest(BaseModel):
    question: str
