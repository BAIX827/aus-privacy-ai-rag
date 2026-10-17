from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from privrag import config
from privrag.rag import answer
from privrag.retrieve import Retriever
from pydantic import BaseModel, Field

app = FastAPI(title="Australian privacy & AI guidance Q&A")
STATIC = Path(__file__).parent / "static"


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@lru_cache(maxsize=1)
def retriever() -> Retriever:
    return Retriever()


@app.get("/health")
def health():
    return {"status": "ok", "backend": config.LLM_BACKEND}


@app.post("/ask")
def ask(q: Question):
    return answer(q.question, retriever())


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
