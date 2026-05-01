# src/api/main.py
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ..generation.generator import Generator

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="RAG Eval Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

generator = Generator()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: list[dict]
    latency_ms: int
    cost_usd: float
    model: str

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = generator.answer(request.question)
    return QueryResponse(**result)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}