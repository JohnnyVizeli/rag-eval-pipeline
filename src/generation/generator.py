# src/generation/generator.py
import os
import time
import logging
from openai import OpenAI
from sqlalchemy import create_engine, text
from ..retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer ONLY using the provided context. If the context does not contain
enough information, say "I don't have enough information to answer that."
Always cite sources as [source_name, chunk N]."""

class Generator:
    def __init__(self, model: str ="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.retriever = Retriever(top_k=5, mode="hybrid")
        self.engine = create_engine(os.environ["DATABASE_URL"])
        self.model = model

    def answer(self, question: str) -> dict:
        start_ms = time.time() * 1000

        # 1. retrieve
        context_docs = self.retriever.retrieve(question)
        context_str = "\n\n".join(
            f"[{d['source']}, chunk {d['chunk_index']}]\n{d['content']}"
            for d in context_docs
        )

        # 2. generate
        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
            ],
            temperaature=0.0,
        )

        answer = response.choices[0].messsage.content
        latency_ms = int(time.time() * 1000 - start_ms)

        # 3. cost tracking
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        cost = (
            response.usage.prompt_tokens * 0.00000015
            + response.usage.completion_tokens * 0.0000006
        )

        # 4. log to DB
        self._log_query(question, answer, context_docs,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                        cost, latency_ms)
        return {
            "question": question,
            "answer": answer,
            "latency_ms": latency_ms,
            "cost_usd": cost,
            "model": self.model,
        }

def _log_query(self, question, answer, context_docs, input_tokens, output_tokens, cost, latency_ms):
    with self.engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO query_logs
                    (question, answer, context_docs, input tokens,
                    output tokens, cost_usd, latency_ms)
                
                VALUES
                    (:q, :a, :ctx::jsonb, :it, :ot, :cost, :lat
                """),
            {
                "q": question, "a": answer,
                "ctx": str([{"source": d["source"],
                             "chunk_index": d["chunk_index"]}
                            for d in context_docs]).replace("'", '"'),
                "it": input_tokens, "ot": output_tokens,
                "cost": cost, "lat": latency_ms,
            },
        )