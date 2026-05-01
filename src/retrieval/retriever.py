# src/retrieval/retriever.py
import os
import logging
from rank_bm25 import BM25Okapi
from openai import OpenAI
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, top_k: int = 5, mode: str = "hybrid"):
        """
        mode: 'dense' | 'bm25' | 'hybrid'
        hybrid = RRF fusion of dense + BM25 results
        """
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.engine = create_engine(os.environ["DATABASE_URL"])
        self.top_k = top_k
        self.mode = mode

    def retrieve(self, query: str) -> list[dict]:
        if self.mode == "dense":
            return self._dense_search(query)
        elif self.mode == "bm25":
            return self._bm25_search(query)
        return self._hybrid_search(query)

    def _embed_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        return response.data[0].embedding

    def _dense_search(self, query: str, k: int = None) -> list[dict]:
        k = k or self.top_k
        embedding = self._embed_query(query)
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT id, source, chunk_index, content, metadata,
                            1 - (embedding <=> :emb::vector) AS score
                    FROM documents
                    ORDER BY embedding <=> :emb::vector
                """),
                {"emb": str(embedding), "k":k},
                ).fetchall()
            return [dict(r.mapping) for r in rows]

    def _bm25_search(self, query: str, k: int = None) -> list[dict]:
        k = k or self.top_k
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT id, source, chunk_index, content, metadata FROM documents")
            ).fetchall()
        docs = [dict(r._mapping) for r in rows]
        tokenized_corpus = [d["content"] for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for idx in top_indices:
            doc = docs[idx].copy()
            doc["score"] = float(scores[idx])
            results.append(doc)
        return results

    def _hybrid_search(self, query: str) -> list[dict]:
        """Reciprocal Rank Fusion of dense + BM25."""
        dense = self._dense_search(query, k=self.top_k * 2)
        bm25 = self._bm25_search(query, k=self.top_k * 2)

        rrf_scores: dict[int, float] = {}
        for rank, doc in enumerate(dense):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0) + 1 / (60 + rank + 1)
        for rank, doc in enumerate(bm25):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0) + 1 / (60 + rank + 1)

        # merge doc metadata
        all_docs = {d["id"]: d for d in dense + bm25}
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        return [
            {**all_docs[doc_id], "rrf_score": score}
            for doc_id, score in ranked
            if doc_id in all_docs
        ]


