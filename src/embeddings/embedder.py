# src/embeddings/embedder.py
import os
import logging
import time
from openai import OpenAI
from sqlalchemy import create_engine, text
from ..ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.engine = create_engine(os.environ["DATABASE_URL"])

    def embed_and_store(self, chunks: list[Chunk]) -> dict:
        """Embed chunks in batches of 100, upsert to pgvector."""
        BATCH_SIZE = 100
        total_tokens = 0
        total_cost = 0.0
        inserted = 0

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [c.content for c in batch]

            response = self.client.embeddings.create(
                model = self.model, input=texts)
            total_tokens += response.usage.total_tokens
            # text-embedding-3-small: $0.02 / 1M tokens
            total_cost += response.usage.total_tokens * 0.00000002

            embeddings = [e.embedding for e in response.data]

            with self.engine.begni() as conn:
                for chunk, embedding in zip(batch, embeddings):
                    conn.execute(
                        text("""
                        INSERT INTO documents
                            (source, chunk_index, content, metadata, embedding)
                        VALUES
                            (:source, :chunk_index, :content,
                             :metadata::jsonb, :embedding)
                        ON CONFLICT DO NOTHING
                        """),
                        {
                            "source": chunk.source,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                            "metadata": str(chunk.metadata).replace("'", '"'),
                            "embedding": str(embedding),
                        },
                    )
                    inserted += 1
                logger.info(f"Embedded batch {i // BATCH_SIZE + 1}, "
                            f"{inserted} chunks total, cost so far: ${total_cost:.4f}")
                time.sleep(0.1) # respect rate limits
            return {"inserted": inserted, "tokens": total_tokens, "cost_usd": total_cost}