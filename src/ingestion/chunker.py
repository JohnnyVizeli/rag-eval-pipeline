# src/ingestion/chunker.py
import tiktoken
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sympy.printing.llvmjitcode import current_link_suffix
from torch.backends.opt_einsum import strategy

from .loader import RawDocument

@dataclass
class Chunk:
    source: str
    chunk_index: int
    content: str
    metadata: dict
    token_count: int

class Chunker:
    """
    Two strategies:
        recursive - LangChain RecursiveCharacterTextSplitter (default)
        semantic - split on sentence boundaries, merge to target token count
    """

    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        model: str = "text-embedding-3-small",
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.encoding_for_model("gpt-4o")

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_len,
            separators=["\n\n", "\n", ".",","""],
        )
    def _token_len(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        if self.strategy == "recursive":
            return self._recursive_chunk(doc)
        elif self.strategy == "semantic":
            return self._semantic_chunk(doc)
        raise ValueError(f"Unknown strategy {self.strategy}")

    def _recursive_chunk(self, doc: RawDocument) -> list[Chunk]:
        texts = self._splitter.split_text(doc.content)
        return [
            Chunk(
                source=doc.source,
                chunk_index=i,
                content=text,
                metadata={**doc.metadata, "strategy": "recursive"},
                token_count=self._token_len(text),
            )
            for i, text in enumerate(texts)
            if text.strip()
        ]
    def _semantic_chunk(self, doc: RawDocument) -> list[Chunk]:
        """Merge sentences until we hit target token count"""

        import re
        sentences = re.split(r'(?<=[.!?])\s+', doc.content)
        chunks, current, current_tokens = [], [], 0

        for sentence in sentences:
            stokens = self._token_len(sentence)
            if current_tokens + stokens > self.chunk_size and current:
                text = " ".join(current)
                chunks.append(Chunk(
                    source=doc.source,
                    chunk_index=len(chunks),
                    content=text,
                    metadata={**doc.metadata, "strategy": "semantic"},
                    token_count=self._token_len(text),
                ))
                # overlap: keep last sentence
                current = current[-1:] + [sentence]
                current_tokens = sum(self._token_len(s) for s in current)
            else:
                current.append(sentence)
                current_tokens += stokens

        if current:
            text = " ".join(current)
            chunks.append(Chunk(
                source=doc.source,
                chunk_index=len(chunks),
                content=text,
                metadata={**doc.metadata, "strategy": "semantic"},
                token_count=self._token_len(text),
            ))
        return chunks





