# src/ingestion/loader.py
import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

logger = logging.getLogger(__name__)

@dataclass
class RawDocument:
    source: str
    content: str
    metadata: dict

class DocumentLoader:
    """Load raw documents from a directory of .txt/.pdf/.md files."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load_all(self) -> list[RawDocument]:
        docs = []
        for path in self.data_dir.rglob("*"):
            if path.suffix in {".txt", ".md"}:
                docs.append(self._load_text(path))
            elif path.suffix == ".pdf":
                docs.append(self._load_pdf(path))
        logger.info(f"Loaded {len(docs)} documents from {self.data_dir}")
        return docs


