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

    def _load_text(self, path: Path) -> RawDocument:
        content = path.read_text(encoding="utf-8", errors="ignore")
        return RawDocument(
            source=str(path.relative_to(self.data_dir)),
            content=content,
            metadata={"file_type": path.suffix, "size_bytes": path.stat().st_size},
        )

    def _load_pdf(self, path: Path) -> RawDocument:
        # pip install pypdf
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        content = "\n\n".join(page.extractText() for page in reader.pages)
        return RawDocument(
            source=str(path.relative_to(self.data_dir)),
            content=content,
            metadata={"file_type": ".pdf", "num_pages": len(reader.pages)},
        )


