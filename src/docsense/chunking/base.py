"""Base interface for chunking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from docsense.ingestion.loader import Document


@dataclass
class Chunk:
    """A chunk of text derived from a document."""

    text: str
    doc_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}::chunk_{self.chunk_index}"


class ChunkingStrategy(ABC):
    """Base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        ...

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks
