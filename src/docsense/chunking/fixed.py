"""Fixed-size chunking strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docsense.chunking.base import Chunk, ChunkingStrategy

if TYPE_CHECKING:
    from docsense.ingestion.loader import Document


class FixedSizeChunker(ChunkingStrategy):
    """Split documents into fixed-size character chunks with overlap.

    Breaks on whitespace boundaries to avoid cutting mid-word.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                # Find the last whitespace before the cutoff
                break_at = text.rfind(" ", start, end)
                if break_at > start:
                    end = break_at

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        doc_id=document.doc_id,
                        chunk_index=len(chunks),
                        metadata={"strategy": "fixed", "char_start": start},
                    )
                )

            # Advance by chunk_size minus overlap
            start = end
            if self.chunk_overlap > 0 and start < len(text):
                start = max(start - self.chunk_overlap, chunks[-1].metadata["char_start"] + 1)

        return chunks
