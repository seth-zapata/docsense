"""Recursive text splitting chunking strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docsense.chunking.base import Chunk, ChunkingStrategy

if TYPE_CHECKING:
    from docsense.ingestion.loader import Document

# Separators in order of preference: try the coarsest split first,
# fall back to finer-grained splits if chunks are still too large.
DEFAULT_SEPARATORS = [
    "\n\n",  # paragraph break
    "\n",  # line break
    ". ",  # sentence boundary
    " ",  # word boundary
]


class RecursiveChunker(ChunkingStrategy):
    """Split text recursively using a hierarchy of separators.

    Tries paragraph breaks first, then line breaks, then sentences,
    then words. Produces chunks close to the target size while
    respecting natural text boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        raw_chunks = self._split_recursive(text, self.separators)

        # Merge small fragments and apply overlap
        merged = self._merge_with_overlap(raw_chunks)

        return [
            Chunk(
                text=t,
                doc_id=document.doc_id,
                chunk_index=i,
                metadata={"strategy": "recursive"},
            )
            for i, t in enumerate(merged)
        ]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text, trying coarser separators first."""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # No separators left — hard cut at chunk_size
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        remaining_seps = separators[1:]
        parts = text.split(sep)

        results = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                results.append(part)
            else:
                results.extend(self._split_recursive(part, remaining_seps))

        return results

    def _merge_with_overlap(self, fragments: list[str]) -> list[str]:
        """Merge small fragments up to chunk_size, then apply overlap."""
        if not fragments:
            return []

        merged = []
        current = fragments[0]

        for frag in fragments[1:]:
            combined = current + "\n\n" + frag
            if len(combined) <= self.chunk_size:
                current = combined
            else:
                merged.append(current)
                # Overlap: carry the tail of the previous chunk
                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                    tail = current[-self.chunk_overlap :]
                    # Break at word boundary
                    space_idx = tail.find(" ")
                    if space_idx > 0:
                        tail = tail[space_idx + 1 :]
                    current = tail + "\n\n" + frag
                    if len(current) > self.chunk_size:
                        current = frag
                else:
                    current = frag

        if current:
            merged.append(current)

        return merged
