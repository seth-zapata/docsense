"""Sparse retrieval using BM25."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from docsense.retrieval.dense import RetrievalResult

if TYPE_CHECKING:
    from docsense.chunking.base import Chunk


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class SparseRetriever:
    """BM25-based sparse retrieval."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        corpus = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        if self._bm25 is None or not self.chunks:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=self.chunks[idx], score=float(score))
            for idx, score in ranked
            if score > 0
        ]

    @property
    def size(self) -> int:
        return len(self.chunks)
