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
    """BM25-based sparse retrieval.

    Note on incremental updates: rank_bm25's ``BM25Okapi`` doesn't support
    incremental updates — adding a single chunk requires recomputing the
    full IDF table over the entire corpus. We cache per-chunk tokens so
    repeated ``add()`` calls don't re-tokenize the existing corpus, but the
    ``BM25Okapi`` index itself is rebuilt on every ``add()``. For one-shot
    indexing this is fine; for high-frequency incremental updates a
    different sparse-retrieval library would be needed.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._tokenized.extend(_tokenize(c.text) for c in chunks)
        self._bm25 = BM25Okapi(self._tokenized)

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
