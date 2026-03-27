"""Dense retrieval using FAISS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import faiss
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from docsense.chunking.base import Chunk


@dataclass
class RetrievalResult:
    """A single retrieval result with score."""

    chunk: Chunk
    score: float


class DenseRetriever:
    """FAISS-backed dense vector retrieval."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: NDArray[np.float32]) -> None:
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def search(
        self, query_embedding: NDArray[np.float32], top_k: int = 20
    ) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:
                continue
            results.append(RetrievalResult(chunk=self.chunks[idx], score=float(score)))
        return results

    @property
    def size(self) -> int:
        return self.index.ntotal
