"""Hybrid retrieval with reciprocal rank fusion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docsense.retrieval.dense import RetrievalResult

if TYPE_CHECKING:
    from docsense.chunking.base import Chunk
    from docsense.config import RetrievalConfig
    from docsense.embedding.embedder import Embedder
    from docsense.retrieval.dense import DenseRetriever
    from docsense.retrieval.sparse import SparseRetriever


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    weights: list[float],
    k: int = 60,
) -> list[RetrievalResult]:
    """Fuse multiple ranked lists using weighted reciprocal rank fusion.

    RRF score = sum(weight_i / (k + rank_i)) for each list where the chunk appears.
    """
    chunk_scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for results, weight in zip(result_lists, weights, strict=True):
        for rank, result in enumerate(results):
            cid = result.chunk.chunk_id
            chunk_map[cid] = result.chunk
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + weight / (k + rank + 1)

    fused = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    return [RetrievalResult(chunk=chunk_map[cid], score=score) for cid, score in fused]


class HybridRetriever:
    """Combines dense and sparse retrieval with reciprocal rank fusion."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        embedder: Embedder,
        config: RetrievalConfig,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.embedder = embedder
        self.config = config

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or self.config.top_k

        query_embedding = self.embedder.embed_query(query)
        dense_results = self.dense.search(query_embedding, top_k=top_k)
        sparse_results = self.sparse.search(query, top_k=top_k)

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            weights=[self.config.dense_weight, self.config.sparse_weight],
        )
        return fused[:top_k]
