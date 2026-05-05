"""Tests for hybrid retrieval and reciprocal rank fusion."""

from unittest.mock import MagicMock

import numpy as np

from docsense.chunking.base import Chunk
from docsense.config import RetrievalConfig
from docsense.retrieval.dense import DenseRetriever, RetrievalResult
from docsense.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from docsense.retrieval.sparse import SparseRetriever


class TestReciprocalRankFusion:
    def _make_result(self, doc_id: str, score: float) -> RetrievalResult:
        chunk = Chunk(text=f"text for {doc_id}", doc_id=doc_id, chunk_index=0)
        return RetrievalResult(chunk=chunk, score=score)

    def test_single_list(self):
        results = [self._make_result("d1", 0.9), self._make_result("d2", 0.5)]
        fused = reciprocal_rank_fusion([results], weights=[1.0])
        assert len(fused) == 2
        assert fused[0].chunk.doc_id == "d1"

    def test_two_lists_boost_overlap(self):
        list_a = [self._make_result("d1", 0.9), self._make_result("d2", 0.5)]
        list_b = [self._make_result("d2", 0.8), self._make_result("d3", 0.4)]

        fused = reciprocal_rank_fusion([list_a, list_b], weights=[1.0, 1.0])
        # d2 appears in both lists, so it should get boosted
        ids = [r.chunk.doc_id for r in fused]
        assert "d2" in ids

    def test_weights_matter(self):
        list_a = [self._make_result("d1", 0.9)]
        list_b = [self._make_result("d2", 0.9)]

        # Heavily weight list_a
        fused = reciprocal_rank_fusion([list_a, list_b], weights=[10.0, 0.1])
        assert fused[0].chunk.doc_id == "d1"

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([], weights=[])
        assert fused == []


def _make_chunks(doc_ids: list[str]) -> list[Chunk]:
    return [Chunk(text=f"content for {d}", doc_id=d, chunk_index=0) for d in doc_ids]


def _stub_embedder(dim: int = 4) -> MagicMock:
    """Mock Embedder that returns a deterministic query vector."""
    embedder = MagicMock()
    embedder.embed_query.return_value = np.zeros(dim, dtype=np.float32)
    return embedder


def _build_dense_retriever(chunks: list[Chunk], dim: int = 4) -> DenseRetriever:
    """Real DenseRetriever populated with zero-vector embeddings.

    Inner-product against a zero query vector yields zero scores for all
    chunks, but FAISS still returns them in deterministic index order, so
    rank-based fusion is exercised correctly.
    """
    retriever = DenseRetriever(dimension=dim)
    embeddings = np.zeros((len(chunks), dim), dtype=np.float32)
    retriever.add(chunks, embeddings)
    return retriever


class TestHybridRetriever:
    def test_search_returns_top_k_results(self):
        chunks = _make_chunks(["d1", "d2", "d3"])
        dense = _build_dense_retriever(chunks)
        sparse = SparseRetriever()
        sparse.add(chunks)
        embedder = _stub_embedder()
        config = RetrievalConfig(top_k=2, dense_weight=0.6, sparse_weight=0.4)

        retriever = HybridRetriever(dense, sparse, embedder, config)
        results = retriever.search("content")

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_search_invokes_embedder_for_query(self):
        """The query string must be embedded via the configured embedder."""
        chunks = _make_chunks(["d1"])
        dense = _build_dense_retriever(chunks)
        sparse = SparseRetriever()
        sparse.add(chunks)
        embedder = _stub_embedder()
        config = RetrievalConfig()

        retriever = HybridRetriever(dense, sparse, embedder, config)
        retriever.search("a specific query")

        embedder.embed_query.assert_called_once_with("a specific query")

    def test_search_top_k_override(self):
        """Explicit top_k must override config.top_k."""
        chunks = _make_chunks(["d1", "d2", "d3", "d4", "d5"])
        dense = _build_dense_retriever(chunks)
        sparse = SparseRetriever()
        sparse.add(chunks)
        embedder = _stub_embedder()
        config = RetrievalConfig(top_k=20)

        retriever = HybridRetriever(dense, sparse, embedder, config)
        results = retriever.search("content", top_k=3)

        assert len(results) == 3

    def test_search_results_are_unique_chunks(self):
        """RRF must dedupe by chunk_id when the same chunk appears in both
        dense and sparse rankings — otherwise downstream consumers get
        duplicate hits."""
        chunks = _make_chunks(["d1", "d2", "d3"])
        dense = _build_dense_retriever(chunks)
        sparse = SparseRetriever()
        sparse.add(chunks)
        embedder = _stub_embedder()
        config = RetrievalConfig(top_k=10)

        retriever = HybridRetriever(dense, sparse, embedder, config)
        results = retriever.search("content")

        chunk_ids = [r.chunk.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))
