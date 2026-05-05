"""End-to-end test of HybridRetriever wired with CrossEncoderReranker.

Composes real DenseRetriever (FAISS), SparseRetriever (BM25), and
CrossEncoderReranker with the cross-encoder model mocked. The Embedder is
also mocked since it's a thin sentence-transformers wrapper that's
extensively unit-tested elsewhere.

These tests verify the *wiring* — that the right candidate counts flow
between stages, that cross-encoder scores replace fusion scores, that
top_k slicing happens at the right point. Not a behavioral test of the
underlying models.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from docsense.chunking.base import Chunk
from docsense.config import RerankingConfig, RetrievalConfig
from docsense.reranking.reranker import CrossEncoderReranker
from docsense.retrieval.dense import DenseRetriever
from docsense.retrieval.hybrid import HybridRetriever
from docsense.retrieval.sparse import SparseRetriever


def _make_chunks(doc_ids: list[str]) -> list[Chunk]:
    return [Chunk(text=f"content for {d}", doc_id=d, chunk_index=0) for d in doc_ids]


def _stub_embedder(dim: int = 4) -> MagicMock:
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


def _make_reranker(scores: list[float]) -> CrossEncoderReranker:
    """A real CrossEncoderReranker with predict mocked.

    `scores` must be at least as long as the largest pair list the test
    will hand the reranker. The mock truncates to the actual call length
    so the reranker's zip(..., strict=True) doesn't trip.
    """
    config = RerankingConfig(model_name="dummy", device="cpu", batch_size=10)
    reranker = CrossEncoderReranker(config)
    mock_model = MagicMock()

    def predict(pairs: list, **_: object) -> np.ndarray:
        return np.array(scores[: len(pairs)], dtype=np.float32)

    mock_model.predict.side_effect = predict
    reranker._model = mock_model
    return reranker


class TestHybridRerankIntegration:
    def test_reranker_invoked_with_candidate_pool(self):
        """When wired, the reranker receives the fused candidate list (up to
        rerank_candidates) — not just the final top_k."""
        chunks = _make_chunks([f"d{i}" for i in range(10)])
        dense = _build_dense_retriever(chunks)
        sparse = SparseRetriever()
        sparse.add(chunks)
        embedder = _stub_embedder()
        config = RetrievalConfig(top_k=3, rerank_candidates=8)
        # Provide enough scores for any plausible candidate count
        reranker = _make_reranker([0.1] * 10)

        retriever = HybridRetriever(dense, sparse, embedder, config, reranker=reranker)
        retriever.search("content")

        # The reranker's predict was called once with the fused candidate pairs
        predict_call = reranker._model.predict.call_args  # type: ignore[union-attr]
        passed_pairs = predict_call.args[0]
        # candidate_k = max(rerank_candidates=8, top_k=3) = 8; capped at how many
        # unique fused chunks exist (10 here, so 8 are reranker input)
        assert len(passed_pairs) == 8

    def test_final_result_count_is_top_k_not_candidate_k(self):
        """Output must be sliced to top_k after reranking, not candidate_k."""
        chunks = _make_chunks([f"d{i}" for i in range(10)])
        dense = _build_dense_retriever(chunks)
        sparse = SparseRetriever()
        sparse.add(chunks)
        embedder = _stub_embedder()
        config = RetrievalConfig(top_k=3, rerank_candidates=8)
        reranker = _make_reranker([0.1] * 10)

        retriever = HybridRetriever(dense, sparse, embedder, config, reranker=reranker)
        results = retriever.search("content")

        assert len(results) == 3

    def test_cross_encoder_scores_replace_fusion_scores(self):
        """The scores in the returned RetrievalResults must be the
        cross-encoder's scores, not the RRF fusion scores."""
        chunks = _make_chunks(["d1", "d2", "d3", "d4"])
        dense = _build_dense_retriever(chunks)
        sparse = SparseRetriever()
        sparse.add(chunks)
        embedder = _stub_embedder()
        config = RetrievalConfig(top_k=2, rerank_candidates=4)
        # Cross-encoder gives high score to d3 and d1 specifically
        reranker = _make_reranker([0.9, 0.2, 0.95, 0.1])

        retriever = HybridRetriever(dense, sparse, embedder, config, reranker=reranker)
        results = retriever.search("content")

        # All RRF scores would have been < 0.05 (k=60 in RRF). If we see
        # scores in the [0.9, 1.0] range, they're cross-encoder scores.
        assert all(r.score >= 0.5 for r in results)
        # And the highest-scored result reflects the cross-encoder ranking
        top_score = max(r.score for r in results)
        assert 0.9 <= top_score <= 1.0

    def test_top_k_override_propagates_through_rerank(self):
        """A caller passing top_k= explicitly must override config.top_k both
        for what's returned and for how the reranker is invoked."""
        chunks = _make_chunks([f"d{i}" for i in range(10)])
        dense = _build_dense_retriever(chunks)
        sparse = SparseRetriever()
        sparse.add(chunks)
        embedder = _stub_embedder()
        config = RetrievalConfig(top_k=5, rerank_candidates=8)
        reranker = _make_reranker([0.1] * 10)

        retriever = HybridRetriever(dense, sparse, embedder, config, reranker=reranker)
        results = retriever.search("content", top_k=2)

        assert len(results) == 2

    def test_top_k_above_rerank_candidates_floors_to_top_k(self):
        """When the caller's top_k exceeds config.rerank_candidates, the
        candidate pool must scale up to at least top_k so the reranker has
        enough items to choose from."""
        chunks = _make_chunks([f"d{i}" for i in range(15)])
        dense = _build_dense_retriever(chunks)
        sparse = SparseRetriever()
        sparse.add(chunks)
        embedder = _stub_embedder()
        # rerank_candidates=8 but caller asks for 12
        config = RetrievalConfig(top_k=5, rerank_candidates=8)
        reranker = _make_reranker([0.1] * 15)

        retriever = HybridRetriever(dense, sparse, embedder, config, reranker=reranker)
        results = retriever.search("content", top_k=12)

        # Reranker saw at least 12 candidates
        passed_pairs = reranker._model.predict.call_args.args[0]  # type: ignore[union-attr]
        assert len(passed_pairs) >= 12
        assert len(results) == 12

    def test_without_reranker_behavior_unchanged(self):
        """Sanity check: HybridRetriever(reranker=None) preserves the
        pre-reranker behavior. RRF scores survive, candidate_k = top_k."""
        chunks = _make_chunks([f"d{i}" for i in range(10)])
        dense = _build_dense_retriever(chunks)
        sparse = SparseRetriever()
        sparse.add(chunks)
        embedder = _stub_embedder()
        config = RetrievalConfig(top_k=3, rerank_candidates=8)

        # Note: no reranker passed
        retriever = HybridRetriever(dense, sparse, embedder, config)
        results = retriever.search("content")

        assert len(results) == 3
        # RRF scores are ≪ 1: weight_dense / (60 + rank+1) max is ~0.01.
        # With weights summing to 1, total max is ~0.0166.
        assert all(r.score < 0.1 for r in results)
