"""Tests for the CrossEncoderReranker wrapper.

The actual cross-encoder is mocked. We're verifying the rerank logic —
sort-by-score, top_k slicing, empty-input short-circuit, and the
score-to-RetrievalResult mapping. The model itself is sentence-transformers'
problem, not ours.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from docsense.chunking.base import Chunk
from docsense.config import RerankingConfig
from docsense.reranking.reranker import CrossEncoderReranker
from docsense.retrieval.dense import RetrievalResult


def _make_results(*texts_and_scores: tuple[str, float]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=Chunk(text=text, doc_id=f"d{i}", chunk_index=0),
            score=score,
        )
        for i, (text, score) in enumerate(texts_and_scores)
    ]


def _make_mock_model_with_scores(scores: list[float]) -> MagicMock:
    """Mock CrossEncoder whose predict() returns the given scores."""
    model = MagicMock()
    model.predict.return_value = np.array(scores, dtype=np.float32)
    return model


class TestCrossEncoderReranker:
    def test_lazy_model_load(self):
        config = RerankingConfig(model_name="dummy", device="cpu")
        reranker = CrossEncoderReranker(config)
        assert reranker._model is None

    def test_empty_results_short_circuit(self):
        """Reranking an empty list must not invoke the model at all."""
        config = RerankingConfig(model_name="dummy", device="cpu")
        reranker = CrossEncoderReranker(config)
        mock_model = _make_mock_model_with_scores([])
        reranker._model = mock_model

        out = reranker.rerank("any query", [])

        assert out == []
        mock_model.predict.assert_not_called()

    def test_reorders_by_cross_encoder_score(self):
        """Initial order should be replaced by the cross-encoder's ranking."""
        config = RerankingConfig(model_name="dummy", device="cpu", batch_size=10)
        reranker = CrossEncoderReranker(config)
        # Initial order: A first by retrieval score, but cross-encoder
        # ranks C > A > B.
        results = _make_results(
            ("text A", 0.9),
            ("text B", 0.8),
            ("text C", 0.7),
        )
        reranker._model = _make_mock_model_with_scores([0.5, 0.1, 0.95])

        out = reranker.rerank("query", results)

        assert [r.chunk.text for r in out] == ["text C", "text A", "text B"]

    def test_returned_scores_are_cross_encoder_scores(self):
        """Output scores must be the cross-encoder scores, not the originals."""
        config = RerankingConfig(model_name="dummy", device="cpu", batch_size=10)
        reranker = CrossEncoderReranker(config)
        results = _make_results(("a", 0.5), ("b", 0.5))
        reranker._model = _make_mock_model_with_scores([0.7, 0.3])

        out = reranker.rerank("query", results)

        assert out[0].score == pytest.approx(0.7)
        assert out[1].score == pytest.approx(0.3)

    def test_top_k_slices_output(self):
        """Explicit top_k must trim the reranked list."""
        config = RerankingConfig(model_name="dummy", device="cpu", batch_size=10)
        reranker = CrossEncoderReranker(config)
        results = _make_results(("a", 0.0), ("b", 0.0), ("c", 0.0), ("d", 0.0))
        reranker._model = _make_mock_model_with_scores([0.1, 0.4, 0.2, 0.3])

        out = reranker.rerank("query", results, top_k=2)

        assert len(out) == 2
        assert [r.chunk.text for r in out] == ["b", "d"]

    def test_top_k_defaults_to_batch_size_when_none(self):
        """When top_k is None the reranker uses config.batch_size."""
        config = RerankingConfig(model_name="dummy", device="cpu", batch_size=2)
        reranker = CrossEncoderReranker(config)
        results = _make_results(("a", 0.0), ("b", 0.0), ("c", 0.0), ("d", 0.0))
        reranker._model = _make_mock_model_with_scores([0.1, 0.4, 0.2, 0.3])

        out = reranker.rerank("query", results)

        assert len(out) == 2

    def test_predict_receives_query_chunk_pairs(self):
        """The cross-encoder must see (query, chunk_text) pairs as input."""
        config = RerankingConfig(model_name="dummy", device="cpu", batch_size=10)
        reranker = CrossEncoderReranker(config)
        results = _make_results(("first", 0.0), ("second", 0.0))
        mock_model = _make_mock_model_with_scores([0.0, 0.0])
        reranker._model = mock_model

        reranker.rerank("the query", results)

        passed_pairs = mock_model.predict.call_args.args[0]
        assert passed_pairs == [("the query", "first"), ("the query", "second")]

    def test_chunks_are_preserved(self):
        """Reranked output should keep the same Chunk objects, not copies."""
        config = RerankingConfig(model_name="dummy", device="cpu", batch_size=10)
        reranker = CrossEncoderReranker(config)
        results = _make_results(("a", 0.0), ("b", 0.0))
        original_chunks = [r.chunk for r in results]
        reranker._model = _make_mock_model_with_scores([0.3, 0.7])

        out = reranker.rerank("query", results)

        assert out[0].chunk is original_chunks[1]
        assert out[1].chunk is original_chunks[0]

    def test_model_property_constructs_with_config_values(self):
        config = RerankingConfig(model_name="some/cross-encoder", device="cuda")
        reranker = CrossEncoderReranker(config)

        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_ce.return_value = _make_mock_model_with_scores([])
            _ = reranker.model

            mock_ce.assert_called_once_with("some/cross-encoder", device="cuda")
