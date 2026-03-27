"""Tests for retrieval evaluation metrics."""

import pytest

from docsense.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_no_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"d", "e"}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        assert precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0

    def test_k_less_than_retrieved(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, k=1) == 1.0
        assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_partial_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "d"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["a"], set(), k=1) == 0.0


class TestMRR:
    def test_first_is_relevant(self):
        assert mean_reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_second_is_relevant(self):
        assert mean_reciprocal_rank(["a", "b"], {"b"}) == 0.5

    def test_none_relevant(self):
        assert mean_reciprocal_rank(["a", "b"], {"c"}) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, k=2) == 1.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a"], set(), k=1) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k(["a"], {"a"}, k=0) == 0.0

    def test_no_relevant_retrieved(self):
        assert ndcg_at_k(["a", "b"], {"c"}, k=2) == 0.0
