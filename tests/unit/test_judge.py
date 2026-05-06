"""Unit tests for the LLMJudge ABC and JudgeScore model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from docsense.evaluation.judge import JudgeScore, LLMJudge


class MockJudge(LLMJudge):
    """Test double that returns whatever scores the test set on it."""

    def __init__(
        self,
        faith_score: float = 1.0,
        rel_score: float = 1.0,
        rationale: str = "mock",
    ) -> None:
        self.faith_score = faith_score
        self.rel_score = rel_score
        self.rationale = rationale
        self.faith_calls: list[tuple[str, str, str]] = []
        self.rel_calls: list[tuple[str, str]] = []

    def judge_faithfulness(self, question: str, context: str, answer: str) -> JudgeScore:
        self.faith_calls.append((question, context, answer))
        return JudgeScore(metric="faithfulness", score=self.faith_score, rationale=self.rationale)

    def judge_relevance(self, question: str, answer: str) -> JudgeScore:
        self.rel_calls.append((question, answer))
        return JudgeScore(metric="relevance", score=self.rel_score, rationale=self.rationale)


class TestJudgeScore:
    def test_valid_score_constructs(self):
        score = JudgeScore(metric="faithfulness", score=0.75, rationale="ok")
        assert score.score == 0.75
        assert score.metric == "faithfulness"

    def test_score_above_one_rejected(self):
        with pytest.raises(ValidationError):
            JudgeScore(metric="faithfulness", score=1.1, rationale="bad")

    def test_score_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            JudgeScore(metric="faithfulness", score=-0.1, rationale="bad")

    def test_invalid_metric_rejected(self):
        with pytest.raises(ValidationError):
            JudgeScore(metric="not_a_metric", score=0.5, rationale="bad")  # type: ignore[arg-type]

    def test_anchor_values_all_valid(self):
        """All five anchor values must construct cleanly. Sanity check
        that the [0, 1] bound includes the endpoints."""
        for anchor in (0.0, 0.25, 0.5, 0.75, 1.0):
            JudgeScore(metric="relevance", score=anchor, rationale="ok")


class TestLLMJudgeABC:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            LLMJudge()  # type: ignore[abstract]

    def test_mock_subclass_satisfies_interface(self):
        judge = MockJudge(faith_score=0.5, rel_score=0.75)
        f = judge.judge_faithfulness("q", "ctx", "ans")
        r = judge.judge_relevance("q", "ans")
        assert f.score == 0.5
        assert f.metric == "faithfulness"
        assert r.score == 0.75
        assert r.metric == "relevance"

    def test_mock_records_calls(self):
        """Confirms the test double's call-recording works — used by
        downstream tests that need to assert what was sent to the judge."""
        judge = MockJudge()
        judge.judge_faithfulness("q1", "ctx1", "a1")
        judge.judge_relevance("q1", "a1")
        assert judge.faith_calls == [("q1", "ctx1", "a1")]
        assert judge.rel_calls == [("q1", "a1")]
