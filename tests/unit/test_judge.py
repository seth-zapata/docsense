"""Unit tests for the LLMJudge ABC and JudgeScore + ClaimAttribution models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from docsense.evaluation.judge import (
    ClaimAttribution,
    JudgeScore,
    LLMJudge,
    RefusalJudgment,
)
from docsense.generation.types import ChunkRef


class MockJudge(LLMJudge):
    """Test double that returns whatever scores the test set on it."""

    def __init__(
        self,
        faith_score: float = 1.0,
        rel_score: float = 1.0,
        refused: bool = True,
        rationale: str = "mock",
    ) -> None:
        self.faith_score = faith_score
        self.rel_score = rel_score
        self.refused = refused
        self.rationale = rationale
        self.faith_calls: list[tuple[str, list[ChunkRef], str]] = []
        self.rel_calls: list[tuple[str, str]] = []
        self.refusal_calls: list[tuple[str, str]] = []

    def judge_faithfulness(self, question: str, chunks: list[ChunkRef], answer: str) -> JudgeScore:
        self.faith_calls.append((question, chunks, answer))
        return JudgeScore(metric="faithfulness", score=self.faith_score, rationale=self.rationale)

    def judge_relevance(self, question: str, answer: str) -> JudgeScore:
        self.rel_calls.append((question, answer))
        return JudgeScore(metric="relevance", score=self.rel_score, rationale=self.rationale)

    def judge_refusal(self, question: str, answer: str) -> RefusalJudgment:
        self.refusal_calls.append((question, answer))
        return RefusalJudgment(refused=self.refused, rationale=self.rationale)


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

    def test_claim_attributions_empty_by_default(self):
        """JudgeScore for relevance (or any non-claim-level metric) has
        an empty claim_attributions list."""
        score = JudgeScore(metric="relevance", score=0.75, rationale="ok")
        assert score.claim_attributions == []

    def test_claim_attributions_round_trip(self):
        """Faithfulness JudgeScore carries per-claim attribution data
        that survives model_dump / model_validate."""
        attrs = [
            ClaimAttribution(
                claim_idx=1, claim_text="alpha", supporting_chunk_idx=2, rationale="r1"
            ),
            ClaimAttribution(
                claim_idx=2, claim_text="beta", supporting_chunk_idx=None, rationale="r2"
            ),
        ]
        score = JudgeScore(
            metric="faithfulness",
            score=0.5,
            rationale="1 of 2 claims supported",
            claim_attributions=attrs,
        )
        dumped = score.model_dump()
        rehydrated = JudgeScore.model_validate(dumped)
        assert len(rehydrated.claim_attributions) == 2
        assert rehydrated.claim_attributions[0].supporting_chunk_idx == 2
        assert rehydrated.claim_attributions[1].supporting_chunk_idx is None

    def test_duplicate_claim_idx_rejected(self):
        """Pin the validator: duplicate claim_idx values would silently
        double-count in score aggregation if not caught at construction."""
        attrs = [
            ClaimAttribution(claim_idx=1, claim_text="a", supporting_chunk_idx=1),
            ClaimAttribution(claim_idx=1, claim_text="b", supporting_chunk_idx=2),  # duplicate idx
        ]
        with pytest.raises(ValidationError, match="Duplicate claim_idx"):
            JudgeScore(
                metric="faithfulness",
                score=1.0,
                rationale="bad",
                claim_attributions=attrs,
            )


class TestClaimAttribution:
    def test_minimal_construction(self):
        attr = ClaimAttribution(claim_idx=1, claim_text="alpha")
        assert attr.claim_idx == 1
        assert attr.supporting_chunk_idx is None
        assert attr.rationale is None

    def test_supported_construction(self):
        attr = ClaimAttribution(
            claim_idx=3,
            claim_text="beta",
            supporting_chunk_idx=2,
            rationale="chunk 2 says beta",
        )
        assert attr.supporting_chunk_idx == 2
        assert attr.rationale == "chunk 2 says beta"

    def test_zero_claim_idx_rejected(self):
        """claim_idx is 1-indexed; 0 would conflate with no-claim."""
        with pytest.raises(ValidationError):
            ClaimAttribution(claim_idx=0, claim_text="alpha")

    def test_negative_chunk_idx_allowed_via_none(self):
        """We don't accept negative chunk_idx — use None for unsupported."""
        # Negative values aren't explicitly forbidden by the schema, but
        # the parser is responsible for emitting None rather than -1 or 0
        # for "unsupported." Document that contract here.
        attr = ClaimAttribution(claim_idx=1, claim_text="x", supporting_chunk_idx=None)
        assert attr.supporting_chunk_idx is None


class TestLLMJudgeABC:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            LLMJudge()  # type: ignore[abstract]

    def test_mock_subclass_satisfies_interface(self):
        judge = MockJudge(faith_score=0.5, rel_score=0.75, refused=True)
        chunks = [ChunkRef(doc_id="a.md", chunk_id="1", score=1.0, text="alpha")]
        f = judge.judge_faithfulness("q", chunks, "ans")
        r = judge.judge_relevance("q", "ans")
        ref = judge.judge_refusal("q", "ans")
        assert f.score == 0.5
        assert f.metric == "faithfulness"
        assert r.score == 0.75
        assert r.metric == "relevance"
        assert ref.refused is True

    def test_mock_records_calls(self):
        """Confirms the test double's call-recording works — used by
        downstream tests that need to assert what was sent to the judge."""
        judge = MockJudge()
        chunks = [ChunkRef(doc_id="a.md", chunk_id="1", score=1.0, text="alpha")]
        judge.judge_faithfulness("q1", chunks, "a1")
        judge.judge_relevance("q1", "a1")
        judge.judge_refusal("q1", "a1")
        assert judge.faith_calls == [("q1", chunks, "a1")]
        assert judge.rel_calls == [("q1", "a1")]
        assert judge.refusal_calls == [("q1", "a1")]


class TestRefusalJudgment:
    def test_refused_construction(self):
        j = RefusalJudgment(refused=True, rationale="model said don't have context")
        assert j.refused is True
        assert "don't have context" in j.rationale

    def test_not_refused_construction(self):
        j = RefusalJudgment(refused=False, rationale="model attempted to answer")
        assert j.refused is False

    def test_round_trip(self):
        j = RefusalJudgment(refused=True, rationale="r")
        rehydrated = RefusalJudgment.model_validate(j.model_dump())
        assert rehydrated.refused is True
        assert rehydrated.rationale == "r"
