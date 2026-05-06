"""Unit tests for rule-based evals.

The Answer fixtures use a single retrieved chunk so citation-marker
range checks have a stable upper bound. Refusal-pattern coverage is
parametrized over a handful of phrasings the LLM might produce.
"""

from __future__ import annotations

import pytest

from docsense.evaluation.rule_based import (
    check_citations_grounded,
    check_no_answer_behavior,
)
from docsense.generation.types import Answer, ChunkRef, Citation, GenerationMetadata


def _make_answer(text: str, n_chunks: int = 3, citations: list[int] | None = None) -> Answer:
    """Build a synthetic Answer with N retrieved chunks (1-indexed)."""
    retrieved = [
        ChunkRef(doc_id=f"doc{i}.md", chunk_id=str(i), score=1.0 - i * 0.01, text=f"chunk {i}")
        for i in range(1, n_chunks + 1)
    ]
    cits = [Citation(doc_id=f"doc{idx}.md", chunk_id=str(idx)) for idx in (citations or [])]
    return Answer(
        text=text,
        citations=cits,
        retrieved_chunks=retrieved,
        metadata=GenerationMetadata(model_name="test", latency_ms=1.0),
    )


class TestCheckNoAnswerBehavior:
    @pytest.mark.parametrize(
        "text",
        [
            "I don't know the answer to that question.",
            "I cannot determine this from the provided context.",
            "The provided context doesn't contain relevant information.",
            "There is no relevant information in the context to answer this.",
            "I'm unable to answer that based on the given context.",
            "I apologize, but I'm unable to find that in the documentation.",
            "Insufficient context to provide a meaningful answer.",
            # Discovered during first Block 1B eval — all 8 no-answer
            # queries opened with this exact phrasing but went undetected
            # because the "have" verb wasn't in the negative-modal verb
            # list. The dont_have_context pattern fixes this.
            "I don't have enough context to answer that.",
            "I do not have the information needed.",
            "The model doesn't have sufficient context for that question.",
        ],
    )
    def test_recognizes_common_refusals(self, text: str):
        answer = _make_answer(text)
        result = check_no_answer_behavior(answer, expected_refusal=True)
        assert result.refused is True
        assert result.correct is True
        assert result.matched_pattern is not None

    def test_normal_answer_does_not_match(self):
        answer = _make_answer(
            "To install transformers, run `pip install transformers` in your terminal."
        )
        result = check_no_answer_behavior(answer, expected_refusal=False)
        assert result.refused is False
        assert result.correct is True
        assert result.matched_pattern is None

    def test_refusal_when_not_expected_marked_incorrect(self):
        """An over-eager refusal on an in-corpus question is a real
        failure mode worth flagging — not just a mirror of the happy
        path. expected_refusal=False but we got a refusal: incorrect."""
        answer = _make_answer("I don't know how to install transformers.")
        result = check_no_answer_behavior(answer, expected_refusal=False)
        assert result.refused is True
        assert result.correct is False

    def test_no_refusal_when_expected_marked_incorrect(self):
        """The other failure mode: model confabulated an answer to an
        off-corpus question instead of refusing."""
        answer = _make_answer("AWS Lambda's free tier includes 1 million requests per month.")
        result = check_no_answer_behavior(answer, expected_refusal=True)
        assert result.refused is False
        assert result.correct is False


class TestCheckCitationsGrounded:
    def test_no_markers_in_text(self):
        """The smoke-run scenario: model produced a clean answer but
        ignored the cite-by-[N] directive."""
        answer = _make_answer("To install transformers, run pip install.")
        result = check_citations_grounded(answer)
        assert result.n_markers_in_text == 0
        assert result.n_resolved_citations == 0
        assert result.any_marker_present is False
        assert result.all_markers_in_range is False

    def test_all_markers_in_range(self):
        answer = _make_answer(
            "First do this [1] then do that [2].",
            n_chunks=3,
            citations=[1, 2],
        )
        result = check_citations_grounded(answer)
        assert result.n_markers_in_text == 2
        assert result.n_resolved_citations == 2
        assert result.all_markers_in_range is True
        assert result.any_marker_present is True

    def test_duplicate_marker_counted_once_in_resolved(self):
        """parse_citations dedupes [1] [1] to one Citation, but the
        text still contains two markers. The check surfaces that
        discrepancy."""
        answer = _make_answer(
            "See [1] and also [1] for details.",
            n_chunks=3,
            citations=[1],
        )
        result = check_citations_grounded(answer)
        assert result.n_markers_in_text == 2
        assert result.n_resolved_citations == 1
        assert result.all_markers_in_range is True

    def test_out_of_range_marker_flagged(self):
        """Model cited [7] but only 3 chunks were retrieved.
        parse_citations drops it, but the raw marker count records it
        and all_markers_in_range goes False."""
        answer = _make_answer(
            "First [1] then later [7].",
            n_chunks=3,
            citations=[1],
        )
        result = check_citations_grounded(answer)
        assert result.n_markers_in_text == 2
        assert result.n_resolved_citations == 1
        assert result.all_markers_in_range is False
        assert result.any_marker_present is True
