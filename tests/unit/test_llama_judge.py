"""Unit tests for LlamaJudge — parser robustness + integration shape.

The model load is never exercised here. Tests subclass LlamaJudge and
override ``_run_inference`` to inject canned strings, the same pattern
``test_generator.py`` uses.
"""

from __future__ import annotations

import pytest

from docsense.config import JudgeConfig
from docsense.evaluation.llama_judge import (
    LlamaJudge,
    _snap_to_anchor,
    parse_claim_attributions,
    parse_claims,
    parse_relevance_response,
)
from docsense.generation.types import ChunkRef


class StubJudge(LlamaJudge):
    """LlamaJudge that returns a queued sequence of inference responses.

    Each ``_run_inference`` call pops the next queued response. Lets a
    test orchestrate a multi-call flow (e.g., faithfulness = extract +
    attribute → 2 calls) without loading the real model.
    """

    def __init__(self) -> None:
        super().__init__(JudgeConfig())
        self._queue: list[str] = []
        self.calls: list[list[dict[str, str]]] = []
        self.max_new_tokens_seen: list[int | None] = []

    def queue(self, *responses: str) -> None:
        self._queue.extend(responses)

    def _run_inference(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        self.calls.append(messages)
        # Track per-call max_new_tokens override so tests can pin the
        # extraction vs attribution sizing contract.
        self.max_new_tokens_seen.append(max_new_tokens)
        if not self._queue:
            return ""
        return self._queue.pop(0)


def _chunks(*texts: str) -> list[ChunkRef]:
    return [
        ChunkRef(doc_id=f"doc{i}.md", chunk_id=str(i), score=1.0, text=text)
        for i, text in enumerate(texts, start=1)
    ]


# --------------------------------------------------------------------
# Snap-to-anchor (relevance scoring helper)
# --------------------------------------------------------------------


class TestSnapToAnchor:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (0.0, 0.0),
            (0.05, 0.0),
            (0.13, 0.25),
            (0.25, 0.25),
            (0.4, 0.5),
            (0.7, 0.75),
            (0.8, 0.75),
            (0.9, 1.0),
            (1.0, 1.0),
        ],
    )
    def test_snaps_to_nearest_anchor(self, raw: float, expected: float):
        assert _snap_to_anchor(raw) == expected

    def test_clamps_above_one(self):
        assert _snap_to_anchor(1.5) == 1.0

    def test_clamps_below_zero(self):
        assert _snap_to_anchor(-0.3) == 0.0


# --------------------------------------------------------------------
# Relevance parser (unchanged behavior, renamed function)
# --------------------------------------------------------------------


class TestParseRelevanceResponse:
    def test_clean_response(self):
        text = "SCORE: 0.75\nRATIONALE: directly addresses the question"
        score = parse_relevance_response(text, "relevance")
        assert score.score == 0.75
        assert "directly addresses" in score.rationale

    def test_lowercase_keys_accepted(self):
        text = "score: 1.0\nrationale: spot on"
        score = parse_relevance_response(text, "relevance")
        assert score.score == 1.0

    def test_off_anchor_value_snapped(self):
        text = "SCORE: 0.7\nRATIONALE: pretty good"
        score = parse_relevance_response(text, "relevance")
        assert score.score == 0.75

    def test_no_score_returns_parse_failure(self):
        text = "Sorry, I don't know how to answer this."
        score = parse_relevance_response(text, "relevance")
        assert score.score == 0.0
        assert "PARSE_FAILED" in score.rationale

    def test_score_present_but_no_rationale(self):
        text = "SCORE: 1.0"
        score = parse_relevance_response(text, "relevance")
        assert score.score == 1.0
        assert "no RATIONALE" in score.rationale

    def test_obviously_invalid_score_flagged(self):
        text = "SCORE: 5.0\nRATIONALE: out of range"
        score = parse_relevance_response(text, "relevance")
        assert score.score == 0.0
        assert "PARSE_FAILED" in score.rationale


# --------------------------------------------------------------------
# Claim extraction parser
# --------------------------------------------------------------------


class TestParseClaims:
    def test_clean_numbered_list(self):
        text = (
            "1. AutoModel.from_pretrained accepts a model name.\n"
            "2. The default cache is ~/.cache/huggingface.\n"
            "3. trust_remote_code=True allows custom model code execution."
        )
        claims = parse_claims(text)
        assert len(claims) == 3
        assert "AutoModel.from_pretrained" in claims[0]
        assert "trust_remote_code" in claims[2]

    def test_no_claims_sentinel(self):
        """Refusal answers / pure code should produce empty claims list."""
        assert parse_claims("NO_CLAIMS") == []
        assert parse_claims("There are no claims to extract.\n\nNO_CLAIMS") == []
        assert parse_claims("no_claims") == []  # case-insensitive

    def test_multiline_claim(self):
        """A claim that wraps to a second line should be captured whole."""
        text = (
            "1. AutoModel.from_pretrained accepts either a model name from\n"
            "the HuggingFace Hub or a local directory path.\n"
            "2. The cache directory defaults to ~/.cache/huggingface."
        )
        claims = parse_claims(text)
        assert len(claims) == 2
        assert "local directory path" in claims[0]

    def test_empty_input(self):
        assert parse_claims("") == []

    def test_handles_extra_whitespace(self):
        text = "  1.   leading whitespace claim   \n  2.   another   "
        claims = parse_claims(text)
        assert len(claims) == 2
        assert claims[0] == "leading whitespace claim"


# --------------------------------------------------------------------
# Claim attribution parser
# --------------------------------------------------------------------


class TestParseClaimAttributions:
    def test_clean_attribution(self):
        text = (
            "CLAIM 1: chunk 2 | RATIONALE: chunk 2 directly states this.\n"
            "CLAIM 2: none | RATIONALE: no chunk discusses this point.\n"
            "CLAIM 3: chunk 1 | RATIONALE: chunk 1 paraphrases the same idea."
        )
        attrs = parse_claim_attributions(text, claims=["a", "b", "c"], n_chunks=3)
        assert len(attrs) == 3
        assert attrs[0].supporting_chunk_idx == 2
        assert attrs[1].supporting_chunk_idx is None
        assert attrs[2].supporting_chunk_idx == 1
        assert "directly states" in attrs[0].rationale  # type: ignore[operator]

    def test_out_of_range_chunk_flagged(self):
        """Model emitted chunk 7 when only 3 chunks exist — treat as
        unsupported and flag with OUT_OF_RANGE marker."""
        text = "CLAIM 1: chunk 7 | RATIONALE: misattributed.\nCLAIM 2: chunk 1 | RATIONALE: ok."
        attrs = parse_claim_attributions(text, claims=["a", "b"], n_chunks=3)
        assert attrs[0].supporting_chunk_idx is None
        assert "OUT_OF_RANGE" in attrs[0].rationale  # type: ignore[operator]
        assert attrs[1].supporting_chunk_idx == 1

    def test_missing_attribution_line_flagged(self):
        """LLM only emitted lines for claims 1 and 3; claim 2 is missing.
        Parser should fill in claim 2 with a parse-failure marker."""
        text = "CLAIM 1: chunk 1 | RATIONALE: ok.\nCLAIM 3: chunk 2 | RATIONALE: ok."
        attrs = parse_claim_attributions(text, claims=["a", "b", "c"], n_chunks=3)
        assert len(attrs) == 3
        assert attrs[0].supporting_chunk_idx == 1
        assert attrs[1].supporting_chunk_idx is None
        assert "PARSE_FAILED" in attrs[1].rationale  # type: ignore[operator]
        assert attrs[2].supporting_chunk_idx == 2

    def test_rationale_optional(self):
        """Lines without a RATIONALE pipe should still parse cleanly."""
        text = "CLAIM 1: chunk 2\nCLAIM 2: none"
        attrs = parse_claim_attributions(text, claims=["a", "b"], n_chunks=3)
        assert attrs[0].supporting_chunk_idx == 2
        assert attrs[0].rationale is None
        assert attrs[1].supporting_chunk_idx is None
        assert attrs[1].rationale is None

    def test_lowercase_keywords_accepted(self):
        text = "claim 1: chunk 2 | rationale: ok"
        attrs = parse_claim_attributions(text, claims=["a"], n_chunks=3)
        assert attrs[0].supporting_chunk_idx == 2

    def test_empty_claims_list(self):
        attrs = parse_claim_attributions("anything", claims=[], n_chunks=3)
        assert attrs == []


# --------------------------------------------------------------------
# Faithfulness end-to-end (claim-level decomposition)
# --------------------------------------------------------------------


class TestJudgeFaithfulnessClaimLevel:
    def test_full_flow_supported_claims(self):
        """Two LLM calls (extract + attribute) → JudgeScore with
        per-claim attributions and supported/total fraction."""
        judge = StubJudge()
        # Call 1 response: extract two claims
        # Call 2 response: attribute both to chunks
        judge.queue(
            "1. Alpha is foo.\n2. Beta is bar.",
            "CLAIM 1: chunk 1 | RATIONALE: alpha verbatim.\n"
            "CLAIM 2: chunk 2 | RATIONALE: beta paraphrased.",
        )
        result = judge.judge_faithfulness(
            "What are alpha and beta?",
            chunks=_chunks("Alpha is foo.", "Beta is bar."),
            answer="Alpha is foo. Beta is bar.",
        )
        assert result.metric == "faithfulness"
        assert result.score == 1.0
        assert len(result.claim_attributions) == 2
        assert result.claim_attributions[0].supporting_chunk_idx == 1
        assert result.claim_attributions[1].supporting_chunk_idx == 2
        assert "2 of 2" in result.rationale

    def test_partial_support(self):
        judge = StubJudge()
        judge.queue(
            "1. Supported claim.\n2. Unsupported claim.\n3. Also unsupported.",
            "CLAIM 1: chunk 1 | RATIONALE: yes.\n"
            "CLAIM 2: none | RATIONALE: no chunk.\n"
            "CLAIM 3: none | RATIONALE: no chunk.",
        )
        result = judge.judge_faithfulness(
            "q?",
            chunks=_chunks("a", "b"),
            answer="a b c",
        )
        assert result.score == pytest.approx(1 / 3)
        assert "1 of 3" in result.rationale

    def test_no_claims_extracted(self):
        """When extraction returns NO_CLAIMS, attribution is skipped
        and the score is 0.0 with a flag in the rationale. This
        shouldn't happen on real in-corpus answers but the eval driver
        should still get a typed JudgeScore back."""
        judge = StubJudge()
        judge.queue("NO_CLAIMS")
        result = judge.judge_faithfulness(
            "q?",
            chunks=_chunks("a"),
            answer="(refusal)",
        )
        assert result.score == 0.0
        assert "NO_CLAIMS_EXTRACTED" in result.rationale
        assert result.claim_attributions == []
        # Only one inference call was made (no attribution call after empty extract).
        assert len(judge.calls) == 1

    def test_extract_claims_includes_question(self):
        """The extract-claims prompt should embed the question so the
        LLM can decide which content is "factual" relative to context.
        Pin this so a future prompt refactor doesn't drop the question."""
        judge = StubJudge()
        judge.queue("NO_CLAIMS")  # short-circuits the rest
        judge.judge_faithfulness("Question about X", chunks=_chunks("x"), answer="ans")
        msgs = judge.calls[0]
        assert "Question about X" in msgs[1]["content"]

    def test_extract_and_attribute_use_larger_token_budgets(self):
        """Pin the per-call max_new_tokens override contract: extraction
        and attribution need substantially larger token budgets than the
        default JudgeConfig.max_new_tokens (256). The default is sized for
        relevance; faithfulness flows would silently truncate output without
        the override (caught in the first PR-B run — long answers had
        attribution output cut off mid-list, producing PARSE_FAILED on
        every claim past the cap)."""
        judge = StubJudge()
        judge.queue("1. claim", "CLAIM 1: chunk 1 | RATIONALE: ok.")
        judge.judge_faithfulness("q", chunks=_chunks("a"), answer="ans")
        # Two calls: extraction then attribution.
        assert len(judge.max_new_tokens_seen) == 2
        extraction_budget, attribution_budget = judge.max_new_tokens_seen
        assert extraction_budget is not None
        assert attribution_budget is not None
        assert extraction_budget > judge.config.max_new_tokens
        assert attribution_budget > judge.config.max_new_tokens
        # Attribution needs more headroom than extraction (~75 tok/line × N
        # claims vs ~30 tok/claim).
        assert attribution_budget > extraction_budget

    def test_relevance_uses_default_token_budget(self):
        """Relevance keeps the default JudgeConfig.max_new_tokens budget
        — single SCORE+RATIONALE response is small."""
        judge = StubJudge()
        judge.queue("SCORE: 0.75\nRATIONALE: ok")
        judge.judge_relevance("q?", "ans")
        assert judge.max_new_tokens_seen == [None]  # i.e., no override

    def test_attribute_call_includes_chunks_and_claims(self):
        """The attribution prompt must include both the chunks block
        and the claims block — drift would mean the judge can't see
        what it's attributing claims against."""
        judge = StubJudge()
        judge.queue(
            "1. Single claim.",
            "CLAIM 1: chunk 1 | RATIONALE: ok.",
        )
        judge.judge_faithfulness(
            "q?",
            chunks=_chunks("alpha chunk text", "beta chunk text"),
            answer="ans",
        )
        attribute_msg = judge.calls[1][1]["content"]
        assert "alpha chunk text" in attribute_msg
        assert "beta chunk text" in attribute_msg
        assert "[1]" in attribute_msg
        assert "[2]" in attribute_msg
        assert "1. Single claim." in attribute_msg


# --------------------------------------------------------------------
# Relevance end-to-end (still single judgment)
# --------------------------------------------------------------------


class TestJudgeRelevance:
    def test_clean_response(self):
        judge = StubJudge()
        judge.queue("SCORE: 1.0\nRATIONALE: direct hit")
        result = judge.judge_relevance("What is X?", "X is foo.")
        assert result.score == 1.0
        assert result.metric == "relevance"
        assert result.claim_attributions == []  # relevance never decomposes

    def test_relevance_does_not_send_chunks(self):
        """Pin the contract difference: relevance prompt has only
        QUESTION and ANSWER, not CHUNKS or CONTEXT."""
        judge = StubJudge()
        judge.queue("SCORE: 0.75\nRATIONALE: ok")
        judge.judge_relevance("q?", "ans")
        user_content = judge.calls[0][1]["content"]
        assert "QUESTION" in user_content
        assert "ANSWER" in user_content
        assert "CHUNKS" not in user_content
        assert "CONTEXT" not in user_content

    def test_garbage_does_not_crash(self):
        judge = StubJudge()
        judge.queue("I'm sorry, as an AI language model I cannot...")
        result = judge.judge_relevance("q?", "ans")
        assert result.score == 0.0
        assert "PARSE_FAILED" in result.rationale
