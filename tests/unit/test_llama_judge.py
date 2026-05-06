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
    _AttributionEntry,
    _AttributionsResponse,
    _ClaimsResponse,
    _extract_json_block,
    _parse_json_response,
    _post_process_attributions,
    _post_process_claims,
    _post_process_refusal,
    _post_process_relevance,
    _RefusalResponse,
    _RelevanceResponse,
    _snap_to_anchor,
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
# JSON helpers (added 2026-05-06 for PR C.2 — judge-output-via-JSON)
# --------------------------------------------------------------------


class TestExtractJsonBlock:
    def test_clean_json_passthrough(self):
        assert _extract_json_block('{"a": 1}') == '{"a": 1}'

    def test_strips_markdown_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert _extract_json_block(text) == '{"a": 1}'

    def test_strips_unlabeled_fence(self):
        text = '```\n{"a": 1}\n```'
        assert _extract_json_block(text) == '{"a": 1}'

    def test_extracts_from_prose_preamble(self):
        """LLMs sometimes preface output with 'Here is my response:'.
        Extract the JSON block out of the surrounding prose."""
        text = 'Here is my response:\n{"a": 1, "b": "x"}\nThanks!'
        assert _extract_json_block(text) == '{"a": 1, "b": "x"}'

    def test_returns_text_when_no_json_present(self):
        """No braces, no fences — pass through. _parse_json_response
        will then fail to json.loads it, which is the right behavior."""
        assert _extract_json_block("just prose, sorry") == "just prose, sorry"


class TestParseJsonResponse:
    def test_clean_response_validates(self):
        text = '{"score": 0.75, "rationale": "ok"}'
        parsed = _parse_json_response(text, _RelevanceResponse)
        assert parsed is not None
        assert parsed.score == 0.75

    def test_fence_wrapped_response_parses(self):
        text = '```json\n{"score": 1.0, "rationale": "great"}\n```'
        parsed = _parse_json_response(text, _RelevanceResponse)
        assert parsed is not None
        assert parsed.score == 1.0

    def test_invalid_json_returns_none(self):
        text = "not json at all"
        assert _parse_json_response(text, _RelevanceResponse) is None

    def test_missing_field_returns_none(self):
        """Pydantic rejects { } missing 'score' — that's a validation
        error, returns None so the caller falls back."""
        text = '{"rationale": "ok"}'
        assert _parse_json_response(text, _RelevanceResponse) is None

    def test_out_of_range_score_returns_none(self):
        """Field(ge=0.0, le=1.0) catches LLM-emitted 5.0; returns
        None to trigger retry rather than silently clamping."""
        text = '{"score": 5.0, "rationale": "out of range"}'
        assert _parse_json_response(text, _RelevanceResponse) is None

    def test_claims_response(self):
        text = '{"claims": ["alpha", "beta"]}'
        parsed = _parse_json_response(text, _ClaimsResponse)
        assert parsed is not None
        assert parsed.claims == ["alpha", "beta"]

    def test_empty_claims_response(self):
        """The empty-claims signal: the JSON parsed cleanly, the list
        is empty. Distinguishes from a parse failure (where None
        would be returned)."""
        text = '{"claims": []}'
        parsed = _parse_json_response(text, _ClaimsResponse)
        assert parsed is not None
        assert parsed.claims == []

    def test_attributions_response(self):
        text = """{
          "attributions": [
            {"claim_idx": 1, "supporting_chunk_idx": 2, "rationale": "ok"},
            {"claim_idx": 2, "supporting_chunk_idx": null, "rationale": "no chunk"}
          ]
        }"""
        parsed = _parse_json_response(text, _AttributionsResponse)
        assert parsed is not None
        assert len(parsed.attributions) == 2
        assert parsed.attributions[0].supporting_chunk_idx == 2
        assert parsed.attributions[1].supporting_chunk_idx is None

    def test_attribution_zero_idx_rejected(self):
        """claim_idx is 1-indexed; 0 would conflate with no-claim."""
        text = '{"attributions": [{"claim_idx": 0, "supporting_chunk_idx": 1}]}'
        assert _parse_json_response(text, _AttributionsResponse) is None

    def test_refusal_response(self):
        text = '{"refused": true, "rationale": "model said it"}'
        parsed = _parse_json_response(text, _RefusalResponse)
        assert parsed is not None
        assert parsed.refused is True


class TestResponseSchemas:
    """Spot-pin a few field-validator behaviors directly on the schemas."""

    def test_attribution_entry_optional_rationale(self):
        entry = _AttributionEntry(claim_idx=1, supporting_chunk_idx=2)
        assert entry.rationale is None

    def test_attribution_entry_optional_chunk(self):
        entry = _AttributionEntry(claim_idx=1)
        assert entry.supporting_chunk_idx is None


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
# Post-processing helpers (replace the old regex parsers)
# --------------------------------------------------------------------


class TestPostProcessClaims:
    def test_extracts_claim_strings(self):
        response = _ClaimsResponse(claims=["alpha", "beta", "gamma"])
        assert _post_process_claims(response) == ["alpha", "beta", "gamma"]

    def test_strips_whitespace(self):
        response = _ClaimsResponse(claims=["  padded  ", "\tindented", "trailing\n"])
        assert _post_process_claims(response) == ["padded", "indented", "trailing"]

    def test_drops_empty_claims(self):
        """If the LLM emits {"claims": ["valid", "", " "]}, drop the
        whitespace-only entries — they're not real claims."""
        response = _ClaimsResponse(claims=["valid claim", "", "   "])
        assert _post_process_claims(response) == ["valid claim"]

    def test_empty_claims_list_passes_through(self):
        """Empty list signals 'no factual claims' — pass through, not
        the same as parse failure."""
        response = _ClaimsResponse(claims=[])
        assert _post_process_claims(response) == []

    def test_parse_failure_returns_empty_list(self):
        """None response (parse failure after retry) → empty list.
        judge_faithfulness distinguishes from intentional empty via
        the rationale on the resulting JudgeScore (NO_CLAIMS_EXTRACTED
        marker applies in both cases since downstream behavior is
        the same)."""
        assert _post_process_claims(None) == []


class TestPostProcessAttributions:
    def test_clean_response_one_per_claim(self):
        response = _AttributionsResponse(
            attributions=[
                _AttributionEntry(claim_idx=1, supporting_chunk_idx=2, rationale="found in 2"),
                _AttributionEntry(claim_idx=2, supporting_chunk_idx=None, rationale="no chunk"),
                _AttributionEntry(
                    claim_idx=3, supporting_chunk_idx=1, rationale="paraphrased in 1"
                ),
            ]
        )
        attrs = _post_process_attributions(response, claims=["a", "b", "c"], n_chunks=3)
        assert len(attrs) == 3
        assert attrs[0].supporting_chunk_idx == 2
        assert attrs[1].supporting_chunk_idx is None
        assert attrs[2].supporting_chunk_idx == 1
        assert attrs[0].rationale == "found in 2"

    def test_out_of_range_chunk_flagged(self):
        """LLM emitted chunk 7 when only 3 exist — treat as unsupported
        and flag with OUT_OF_RANGE marker. Same semantic as the prior
        regex parser, just sourced from the JSON path."""
        response = _AttributionsResponse(
            attributions=[
                _AttributionEntry(claim_idx=1, supporting_chunk_idx=7, rationale="misattributed"),
                _AttributionEntry(claim_idx=2, supporting_chunk_idx=1, rationale="ok"),
            ]
        )
        attrs = _post_process_attributions(response, claims=["a", "b"], n_chunks=3)
        assert attrs[0].supporting_chunk_idx is None
        assert "OUT_OF_RANGE" in attrs[0].rationale  # type: ignore[operator]
        assert attrs[1].supporting_chunk_idx == 1

    def test_missing_claim_entry_flagged(self):
        """LLM only emitted entries for claims 1 and 3; claim 2 missing
        from the JSON. Post-processing fills it with PARSE_FAILED."""
        response = _AttributionsResponse(
            attributions=[
                _AttributionEntry(claim_idx=1, supporting_chunk_idx=1),
                _AttributionEntry(claim_idx=3, supporting_chunk_idx=2),
            ]
        )
        attrs = _post_process_attributions(response, claims=["a", "b", "c"], n_chunks=3)
        assert len(attrs) == 3
        assert attrs[0].supporting_chunk_idx == 1
        assert attrs[1].supporting_chunk_idx is None
        assert "PARSE_FAILED" in attrs[1].rationale  # type: ignore[operator]
        assert attrs[2].supporting_chunk_idx == 2

    def test_extra_attributions_ignored(self):
        """LLM emitted an entry for claim_idx=99 we didn't ask about;
        silently drop it. Don't fail or insert phantom attributions."""
        response = _AttributionsResponse(
            attributions=[
                _AttributionEntry(claim_idx=1, supporting_chunk_idx=1),
                _AttributionEntry(claim_idx=99, supporting_chunk_idx=2, rationale="who?"),
            ]
        )
        attrs = _post_process_attributions(response, claims=["a"], n_chunks=3)
        assert len(attrs) == 1
        assert attrs[0].supporting_chunk_idx == 1

    def test_optional_rationale_preserved(self):
        response = _AttributionsResponse(
            attributions=[_AttributionEntry(claim_idx=1, supporting_chunk_idx=1)]
        )
        attrs = _post_process_attributions(response, claims=["a"], n_chunks=3)
        assert attrs[0].rationale is None

    def test_empty_claims_list_returns_empty(self):
        response = _AttributionsResponse(attributions=[])
        attrs = _post_process_attributions(response, claims=[], n_chunks=3)
        assert attrs == []

    def test_parse_failure_marks_every_claim(self):
        """None response → every claim gets a PARSE_FAILED marker.
        Matches the prior regex parser's behavior so downstream
        aggregation (n_claim_parse_failures) keeps working."""
        attrs = _post_process_attributions(None, claims=["a", "b", "c"], n_chunks=3)
        assert len(attrs) == 3
        assert all(a.supporting_chunk_idx is None for a in attrs)
        assert all("PARSE_FAILED" in (a.rationale or "") for a in attrs)


class TestPostProcessRelevance:
    def test_snaps_to_anchor(self):
        """LLM emitted 0.7 instead of an anchor — snap to 0.75 silently."""
        response = _RelevanceResponse(score=0.7, rationale="pretty good")
        result = _post_process_relevance(response)
        assert result.score == 0.75
        assert result.metric == "relevance"

    def test_already_at_anchor(self):
        response = _RelevanceResponse(score=1.0, rationale="on point")
        assert _post_process_relevance(response).score == 1.0

    def test_parse_failure_returns_zero(self):
        """None → score=0.0 with PARSE_FAILED marker. Same conservative
        default as the prior regex parser."""
        result = _post_process_relevance(None)
        assert result.score == 0.0
        assert "PARSE_FAILED" in result.rationale


class TestPostProcessRefusal:
    def test_refused_pass_through(self):
        response = _RefusalResponse(refused=True, rationale="model said it")
        result = _post_process_refusal(response)
        assert result.refused is True
        assert result.rationale == "model said it"

    def test_not_refused_pass_through(self):
        response = _RefusalResponse(refused=False, rationale="model attempted")
        assert _post_process_refusal(response).refused is False

    def test_parse_failure_conservative_default(self):
        """None → refused=False with PARSE_FAILED. Conservative default
        prevents parse failures from silently inflating the refusal rate."""
        result = _post_process_refusal(None)
        assert result.refused is False
        assert "PARSE_FAILED" in result.rationale


# --------------------------------------------------------------------
# Faithfulness end-to-end (claim-level decomposition)
# --------------------------------------------------------------------


class TestJudgeFaithfulnessClaimLevel:
    """Integration tests via StubJudge. The judge now expects JSON
    output for both extract_claims and attribute_claims, so test
    fixtures inject JSON strings rather than the prior numbered-list
    + 'CLAIM N: chunk K' formats."""

    def test_full_flow_supported_claims(self):
        judge = StubJudge()
        judge.queue(
            '{"claims": ["Alpha is foo.", "Beta is bar."]}',
            (
                '{"attributions": ['
                '{"claim_idx": 1, "supporting_chunk_idx": 1, "rationale": "alpha verbatim"},'
                '{"claim_idx": 2, "supporting_chunk_idx": 2, "rationale": "beta paraphrased"}'
                "]}"
            ),
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
            '{"claims": ["Supported.", "Unsupported.", "Also unsupported."]}',
            (
                '{"attributions": ['
                '{"claim_idx": 1, "supporting_chunk_idx": 1, "rationale": "yes"},'
                '{"claim_idx": 2, "supporting_chunk_idx": null, "rationale": "no chunk"},'
                '{"claim_idx": 3, "supporting_chunk_idx": null, "rationale": "no chunk"}'
                "]}"
            ),
        )
        result = judge.judge_faithfulness("q?", chunks=_chunks("a", "b"), answer="a b c")
        assert result.score == pytest.approx(1 / 3)
        assert "1 of 3" in result.rationale

    def test_no_claims_extracted(self):
        """Empty claims array → JudgeScore with NO_CLAIMS_EXTRACTED
        marker, no attribution call. Pin the early-exit contract."""
        judge = StubJudge()
        judge.queue('{"claims": []}')
        result = judge.judge_faithfulness("q?", chunks=_chunks("a"), answer="(refusal)")
        assert result.score == 0.0
        assert "NO_CLAIMS_EXTRACTED" in result.rationale
        assert result.claim_attributions == []
        # Only one inference call was made (no attribution call after
        # an empty extract).
        assert len(judge.calls) == 1

    def test_extract_claims_parse_failure_retries_then_falls_back(self):
        """If JSON parse fails on the first call, the helper retries
        once with a clarifying message. If the retry also fails, the
        flow short-circuits the same as the empty-claims case."""
        judge = StubJudge()
        judge.queue("not valid json", "still not valid json")
        result = judge.judge_faithfulness("q?", chunks=_chunks("a"), answer="ans")
        assert result.score == 0.0
        assert "NO_CLAIMS_EXTRACTED" in result.rationale
        # Two calls — initial + retry — for extraction. Then early exit
        # because empty claims, so no attribution call.
        assert len(judge.calls) == 2

    def test_extract_claims_includes_question(self):
        """Pin: the extract-claims prompt embeds the question."""
        judge = StubJudge()
        judge.queue('{"claims": []}')
        judge.judge_faithfulness("Question about X", chunks=_chunks("x"), answer="ans")
        msgs = judge.calls[0]
        assert "Question about X" in msgs[1]["content"]

    def test_extract_and_attribute_use_larger_token_budgets(self):
        """Pin the per-call max_new_tokens override: extraction and
        attribution need bigger budgets than relevance default (256)."""
        judge = StubJudge()
        judge.queue(
            '{"claims": ["one"]}',
            '{"attributions": [{"claim_idx": 1, "supporting_chunk_idx": 1}]}',
        )
        judge.judge_faithfulness("q", chunks=_chunks("a"), answer="ans")
        assert len(judge.max_new_tokens_seen) == 2
        extraction_budget, attribution_budget = judge.max_new_tokens_seen
        assert extraction_budget is not None
        assert attribution_budget is not None
        assert extraction_budget > judge.config.max_new_tokens
        assert attribution_budget > judge.config.max_new_tokens
        assert attribution_budget > extraction_budget

    def test_relevance_uses_default_token_budget(self):
        """Relevance keeps the default JudgeConfig.max_new_tokens —
        small JSON response."""
        judge = StubJudge()
        judge.queue('{"score": 0.75, "rationale": "ok"}')
        judge.judge_relevance("q?", "ans")
        assert judge.max_new_tokens_seen == [None]

    def test_attribute_call_includes_chunks_and_claims(self):
        judge = StubJudge()
        judge.queue(
            '{"claims": ["Single claim."]}',
            '{"attributions": [{"claim_idx": 1, "supporting_chunk_idx": 1}]}',
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

    def test_attribution_parse_failure_retries(self):
        """Pin the retry path on the attribution call. Initial response
        is malformed; the retry succeeds and we get a real JudgeScore."""
        judge = StubJudge()
        judge.queue(
            '{"claims": ["one"]}',  # extract: clean
            "garbage attribution output",  # attribute: malformed
            '{"attributions": [{"claim_idx": 1, "supporting_chunk_idx": 1, "rationale": "after retry"}]}',
        )
        result = judge.judge_faithfulness("q?", chunks=_chunks("a"), answer="ans")
        assert result.score == 1.0
        assert result.claim_attributions[0].rationale == "after retry"
        # 3 calls: extract (1) + attribute initial (1) + attribute retry (1).
        assert len(judge.calls) == 3


# --------------------------------------------------------------------
# Relevance end-to-end (single LLM call, JSON output)
# --------------------------------------------------------------------


class TestJudgeRelevance:
    def test_clean_response(self):
        judge = StubJudge()
        judge.queue('{"score": 1.0, "rationale": "direct hit"}')
        result = judge.judge_relevance("What is X?", "X is foo.")
        assert result.score == 1.0
        assert result.metric == "relevance"
        assert result.claim_attributions == []  # relevance never decomposes

    def test_off_anchor_score_snapped(self):
        """LLM emitted 0.7 — snap to 0.75 silently in post-processing."""
        judge = StubJudge()
        judge.queue('{"score": 0.7, "rationale": "pretty good"}')
        result = judge.judge_relevance("q?", "ans")
        assert result.score == 0.75

    def test_relevance_does_not_send_chunks(self):
        judge = StubJudge()
        judge.queue('{"score": 0.75, "rationale": "ok"}')
        judge.judge_relevance("q?", "ans")
        user_content = judge.calls[0][1]["content"]
        assert "QUESTION" in user_content
        assert "ANSWER" in user_content
        assert "CHUNKS" not in user_content
        assert "CONTEXT" not in user_content

    def test_parse_failure_falls_back(self):
        """Garbage response after retry → score=0.0 with PARSE_FAILED."""
        judge = StubJudge()
        judge.queue("not json", "still not json")
        result = judge.judge_relevance("q?", "ans")
        assert result.score == 0.0
        assert "PARSE_FAILED" in result.rationale

    def test_first_response_invalid_retry_succeeds(self):
        """Pin retry path: malformed first response; valid second."""
        judge = StubJudge()
        judge.queue("garbage", '{"score": 1.0, "rationale": "ok"}')
        result = judge.judge_relevance("q?", "ans")
        assert result.score == 1.0
        assert len(judge.calls) == 2


# --------------------------------------------------------------------
# Refusal end-to-end (single LLM call, JSON output)
# --------------------------------------------------------------------


class TestJudgeRefusalEndToEnd:
    def test_refusal_yes(self):
        judge = StubJudge()
        judge.queue('{"refused": true, "rationale": "model said it does not have context"}')
        result = judge.judge_refusal("Off-corpus q?", "I don't have enough context.")
        assert result.refused is True
        assert "does not have context" in result.rationale

    def test_refusal_no(self):
        judge = StubJudge()
        judge.queue('{"refused": false, "rationale": "model attempted an answer"}')
        result = judge.judge_refusal("In-corpus q?", "AutoModel takes a name.")
        assert result.refused is False

    def test_refusal_uses_default_token_budget(self):
        judge = StubJudge()
        judge.queue('{"refused": false, "rationale": "ok"}')
        judge.judge_refusal("q?", "ans")
        assert judge.max_new_tokens_seen == [None]

    def test_refusal_prompt_no_chunks(self):
        judge = StubJudge()
        judge.queue('{"refused": true, "rationale": "ok"}')
        judge.judge_refusal("q?", "ans")
        user_content = judge.calls[0][1]["content"]
        assert "QUESTION" in user_content
        assert "ANSWER" in user_content
        assert "CHUNKS" not in user_content
        assert "CONTEXT" not in user_content

    def test_garbage_response_falls_back_to_not_refused(self):
        judge = StubJudge()
        judge.queue("garbage", "still garbage")
        result = judge.judge_refusal("q?", "ans")
        assert result.refused is False
        assert "PARSE_FAILED" in result.rationale
