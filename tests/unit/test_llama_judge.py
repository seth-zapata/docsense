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
    parse_judge_response,
)


class StubJudge(LlamaJudge):
    """LlamaJudge that returns whatever string the test queues up."""

    def __init__(self) -> None:
        super().__init__(JudgeConfig())
        self._stub_response = ""
        self.calls: list[list[dict[str, str]]] = []

    def queue(self, response: str) -> None:
        self._stub_response = response

    def _run_inference(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self._stub_response


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


class TestParseJudgeResponse:
    def test_clean_response(self):
        text = "SCORE: 0.75\nRATIONALE: The answer covers most claims."
        score = parse_judge_response(text, "faithfulness")
        assert score.score == 0.75
        assert score.metric == "faithfulness"
        assert "covers most claims" in score.rationale

    def test_lowercase_keys_accepted(self):
        text = "score: 1.0\nrationale: spot on"
        score = parse_judge_response(text, "relevance")
        assert score.score == 1.0
        assert score.rationale == "spot on"

    def test_off_anchor_value_snapped(self):
        """LLM emitted 0.7 instead of an anchor — should snap to 0.75."""
        text = "SCORE: 0.7\nRATIONALE: pretty good"
        score = parse_judge_response(text, "relevance")
        assert score.score == 0.75

    def test_score_without_colon(self):
        """Tolerant of slightly off-format output."""
        text = "SCORE 0.5\nRATIONALE missing colon"
        score = parse_judge_response(text, "faithfulness")
        assert score.score == 0.5

    def test_leading_dot_decimal(self):
        text = "SCORE: .25\nRATIONALE: leading dot"
        score = parse_judge_response(text, "relevance")
        assert score.score == 0.25

    def test_multiline_rationale(self):
        text = (
            "SCORE: 0.5\nRATIONALE: First, the answer addresses\n"
            "the question, but second, it omits a key detail.\n"
            "\n"
            "(some trailing model noise)"
        )
        score = parse_judge_response(text, "faithfulness")
        assert score.score == 0.5
        # Multi-line content captured, trailing noise after blank line not.
        assert "First" in score.rationale
        assert "omits a key detail" in score.rationale
        assert "trailing model noise" not in score.rationale

    def test_no_score_returns_parse_failure(self):
        """Robust to a judge response that's missing the score line:
        return a JudgeScore at 0.0 with a parse-failure rationale, not
        an exception. The report can flag it for human spot-check."""
        text = "Sorry, I don't know how to answer this."
        score = parse_judge_response(text, "faithfulness")
        assert score.score == 0.0
        assert "PARSE_FAILED" in score.rationale

    def test_score_present_but_no_rationale(self):
        text = "SCORE: 1.0"
        score = parse_judge_response(text, "relevance")
        assert score.score == 1.0
        assert "no RATIONALE" in score.rationale

    def test_obviously_invalid_score_flagged_as_parse_failure(self):
        """A "SCORE: 5.0" output means the LLM ignored the [0, 1] scale
        entirely — flag it for review rather than silently clamping to
        1.0. The score regex only accepts values in [0, 1], so anything
        outside falls through to the parse-failure path."""
        text = "SCORE: 5.0\nRATIONALE: out of range"
        score = parse_judge_response(text, "faithfulness")
        assert score.score == 0.0
        assert "PARSE_FAILED" in score.rationale


class TestLlamaJudgeIntegration:
    def test_judge_faithfulness_passes_context(self):
        """Verifies the faithfulness path includes context in the user
        message — relevance does not, and that's a meaningful contract
        difference worth pinning."""
        judge = StubJudge()
        judge.queue("SCORE: 0.75\nRATIONALE: ok")
        result = judge.judge_faithfulness("What is X?", "Context X is foo.", "X is foo.")
        assert result.score == 0.75
        assert result.metric == "faithfulness"

        # Last call's user message includes context
        msgs = judge.calls[-1]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "Context X is foo." in msgs[1]["content"]

    def test_judge_relevance_does_not_pass_context(self):
        judge = StubJudge()
        judge.queue("SCORE: 1.0\nRATIONALE: direct hit")
        result = judge.judge_relevance("What is X?", "X is foo.")
        assert result.score == 1.0
        assert result.metric == "relevance"

        msgs = judge.calls[-1]
        # Relevance prompt should not mention context — verify by
        # checking the user message contains QUESTION and ANSWER but
        # not CONTEXT.
        user_content = msgs[1]["content"]
        assert "QUESTION" in user_content
        assert "ANSWER" in user_content
        assert "CONTEXT" not in user_content

    def test_garbage_judge_response_does_not_crash(self):
        """End-to-end robustness: a misbehaving judge that emits free
        prose still produces a valid JudgeScore (with the parse-failure
        marker) so the eval driver can keep going."""
        judge = StubJudge()
        judge.queue("I'm sorry, as an AI language model I cannot...")
        result = judge.judge_faithfulness("q", "ctx", "a")
        assert result.score == 0.0
        assert "PARSE_FAILED" in result.rationale
