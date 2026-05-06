"""Rule-based evals over Answer objects — no LLM judge required.

Two cheap, deterministic checks that complement the LLM-judge metrics:

- ``check_no_answer_behavior``: pattern-matches refusal phrases in
  the answer text. For an off-corpus question the pipeline should
  refuse; for an in-corpus question it shouldn't.
- ``check_citations_grounded``: verifies the answer text contains
  ``[N]`` markers and that they point at valid retrieved chunks.

These are pattern-match heuristics, not semantic evals — they will
miss novel refusal phrasings and they cannot tell whether a citation
was used appropriately. Pair with the LLM-judge for the cases where
that nuance matters. The point of having both is that rule-based
checks are zero-cost to run on every Answer, while LLM-judge calls
are expensive and we run them only on the metrics that need them.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from docsense.generation.types import Answer

# Refusal-phrase regexes. Case-insensitive, applied to the lowered
# answer text. Each pattern is anchored with word boundaries where
# possible so it won't match inside an unrelated word. Order doesn't
# matter for correctness; the matcher reports the first hit purely so
# the eval report can surface which phrase fired.
_REFUSAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "no_information",
        re.compile(r"\b(?:no|not enough|insufficient)\s+(?:information|context|details?)\b"),
    ),
    # Possessive form — "I don't have enough context to answer".
    # Discovered as a coverage gap during the first Block 1B eval run:
    # all 8 no-answer queries opened with this exact phrase, but only
    # the 3 with a follow-up "The provided context does not..." sentence
    # were flagged as refusals. The "have" verb wasn't in cannot_action's
    # action-verb list because "I don't have time" / "I don't have a
    # pen" aren't refusals on their own — but "don't have <quantifier>
    # context/information" is unambiguous.
    (
        "dont_have_context",
        re.compile(
            r"\b(?:do\s+not|don'?t|doesn'?t|does\s+not)\s+have\s+"
            r"(?:enough|the|sufficient|adequate|any|relevant)?\s*"
            r"(?:context|information|details?)\b"
        ),
    ),
    # Negative-modal + action verb. Covers "I don't know", "cannot
    # determine", "I'm unable to answer", "can't tell", etc. — the
    # bulk of LLM refusal phrasings collapse here.
    (
        "cannot_action",
        re.compile(
            r"\b(?:i'?m\s+|i\s+am\s+|i\s+)?(?:do\s+not|don'?t|cannot|can'?t|am\s+unable\s+to|unable\s+to)\s+"
            r"(?:know|determine|tell|say|answer|provide|find|help|locate)\b"
        ),
    ),
    (
        "not_in_context",
        re.compile(
            r"\b(?:not|isn'?t)\s+(?:in|covered|mentioned|addressed|found)\s+(?:in\s+)?(?:the\s+)?(?:provided\s+)?context\b"
        ),
    ),
    (
        "context_doesnt",
        re.compile(
            r"\bcontext\s+(?:does\s+not|doesn'?t)\s+(?:contain|cover|mention|provide|include|address)\b"
        ),
    ),
    ("no_relevant", re.compile(r"\bno\s+relevant\s+(?:information|context|content|details?)\b")),
    (
        "apologize_unable",
        re.compile(r"\b(?:i\s+)?apologi[sz]e\b.{0,40}\b(?:unable|cannot|can'?t)\b"),
    ),
)


class NoAnswerCheck(BaseModel):
    """Result of testing whether an Answer refused as expected.

    ``refused`` is the heuristic verdict; ``correct`` is whether that
    verdict matched what the eval set declared as the expected
    behavior. ``matched_pattern`` names the regex that fired (if any),
    so an analyst can spot-check whether the heuristic is firing on
    the right kind of phrase or whether the LLM is producing some
    novel refusal phrasing that the patterns miss.
    """

    refused: bool
    expected_refusal: bool
    correct: bool
    matched_pattern: str | None


class CitationCheck(BaseModel):
    """Result of inspecting an Answer's citation behavior.

    The pydantic invariant on ``Answer`` already guarantees every
    entry in ``citations`` points at a real retrieved chunk. This
    check goes further:

    - ``n_markers_in_text``: count of raw ``[N]`` markers in the
      answer text. Captures the "did the model cite at all?"
      question — the smoke run showed Qwen producing zero markers
      despite the prompt directive, and this is the metric that
      surfaces that.
    - ``n_resolved_citations``: ``len(answer.citations)``, which is
      the deduplicated count of in-range markers. Compare against
      ``n_markers_in_text`` to detect out-of-range hallucinations.
    - ``all_markers_in_range``: every ``[N]`` in the text is a valid
      1-indexed chunk reference.
    - ``any_marker_present``: convenience boolean, identical to
      ``n_markers_in_text > 0``.
    """

    n_markers_in_text: int
    n_resolved_citations: int
    all_markers_in_range: bool
    any_marker_present: bool


_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


def check_no_answer_behavior(answer: Answer, expected_refusal: bool) -> NoAnswerCheck:
    """Test whether ``answer.text`` exhibits a refusal phrase.

    For an off-corpus question (``expected_refusal=True``), a
    well-behaved system refuses. For an in-corpus question
    (``expected_refusal=False``), the system should answer instead.
    The check returns a typed result rather than a raw bool so the
    eval report can record which pattern matched (or "none") for
    later inspection.
    """
    lowered = answer.text.lower()
    matched: str | None = None
    for name, pattern in _REFUSAL_PATTERNS:
        if pattern.search(lowered):
            matched = name
            break

    refused = matched is not None
    return NoAnswerCheck(
        refused=refused,
        expected_refusal=expected_refusal,
        correct=(refused == expected_refusal),
        matched_pattern=matched,
    )


def check_citations_grounded(answer: Answer) -> CitationCheck:
    """Inspect citation behavior in ``answer``.

    Returns the populated ``CitationCheck`` — see that model's
    docstring for field semantics. This function never raises, even
    if ``answer.text`` contains no markers at all; the caller
    aggregates many results into the eval report and one zero-marker
    Answer shouldn't break the loop.
    """
    n_chunks = len(answer.retrieved_chunks)
    markers = _CITATION_MARKER_RE.findall(answer.text)
    n_markers = len(markers)
    all_in_range = n_markers > 0 and all(1 <= int(m) <= n_chunks for m in markers)

    return CitationCheck(
        n_markers_in_text=n_markers,
        n_resolved_citations=len(answer.citations),
        all_markers_in_range=all_in_range,
        any_marker_present=n_markers > 0,
    )
