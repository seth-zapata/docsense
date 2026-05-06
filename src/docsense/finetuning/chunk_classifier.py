"""Heuristic chunk-affinity classifier for training-data seeding.

For Block 3B.2 we sample chunks from the corpus stratified by question
type. Given a chunk's text, ``classify_chunk()`` returns the set of
types it has affinity for — multi-label, since a chunk with both code
and advisory phrasing legitimately seeds either procedural or
best-practice questions.

The classifier is intentionally heuristic and lossy: it under-includes
ambiguous chunks rather than over-includes. Chunks with no detected
affinity are simply unsampled — we have ~12K chunks in the corpus and
only need ~800 examples, so coverage isn't the bottleneck. False
positives are tolerable because the downstream Haiku query generator
and rule-based filters catch malformed queries before distillation.

Refusal isn't a chunk affinity; refusal queries are seeded separately
in the query generator (off-corpus topic seeds + retrieval-failure
pairing). The classifier never returns ``QuestionType.REFUSAL``.
"""

from __future__ import annotations

import re
from enum import StrEnum


class QuestionType(StrEnum):
    """Question shape a training example targets.

    Used at three layers of the pipeline:

    1. Chunk classification (this file): "this chunk has affinity for
       procedural questions"
    2. Query generation: "generate a procedural question for this chunk"
    3. Query-pool serialization: "this query was generated as procedural"
    """

    PROCEDURAL = "procedural"
    COMPARISON = "comparison"
    BEST_PRACTICE = "best_practice"
    POINTER = "pointer"
    REFUSAL = "refusal"


# Procedural — code blocks. We don't match a lone triple-backtick because
# chunks that start mid-block can contain only the orphan closing fence
# (descriptive prose with a hanging close, not procedural code).
_TRIPLE_BACKTICK_RE = re.compile(r"```")
_LANG_HINT_RE = re.compile(r"```(?:py|python|bash|sh|json|yaml|toml)\b", re.IGNORECASE)
_REPL_PROMPT_RE = re.compile(r"^>>> ", re.MULTILINE)


_COMPARISON_RE = re.compile(
    "|".join(
        [
            r"\btwo (?:general )?types\b",
            r"\bvs\.?(?:\s|$)",
            r"\bversus\b",
            r"\binstead of\b",
            r"\bcompared (?:to|with)\b",
            r"\bdiffer(?:s|ence|ent)\b",
            r"\bdistinct(?:ion)?\b",
            r"\bunlike\b",
            r"\bwhereas\b",
        ]
    ),
    re.IGNORECASE,
)


# "should" alone is too noisy ("this should not happen", "the model should
# produce X" — both descriptive, not advisory). Restrict to direct-advisory
# contexts: "you should" or passive-advisory "should be {verb}".
_BEST_PRACTICE_PHRASE_RE = re.compile(
    "|".join(
        [
            r"\bwe (?:advise|recommend)\b",
            r"\brecommended\b",
            r"\bbest practice\b",
            r"\byou should\b",
            r"\bshould be (?:set|used|called|configured|applied)\b",
            r"\bmake sure (?:to|that)\b",
            r"\bprefer(?:red)?\b",
            r"\bleads to (?:more|better)\b",
        ]
    ),
    re.IGNORECASE,
)
_BEST_PRACTICE_HEADER_RE = re.compile(
    r"(?:^|\n)\s*#*\s*(?:usage )?tips?\s*$|"
    r"(?:^|\n)\s*#*\s*best practices?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


_POINTER_RE = re.compile(
    "|".join(
        [
            r"\b(?:a )?good (?:starting point|place to start)\b",
            r"\bstarting point\b",
            r"\brefer to (?:the )?",
            r"\bsee (?:the )?(?:[\w\-]+ )?(?:repository|repo|script|notebook|guide|docs?)\b",
            r"\bfor more (?:details|information|context)\b",
            r"\blook at (?:the )?",
            r"\bcopy(?:,)? adapt(?:,)? and reuse\b",
            r"\bconversion script\b",
        ]
    ),
    re.IGNORECASE,
)


def is_procedural(text: str) -> bool:
    """True iff the chunk contains an executable code block."""
    if len(_TRIPLE_BACKTICK_RE.findall(text)) >= 2:
        return True
    if _LANG_HINT_RE.search(text):
        return True
    return bool(_REPL_PROMPT_RE.search(text))


def is_comparison(text: str) -> bool:
    """True iff the chunk uses two-concept comparison language."""
    return bool(_COMPARISON_RE.search(text))


def is_best_practice(text: str) -> bool:
    """True iff the chunk contains advisory phrasing or a tips header."""
    return bool(_BEST_PRACTICE_PHRASE_RE.search(text)) or bool(
        _BEST_PRACTICE_HEADER_RE.search(text)
    )


def is_pointer(text: str) -> bool:
    """True iff the chunk directs the reader to a specific resource."""
    return bool(_POINTER_RE.search(text))


def classify_chunk(text: str) -> set[QuestionType]:
    """Return the set of question types this chunk has affinity for.

    Multi-label. Empty set means no heuristic matched — the chunk
    simply won't be sampled by the type-aware seeder. ``REFUSAL`` is
    never returned; refusal queries are seeded separately.
    """
    affinities: set[QuestionType] = set()
    if is_procedural(text):
        affinities.add(QuestionType.PROCEDURAL)
    if is_comparison(text):
        affinities.add(QuestionType.COMPARISON)
    if is_best_practice(text):
        affinities.add(QuestionType.BEST_PRACTICE)
    if is_pointer(text):
        affinities.add(QuestionType.POINTER)
    return affinities
