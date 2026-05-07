"""Quality filters for the Block 3B.2 query pool.

Four filters operating on ``list[GeneratedQuery]`` (pre-retrieval):

1. ``filter_by_length`` — drops queries below a word-count floor
   (default 5). Catches outlier short queries the LLM produces despite
   the prompt's "5-15 words" instruction.

2. ``filter_by_type_shape`` — drops queries whose phrasing doesn't
   match their assigned ``question_type`` shape. Reduces type drift
   (e.g., a procedural-shaped question landing in the comparison
   bucket because the chunk classifier flagged the chunk as
   comparison-affine but the chunk content didn't lend itself to
   actual comparison framing). Type accuracy matters for downstream
   eval breakdowns ("did fine-tuning improve comparison answers?").

3. ``filter_duplicates`` — embedding-based dedupe within each question
   type. A procedural query and a comparison query may legitimately
   embed similarly while answering different question shapes;
   deduping them would lose diversity. Stratification preserves the
   type distribution we engineered in 3B.2.

4. ``filter_eval_contamination`` — drops any query similar to an
   existing eval query (curated_eval, no_answer, structural). Hard
   threshold (default 0.7) — even a few contaminated examples destroy
   the Phase 3 fine-tuning eval. NOT type-stratified: cross-type
   overlap with eval queries is still contamination.

Each filter returns a ``FilterReport`` with ``kept`` and ``dropped``
lists. Chaining is explicit (caller passes ``report.kept`` to the
next filter); no magic composition keeps the audit trail visible.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from docsense.finetuning.chunk_classifier import QuestionType

if TYPE_CHECKING:
    from docsense.finetuning.query_generation import GeneratedQuery

# Embeds a list of strings; returns a (n, d) ndarray. Caller owns the
# embedder model — typically ``SentenceTransformer.encode``.
Embedder = Callable[[list[str]], np.ndarray]


@dataclass
class FilterReport:
    """Outcome of one filter pass.

    ``kept`` are the queries that survived. ``dropped`` is a list of
    (query, reason) tuples — preserved so the seeder's audit log can
    record exactly why each query was rejected.
    """

    kept: list[GeneratedQuery] = field(default_factory=list)
    dropped: list[tuple[GeneratedQuery, str]] = field(default_factory=list)

    @property
    def kept_count(self) -> int:
        return len(self.kept)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    @property
    def total(self) -> int:
        return self.kept_count + self.dropped_count

    def reason_summary(self) -> dict[str, int]:
        """Counts of dropped queries by reason. Useful for log lines."""
        counts: dict[str, int] = {}
        for _, reason in self.dropped:
            counts[reason] = counts.get(reason, 0) + 1
        return counts


# Per-type shape patterns. Calibrated against the first Stage 1 run's
# query distribution; intent is to catch obvious type drift (procedural-
# shaped queries in the comparison bucket, etc.) without false-rejecting
# legitimate queries with overlapping vocabulary. Each pattern requires
# AT LEAST ONE shape-distinctive token. REFUSAL is exempt — refusal
# queries are valid in any shape as long as they're off-corpus.
_PROCEDURAL_SHAPE_RE = re.compile(r"^\s*how\b", re.IGNORECASE)
_COMPARISON_SHAPE_RE = re.compile(
    # vocabulary: difference / differs / different / vs / versus / compared / instead of
    r"\b(differ(s|ence|ent)?|vs\.?|versus|compared\s+(to|with)|instead\s+of)\b"
    # decision framing: "should I use X or Y" / "when should I use X vs Y"
    r"|^\s*should\s+i\s+use\b"
    r"|^\s*when\s+(should|do)\s+i\s+use\b",
    re.IGNORECASE,
)
_BEST_PRACTICE_SHAPE_RE = re.compile(
    r"\b(recommend(ed)?|best\s+(way|practice|approach)|"
    r"should\s+i|how\s+should\s+i|preferred?|advise[sd]?)\b",
    re.IGNORECASE,
)
_POINTER_SHAPE_RE = re.compile(
    r"\bwhere\b|\b(starting\s+point|good\s+place\s+to\s+start|"
    r"find\s+(the|an?)|locate\s+the)\b",
    re.IGNORECASE,
)


def _matches_type_shape(query: str, qt: QuestionType) -> bool:
    """True iff ``query``'s phrasing matches the expected shape for ``qt``."""
    if qt == QuestionType.PROCEDURAL:
        return bool(_PROCEDURAL_SHAPE_RE.search(query))
    if qt == QuestionType.COMPARISON:
        return bool(_COMPARISON_SHAPE_RE.search(query))
    if qt == QuestionType.BEST_PRACTICE:
        return bool(_BEST_PRACTICE_SHAPE_RE.search(query))
    if qt == QuestionType.POINTER:
        return bool(_POINTER_SHAPE_RE.search(query))
    # REFUSAL: any phrasing is acceptable; the chunk-mismatch property
    # (off-corpus topic OR mismatched chunks) is what defines refusals.
    return True


def filter_by_type_shape(queries: Sequence[GeneratedQuery]) -> FilterReport:
    """Drop queries whose phrasing doesn't match their assigned type.

    The chunk classifier sometimes flags chunks as comparison-affine
    or pointer-affine based on heuristic vocabulary in the chunk, but
    the chunk's actual content may not support that question shape.
    When the generator is asked to produce a comparison question for
    such a chunk, it tends to produce a procedural-shaped question
    instead (Haiku does its best with what's there).

    These mis-typed queries are still good queries — they just shouldn't
    sit in the wrong bucket. Type accuracy matters for downstream eval
    breakdowns and for the proportional sampling we engineered.

    REFUSAL queries are exempt — their defining property is chunk
    mismatch (off-corpus topic OR mismatched chunks), not phrasing.
    """
    report = FilterReport()
    for q in queries:
        if _matches_type_shape(q.query, q.question_type):
            report.kept.append(q)
        else:
            report.dropped.append((q, f"type_mismatch_{q.question_type.value}"))
    return report


def filter_by_length(
    queries: Sequence[GeneratedQuery],
    *,
    min_words: int = 5,
) -> FilterReport:
    """Drop queries shorter than ``min_words`` words.

    Word count via plain ``str.split()``. Multi-token identifiers like
    ``AutoModelForCausalLM`` count as one word — that's fine for our
    purpose (filtering vague/trivial queries, not enforcing fluency).
    """
    report = FilterReport()
    for q in queries:
        if len(q.query.split()) < min_words:
            report.dropped.append((q, f"length<{min_words}_words"))
        else:
            report.kept.append(q)
    return report


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize so cosine similarity reduces to a dot product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def filter_duplicates(
    queries: Sequence[GeneratedQuery],
    embedder: Embedder,
    *,
    threshold: float = 0.85,
) -> FilterReport:
    """Drop near-duplicate queries within each question type.

    Stratification: dedupe is per-type. A procedural query and a
    comparison query that happen to embed similarly stay — they're
    answering different question shapes, so the training signals are
    distinct.

    Pair-resolution policy: when two queries are near-duplicates, keep
    the FIRST one encountered in input order. Stable across runs given
    a stable input order.

    Cosine threshold defaults to 0.85, calibrated for "obvious
    paraphrase" — tighter and we'd miss real duplicates, looser and
    we'd over-prune diverse-but-related queries.
    """
    report = FilterReport()
    if not queries:
        return report

    # Group by question_type; dedupe within each group only.
    by_type: dict[str, list[tuple[int, GeneratedQuery]]] = {}
    for idx, q in enumerate(queries):
        by_type.setdefault(q.question_type.value, []).append((idx, q))

    drop_reason: dict[int, str] = {}
    for type_name, group in by_type.items():
        if len(group) < 2:
            continue
        texts = [q.query for _, q in group]
        embeddings = _normalize(embedder(texts))
        sims = embeddings @ embeddings.T
        np.fill_diagonal(sims, -np.inf)

        kept_local: list[int] = []
        for i in range(len(group)):
            is_dup = any(sims[i, k] > threshold for k in kept_local)
            if is_dup:
                drop_reason[group[i][0]] = f"duplicate_within_{type_name}"
            else:
                kept_local.append(i)

    for idx, q in enumerate(queries):
        if idx in drop_reason:
            report.dropped.append((q, drop_reason[idx]))
        else:
            report.kept.append(q)
    return report


def filter_eval_contamination(
    queries: Sequence[GeneratedQuery],
    eval_query_texts: Sequence[str],
    embedder: Embedder,
    *,
    threshold: float = 0.7,
) -> FilterReport:
    """Drop queries too similar to existing eval queries.

    NOT type-stratified — a procedural training query that overlaps
    with a comparison eval query is still contamination. Phase 3's
    fine-tuning eval compares fine-tuned vs base on those eval
    queries; if any of them appear (even paraphrased) in training,
    the comparison is meaningless.

    Default threshold 0.7 is intentionally loose. Eval contamination
    is asymmetric: false positives (rejecting a legitimate query) are
    cheap (we have many candidates), false negatives (accepting a
    contaminated query) destroy the eval. Skew toward false-positive.
    """
    report = FilterReport()
    if not queries:
        return report
    if not eval_query_texts:
        # Nothing to check against; every query passes.
        report.kept = list(queries)
        return report

    train_texts = [q.query for q in queries]
    train_emb = _normalize(embedder(train_texts))
    eval_emb = _normalize(embedder(list(eval_query_texts)))

    # (n_train, n_eval) similarity matrix → max similarity per train query.
    sims = train_emb @ eval_emb.T
    max_sim_per_train = sims.max(axis=1)

    for q, max_sim in zip(queries, max_sim_per_train, strict=True):
        if max_sim > threshold:
            report.dropped.append((q, f"eval_contamination_sim={max_sim:.3f}"))
        else:
            report.kept.append(q)
    return report
