"""Generate evaluation queries programmatically from document structure.

This module exists to complement the hand-curated queries in
:mod:`docsense.evaluation.eval_queries`. Hand-curated queries reward
retrieval that mirrors the curator's mental model of the corpus — they are
a useful eval but a biased one. By contrast, structural queries are derived
deterministically from each document's ``##``-level subheadings, with no
human selection in the loop. Comparing the two evaluations gives a stronger
signal about retrieval quality than either alone:

- Agreement between the two on relative strategy ranking → high confidence
  in the bakeoff conclusions.
- Disagreement → a flag worth investigating.

The structural set is intentionally noisier than the curated set (some
headings are still vague even after filtering), but it's noisy *without*
the curator's prior, which is the whole point.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docsense.evaluation.eval_queries import EvalQuery
    from docsense.ingestion.loader import Document


# Markdown ``## ...`` header pattern. Captures the heading text after the
# hashes, before any ``[[anchor]]`` syntax sometimes used in HF docs.
_H2_RE = re.compile(r"^##\s+(.+?)(?:\s*\[\[[^\]]+\]\])?\s*$", re.MULTILINE)

# Headings that appear across many docs and don't make discriminative
# eval queries. Lowercased for case-insensitive matching.
_GENERIC_HEADINGS = frozenset(
    {
        "overview",
        "introduction",
        "usage",
        "setup",
        "installation",
        "examples",
        "example",
        "resources",
        "references",
        "license",
        "notes",
        "note",
        "see also",
        "citation",
        "citations",
        "further reading",
        "quickstart",
        "quick start",
        "background",
        "configuration",
        "config",
        "arguments",
        "parameters",
        "api",
        "api reference",
        "supported models",
        "tips",
        "warnings",
        "limitations",
        "summary",
        "conclusion",
        "appendix",
    }
)

# Headings that are "Note:"-style annotations, not section titles.
_ANNOTATION_PREFIXES = ("note:", "tip:", "warning:", "todo:", "fixme:")


def _is_meaningful_heading(heading: str) -> bool:
    """Filter heading text down to ones likely to be discriminative queries."""
    text = heading.strip()
    if not text:
        return False

    lower = text.lower()
    if lower in _GENERIC_HEADINGS:
        return False
    if any(lower.startswith(p) for p in _ANNOTATION_PREFIXES):
        return False

    word_count = len(text.split())
    return not (word_count < 2 or len(text) < 12)


def extract_meaningful_headings(document: Document) -> list[str]:
    """Return all ``##`` subheadings from a document that pass the meaningful-
    heading filter, with markdown anchor syntax stripped."""
    return [h for h in _H2_RE.findall(document.content) if _is_meaningful_heading(h)]


def generate_structural_queries(
    documents: list[Document],
    n_queries: int = 30,
    seed: int = 42,
) -> list[EvalQuery]:
    """Sample structural eval queries from document headings.

    For each document with at least one meaningful subheading, the longest
    such heading is taken as a candidate query whose expected relevant doc
    is the source document. ``n_queries`` candidates are sampled (without
    replacement) using ``seed`` for reproducibility.

    Returns the same ``(query, [doc_id_prefix])`` shape used by the curated
    eval set, so the same evaluation pipeline can consume both.
    """
    rng = random.Random(seed)

    candidates: list[EvalQuery] = []
    for doc in documents:
        headings = extract_meaningful_headings(doc)
        if not headings:
            continue
        # Longest heading per doc — typically the most specific / least
        # ambiguous candidate available.
        chosen = max(headings, key=len)
        candidates.append((chosen, [doc.doc_id]))

    if n_queries >= len(candidates):
        return candidates

    return rng.sample(candidates, n_queries)
