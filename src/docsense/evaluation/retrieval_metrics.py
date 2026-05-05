"""Retrieval quality metrics: precision@k, recall@k, MRR, nDCG.

All metrics here assume `retrieved_ids` and `relevant_ids` are at the **same
granularity**. If you retrieve at chunk level but judge relevance at document
level, multiple chunks from the same relevant document will each count as a
hit, which inflates recall (can exceed 1.0) and nDCG. Use
:func:`deduplicate_preserving_order` to convert a chunk-level ranking into a
document-level ranking before passing it to these functions.
"""

from __future__ import annotations

import numpy as np


def deduplicate_preserving_order(items: list[str]) -> list[str]:
    """Return unique items in first-occurrence order.

    Useful for converting a chunk-level retrieval ranking (where multiple
    chunks share a doc_id) into a document-level ranking suitable for the
    metrics in this module.
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant documents found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal of the rank of the first relevant document."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized discounted cumulative gain at k."""
    if not relevant_ids or k == 0:
        return 0.0

    top_k = retrieved_ids[:k]
    gains = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in top_k]

    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))

    n_relevant_in_k = min(len(relevant_ids), k)
    ideal_gains = [1.0] * n_relevant_in_k + [0.0] * (k - n_relevant_in_k)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))

    return dcg / idcg if idcg > 0 else 0.0
