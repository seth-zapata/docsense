"""Retrieval quality metrics: precision@k, recall@k, MRR, nDCG."""

from __future__ import annotations

import numpy as np


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

    ideal_gains = sorted(gains, reverse=True)
    n_relevant_in_k = min(len(relevant_ids), k)
    ideal_gains = [1.0] * n_relevant_in_k + [0.0] * (k - n_relevant_in_k)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))

    return dcg / idcg if idcg > 0 else 0.0
