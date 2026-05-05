#!/usr/bin/env python3
"""Reproducible chunking-strategy bakeoff runner.

Loads the FAISS indices built by `scripts/build_index.py`, runs the
curated eval queries against each strategy at the requested ``eval_k``,
and writes a JSON report to ``evaluations/reports/<filename>.json``.

When a baseline file is provided (default:
``evaluations/baselines/phase1_chunking.json``), prints a markdown
delta table comparing this run to the baseline.

This script intentionally mirrors the methodology in
``notebooks/chunking_comparison.ipynb`` — same eval set, same metrics,
same chunk-vs-doc dedup at the eval boundary — so its output can be
checked against the committed Phase 1 baseline within tolerance. The
notebook stays as the GitHub-rendered artifact; this script is the
runner you actually invoke during dev.

Usage:
    python scripts/run_bakeoff.py
    python scripts/run_bakeoff.py --eval-k 10 --strategies recursive header
    python scripts/run_bakeoff.py --eval-k 5 --out evaluations/reports/k5.json

Defaults:
    --eval-k 10  (matches the committed Phase 1 baseline)
    --strategies fixed recursive header
    --baseline evaluations/baselines/phase1_chunking.json
    --out      evaluations/reports/bakeoff-<UTC-date>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from docsense.config import DATA_DIR, Settings
from docsense.embedding.embedder import Embedder
from docsense.evaluation.eval_queries import CURATED_QUERIES, EvalQuery
from docsense.evaluation.retrieval_metrics import (
    deduplicate_preserving_order,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from docsense.reranking.reranker import CrossEncoderReranker
from docsense.retrieval.dense import DenseRetriever
from docsense.retrieval.hybrid import HybridRetriever
from docsense.retrieval.sparse import SparseRetriever

if TYPE_CHECKING:
    from docsense.chunking.base import Chunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = DATA_DIR / "index"
DEFAULT_BASELINE = PROJECT_ROOT / "evaluations" / "baselines" / "phase1_chunking.json"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "evaluations" / "reports"

ALL_STRATEGIES = ("fixed", "recursive", "header")
METRIC_KS = (1, 3, 5, 10)


def _is_relevant(doc_id: str, prefixes: list[str]) -> bool:
    return any(doc_id.startswith(p) for p in prefixes)


def _load_index_and_chunks(strategy: str) -> tuple[faiss.Index, list[Chunk]]:
    idx_dir = INDEX_DIR / strategy
    if not idx_dir.exists():
        msg = (
            f"No index at {idx_dir}. Run "
            f"`python scripts/build_index.py --strategy {strategy}` first."
        )
        raise FileNotFoundError(msg)

    index = faiss.read_index(str(idx_dir / "index.faiss"))
    with open(idx_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)  # noqa: S301
    return index, chunks


def _eval_strategy_dense(
    strategy: str,
    embedder: Embedder,
    eval_set: list[EvalQuery],
    eval_k: int,
) -> dict:
    """Run dense-only retrieval for one strategy; return metrics dict.

    Matches the Phase 1 baseline methodology: raw FAISS index search, no
    BM25, no fusion, no reranker. Over-retrieves at chunk granularity, then
    dedupes to ~eval_k unique doc_ids.
    """
    index, chunks = _load_index_and_chunks(strategy)
    logger.info("  %s: index=%d vectors, %d chunks", strategy, index.ntotal, len(chunks))

    # Over-retrieve at chunk granularity, then dedupe to ~eval_k unique doc_ids.
    # Multiplier of 4 matches the notebook; high enough that dedup rarely
    # leaves us short of eval_k.
    chunk_k = min(eval_k * 4, index.ntotal)

    per_query: list[dict] = []
    for query, prefixes in eval_set:
        query_emb = embedder.embed_texts([query])
        _scores, idxs = index.search(query_emb, chunk_k)

        chunk_doc_ids: list[str] = []
        for idx in idxs[0]:
            if idx == -1:
                break
            chunk_doc_ids.append(chunks[idx].doc_id)
        retrieved = deduplicate_preserving_order(chunk_doc_ids)

        relevant = {c.doc_id for c in chunks if _is_relevant(c.doc_id, prefixes)}
        per_query.append({"retrieved": retrieved, "relevant": relevant})

    return _aggregate_metrics(per_query, eval_k=eval_k, chunks_total=len(chunks))


def _eval_strategy_rerank(
    strategy: str,
    embedder: Embedder,
    eval_set: list[EvalQuery],
    eval_k: int,
    settings: Settings,
) -> dict:
    """Run the full HybridRetriever + cross-encoder pipeline for one strategy.

    Methodological note: this differs from the dense-only path in *three*
    ways simultaneously — it adds BM25, adds RRF fusion of dense+sparse,
    and adds the cross-encoder reranker. The bakeoff therefore measures
    the combined effect of "the production retrieval stack" vs. the
    Phase 1 dense-only baseline. A future `--hybrid` flag (no rerank)
    would let us ablate the reranker contribution alone.
    """
    index, chunks = _load_index_and_chunks(strategy)
    logger.info("  %s: index=%d vectors, %d chunks", strategy, index.ntotal, len(chunks))

    dense = DenseRetriever(dimension=index.d)
    dense.index = index
    dense.chunks = chunks

    sparse = SparseRetriever()
    sparse.add(chunks)

    reranker = CrossEncoderReranker(settings.reranking)
    hybrid = HybridRetriever(dense, sparse, embedder, settings.retrieval, reranker=reranker)

    # Over-retrieve to give the cross-encoder a richer pool *and* leave
    # enough headroom that dedup yields ~eval_k unique doc_ids. Same
    # multiplier (×4) as the dense-only path so the eval is symmetric.
    eval_top_k = min(eval_k * 4, index.ntotal)

    per_query: list[dict] = []
    for query, prefixes in eval_set:
        results = hybrid.search(query, top_k=eval_top_k)
        retrieved = deduplicate_preserving_order([r.chunk.doc_id for r in results])
        relevant = {c.doc_id for c in chunks if _is_relevant(c.doc_id, prefixes)}
        per_query.append({"retrieved": retrieved, "relevant": relevant})

    return _aggregate_metrics(per_query, eval_k=eval_k, chunks_total=len(chunks))


def _aggregate_metrics(per_query: list[dict], *, eval_k: int, chunks_total: int) -> dict:
    metrics: dict[str, float] = {}
    for k in METRIC_KS:
        if k > eval_k:
            continue
        metrics[f"P@{k}"] = float(
            np.mean([precision_at_k(q["retrieved"], q["relevant"], k) for q in per_query])
        )
        metrics[f"Recall@{k}"] = float(
            np.mean([recall_at_k(q["retrieved"], q["relevant"], k) for q in per_query])
        )
        metrics[f"nDCG@{k}"] = float(
            np.mean([ndcg_at_k(q["retrieved"], q["relevant"], k) for q in per_query])
        )

    metrics["MRR"] = float(
        np.mean([mean_reciprocal_rank(q["retrieved"], q["relevant"]) for q in per_query])
    )
    # Hit rate at top-5: fraction of queries with at least one relevant doc in top 5
    metrics["hit_rate_top_5"] = float(
        np.mean([any(d in q["relevant"] for d in q["retrieved"][:5]) for q in per_query])
    )
    return {"chunks_total": chunks_total, "metrics": metrics}


def _format_metric(v: float) -> str:
    return f"{v:.3f}"


def _format_delta(current: float, baseline: float) -> str:
    delta = current - baseline
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"


def _print_markdown_delta(results: dict, baseline: dict) -> None:
    """Print a markdown comparison table of current results vs. baseline."""
    metrics_to_show = ("MRR", "P@1", "Recall@10", "nDCG@10", "hit_rate_top_5")
    print()
    print(f"## Bakeoff vs baseline ({baseline.get('description', 'baseline')[:60]}...)")
    print()
    print("| Strategy | Metric | Baseline | This run | Δ |")
    print("|---|---|---:|---:|---:|")
    for strategy in results["strategies"]:
        if strategy not in baseline.get("strategies", {}):
            continue
        for metric in metrics_to_show:
            base = baseline["strategies"][strategy]["metrics"].get(metric)
            curr = results["strategies"][strategy]["metrics"].get(metric)
            if base is None or curr is None:
                continue
            print(
                f"| {strategy} | {metric} | {_format_metric(base)} | "
                f"{_format_metric(curr)} | {_format_delta(curr, base)} |"
            )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--eval-k",
        type=int,
        default=10,
        help=(
            "Metric horizon: compute Recall@k, nDCG@k, etc. up to this value. "
            "Default 10 matches the Phase 1 baseline."
        ),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=ALL_STRATEGIES,
        default=list(ALL_STRATEGIES),
        help="Which chunking strategies to evaluate (default: all three).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Path to a baseline JSON to diff against. Pass empty to skip.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Use the full HybridRetriever + cross-encoder pipeline instead of "
            "dense-only. Note: this also enables BM25 + RRF fusion, so the "
            "comparison vs Phase 1 conflates three additions; see the "
            "methodology note in the report."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: evaluations/reports/bakeoff-<UTC-date>.json).",
    )
    args = parser.parse_args()

    if args.eval_k <= 0:
        parser.error("--eval-k must be positive")

    settings = Settings()
    embedder = Embedder(settings.embedding)
    logger.info("Embedder: %s", settings.embedding.model_name)
    logger.info("eval_k=%d, strategies=%s", args.eval_k, args.strategies)

    pipeline = "hybrid+rerank" if args.rerank else "dense-only"
    suffix = f"-{pipeline.replace('+', '-')}"
    out_path = args.out or DEFAULT_REPORTS_DIR / (
        f"bakeoff-{datetime.now(tz=UTC).strftime('%Y%m%d')}{suffix}.json"
    )

    results: dict = {
        "schema_version": 1,
        "captured_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "eval_k": args.eval_k,
        "pipeline": pipeline,
        "embedding_model": settings.embedding.model_name,
        "reranker_model": (settings.reranking.model_name if args.rerank else None),
        "retrieval": {
            "rerank_candidates": (settings.retrieval.rerank_candidates if args.rerank else None),
            "dense_weight": (settings.retrieval.dense_weight if args.rerank else None),
            "sparse_weight": (settings.retrieval.sparse_weight if args.rerank else None),
        },
        "eval_set": "curated",
        "eval_set_size": len(CURATED_QUERIES),
        "strategies": {},
    }
    for strategy in args.strategies:
        logger.info("Evaluating %s ...", strategy)
        if args.rerank:
            results["strategies"][strategy] = _eval_strategy_rerank(
                strategy, embedder, CURATED_QUERIES, eval_k=args.eval_k, settings=settings
            )
        else:
            results["strategies"][strategy] = _eval_strategy_dense(
                strategy, embedder, CURATED_QUERIES, eval_k=args.eval_k
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=False) + "\n")
    logger.info("Report written to %s", out_path)

    if args.baseline and args.baseline != Path() and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text())
        _print_markdown_delta(results, baseline)
    else:
        logger.warning("No baseline at %s; skipping delta table.", args.baseline)

    return 0


if __name__ == "__main__":
    sys.exit(main())
