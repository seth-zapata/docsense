#!/usr/bin/env python3
"""Build the Stage 1 query pool for Block 3B.2 training-data seeding.

Pipeline:

  1. Load the indexed corpus chunks.
  2. Classify chunks by question-type affinity (chunk_classifier).
  3. Sample chunks per type per quotas (default 240/160/160/120).
  4. Generate in-corpus queries via Haiku 4.5 — RESUMABLE.
  5. Generate off-corpus refusal queries via Haiku 4.5 — RESUMABLE.
  6. Synthesize retrieval-failure refusals (no LLM) — pairing in-corpus
     queries with mismatched chunks.
  7. Run hybrid retrieval per query → populate retrieved_chunks.
  8. Apply quality filters (length → dedupe → eval-contamination).
  9. Write artifacts: query_pool.jsonl, filter_report.md, preview.md.

Resumability: per-query JSONL append at ``<output>/raw_queries.jsonl``.
On restart, the script reads the file, builds a set of task keys
already done, and skips them. Generation can be killed and resumed
without losing prior progress.

Hand-review checkpoint: ``<output>/preview.md`` contains a stratified
random sample of 30 queries across types — eyeball before running
distillation (Block 3B.2.f). If the preview reveals systematic
quality issues, delete the pool and re-run with adjustments.

Usage:
  python scripts/build_query_pool.py
  python scripts/build_query_pool.py --output-dir /tmp/pool --strategy header
  python scripts/build_query_pool.py --no-resume       # start fresh
  python scripts/build_query_pool.py --dry-run         # plan only, no API calls
  python scripts/build_query_pool.py --limit 20        # smoke test (small quotas)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anthropic
import faiss

from docsense.config import DATA_DIR, Settings
from docsense.embedding.embedder import Embedder
from docsense.evaluation.eval_queries import CURATED_QUERIES
from docsense.evaluation.no_answer_queries import NO_ANSWER_QUERIES
from docsense.finetuning.chunk_classifier import QuestionType, classify_chunk
from docsense.finetuning.query_filters import (
    FilterReport,
    filter_by_length,
    filter_duplicates,
    filter_eval_contamination,
)
from docsense.finetuning.query_generation import (
    GeneratedQuery,
    TypeAwareQueryGenerator,
)
from docsense.finetuning.refusal_seeds import RefusalSeeds, load_default_seeds
from docsense.generation.types import ChunkRef
from docsense.retrieval.dense import DenseRetriever
from docsense.retrieval.hybrid import HybridRetriever
from docsense.retrieval.sparse import SparseRetriever

if TYPE_CHECKING:
    from docsense.chunking.base import Chunk


logger = logging.getLogger(__name__)


# Default quotas from docs/phase-3-block-3b2-plan.md (30/20/20/15/15
# of 800 = 240/160/160/120 in-corpus + 120 refusal).
DEFAULT_QUOTAS: dict[QuestionType, int] = {
    QuestionType.PROCEDURAL: 240,
    QuestionType.COMPARISON: 160,
    QuestionType.BEST_PRACTICE: 160,
    QuestionType.POINTER: 120,
}
# Refusal split: 80 off-corpus + 40 retrieval-failure (per plan doc).
# 30 seeds × 3 = 90 ≈ 80 (slight buffer).
DEFAULT_REFUSALS_PER_SEED = 3
DEFAULT_RETRIEVAL_FAILURE_COUNT = 40

DEFAULT_OUTPUT_DIR = Path("evaluations/datasets/training/query_pool")
INDEX_DIR = DATA_DIR / "index"
DEFAULT_STRATEGY = "header"

# When populating retrieved_chunks, fetch this many. Matches the
# top_k used by the inference pipeline so train and serve see the
# same retrieval depth.
RETRIEVAL_TOP_K = 5

# Where to look for the API key. Mirrors build_training_dataset.py's
# convention (mode 600 file in $HOME).
KEY_FILE = Path.home() / ".anthropic-key"


# --------------------------------------------------------------------
# Data shapes
# --------------------------------------------------------------------


@dataclass(frozen=True)
class TaskKey:
    """Identifies a generation task for resumability dedup.

    For in-corpus types: ``(question_type, chunk_id)``. For off-corpus
    refusals: ``(REFUSAL, "<seed_id>::<seq>")``. The string form is
    persisted in raw_queries.jsonl alongside each generated query.
    """

    question_type: QuestionType
    identifier: str

    def to_str(self) -> str:
        return f"{self.question_type.value}::{self.identifier}"


# --------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------


def _read_anthropic_key() -> str:
    if not KEY_FILE.exists():
        msg = (
            f"No Anthropic API key found at {KEY_FILE}. Create it with "
            f"`echo $ANTHROPIC_API_KEY > {KEY_FILE} && chmod 600 {KEY_FILE}`."
        )
        raise SystemExit(msg)
    return KEY_FILE.read_text().strip()


def _load_chunks_and_index(strategy: str) -> tuple[Any, list[Chunk]]:
    """Load the FAISS index + chunks pickle for ``strategy``."""
    idx_dir = INDEX_DIR / strategy
    if not idx_dir.exists():
        msg = (
            f"No index at {idx_dir}. Run "
            f"`python scripts/build_index.py --strategy {strategy}` first."
        )
        raise SystemExit(msg)
    index = faiss.read_index(str(idx_dir / "index.faiss"))
    with open(idx_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)  # noqa: S301
    return index, chunks


def _build_hybrid_retriever(
    index: Any,
    chunks: list[Chunk],
    settings: Settings,
) -> HybridRetriever:
    """Build a HybridRetriever WITHOUT the cross-encoder reranker.

    Train/serve consistency: the distilled LLM is fed the post-RRF top-K
    at inference, NOT the reranked top-K, because the reranker is too
    slow for the ~12-16 s p50 latency budget and is currently disabled
    in the production path. Match that here so training-time chunks =
    serving-time chunks.
    """
    embedder = Embedder(settings.embedding)
    dense = DenseRetriever(dimension=index.d)
    dense.index = index
    dense.chunks = chunks
    sparse = SparseRetriever()
    sparse.add(chunks)
    return HybridRetriever(dense, sparse, embedder, settings.retrieval, reranker=None)


def _load_eval_query_texts() -> list[str]:
    """Concatenate all eval query texts (curated + no-answer) for the
    contamination filter. Structural queries are dynamically generated
    so they're not in the static contamination check — acceptable
    because structural eval is small and overlap risk is low."""
    return [q[0] for q in CURATED_QUERIES] + list(NO_ANSWER_QUERIES)


# --------------------------------------------------------------------
# Resumability — read/write raw_queries.jsonl
# --------------------------------------------------------------------


def _serialize_raw_query(query: GeneratedQuery, task_key: TaskKey) -> str:
    """Serialize a GeneratedQuery + its task key as one JSONL line."""
    payload = {
        "task_key": task_key.to_str(),
        "query": query.model_dump(mode="json"),
    }
    return json.dumps(payload)


def _read_raw_queries(raw_path: Path) -> list[tuple[TaskKey, GeneratedQuery]]:
    """Read all (task_key, query) tuples from raw_queries.jsonl."""
    if not raw_path.exists():
        return []
    entries: list[tuple[TaskKey, GeneratedQuery]] = []
    for line in raw_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        qt_str, identifier = obj["task_key"].split("::", 1)
        key = TaskKey(QuestionType(qt_str), identifier)
        query = GeneratedQuery.model_validate(obj["query"])
        entries.append((key, query))
    return entries


def _load_done_task_keys(raw_path: Path) -> set[str]:
    """Return the set of task_key strings already in the raw file."""
    return {key.to_str() for key, _ in _read_raw_queries(raw_path)}


def _append_raw_query(
    raw_path: Path,
    task_key: TaskKey,
    query: GeneratedQuery,
) -> None:
    """Append one generated query to raw_queries.jsonl. Created if absent."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a") as f:
        f.write(_serialize_raw_query(query, task_key) + "\n")


# --------------------------------------------------------------------
# Sampling + classification
# --------------------------------------------------------------------


def _classify_corpus(chunks: list[Chunk]) -> dict[QuestionType, list[Chunk]]:
    """Group chunks by their classifier-detected affinities (multi-label)."""
    by_type: dict[QuestionType, list[Chunk]] = {
        QuestionType.PROCEDURAL: [],
        QuestionType.COMPARISON: [],
        QuestionType.BEST_PRACTICE: [],
        QuestionType.POINTER: [],
    }
    for chunk in chunks:
        for qt in classify_chunk(chunk.text):
            by_type[qt].append(chunk)
    return by_type


def _sample_chunks_per_type(
    chunks_by_type: dict[QuestionType, list[Chunk]],
    quotas: dict[QuestionType, int],
    seed: int = 42,
) -> dict[QuestionType, list[Chunk]]:
    """Random-sample chunks per type per quota (without replacement).

    If a quota exceeds available chunks, take all available and log a
    warning — the script continues with the smaller set rather than
    failing. Caller should decide whether to retry with smaller quotas
    or skip the affected type.

    Sampling is uniform, NOT stratified across docs. The plan calls
    for doc-stratification ("not 200 questions all about Trainer") but
    we start uniform — re-evaluate after the first run if concentration
    is an issue.
    """
    rng = random.Random(seed)
    sampled: dict[QuestionType, list[Chunk]] = {}
    for qt, n in quotas.items():
        candidates = chunks_by_type.get(qt, [])
        if not candidates:
            logger.warning("No chunks classified as %s — skipping", qt.value)
            sampled[qt] = []
            continue
        if len(candidates) < n:
            logger.warning(
                "Quota %d exceeds available chunks %d for %s — taking all available",
                n,
                len(candidates),
                qt.value,
            )
            sampled[qt] = list(candidates)
            continue
        sampled[qt] = rng.sample(candidates, n)
    return sampled


def _chunk_to_chunkref(chunk: Chunk) -> ChunkRef:
    """Convert a corpus Chunk (dataclass) to a ChunkRef (pydantic).

    ``score=0.0`` because seed chunks aren't scored — they're picked by
    the sampler, not by retrieval. Retrieved chunks (post-retrieval)
    get real scores from the hybrid retriever.
    """
    return ChunkRef(
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        score=0.0,
        text=chunk.text,
    )


# --------------------------------------------------------------------
# Generation drivers (resumable)
# --------------------------------------------------------------------


def _generate_in_corpus_queries(
    generator: TypeAwareQueryGenerator | None,
    sampled: dict[QuestionType, list[Chunk]],
    raw_path: Path,
    done: set[str],
    *,
    dry_run: bool = False,
) -> int:
    """For each (chunk, qt) pair, call generator if not in ``done``.

    Each successful generation is appended to raw_path immediately so a
    crash mid-batch only loses the in-flight call. Parse failures are
    logged + skipped (not raised) so one bad chunk doesn't abort the
    whole run.
    """
    n_done = 0
    for qt, chunks in sampled.items():
        for chunk in chunks:
            chunk_ref = _chunk_to_chunkref(chunk)
            key = TaskKey(qt, chunk.chunk_id)
            if key.to_str() in done:
                continue
            if dry_run:
                logger.info("[dry-run] would generate %s", key.to_str())
                n_done += 1
                continue
            try:
                gq = generator.generate_for_chunk(chunk_ref, qt)
            except RuntimeError as e:
                logger.warning("parse failure for %s: %s — skipping", key.to_str(), e)
                continue
            _append_raw_query(raw_path, key, gq)
            n_done += 1
            if n_done % 50 == 0:
                logger.info("  generated %d in-corpus queries so far", n_done)
    return n_done


def _generate_off_corpus_refusals(
    generator: TypeAwareQueryGenerator | None,
    seeds: RefusalSeeds,
    n_per_seed: int,
    raw_path: Path,
    done: set[str],
    *,
    dry_run: bool = False,
) -> int:
    """For each seed, generate ``n_per_seed`` queries via topic prompt."""
    n_done = 0
    for seed in seeds.seeds:
        for seq in range(n_per_seed):
            key = TaskKey(QuestionType.REFUSAL, f"{seed.id}::{seq}")
            if key.to_str() in done:
                continue
            if dry_run:
                logger.info("[dry-run] would generate %s", key.to_str())
                n_done += 1
                continue
            try:
                gq = generator.generate_for_topic(seed.topic)
            except RuntimeError as e:
                logger.warning("parse failure for %s: %s — skipping", key.to_str(), e)
                continue
            _append_raw_query(raw_path, key, gq)
            n_done += 1
    return n_done


def _synthesize_retrieval_failure_refusals(
    in_corpus_queries: list[GeneratedQuery],
    chunks: list[Chunk],
    n_target: int,
    seed: int = 42,
) -> list[GeneratedQuery]:
    """Synthesize refusal queries by pairing in-corpus queries with
    chunks from a *different* doc topic.

    Method: pick ``n_target`` random in-corpus queries. For each, find
    chunks whose ``doc_id`` shares NO leading path component with the
    query's seed chunks (e.g., a query seeded from ``model_doc/llava.md``
    gets paired with chunks from ``quantization/`` or ``trainer.md``).
    The result is a refusal-shaped pool entry: query is realistic,
    chunks are tangentially related, no answer is derivable.

    This is the highest-leverage refusal signal — teaches the production
    failure mode where retrieval returns topically-adjacent-but-non-
    answering chunks. Without it, Qwen learns "if there are chunks,
    answer using them".
    """
    rng = random.Random(seed)
    pool = [q for q in in_corpus_queries if q.seed_chunks]
    if not pool:
        logger.warning("No in-corpus queries to synthesize retrieval-failure refusals from")
        return []
    sampled_queries = rng.sample(pool, min(n_target, len(pool)))

    # Pre-group chunks by top-level doc family to find mismatches fast.
    by_family: dict[str, list[Chunk]] = {}
    for c in chunks:
        family = c.doc_id.split("/")[0] if "/" in c.doc_id else c.doc_id
        by_family.setdefault(family, []).append(c)

    refusals: list[GeneratedQuery] = []
    for original in sampled_queries:
        seed_family = (
            original.seed_chunks[0].doc_id.split("/")[0]
            if "/" in original.seed_chunks[0].doc_id
            else original.seed_chunks[0].doc_id
        )
        # Pool of chunks from any OTHER doc family.
        mismatched_families = [f for f in by_family if f != seed_family]
        if not mismatched_families:
            continue
        # Pick 5 random chunks from random mismatched families.
        mismatched_chunks: list[Chunk] = []
        for _ in range(5):
            fam = rng.choice(mismatched_families)
            mismatched_chunks.append(rng.choice(by_family[fam]))

        refusals.append(
            GeneratedQuery(
                query=original.query,
                question_type=QuestionType.REFUSAL,
                seed_chunks=[_chunk_to_chunkref(c) for c in mismatched_chunks],
                seed_topic=None,
                metadata={
                    **original.metadata,
                    "synthesized_from": original.metadata.get("captured_at", "unknown"),
                    "synthesis_method": "retrieval_failure_pairing",
                    "seed_family_avoided": seed_family,
                },
            )
        )
    return refusals


# --------------------------------------------------------------------
# Retrieval + filtering
# --------------------------------------------------------------------


@dataclass
class PoolEntry:
    """One entry in the final query pool — generated query + retrieved chunks.

    Kept separate from GeneratedQuery because retrieved_chunks comes
    from the hybrid retriever (not the generator) and conceptually
    belongs to the seeder's stage.
    """

    generated: GeneratedQuery
    retrieved_chunks: list[ChunkRef]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the pool JSONL shape consumed by the distillation script."""
        return {
            "query": self.generated.query,
            "question_type": self.generated.question_type.value,
            "expected_refusal": self.generated.question_type == QuestionType.REFUSAL,
            "retrieved_chunks": [c.model_dump(mode="json") for c in self.retrieved_chunks],
            "seed_chunks": [c.model_dump(mode="json") for c in self.generated.seed_chunks],
            "seed_topic": self.generated.seed_topic,
            "metadata": self.generated.metadata,
        }


def _populate_retrieved_chunks(
    queries: list[GeneratedQuery],
    retriever: HybridRetriever,
    *,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[PoolEntry]:
    """For each query, run retrieval and bundle results into a PoolEntry.

    For retrieval-failure refusals (seed_chunks already populated with
    mismatched chunks), we use those as ``retrieved_chunks`` directly —
    re-running retrieval would just re-fetch topically-relevant chunks
    and undo the mismatch we engineered.
    """
    entries: list[PoolEntry] = []
    for q in queries:
        is_synthetic_refusal = (
            q.question_type == QuestionType.REFUSAL
            and q.seed_chunks
            and q.metadata.get("synthesis_method") == "retrieval_failure_pairing"
        )
        if is_synthetic_refusal:
            entries.append(PoolEntry(generated=q, retrieved_chunks=list(q.seed_chunks)))
            continue
        results = retriever.search(q.query, top_k=top_k)
        retrieved = [
            ChunkRef(
                doc_id=r.chunk.doc_id,
                chunk_id=r.chunk.chunk_id,
                score=r.score,
                text=r.chunk.text,
            )
            for r in results
        ]
        entries.append(PoolEntry(generated=q, retrieved_chunks=retrieved))
    return entries


# --------------------------------------------------------------------
# Markdown artifact generators
# --------------------------------------------------------------------


def _format_filter_report_markdown(
    *,
    initial_count: int,
    after_length: FilterReport,
    after_dedupe: FilterReport,
    after_contamination: FilterReport,
) -> str:
    """Per-stage filter audit log for the seeder's reproducibility trail."""
    lines = [
        "# Stage 1 filter report",
        "",
        f"Started with **{initial_count}** generated queries.",
        "",
        "## Filter pass-through",
        "",
        "| Stage | Kept | Dropped | Reasons |",
        "|---|---:|---:|---|",
    ]
    for label, report in [
        ("length floor", after_length),
        ("dedupe (within-type)", after_dedupe),
        ("eval contamination", after_contamination),
    ]:
        reasons = ", ".join(f"{r}={n}" for r, n in sorted(report.reason_summary().items()))
        lines.append(
            f"| {label} | {report.kept_count} | {report.dropped_count} | {reasons or '—'} |"
        )
    lines.append("")
    lines.append(f"Final pool: **{after_contamination.kept_count}** queries.")
    return "\n".join(lines) + "\n"


def _format_preview_markdown(
    pool: list[PoolEntry],
    n_sample: int = 30,
    seed: int = 42,
) -> str:
    """Stratified random sample of ``n_sample`` queries, formatted for review.

    Sample sizes are proportional to type counts in the pool (e.g., if
    procedural is 30% of the pool, ~30% of the sample is procedural).
    """
    rng = random.Random(seed)
    by_type: dict[QuestionType, list[PoolEntry]] = {}
    for entry in pool:
        by_type.setdefault(entry.generated.question_type, []).append(entry)

    total = len(pool)
    lines = [
        f"# Stage 1 query-pool preview ({n_sample} sample of {total})",
        "",
        "Stratified random sample for hand-review. Eyeball check:",
        "",
        "- Are queries realistic? (would a developer actually ask this?)",
        "- Are queries answerable from the retrieved chunks?",
        "- Are retrieved chunks reasonable (not all garbage)?",
        "",
        "If a systematic quality issue surfaces, delete the pool and",
        "re-run with adjustments.",
        "",
    ]

    for qt, entries in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        per_type_n = max(1, round(n_sample * len(entries) / total)) if total else 0
        per_type_n = min(per_type_n, len(entries))
        sampled = rng.sample(entries, per_type_n)
        lines.append(f"## {qt.value} (sample {per_type_n} of {len(entries)})")
        lines.append("")
        for i, entry in enumerate(sampled, 1):
            lines.append(f"**{i}.** {entry.generated.query}")
            lines.append("")
            top_doc_ids = [c.doc_id for c in entry.retrieved_chunks[:3]]
            lines.append(f"  - top-3 retrieved doc_ids: `{'`, `'.join(top_doc_ids) or '(none)'}`")
            if entry.generated.seed_topic:
                lines.append(f"  - seed topic: `{entry.generated.seed_topic}`")
            elif entry.generated.seed_chunks:
                lines.append(f"  - seed chunk: `{entry.generated.seed_chunks[0].doc_id}`")
            lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------


def main() -> int:  # noqa: PLR0915 — orchestration script, sequential by nature
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--strategy",
        choices=("fixed", "recursive", "header"),
        default=DEFAULT_STRATEGY,
        help=f"Chunking strategy whose index to load (default: {DEFAULT_STRATEGY})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Delete existing raw_queries.jsonl and start fresh",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only — no API calls, no artifacts written",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap quotas to this many per type (smoke testing). Skips refusals when set.",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5",
        help="Anthropic model for query generation (default: claude-haiku-4-5)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_queries.jsonl"

    if args.no_resume and raw_path.exists():
        logger.info("--no-resume: deleting existing %s", raw_path)
        raw_path.unlink()

    quotas = dict(DEFAULT_QUOTAS)
    if args.limit is not None:
        quotas = {qt: min(args.limit, n) for qt, n in quotas.items()}
        logger.info("--limit: capping quotas to %d per type", args.limit)

    # 1. Load corpus
    settings = Settings()
    logger.info("Loading index + chunks (strategy=%s)...", args.strategy)
    index, chunks = _load_chunks_and_index(args.strategy)
    logger.info("  loaded %d chunks", len(chunks))

    # 2. Classify chunks
    logger.info("Classifying chunks by question-type affinity...")
    chunks_by_type = _classify_corpus(chunks)
    for qt, group in chunks_by_type.items():
        logger.info("  %s: %d candidate chunks", qt.value, len(group))

    # 3. Sample per type
    sampled = _sample_chunks_per_type(chunks_by_type, quotas, seed=args.seed)
    total_in_corpus_target = sum(len(v) for v in sampled.values())
    logger.info("Sampled %d chunks total for in-corpus generation", total_in_corpus_target)

    # 4. Resume bookkeeping
    done = _load_done_task_keys(raw_path)
    if done:
        logger.info("Resuming: %d tasks already done", len(done))

    # 5. Set up generator (skip in dry-run)
    generator: TypeAwareQueryGenerator | None = None
    if not args.dry_run:
        client = anthropic.Anthropic(api_key=_read_anthropic_key())
        generator = TypeAwareQueryGenerator(client, model=args.model)

    # 6. In-corpus generation (resumable)
    logger.info("Generating in-corpus queries...")
    t0 = time.time()
    n_in_corpus = _generate_in_corpus_queries(
        generator,
        sampled,
        raw_path,
        done,
        dry_run=args.dry_run,
    )
    logger.info("  +%d in-corpus queries (%.1f s)", n_in_corpus, time.time() - t0)

    # 7. Off-corpus refusal generation
    n_refusals = 0
    if args.limit is None:
        logger.info("Generating off-corpus refusal queries...")
        seeds = load_default_seeds()
        t0 = time.time()
        n_refusals = _generate_off_corpus_refusals(
            generator,
            seeds,
            DEFAULT_REFUSALS_PER_SEED,
            raw_path,
            done,
            dry_run=args.dry_run,
        )
        logger.info("  +%d off-corpus refusals (%.1f s)", n_refusals, time.time() - t0)

    if args.dry_run:
        logger.info("Dry run complete. Would have generated %d queries.", n_in_corpus + n_refusals)
        return 0

    # 8. Read all generated queries back
    raw = _read_raw_queries(raw_path)
    logger.info("Total raw queries on disk: %d", len(raw))
    all_queries = [q for _, q in raw]
    in_corpus_queries = [q for q in all_queries if q.question_type != QuestionType.REFUSAL]

    # 9. Synthesize retrieval-failure refusals
    failure_refusals = []
    if args.limit is None:
        logger.info(
            "Synthesizing %d retrieval-failure refusals...", DEFAULT_RETRIEVAL_FAILURE_COUNT
        )
        failure_refusals = _synthesize_retrieval_failure_refusals(
            in_corpus_queries,
            chunks,
            DEFAULT_RETRIEVAL_FAILURE_COUNT,
            seed=args.seed,
        )
        logger.info("  +%d retrieval-failure refusals", len(failure_refusals))

    # 10. Run retrieval per query
    logger.info("Building hybrid retriever (no reranker — train/serve parity)...")
    retriever = _build_hybrid_retriever(index, chunks, settings)
    logger.info(
        "Populating retrieved_chunks for %d queries...", len(all_queries) + len(failure_refusals)
    )
    t0 = time.time()
    pool_entries = _populate_retrieved_chunks(all_queries + failure_refusals, retriever)
    logger.info("  %d entries with retrieval (%.1f s)", len(pool_entries), time.time() - t0)

    # 11. Apply filters
    logger.info("Applying quality filters...")
    embedder = Embedder(settings.embedding)

    queries_only = [pe.generated for pe in pool_entries]
    initial_count = len(queries_only)
    r1 = filter_by_length(queries_only)
    r2 = filter_duplicates(r1.kept, embedder.embed_texts)
    eval_texts = _load_eval_query_texts()
    r3 = filter_eval_contamination(r2.kept, eval_texts, embedder.embed_texts)

    # Build the final pool, preserving alignment between GeneratedQuery
    # and PoolEntry. We map by id() since GeneratedQuery isn't hashable.
    final_keep_ids = {id(q) for q in r3.kept}
    final_pool = [pe for pe in pool_entries if id(pe.generated) in final_keep_ids]
    logger.info(
        "Filtered: %d → %d (length=%d, dedupe=%d, eval=%d)",
        initial_count,
        len(final_pool),
        r1.dropped_count,
        r2.dropped_count,
        r3.dropped_count,
    )

    # 12. Write artifacts
    pool_path = output_dir / "query_pool.jsonl"
    pool_path.write_text("\n".join(json.dumps(pe.to_dict()) for pe in final_pool) + "\n")
    logger.info("Wrote %s (%d entries)", pool_path, len(final_pool))

    report_path = output_dir / "filter_report.md"
    report_path.write_text(
        _format_filter_report_markdown(
            initial_count=initial_count,
            after_length=r1,
            after_dedupe=r2,
            after_contamination=r3,
        )
    )
    logger.info("Wrote %s", report_path)

    preview_path = output_dir / "preview.md"
    preview_path.write_text(_format_preview_markdown(final_pool, n_sample=30, seed=args.seed))
    logger.info("Wrote %s — review before running distillation", preview_path)

    # 13. Final summary
    type_counts = Counter(pe.generated.question_type.value for pe in final_pool)
    logger.info("=== Final pool by type ===")
    for qt_name, n in sorted(type_counts.items()):
        logger.info("  %s: %d", qt_name, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
