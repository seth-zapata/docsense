#!/usr/bin/env python3
"""LLM-judged end-to-end generation eval driver.

Runs the full pipeline (retrieve → rerank → assemble → generate)
against an eval set, then judges each Answer for faithfulness and
relevance using a separate LLM judge. Rule-based checks (no-answer
behavior, citation grounding) run alongside the judge calls.

Sequential model loading is the headline constraint:

    Phase A — setup retrieval stack (embedder, retriever, reranker).
    Phase B — load Generator, run all queries, save Answer JSONs to
              disk, free Generator VRAM.
    Phase C — load LlamaJudge, score each saved Answer, free judge
              VRAM.
    Phase D — aggregate per-query results, write the JSON report.

Generator and judge are both 7-8B Instruct models loaded at NF4.
Each individually fits in ~5-6 GB; together they don't fit in 12 GB.
The save-to-disk handoff between Phases B and C is what makes the
sequential schedule reliable — Phase C reads its inputs from disk
rather than relying on Python-process state survival.

Usage::

    # Full eval, one eval set at a time
    python scripts/run_generation_eval.py --eval-set curated
    python scripts/run_generation_eval.py --eval-set structural
    python scripts/run_generation_eval.py --eval-set no-answer

    # Dry-run with a small N to validate the script before a long run
    python scripts/run_generation_eval.py --eval-set curated --limit 5

    # Force all-GPU placement on a 12 GB shared GPU
    python scripts/run_generation_eval.py --eval-set curated --device cuda:0

Reports land at ``evaluations/reports/generation-<UTC-date>-<set>.json``.
Per-query Answer dumps land at ``data/eval-runs/<run-id>/<set>/`` —
gitignored, kept for debugging across runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import faiss

from docsense.config import (
    DATA_DIR,
    GenerationConfig,
    JudgeConfig,
    Settings,
)
from docsense.embedding.embedder import Embedder
from docsense.evaluation.eval_queries import CURATED_QUERIES
from docsense.evaluation.llama_judge import LlamaJudge
from docsense.evaluation.no_answer_queries import NO_ANSWER_QUERIES
from docsense.evaluation.rule_based import (
    check_citations_grounded,
    check_no_answer_behavior,
)
from docsense.generation.context import ContextAssembler
from docsense.generation.generator import Generator
from docsense.generation.prompt import PromptBuilder
from docsense.generation.types import Answer, ChunkRef
from docsense.reranking.reranker import CrossEncoderReranker
from docsense.retrieval.dense import DenseRetriever
from docsense.retrieval.hybrid import HybridRetriever
from docsense.retrieval.sparse import SparseRetriever

if TYPE_CHECKING:
    from docsense.evaluation.judge import JudgeScore, RefusalJudgment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = DATA_DIR / "index"
EVAL_SETS_DIR = PROJECT_ROOT / "evaluations" / "eval_sets"
REPORTS_DIR = PROJECT_ROOT / "evaluations" / "reports"
EVAL_RUNS_DIR = DATA_DIR / "eval-runs"

ALL_EVAL_SETS = ("curated", "structural", "no-answer")


@dataclass(frozen=True)
class EvalQuery:
    """One eval-set entry, normalized across in-corpus and off-corpus sets.

    ``query_id`` is the stable identifier we key per-query records on.
    ``is_no_answer`` flips judge selection — off-corpus queries skip
    LLM-judge metrics (faithfulness/relevance don't apply when context
    is meant to be irrelevant) and only get the rule-based refusal
    check.
    """

    query_id: str
    text: str
    is_no_answer: bool


@dataclass
class QueryRecord:
    """In-memory carrier from Phase B (generation) into Phase C (judging).

    Phase B fills ``answer`` and ``timing``. Phase C fills the score
    fields if applicable. Phase D dumps the whole thing to the report.

    For in-corpus queries: ``faithfulness`` (claim-level) and
    ``relevance`` (5-anchor absolute) are populated.

    For no-answer queries: ``refusal_judge`` (LLM judgment) is
    populated as the primary measurement. The rule-based check is
    not stored on the record — it's recomputed from ``answer`` in
    the report-build phase since it's a pure-function regex.
    """

    query: EvalQuery
    answer: Answer
    timing: dict[str, float]
    faithfulness: JudgeScore | None = None
    relevance: JudgeScore | None = None
    refusal_judge: RefusalJudgment | None = None


# ---------------------------------------------------------------------------
# Eval-set loading
# ---------------------------------------------------------------------------


def _load_curated() -> list[EvalQuery]:
    return [
        EvalQuery(query_id=f"curated_{i:03d}", text=q, is_no_answer=False)
        for i, (q, _prefixes) in enumerate(CURATED_QUERIES, start=1)
    ]


def _load_structural() -> list[EvalQuery]:
    path = EVAL_SETS_DIR / "structural.json"
    if not path.exists():
        msg = (
            f"Structural eval set not found at {path}. "
            "Generate with: python scripts/generate_structural_queries.py"
        )
        raise FileNotFoundError(msg)
    raw = json.loads(path.read_text())
    return [
        EvalQuery(query_id=f"structural_{i:03d}", text=item["query"], is_no_answer=False)
        for i, item in enumerate(raw, start=1)
    ]


def _load_no_answer() -> list[EvalQuery]:
    return [
        EvalQuery(query_id=f"no_answer_{i:03d}", text=q, is_no_answer=True)
        for i, q in enumerate(NO_ANSWER_QUERIES, start=1)
    ]


def load_eval_set(name: str) -> list[EvalQuery]:
    if name == "curated":
        return _load_curated()
    if name == "structural":
        return _load_structural()
    if name == "no-answer":
        return _load_no_answer()
    msg = f"Unknown eval set: {name!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Phase A — retrieval stack
# ---------------------------------------------------------------------------


def _load_index_and_chunks(strategy: str) -> tuple[faiss.Index, list]:
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


def build_retrieval_stack(settings: Settings, strategy: str) -> tuple[HybridRetriever, int]:
    """Return a configured production-style HybridRetriever (with reranker).

    The eval intentionally uses the production stack — hybrid + cross-
    encoder rerank — because that's what downstream generation will see
    in practice. Running the eval against a different retrieval pipeline
    would measure a system we never serve.
    """
    index, chunks = _load_index_and_chunks(strategy)
    logger.info("Loaded %s index: %d vectors / %d chunks", strategy, index.ntotal, len(chunks))

    embedder = Embedder(settings.embedding)

    dense = DenseRetriever(dimension=index.d)
    dense.index = index
    dense.chunks = chunks

    sparse = SparseRetriever()
    sparse.add(chunks)

    reranker = CrossEncoderReranker(settings.reranking)
    hybrid = HybridRetriever(dense, sparse, embedder, settings.retrieval, reranker=reranker)
    return hybrid, len(chunks)


# ---------------------------------------------------------------------------
# Phase B — generation
# ---------------------------------------------------------------------------


def _free_cuda(label: str) -> None:
    """Release CUDA memory between Phase B and Phase C.

    Calls ``gc.collect()`` first so any lingering Python references to
    the just-deleted model are dropped before ``empty_cache``. Without
    the explicit gc, accelerate hooks and BitsAndBytes parameter
    objects can survive a ``del`` and pin VRAM. Logs the post-free
    allocated bytes so the eval log shows the handoff worked.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated_mb = torch.cuda.memory_allocated() / (1024**2)
            logger.info("After %s: CUDA allocated = %.0f MB", label, allocated_mb)
    except ImportError:
        # CPU-only environment: nothing to free.
        pass


def run_generation_phase(
    queries: list[EvalQuery],
    retriever: HybridRetriever,
    settings: Settings,
    run_dir: Path,
    eval_set: str,
) -> list[QueryRecord]:
    """Run all queries through the pipeline; return per-query records.

    Generator is loaded inside this function so its lifecycle is
    contained — at function exit the local reference goes out of scope
    and ``_free_cuda`` releases the VRAM before Phase C tries to load
    the judge.

    Per-query Answer JSON dumps go to disk as a debugging side-effect.
    The records returned in-memory carry the same data; the disk dumps
    are mainly for post-mortem when something looks off in the report.
    """
    prompt_builder = PromptBuilder()
    assembler = ContextAssembler(max_tokens=settings.generation.max_context_tokens)
    generator = Generator(settings.generation)

    answers_dir = run_dir / eval_set
    answers_dir.mkdir(parents=True, exist_ok=True)

    records: list[QueryRecord] = []
    for i, q in enumerate(queries, start=1):
        logger.info("[%d/%d] %s — %s", i, len(queries), q.query_id, q.text[:80])

        t0 = time.perf_counter()
        results = retriever.search(q.text)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        chunk_refs = [
            ChunkRef(
                doc_id=r.chunk.doc_id,
                chunk_id=r.chunk.chunk_id,
                score=r.score,
                text=r.chunk.text,
            )
            for r in results
        ]

        t0 = time.perf_counter()
        context, included = assembler.assemble(chunk_refs)
        messages = prompt_builder.build(query=q.text, context=context)
        assemble_ms = (time.perf_counter() - t0) * 1000

        answer = generator.generate(messages, included)
        timing = {
            "retrieve_ms": retrieve_ms,
            "assemble_ms": assemble_ms,
            "generate_ms": float(answer.metadata.latency_ms),
        }
        records.append(QueryRecord(query=q, answer=answer, timing=timing))

        # Best-effort dump for debugging; failure to dump shouldn't kill
        # the run since the in-memory record still flows through Phase C.
        try:
            (answers_dir / f"{q.query_id}.json").write_text(answer.model_dump_json(indent=2) + "\n")
        except OSError as exc:
            logger.warning("Could not dump %s answer JSON: %s", q.query_id, exc)

    del generator
    _free_cuda("generator unload")
    return records


# ---------------------------------------------------------------------------
# Phase C — judging
# ---------------------------------------------------------------------------


def run_judging_phase(records: list[QueryRecord], settings: Settings) -> None:
    """Score each record's Answer in-place.

    For in-corpus queries: faithfulness (claim-level decomposition,
    two LLM calls per query) + relevance (single absolute-scale
    judgment).

    For no-answer queries: refusal_judge (single LLM call) — replaces
    regex-only refusal detection as the primary measurement.
    Faithfulness and relevance still skipped since they assume a
    well-formed in-corpus question. The rule-based
    check_no_answer_behavior is recomputed in Phase D from the answer
    text and surfaced alongside the judge verdict; the agreement rate
    between the two is a methodology-health signal (see
    _aggregate_refusal_judgments).

    The citation_check rule-based pass runs in Phase D from Answer
    state — it doesn't need the judge to be loaded.
    """
    judge = LlamaJudge(settings.judge)

    for i, rec in enumerate(records, start=1):
        logger.info("[%d/%d] judging %s", i, len(records), rec.query.query_id)
        if rec.query.is_no_answer:
            rec.refusal_judge = judge.judge_refusal(rec.query.text, rec.answer.text)
        else:
            rec.faithfulness = judge.judge_faithfulness(
                rec.query.text, rec.answer.retrieved_chunks, rec.answer.text
            )
            rec.relevance = judge.judge_relevance(rec.query.text, rec.answer.text)

    del judge
    _free_cuda("judge unload")


# ---------------------------------------------------------------------------
# Phase D — aggregation + report
# ---------------------------------------------------------------------------


def _percentiles(values: list[float]) -> dict[str, float]:
    """Return p50/p95/p99 + mean + n.

    AWS-standard latency reporting. p50 captures typical UX, p95 catches
    the slow tail without being dominated by single outliers, p99 is the
    "1-in-100" tail signal — what users at the long tail experience.
    Mean is included because it's still a useful summary alongside the
    percentiles (and reveals when the distribution is skewed: mean ≫
    p50 means heavy tail).

    p90 and max were intentionally dropped:
    - p90 is a lower-bar version of p95 that the AWS convention
      replaces. Keeping both adds noise without information.
    - max is a single-observation tail signal that's easy to pollute
      with one-off slow events (cold-start cross-encoder warmup, GC
      pause). p99 is the right tail metric in a controlled eval; max
      is recoverable from per_query data in the report if needed.

    Empty input returns a zeroed dict; no NaN propagation.
    """
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "n": 0}
    sorted_vals = sorted(values)

    def _pct(p: float) -> float:
        # Linear interpolation between the two nearest ranks.
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        idx = (len(sorted_vals) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = idx - lo
        return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

    return {
        "p50": round(_pct(0.5), 2),
        "p95": round(_pct(0.95), 2),
        "p99": round(_pct(0.99), 2),
        "mean": round(statistics.fmean(sorted_vals), 2),
        "n": len(sorted_vals),
    }


def _aggregate_relevance_scores(records: list[QueryRecord]) -> dict[str, Any]:
    """Aggregate relevance JudgeScores (5-anchor absolute scale).

    Returns mean + median + anchor distribution + parse-failure count.
    Anchor distribution is the histogram across the five anchors; lets
    an analyst see whether the judge clusters at one value (suspicious)
    or spreads across the scale (healthier signal).
    """
    scores: list[float] = []
    n_parse_failed = 0
    anchor_dist: dict[str, int] = {f"{a:.2f}": 0 for a in (0.0, 0.25, 0.5, 0.75, 1.0)}
    for rec in records:
        score = rec.relevance
        if score is None:
            continue
        scores.append(score.score)
        anchor_dist[f"{score.score:.2f}"] = anchor_dist.get(f"{score.score:.2f}", 0) + 1
        if "PARSE_FAILED" in score.rationale:
            n_parse_failed += 1
    if not scores:
        return {"n": 0}
    return {
        "n": len(scores),
        "mean": round(statistics.fmean(scores), 3),
        "median": round(statistics.median(scores), 3),
        "anchor_distribution": anchor_dist,
        "n_parse_failures": n_parse_failed,
    }


def _aggregate_faithfulness_scores(records: list[QueryRecord]) -> dict[str, Any]:
    """Aggregate faithfulness JudgeScores (claim-level continuous).

    Returns score-level summary + claim-level summary + chunk-usage
    distribution. Three separate lenses on the same data:

    - **score**: per-query aggregate. Mean / median / std / bucketed
      histogram. The bucket histogram replaces the old anchor distribution
      since claim-level scores are continuous (n_supported / n_total),
      not snapped to the 5 anchors.
    - **claims**: per-claim aggregate across all queries. Total
      extracted, total supported, support rate (the "true" faithfulness
      number aggregated over claims rather than per-query),
      mean/max claims per answer, and the count of NO_CLAIMS_EXTRACTED
      / parse-failure cases.
    - **chunk_usage**: which retrieved chunks were cited most often.
      Surfaces retrieval quality from a different angle: a chunk that's
      in the top-5 but never gets cited may be a topical false positive.
    """
    scores: list[float] = []
    n_no_claims = 0
    # Two distinct parse-failure signals to surface separately:
    # - score-level: extraction parser failed to parse the LLM's claim list
    # - claim-level: attribution parser failed to find a line for some claims
    # The score-level case is rare (NO_CLAIMS_EXTRACTED is the much more
    # common path); the claim-level case caught a real bug where attribution
    # output was truncated by max_new_tokens on long answers. Both worth
    # surfacing so the aggregate doesn't lie about the eval's quality.
    n_score_parse_failures = 0
    n_claim_parse_failures = 0
    n_claim_out_of_range = 0
    total_claims = 0
    total_supported = 0
    claims_per_answer: list[int] = []
    chunk_citation_counts: dict[str, int] = {}

    for rec in records:
        score = rec.faithfulness
        if score is None:
            continue
        scores.append(score.score)
        if "NO_CLAIMS_EXTRACTED" in score.rationale:
            n_no_claims += 1
        if "PARSE_FAILED" in score.rationale:
            n_score_parse_failures += 1

        n_claims = len(score.claim_attributions)
        claims_per_answer.append(n_claims)
        total_claims += n_claims
        for attr in score.claim_attributions:
            if attr.supporting_chunk_idx is not None:
                total_supported += 1
                key = str(attr.supporting_chunk_idx)
                chunk_citation_counts[key] = chunk_citation_counts.get(key, 0) + 1
            else:
                chunk_citation_counts["none"] = chunk_citation_counts.get("none", 0) + 1
            if attr.rationale and "PARSE_FAILED" in attr.rationale:
                n_claim_parse_failures += 1
            if attr.rationale and "OUT_OF_RANGE" in attr.rationale:
                n_claim_out_of_range += 1

    if not scores:
        return {"n": 0}

    # Bucketed histogram. Endpoints get their own bucket because
    # "all unsupported" (0.0) and "all supported" (1.0) are
    # qualitatively different from "partial" — splitting them out
    # surfaces those cases at a glance.
    buckets: dict[str, int] = {
        "0.0": 0,
        "0.0-0.25": 0,
        "0.25-0.5": 0,
        "0.5-0.75": 0,
        "0.75-1.0": 0,
        "1.0": 0,
    }
    for s in scores:
        if s == 0.0:
            buckets["0.0"] += 1
        elif s == 1.0:
            buckets["1.0"] += 1
        elif s < 0.25:
            buckets["0.0-0.25"] += 1
        elif s < 0.5:
            buckets["0.25-0.5"] += 1
        elif s < 0.75:
            buckets["0.5-0.75"] += 1
        else:
            buckets["0.75-1.0"] += 1

    score_summary = {
        "n": len(scores),
        "mean": round(statistics.fmean(scores), 3),
        "median": round(statistics.median(scores), 3),
        "stdev": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
        "score_distribution": buckets,
    }
    claims_summary = {
        "total_extracted": total_claims,
        "total_supported": total_supported,
        "support_rate": round(total_supported / total_claims, 3) if total_claims > 0 else 0.0,
        "mean_per_answer": round(statistics.fmean(claims_per_answer), 2)
        if claims_per_answer
        else 0.0,
        "max_per_answer": max(claims_per_answer) if claims_per_answer else 0,
        "n_no_claims_extracted": n_no_claims,
        # Score-level parse failures (extraction-step misbehavior) and
        # claim-level parse failures (attribution-step misbehavior) are
        # distinct signals — the latter often indicates max_new_tokens
        # truncation. Surface both so the aggregate honestly reports
        # eval health.
        "n_score_parse_failures": n_score_parse_failures,
        "n_claim_parse_failures": n_claim_parse_failures,
        "n_claim_out_of_range": n_claim_out_of_range,
    }
    return {
        "score": score_summary,
        "claims": claims_summary,
        "chunk_usage_distribution": chunk_citation_counts,
    }


def _aggregate_refusal_judgments(records: list[QueryRecord]) -> dict[str, Any]:
    """Aggregate refusal data: LLM-judge primary, rule-based guardrail.

    For each off-corpus record, both the LLM judge's verdict
    (``rec.refusal_judge``) and the rule-based pattern check
    (recomputed here from ``rec.answer``) produce a bool. We surface
    three numbers:

    - ``frac_refused_judge``: fraction the LLM judge marked as
      refusals. The primary measurement going forward — robust to
      phrasing drift in a way the regex isn't.
    - ``frac_refused_rule``: fraction the regex marked as refusals.
      Kept as a sanity check; should agree closely on a stable model.
    - ``agreement_rate``: fraction of records where the two methods
      gave the same verdict. Methodology-health signal: a drop here
      between Phase 3 pre-FT and post-FT runs is an early warning
      that fine-tuning shifted refusal phrasing in a way one of the
      methods missed.

    Plus parse-failure counts and pattern-match diagnostics from the
    rule-based side (which patterns fired, on which queries).
    """
    n = len(records)
    if n == 0:
        return {"n": 0}

    judge_verdicts: list[bool] = []
    rule_verdicts: list[bool] = []
    n_judge_parse_failures = 0
    matched_patterns: dict[str, int] = {}
    disagreement_query_ids: list[str] = []

    for rec in records:
        rule_result = check_no_answer_behavior(rec.answer, expected_refusal=True)
        rule_verdicts.append(rule_result.refused)
        if rule_result.matched_pattern is not None:
            matched_patterns[rule_result.matched_pattern] = (
                matched_patterns.get(rule_result.matched_pattern, 0) + 1
            )

        judge = rec.refusal_judge
        if judge is None:
            # Shouldn't happen if Phase C ran cleanly; defensive fallback.
            judge_verdicts.append(False)
            continue
        judge_verdicts.append(judge.refused)
        if "PARSE_FAILED" in judge.rationale:
            n_judge_parse_failures += 1

        if judge.refused != rule_result.refused:
            disagreement_query_ids.append(rec.query.query_id)

    n_judge_refused = sum(judge_verdicts)
    n_rule_refused = sum(rule_verdicts)
    n_agree = sum(1 for j, r in zip(judge_verdicts, rule_verdicts, strict=True) if j == r)

    # All off-corpus queries should refuse (expected_refusal=True).
    # frac_correct = frac_refused_judge by construction here, but we
    # surface it explicitly for clarity in the report.
    return {
        "n": n,
        "frac_refused_judge": round(n_judge_refused / n, 3),
        "frac_refused_rule": round(n_rule_refused / n, 3),
        "frac_correct_judge": round(n_judge_refused / n, 3),
        "frac_correct_rule": round(n_rule_refused / n, 3),
        "agreement_rate": round(n_agree / n, 3),
        "n_disagreements": n - n_agree,
        "disagreement_query_ids": disagreement_query_ids,
        "n_judge_parse_failures": n_judge_parse_failures,
        "rule_matched_patterns": matched_patterns,
    }


def build_report(
    records: list[QueryRecord],
    *,
    eval_set: str,
    settings: Settings,
    chunks_total: int,
    limit_applied: int | None,
) -> dict[str, Any]:
    """Assemble the final report dict from all records."""
    per_query: list[dict[str, Any]] = []
    for rec in records:
        cit = check_citations_grounded(rec.answer)
        entry: dict[str, Any] = {
            "query_id": rec.query.query_id,
            "query": rec.query.text,
            "answer_text": rec.answer.text,
            "is_no_answer_query": rec.query.is_no_answer,
            "n_retrieved_chunks": len(rec.answer.retrieved_chunks),
            "timing": rec.timing,
            "citation_check": cit.model_dump(),
        }
        if rec.query.is_no_answer:
            # Cross-validation: both the rule-based regex and the LLM
            # judge run on the answer. The judge is the primary
            # measurement; the rule is a guardrail and disagreement
            # signal. See _aggregate_refusal_judgments for the
            # agreement-rate computation.
            rule_check = check_no_answer_behavior(rec.answer, expected_refusal=True)
            entry["no_answer_check_rule"] = rule_check.model_dump()
            entry["refusal_judge"] = rec.refusal_judge.model_dump() if rec.refusal_judge else None
        else:
            entry["faithfulness"] = rec.faithfulness.model_dump() if rec.faithfulness else None
            entry["relevance"] = rec.relevance.model_dump() if rec.relevance else None
        per_query.append(entry)

    # Aggregates
    aggregates: dict[str, Any] = {}
    in_corpus_records = [r for r in records if not r.query.is_no_answer]
    no_answer_records = [r for r in records if r.query.is_no_answer]

    if in_corpus_records:
        aggregates["faithfulness"] = _aggregate_faithfulness_scores(in_corpus_records)
        aggregates["relevance"] = _aggregate_relevance_scores(in_corpus_records)
        cits = [check_citations_grounded(r.answer) for r in in_corpus_records]
        aggregates["citation_check"] = {
            "n": len(cits),
            "frac_with_any_marker": round(
                statistics.fmean([1.0 if c.any_marker_present else 0.0 for c in cits]), 3
            ),
            "frac_all_in_range": round(
                statistics.fmean([1.0 if c.all_markers_in_range else 0.0 for c in cits]), 3
            ),
            "mean_n_markers_in_text": round(
                statistics.fmean([c.n_markers_in_text for c in cits]), 3
            ),
        }
    if no_answer_records:
        aggregates["no_answer"] = _aggregate_refusal_judgments(no_answer_records)

    timings_by_stage = {
        "retrieve_ms": [r.timing["retrieve_ms"] for r in records],
        "assemble_ms": [r.timing["assemble_ms"] for r in records],
        "generate_ms": [r.timing["generate_ms"] for r in records],
    }
    timing_percentiles = {stage: _percentiles(values) for stage, values in timings_by_stage.items()}

    return {
        "schema_version": 1,
        "captured_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "eval_set": eval_set,
        "eval_set_size": len(records),
        "limit_applied": limit_applied,
        "config": {
            "chunking_strategy": settings.chunking.strategy,
            "chunks_total": chunks_total,
            "generator_model": settings.generation.model_name,
            # Judge is now used for no-answer queries too (refusal
            # detection moved from regex-only to LLM-judge primary
            # with regex as guardrail). So the judge model is recorded
            # for every eval set.
            "judge_model": settings.judge.model_name,
            "use_4bit_generator": settings.generation.use_4bit_quantization,
            "use_4bit_judge": settings.judge.use_4bit_quantization,
            "rerank_candidates": settings.retrieval.rerank_candidates,
            "top_k": settings.retrieval.top_k,
            "max_context_tokens": settings.generation.max_context_tokens,
            "max_new_tokens_generator": settings.generation.max_new_tokens,
            "max_new_tokens_judge": settings.judge.max_new_tokens,
        },
        "aggregates": aggregates,
        "timing_ms_per_query": timing_percentiles,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _apply_device_override(settings: Settings, device: str | None) -> None:
    if device is None:
        return
    settings.generation = GenerationConfig(**{**settings.generation.model_dump(), "device": device})
    settings.judge = JudgeConfig(**{**settings.judge.model_dump(), "device": device})


def run_one_eval_set(
    eval_set: str,
    *,
    settings: Settings,
    strategy: str,
    limit: int | None,
    run_id: str,
) -> Path:
    """Run Phases A-D for one eval set; return the path to the report.

    Each eval set is a self-contained run (its own retrieval load, its
    own generator load, its own judge load). When --eval-set all is
    used the outer loop just calls this three times.
    """
    queries = load_eval_set(eval_set)
    if limit is not None:
        queries = queries[:limit]
    logger.info("=== Eval set: %s (%d queries) ===", eval_set, len(queries))

    retriever, chunks_total = build_retrieval_stack(settings, strategy)

    run_dir = EVAL_RUNS_DIR / run_id
    records = run_generation_phase(
        queries=queries,
        retriever=retriever,
        settings=settings,
        run_dir=run_dir,
        eval_set=eval_set,
    )

    run_judging_phase(records, settings)

    report = build_report(
        records,
        eval_set=eval_set,
        settings=settings,
        chunks_total=chunks_total,
        limit_applied=limit,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"generation-{run_id}-{eval_set}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    logger.info("Report written to %s", out_path)
    _print_summary(report)
    return out_path


def _print_summary(report: dict[str, Any]) -> None:
    """Brief stdout summary so the operator sees results without
    opening the JSON."""
    print()
    print(f"=== {report['eval_set']} (n={report['eval_set_size']}) ===")
    agg = report["aggregates"]
    # Faithfulness now has nested {score, claims, chunk_usage_distribution}
    # rather than flat {n, mean, anchor_distribution}. The claim-level
    # support_rate is the most informative single number — fraction of
    # all extracted claims that were attributed to a chunk.
    if "faithfulness" in agg and "score" in agg["faithfulness"]:
        sc = agg["faithfulness"]["score"]
        cl = agg["faithfulness"]["claims"]
        print(
            f"  faithfulness: mean={sc['mean']:.3f}  median={sc['median']:.3f}  "
            f"support={cl['total_supported']}/{cl['total_extracted']} claims "
            f"({cl['support_rate']:.2f})  "
            f"parse-fails: score={cl['n_score_parse_failures']} "
            f"claim={cl['n_claim_parse_failures']} oor={cl['n_claim_out_of_range']}"
        )
    if "relevance" in agg and agg["relevance"]["n"] > 0:
        print(
            f"  relevance:    mean={agg['relevance']['mean']:.3f}  "
            f"parse-failures={agg['relevance'].get('n_parse_failures', 0)}"
        )
    if "citation_check" in agg:
        cc = agg["citation_check"]
        print(
            f"  citations:    frac_any_marker={cc['frac_with_any_marker']:.2f}  "
            f"mean_markers/answer={cc['mean_n_markers_in_text']:.2f}"
        )
    if "no_answer" in agg and agg["no_answer"].get("n", 0) > 0:
        no = agg["no_answer"]
        print(
            f"  no-answer:    judge={no['frac_refused_judge']:.2f}  "
            f"rule={no['frac_refused_rule']:.2f}  "
            f"agreement={no['agreement_rate']:.2f}  "
            f"disagreements={no['n_disagreements']}  "
            f"judge-parse-fails={no['n_judge_parse_failures']}"
        )
    timing = report["timing_ms_per_query"]
    print(
        f"  timing/query: retrieve p50={timing['retrieve_ms']['p50']:.0f}ms  "
        f"generate p50={timing['generate_ms']['p50']:.0f}ms  "
        f"generate p95={timing['generate_ms']['p95']:.0f}ms  "
        f"generate p99={timing['generate_ms']['p99']:.0f}ms"
    )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--eval-set",
        choices=(*ALL_EVAL_SETS, "all"),
        required=True,
        help=(
            "Which eval set to evaluate. 'all' runs curated → structural → "
            "no-answer in sequence with a separate retrieval+generator+judge "
            "load per set."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=("fixed", "recursive", "header"),
        default="recursive",
        help="Chunking strategy whose index to load (default: recursive).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap each eval set to the first N queries. Use 5 for a fast "
            "dry-run that still exercises every phase before committing "
            "to a full ~45-minute run."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Override device for both generator and judge. 'auto' is the "
            "default in config; pass 'cuda:0' to force all-GPU placement "
            "on a 12 GB shared GPU (avoids accelerate spilling weights to "
            "CPU and thrashing). 'cpu' for CPU-only runs."
        ),
    )
    args = parser.parse_args()

    settings = Settings()
    _apply_device_override(settings, args.device)

    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    logger.info("Run ID: %s", run_id)
    logger.info("Generator model: %s", settings.generation.model_name)
    logger.info("Judge model:     %s", settings.judge.model_name)
    logger.info("Chunking:        %s", args.strategy)
    logger.info("Limit:           %s", args.limit or "(none)")
    logger.info("Device override: %s", args.device or "(default)")

    eval_sets_to_run: tuple[str, ...] = (
        ALL_EVAL_SETS if args.eval_set == "all" else (args.eval_set,)
    )
    for es in eval_sets_to_run:
        run_one_eval_set(
            es,
            settings=settings,
            strategy=args.strategy,
            limit=args.limit,
            run_id=run_id,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
