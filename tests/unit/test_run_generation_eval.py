"""Unit tests for the eval-driver helpers.

Covers the parts that don't need a real model: eval-set loading,
percentile computation, judge-context formatting, score aggregation,
and end-to-end report assembly using synthetic QueryRecord inputs.

The actual generation/judging phases load models and are exercised
manually via ``scripts/run_generation_eval.py`` — they're behind the
"don't claim from mocked tests" rule.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from docsense.config import Settings
from docsense.evaluation.judge import ClaimAttribution, JudgeScore
from docsense.generation.types import Answer, ChunkRef, GenerationMetadata


def _load_eval_driver():
    """scripts/ isn't a package; load by file path so tests can reach in."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_generation_eval.py"
    spec = importlib.util.spec_from_file_location("run_generation_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_generation_eval"] = module
    spec.loader.exec_module(module)
    return module


driver = _load_eval_driver()


def _make_answer(text: str, n_chunks: int = 3) -> Answer:
    chunks = [
        ChunkRef(doc_id=f"doc{i}.md", chunk_id=str(i), score=1.0, text=f"chunk text {i}")
        for i in range(1, n_chunks + 1)
    ]
    return Answer(
        text=text,
        citations=[],
        retrieved_chunks=chunks,
        metadata=GenerationMetadata(model_name="test", latency_ms=100.0),
    )


def _record(query_id: str, *, is_no_answer: bool, answer_text: str = "ok") -> driver.QueryRecord:
    return driver.QueryRecord(
        query=driver.EvalQuery(query_id=query_id, text="q?", is_no_answer=is_no_answer),
        answer=_make_answer(answer_text),
        timing={"retrieve_ms": 100.0, "assemble_ms": 5.0, "generate_ms": 1000.0},
    )


class TestLoadEvalSet:
    def test_curated_loads_with_correct_id_format(self):
        queries = driver.load_eval_set("curated")
        assert len(queries) >= 10  # the curated set has 20
        assert queries[0].query_id == "curated_001"
        assert all(q.is_no_answer is False for q in queries)

    def test_no_answer_loads_marked_correctly(self):
        queries = driver.load_eval_set("no-answer")
        assert len(queries) == 8
        assert queries[0].query_id == "no_answer_001"
        assert all(q.is_no_answer is True for q in queries)

    def test_unknown_set_raises(self):
        with pytest.raises(ValueError, match="Unknown eval set"):
            driver.load_eval_set("nonexistent")


class TestPercentiles:
    def test_empty_input_safe(self):
        result = driver._percentiles([])
        assert result["n"] == 0
        assert result["p50"] == 0.0
        assert result["p95"] == 0.0
        assert result["p99"] == 0.0

    def test_no_p90_or_max_in_output(self):
        """AWS-standard reporting is p50/p95/p99 + mean. p90 and max
        were intentionally removed to keep the schema focused — pin
        the omission so a future addition is a deliberate decision."""
        result = driver._percentiles([1.0, 2.0, 3.0])
        assert "p90" not in result
        assert "max" not in result
        assert set(result.keys()) == {"p50", "p95", "p99", "mean", "n"}

    def test_single_value(self):
        result = driver._percentiles([42.0])
        assert result["p50"] == 42.0
        assert result["p95"] == 42.0
        assert result["p99"] == 42.0
        assert result["mean"] == 42.0
        assert result["n"] == 1

    def test_p50_is_median_for_odd_count(self):
        result = driver._percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["p50"] == 3.0
        assert result["mean"] == 3.0

    def test_percentiles_monotone_non_decreasing(self):
        """Sanity: p50 ≤ p95 ≤ p99 ≤ underlying max for any input."""
        values = [float(i) for i in range(100)]
        result = driver._percentiles(values)
        assert result["p50"] <= result["p95"]
        assert result["p95"] <= result["p99"]
        assert result["p99"] <= max(values)

    def test_p99_at_n100(self):
        """With exactly 100 values 0-99, p99 lands at the last value
        (or very close — linear-interpolation between rank 98 and 99).
        Documents the small-N caveat for the schema."""
        values = [float(i) for i in range(100)]
        result = driver._percentiles(values)
        assert result["p99"] >= 98.0  # ~99.0 with linear interpolation


class TestAggregateRelevanceScores:
    def test_empty_records_returns_n_zero(self):
        result = driver._aggregate_relevance_scores([])
        assert result == {"n": 0}

    def test_records_without_scores_skipped(self):
        records = [_record("q1", is_no_answer=False), _record("q2", is_no_answer=False)]
        result = driver._aggregate_relevance_scores(records)
        assert result == {"n": 0}

    def test_aggregates_with_anchor_distribution(self):
        records = [_record("q1", is_no_answer=False), _record("q2", is_no_answer=False)]
        records[0].relevance = JudgeScore(metric="relevance", score=1.0, rationale="ok")
        records[1].relevance = JudgeScore(metric="relevance", score=0.5, rationale="ok")
        result = driver._aggregate_relevance_scores(records)
        assert result["n"] == 2
        assert result["mean"] == 0.75
        assert result["anchor_distribution"]["1.00"] == 1
        assert result["anchor_distribution"]["0.50"] == 1
        assert result["n_parse_failures"] == 0

    def test_parse_failures_counted(self):
        records = [_record("q1", is_no_answer=False)]
        records[0].relevance = JudgeScore(
            metric="relevance",
            score=0.0,
            rationale="PARSE_FAILED: no SCORE in response",
        )
        result = driver._aggregate_relevance_scores(records)
        assert result["n_parse_failures"] == 1


class TestAggregateFaithfulnessScores:
    def _faith_score(self, score: float, attrs: list[ClaimAttribution]) -> JudgeScore:
        return JudgeScore(
            metric="faithfulness",
            score=score,
            rationale=f"{int(score * len(attrs))} of {len(attrs)} claims supported.",
            claim_attributions=attrs,
        )

    def test_empty_records_returns_n_zero(self):
        result = driver._aggregate_faithfulness_scores([])
        assert result == {"n": 0}

    def test_aggregate_with_claim_breakdown(self):
        """Two records: one fully supported, one half supported.
        Aggregate score-mean=0.75; total claims 4 (3 supported / 1 not);
        chunk usage shows chunk 1 cited twice, chunk 2 once, none once."""
        rec1 = _record("q1", is_no_answer=False)
        rec2 = _record("q2", is_no_answer=False)
        rec1.faithfulness = self._faith_score(
            1.0,
            [
                ClaimAttribution(claim_idx=1, claim_text="a", supporting_chunk_idx=1),
                ClaimAttribution(claim_idx=2, claim_text="b", supporting_chunk_idx=2),
            ],
        )
        rec2.faithfulness = self._faith_score(
            0.5,
            [
                ClaimAttribution(claim_idx=1, claim_text="a", supporting_chunk_idx=1),
                ClaimAttribution(claim_idx=2, claim_text="c", supporting_chunk_idx=None),
            ],
        )
        result = driver._aggregate_faithfulness_scores([rec1, rec2])

        # Score-level
        assert result["score"]["n"] == 2
        assert result["score"]["mean"] == 0.75
        assert result["score"]["min"] == 0.5
        assert result["score"]["max"] == 1.0

        # Claims-level (cross-query)
        assert result["claims"]["total_extracted"] == 4
        assert result["claims"]["total_supported"] == 3
        assert result["claims"]["support_rate"] == 0.75  # 3/4
        assert result["claims"]["mean_per_answer"] == 2.0
        assert result["claims"]["max_per_answer"] == 2

        # Chunk usage histogram
        assert result["chunk_usage_distribution"]["1"] == 2
        assert result["chunk_usage_distribution"]["2"] == 1
        assert result["chunk_usage_distribution"]["none"] == 1

    def test_score_buckets_split_endpoints(self):
        """Scores at exactly 0.0 and 1.0 land in their own buckets, not
        merged with the inner ranges. Lets reports surface "all
        unsupported" and "all supported" cases at a glance."""
        rec_zero = _record("q1", is_no_answer=False)
        rec_one = _record("q2", is_no_answer=False)
        rec_partial = _record("q3", is_no_answer=False)
        rec_zero.faithfulness = self._faith_score(
            0.0, [ClaimAttribution(claim_idx=1, claim_text="x", supporting_chunk_idx=None)]
        )
        rec_one.faithfulness = self._faith_score(
            1.0, [ClaimAttribution(claim_idx=1, claim_text="y", supporting_chunk_idx=1)]
        )
        rec_partial.faithfulness = self._faith_score(
            0.5,
            [
                ClaimAttribution(claim_idx=1, claim_text="a", supporting_chunk_idx=1),
                ClaimAttribution(claim_idx=2, claim_text="b", supporting_chunk_idx=None),
            ],
        )
        result = driver._aggregate_faithfulness_scores([rec_zero, rec_one, rec_partial])
        assert result["score"]["score_distribution"]["0.0"] == 1
        assert result["score"]["score_distribution"]["1.0"] == 1
        assert result["score"]["score_distribution"]["0.5-0.75"] == 1

    def test_no_claims_extracted_counted(self):
        """A NO_CLAIMS_EXTRACTED case (e.g., extraction parse failure)
        gets counted in the claims aggregate but contributes 0 to
        the score and 0 to the claim totals."""
        rec = _record("q1", is_no_answer=False)
        rec.faithfulness = JudgeScore(
            metric="faithfulness",
            score=0.0,
            rationale="NO_CLAIMS_EXTRACTED: claim-extraction step returned no claims.",
            claim_attributions=[],
        )
        result = driver._aggregate_faithfulness_scores([rec])
        assert result["claims"]["n_no_claims_extracted"] == 1
        assert result["claims"]["total_extracted"] == 0

    def test_per_claim_parse_failures_counted(self):
        """If individual claim attributions carry PARSE_FAILED markers
        (e.g., from max_new_tokens truncation cutting off attribution
        output mid-list), the aggregate must count them. Previously
        n_parse_failures only counted score-level rationales and
        invisibly missed per-claim failures."""
        rec = _record("q1", is_no_answer=False)
        rec.faithfulness = JudgeScore(
            metric="faithfulness",
            score=0.0,
            rationale="0 of 3 claims supported.",
            claim_attributions=[
                ClaimAttribution(
                    claim_idx=1,
                    claim_text="a",
                    supporting_chunk_idx=None,
                    rationale="PARSE_FAILED: no attribution line for claim 1",
                ),
                ClaimAttribution(
                    claim_idx=2,
                    claim_text="b",
                    supporting_chunk_idx=None,
                    rationale="PARSE_FAILED: no attribution line for claim 2",
                ),
                ClaimAttribution(
                    claim_idx=3,
                    claim_text="c",
                    supporting_chunk_idx=None,
                    rationale="OUT_OF_RANGE: chunk 9 not in [1, 5]",
                ),
            ],
        )
        result = driver._aggregate_faithfulness_scores([rec])
        assert result["claims"]["n_claim_parse_failures"] == 2
        assert result["claims"]["n_claim_out_of_range"] == 1
        # Score-level parse failure is separate; this case has no score-level marker.
        assert result["claims"]["n_score_parse_failures"] == 0


class TestBuildReport:
    def test_in_corpus_report_has_all_sections(self):
        records = [_record("q1", is_no_answer=False, answer_text="answer with [1]")]
        records[0].faithfulness = JudgeScore(
            metric="faithfulness",
            score=0.75,
            rationale="3 of 4 claims supported.",
            claim_attributions=[
                ClaimAttribution(claim_idx=1, claim_text="a", supporting_chunk_idx=1),
                ClaimAttribution(claim_idx=2, claim_text="b", supporting_chunk_idx=2),
                ClaimAttribution(claim_idx=3, claim_text="c", supporting_chunk_idx=3),
                ClaimAttribution(claim_idx=4, claim_text="d", supporting_chunk_idx=None),
            ],
        )
        records[0].relevance = JudgeScore(metric="relevance", score=1.0, rationale="ok")

        report = driver.build_report(
            records,
            eval_set="curated",
            settings=Settings(),
            chunks_total=12345,
            limit_applied=None,
        )
        assert report["eval_set"] == "curated"
        assert report["eval_set_size"] == 1
        # Faithfulness now has nested {score, claims, chunk_usage_distribution}.
        assert report["aggregates"]["faithfulness"]["score"]["mean"] == 0.75
        assert report["aggregates"]["faithfulness"]["claims"]["support_rate"] == 0.75
        # Relevance keeps the flat shape.
        assert report["aggregates"]["relevance"]["mean"] == 1.0
        assert "citation_check" in report["aggregates"]
        # Per-query carries the full claim attributions for debugging.
        assert report["per_query"][0]["query_id"] == "q1"
        assert report["per_query"][0]["faithfulness"]["score"] == 0.75
        assert len(report["per_query"][0]["faithfulness"]["claim_attributions"]) == 4
        assert report["config"]["chunks_total"] == 12345
        # Judge model is recorded for in-corpus runs (judge was needed).
        assert report["config"]["judge_model"] is not None

    def test_no_answer_report_uses_rule_based_only(self):
        records = [_record("q1", is_no_answer=True, answer_text="I don't know.")]
        report = driver.build_report(
            records,
            eval_set="no-answer",
            settings=Settings(),
            chunks_total=12345,
            limit_applied=None,
        )
        assert "no_answer_check" in report["aggregates"]
        assert report["aggregates"]["no_answer_check"]["frac_correct_refusal"] == 1.0
        # No faithfulness/relevance for no-answer set.
        assert "faithfulness" not in report["aggregates"]
        assert "relevance" not in report["aggregates"]
        # Judge model is None when no in-corpus records were judged.
        assert report["config"]["judge_model"] is None

    def test_timing_percentiles_computed(self):
        records = [_record(f"q{i}", is_no_answer=False) for i in range(5)]
        for rec in records:
            rec.faithfulness = JudgeScore(
                metric="faithfulness",
                score=1.0,
                rationale="ok",
                claim_attributions=[
                    ClaimAttribution(claim_idx=1, claim_text="a", supporting_chunk_idx=1)
                ],
            )
            rec.relevance = JudgeScore(metric="relevance", score=1.0, rationale="ok")

        report = driver.build_report(
            records,
            eval_set="curated",
            settings=Settings(),
            chunks_total=12345,
            limit_applied=None,
        )
        assert "timing_ms_per_query" in report
        assert report["timing_ms_per_query"]["generate_ms"]["n"] == 5
        assert report["timing_ms_per_query"]["retrieve_ms"]["mean"] > 0
        # Schema-pin: p50/p95/p99 must be present, p90/max must not.
        assert "p99" in report["timing_ms_per_query"]["generate_ms"]
        assert "p95" in report["timing_ms_per_query"]["generate_ms"]
        assert "p90" not in report["timing_ms_per_query"]["generate_ms"]
        assert "max" not in report["timing_ms_per_query"]["generate_ms"]
