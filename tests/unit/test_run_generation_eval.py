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
from docsense.evaluation.judge import JudgeScore
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

    def test_single_value(self):
        result = driver._percentiles([42.0])
        assert result["p50"] == 42.0
        assert result["p90"] == 42.0
        assert result["max"] == 42.0
        assert result["mean"] == 42.0
        assert result["n"] == 1

    def test_p50_is_median_for_odd_count(self):
        result = driver._percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["p50"] == 3.0
        assert result["max"] == 5.0
        assert result["mean"] == 3.0

    def test_p90_above_p50(self):
        """Sanity: percentiles must be monotone non-decreasing."""
        values = [float(i) for i in range(100)]
        result = driver._percentiles(values)
        assert result["p90"] >= result["p50"]
        assert result["max"] >= result["p90"]


class TestFormatContextForJudge:
    def test_numbered_format_matches_prompt_shape(self):
        chunks = [
            ChunkRef(doc_id="a.md", chunk_id="1", score=1.0, text="alpha"),
            ChunkRef(doc_id="b.md", chunk_id="2", score=0.9, text="beta"),
        ]
        rendered = driver._format_context_for_judge(chunks)
        assert "[1]" in rendered
        assert "[2]" in rendered
        assert "a.md" in rendered
        assert "alpha" in rendered
        assert "beta" in rendered


class TestAggregateScores:
    def test_empty_records_returns_n_zero(self):
        result = driver._aggregate_scores([], "faithfulness")
        assert result == {"n": 0}

    def test_records_without_scores_skipped(self):
        records = [_record("q1", is_no_answer=False), _record("q2", is_no_answer=False)]
        # No faithfulness/relevance set on either — aggregation should be empty.
        result = driver._aggregate_scores(records, "faithfulness")
        assert result == {"n": 0}

    def test_aggregates_with_anchor_distribution(self):
        records = [_record("q1", is_no_answer=False), _record("q2", is_no_answer=False)]
        records[0].faithfulness = JudgeScore(metric="faithfulness", score=1.0, rationale="ok")
        records[1].faithfulness = JudgeScore(metric="faithfulness", score=0.5, rationale="ok")
        result = driver._aggregate_scores(records, "faithfulness")
        assert result["n"] == 2
        assert result["mean"] == 0.75
        assert result["anchor_distribution"]["1.00"] == 1
        assert result["anchor_distribution"]["0.50"] == 1
        assert result["n_parse_failures"] == 0

    def test_parse_failures_counted(self):
        records = [_record("q1", is_no_answer=False)]
        records[0].faithfulness = JudgeScore(
            metric="faithfulness",
            score=0.0,
            rationale="PARSE_FAILED: no SCORE in response",
        )
        result = driver._aggregate_scores(records, "faithfulness")
        assert result["n_parse_failures"] == 1


class TestBuildReport:
    def test_in_corpus_report_has_all_sections(self):
        records = [_record("q1", is_no_answer=False, answer_text="answer with [1]")]
        records[0].faithfulness = JudgeScore(metric="faithfulness", score=0.75, rationale="ok")
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
        assert report["aggregates"]["faithfulness"]["mean"] == 0.75
        assert report["aggregates"]["relevance"]["mean"] == 1.0
        assert "citation_check" in report["aggregates"]
        assert report["per_query"][0]["query_id"] == "q1"
        assert report["per_query"][0]["faithfulness"]["score"] == 0.75
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
            rec.faithfulness = JudgeScore(metric="faithfulness", score=1.0, rationale="ok")
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
