"""Unit tests for the Stage 1 query-pool seeder.

Tests the script's pure helpers — sampling, classification grouping,
resumability bookkeeping, retrieval-failure synthesis, markdown
artifact generation, PoolEntry serialization. The full main() loop
isn't tested; it depends on a real index, real Anthropic client, and
real embedder, all out-of-scope for unit tests.

Loaded via importlib because scripts/ isn't a package — same pattern
as test_build_training_dataset.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from docsense.finetuning.chunk_classifier import QuestionType
from docsense.finetuning.query_filters import FilterReport
from docsense.finetuning.query_generation import GeneratedQuery
from docsense.generation.types import ChunkRef


def _load_pool_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_query_pool.py"
    spec = importlib.util.spec_from_file_location("build_query_pool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_query_pool"] = module
    spec.loader.exec_module(module)
    return module


pool = _load_pool_script()


@dataclass
class _StubChunk:
    """Minimal Chunk-like object — matches the duck type the script uses."""

    text: str
    doc_id: str
    chunk_index: int = 0

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}::chunk_{self.chunk_index}"


def _chunk(doc: str, idx: int = 0, text: str = "sample text") -> _StubChunk:
    return _StubChunk(text=text, doc_id=doc, chunk_index=idx)


def _make_query(
    text: str = "how do I save a fine-tuned model?",
    qt: QuestionType = QuestionType.PROCEDURAL,
    *,
    seed_chunks: list[ChunkRef] | None = None,
    seed_topic: str | None = None,
    metadata: dict | None = None,
) -> GeneratedQuery:
    return GeneratedQuery(
        query=text,
        question_type=qt,
        seed_chunks=seed_chunks or [],
        seed_topic=seed_topic,
        metadata=metadata or {},
    )


def _ref(doc: str, idx: int = 0, text: str = "sample text") -> ChunkRef:
    return ChunkRef(doc_id=doc, chunk_id=f"{doc}::chunk_{idx}", score=0.0, text=text)


# --------------------------------------------------------------------
# TaskKey
# --------------------------------------------------------------------


class TestTaskKey:
    def test_to_str_round_trip(self):
        key = pool.TaskKey(QuestionType.PROCEDURAL, "doc.md::chunk_3")
        assert key.to_str() == "procedural::doc.md::chunk_3"

    def test_refusal_key_with_seq(self):
        key = pool.TaskKey(QuestionType.REFUSAL, "aws-lambda::0")
        assert key.to_str() == "refusal::aws-lambda::0"

    def test_keys_with_same_content_equal(self):
        """Frozen dataclass hashes by value — same identifier means
        same key, used for resumability dedup."""
        a = pool.TaskKey(QuestionType.PROCEDURAL, "x")
        b = pool.TaskKey(QuestionType.PROCEDURAL, "x")
        assert a == b
        assert hash(a) == hash(b)


# --------------------------------------------------------------------
# Resumability — _serialize_raw_query / _read_raw_queries / _append_raw_query
# --------------------------------------------------------------------


class TestRawQueryPersistence:
    def test_round_trip_preserves_query_and_key(self, tmp_path):
        path = tmp_path / "raw.jsonl"
        key = pool.TaskKey(QuestionType.PROCEDURAL, "doc.md::chunk_1")
        gq = _make_query(text="how do I X?", qt=QuestionType.PROCEDURAL)

        pool._append_raw_query(path, key, gq)

        loaded = pool._read_raw_queries(path)
        assert len(loaded) == 1
        loaded_key, loaded_gq = loaded[0]
        assert loaded_key == key
        assert loaded_gq.query == "how do I X?"
        assert loaded_gq.question_type == QuestionType.PROCEDURAL

    def test_append_creates_file_if_absent(self, tmp_path):
        path = tmp_path / "subdir" / "raw.jsonl"
        assert not path.exists()
        pool._append_raw_query(
            path,
            pool.TaskKey(QuestionType.COMPARISON, "x"),
            _make_query(qt=QuestionType.COMPARISON),
        )
        assert path.exists()

    def test_load_done_task_keys_skips_done(self, tmp_path):
        """Resume scenario: existing raw file → done set populated."""
        path = tmp_path / "raw.jsonl"
        for i in range(3):
            pool._append_raw_query(
                path,
                pool.TaskKey(QuestionType.PROCEDURAL, f"chunk_{i}"),
                _make_query(qt=QuestionType.PROCEDURAL),
            )
        done = pool._load_done_task_keys(path)
        assert done == {
            "procedural::chunk_0",
            "procedural::chunk_1",
            "procedural::chunk_2",
        }

    def test_load_done_keys_empty_for_missing_file(self, tmp_path):
        assert pool._load_done_task_keys(tmp_path / "nonexistent.jsonl") == set()

    def test_read_raw_queries_skips_blank_lines(self, tmp_path):
        """Defensive: a blank line in the middle of the file shouldn't
        crash the loader (could happen from manual edit)."""
        path = tmp_path / "raw.jsonl"
        pool._append_raw_query(
            path,
            pool.TaskKey(QuestionType.PROCEDURAL, "a"),
            _make_query(qt=QuestionType.PROCEDURAL),
        )
        # Inject a blank line.
        with path.open("a") as f:
            f.write("\n")
        pool._append_raw_query(
            path,
            pool.TaskKey(QuestionType.PROCEDURAL, "b"),
            _make_query(qt=QuestionType.PROCEDURAL),
        )
        loaded = pool._read_raw_queries(path)
        assert len(loaded) == 2


# --------------------------------------------------------------------
# Classification grouping
# --------------------------------------------------------------------


class TestClassifyCorpus:
    def test_groups_chunks_by_affinity(self):
        chunks = [
            _chunk("a.md", text="Example:\n\n```py\nprint('x')\n```"),  # procedural
            _chunk("b.md", text="Use AutoModel instead of LlamaModel."),  # comparison
            _chunk(
                "c.md",
                text='We advise users to use padding_side="left" before generating.',
            ),  # best_practice
            _chunk("d.md", text="A good starting point is the BERT script."),  # pointer
            _chunk("e.md", text="AutoModel loads pretrained weights."),  # no affinity
        ]
        grouped = pool._classify_corpus(chunks)
        assert len(grouped[QuestionType.PROCEDURAL]) == 1
        assert len(grouped[QuestionType.COMPARISON]) == 1
        assert len(grouped[QuestionType.BEST_PRACTICE]) == 1
        assert len(grouped[QuestionType.POINTER]) == 1
        # Refusal is not a chunk affinity — should not be present.
        assert QuestionType.REFUSAL not in grouped


# --------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------


class TestSampleChunksPerType:
    def test_samples_quota_per_type(self):
        chunks_by_type = {
            QuestionType.PROCEDURAL: [_chunk(f"d{i}.md") for i in range(20)],
            QuestionType.COMPARISON: [_chunk(f"c{i}.md") for i in range(15)],
        }
        quotas = {QuestionType.PROCEDURAL: 5, QuestionType.COMPARISON: 3}
        sampled = pool._sample_chunks_per_type(chunks_by_type, quotas)
        assert len(sampled[QuestionType.PROCEDURAL]) == 5
        assert len(sampled[QuestionType.COMPARISON]) == 3

    def test_sampling_is_deterministic_with_seed(self):
        chunks_by_type = {QuestionType.PROCEDURAL: [_chunk(f"d{i}.md") for i in range(20)]}
        quotas = {QuestionType.PROCEDURAL: 5}
        a = pool._sample_chunks_per_type(chunks_by_type, quotas, seed=42)
        b = pool._sample_chunks_per_type(chunks_by_type, quotas, seed=42)
        assert [c.doc_id for c in a[QuestionType.PROCEDURAL]] == [
            c.doc_id for c in b[QuestionType.PROCEDURAL]
        ]

    def test_quota_exceeds_available_takes_all(self):
        chunks_by_type = {QuestionType.PROCEDURAL: [_chunk(f"d{i}.md") for i in range(3)]}
        quotas = {QuestionType.PROCEDURAL: 10}  # too many
        sampled = pool._sample_chunks_per_type(chunks_by_type, quotas)
        assert len(sampled[QuestionType.PROCEDURAL]) == 3

    def test_empty_type_returns_empty(self):
        sampled = pool._sample_chunks_per_type(
            {QuestionType.COMPARISON: []},
            {QuestionType.COMPARISON: 5},
        )
        assert sampled[QuestionType.COMPARISON] == []


# --------------------------------------------------------------------
# Chunk → ChunkRef conversion
# --------------------------------------------------------------------


class TestChunkToChunkRef:
    def test_preserves_text_and_doc_id(self):
        c = _chunk("foo.md", idx=3, text="hello")
        ref = pool._chunk_to_chunkref(c)
        assert ref.doc_id == "foo.md"
        assert ref.text == "hello"
        assert ref.chunk_id == "foo.md::chunk_3"
        # Score=0.0 because seed chunks aren't scored — they're picked
        # by the sampler, not by retrieval.
        assert ref.score == 0.0


# --------------------------------------------------------------------
# Retrieval-failure refusal synthesis
# --------------------------------------------------------------------


class TestSynthesizeRetrievalFailureRefusals:
    def test_pairs_query_with_chunks_from_different_doc_family(self):
        """A query seeded from model_doc/ should be paired with chunks
        from a DIFFERENT family (e.g., quantization/, tasks/)."""
        in_corpus = [
            _make_query(
                text="how do I save?",
                seed_chunks=[_ref("model_doc/llava.md")],
            ),
        ]
        all_chunks = (
            [_chunk("model_doc/llava.md", idx=i) for i in range(5)]
            + [_chunk("quantization/gptq.md", idx=i) for i in range(5)]
            + [_chunk("tasks/asr.md", idx=i) for i in range(5)]
        )
        refusals = pool._synthesize_retrieval_failure_refusals(
            in_corpus, all_chunks, n_target=1, seed=42
        )
        assert len(refusals) == 1
        r = refusals[0]
        assert r.question_type == QuestionType.REFUSAL
        assert r.query == "how do I save?"
        # All paired chunks should be from a family OTHER than model_doc.
        for c in r.seed_chunks:
            family = c.doc_id.split("/")[0]
            assert family != "model_doc"

    def test_metadata_records_synthesis_method(self):
        in_corpus = [_make_query(seed_chunks=[_ref("model_doc/x.md")])]
        all_chunks = [_chunk("other_family/y.md")] * 5
        refusals = pool._synthesize_retrieval_failure_refusals(in_corpus, all_chunks, n_target=1)
        assert refusals[0].metadata["synthesis_method"] == "retrieval_failure_pairing"
        assert refusals[0].metadata["seed_family_avoided"] == "model_doc"

    def test_n_target_capped_by_available_queries(self):
        """If we ask for 100 refusals but only have 3 in-corpus queries,
        we get 3 refusals — no over-sampling."""
        in_corpus = [_make_query(seed_chunks=[_ref(f"model_doc/x{i}.md")]) for i in range(3)]
        all_chunks = [_chunk("other/y.md")] * 5
        refusals = pool._synthesize_retrieval_failure_refusals(in_corpus, all_chunks, n_target=100)
        assert len(refusals) == 3

    def test_no_in_corpus_queries_returns_empty(self):
        refusals = pool._synthesize_retrieval_failure_refusals([], [_chunk("x.md")], n_target=10)
        assert refusals == []

    def test_skips_queries_with_no_seed_chunks(self):
        """Off-corpus refusals (seed_chunks empty, seed_topic set)
        shouldn't be used as the basis for retrieval-failure refusals."""
        in_corpus = [
            _make_query(qt=QuestionType.REFUSAL, seed_topic="aws", seed_chunks=[]),
            _make_query(seed_chunks=[_ref("model_doc/x.md")]),
        ]
        all_chunks = [_chunk("other/y.md")] * 5
        refusals = pool._synthesize_retrieval_failure_refusals(in_corpus, all_chunks, n_target=10)
        assert len(refusals) == 1


# --------------------------------------------------------------------
# PoolEntry serialization
# --------------------------------------------------------------------


class TestPoolEntryToDict:
    def test_in_corpus_shape(self):
        gq = _make_query(
            text="how do I save?",
            qt=QuestionType.PROCEDURAL,
            seed_chunks=[_ref("doc.md")],
            metadata={"gen_model": "claude-haiku-4-5"},
        )
        retrieved = [_ref("doc.md", idx=1, text="retrieved text")]
        entry = pool.PoolEntry(generated=gq, retrieved_chunks=retrieved)
        d = entry.to_dict()
        assert d["query"] == "how do I save?"
        assert d["question_type"] == "procedural"
        assert d["expected_refusal"] is False
        assert len(d["retrieved_chunks"]) == 1
        assert d["retrieved_chunks"][0]["doc_id"] == "doc.md"
        assert len(d["seed_chunks"]) == 1
        assert d["seed_topic"] is None
        assert d["metadata"]["gen_model"] == "claude-haiku-4-5"

    def test_refusal_shape(self):
        gq = _make_query(
            qt=QuestionType.REFUSAL,
            seed_topic="aws-lambda",
            seed_chunks=[],
        )
        entry = pool.PoolEntry(generated=gq, retrieved_chunks=[_ref("x.md")])
        d = entry.to_dict()
        assert d["question_type"] == "refusal"
        assert d["expected_refusal"] is True
        assert d["seed_topic"] == "aws-lambda"
        assert d["seed_chunks"] == []

    def test_round_trip_via_json(self):
        """The dict must be JSON-serializable — pin so a future addition
        of a non-serializable type breaks loudly."""
        gq = _make_query(
            seed_chunks=[_ref("doc.md")],
            metadata={"captured_at": "2026-05-06T00:00:00+00:00"},
        )
        entry = pool.PoolEntry(generated=gq, retrieved_chunks=[_ref("d.md")])
        as_json = json.dumps(entry.to_dict())
        loaded = json.loads(as_json)
        assert loaded["query"] == gq.query


# --------------------------------------------------------------------
# Markdown artifacts
# --------------------------------------------------------------------


class TestFilterReportMarkdown:
    def test_includes_per_stage_counts(self):
        r1 = FilterReport(
            kept=[_make_query()] * 90,
            dropped=[(_make_query(text="short"), "length<5_words")] * 10,
        )
        r2 = FilterReport(
            kept=[_make_query()] * 85,
            dropped=[(_make_query(), "duplicate_within_procedural")] * 5,
        )
        r3 = FilterReport(
            kept=[_make_query()] * 80,
            dropped=[(_make_query(), "eval_contamination_sim=0.812")] * 5,
        )
        md = pool._format_filter_report_markdown(
            initial_count=100,
            after_length=r1,
            after_dedupe=r2,
            after_contamination=r3,
        )
        assert "Started with **100**" in md
        assert "length floor" in md
        assert "dedupe" in md
        assert "eval contamination" in md
        assert "**80**" in md  # final count


class TestPreviewMarkdown:
    def test_stratified_sample_proportional_to_type_counts(self):
        """30% procedural in pool → ~30% of sample is procedural."""
        # 30 procedural + 60 comparison + 10 refusal = 100 total
        pool_entries = (
            [
                pool.PoolEntry(
                    generated=_make_query(text=f"p{i}", qt=QuestionType.PROCEDURAL),
                    retrieved_chunks=[_ref("doc.md")],
                )
                for i in range(30)
            ]
            + [
                pool.PoolEntry(
                    generated=_make_query(text=f"c{i}", qt=QuestionType.COMPARISON),
                    retrieved_chunks=[_ref("doc.md")],
                )
                for i in range(60)
            ]
            + [
                pool.PoolEntry(
                    generated=_make_query(
                        text=f"r{i}",
                        qt=QuestionType.REFUSAL,
                        seed_topic="aws",
                    ),
                    retrieved_chunks=[_ref("doc.md")],
                )
                for i in range(10)
            ]
        )
        md = pool._format_preview_markdown(pool_entries, n_sample=10, seed=42)
        # Should have sections for all three types present in the pool.
        assert "## procedural" in md
        assert "## comparison" in md
        assert "## refusal" in md
        # Title shows sample size.
        assert "preview (10 sample of 100)" in md

    def test_includes_top_doc_ids(self):
        pool_entries = [
            pool.PoolEntry(
                generated=_make_query(text="how do I save?"),
                retrieved_chunks=[
                    _ref("models.md", idx=1),
                    _ref("trainer.md", idx=2),
                ],
            )
        ]
        md = pool._format_preview_markdown(pool_entries, n_sample=1)
        assert "models.md" in md
        assert "trainer.md" in md

    def test_handles_empty_pool(self):
        """Edge case: no queries survived filters — preview should not crash."""
        md = pool._format_preview_markdown([], n_sample=30)
        assert "preview" in md.lower()


# --------------------------------------------------------------------
# Eval query loading (smoke)
# --------------------------------------------------------------------


class TestLoadEvalQueryTexts:
    def test_returns_nonempty_list_of_strings(self):
        """Smoke: real eval modules load cleanly and produce strings."""
        texts = pool._load_eval_query_texts()
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)
        assert all(t for t in texts)
