"""Unit tests for the Anthropic distillation script.

Real API calls aren't tested here (they'd cost money and require the
key). Tests cover the testable surface:

- ``parse_distill_response`` and ``_extract_json_block``: pure parsers
  for the model's JSON output. Same robustness pattern as the eval
  pipeline parsers — strip fences, find JSON, validate shape.
- ``format_user_message``: pure formatting of (query, chunks) into
  the user message body.
- ``distill_one_example`` with a stubbed client: shape of the returned
  TrainingExample, refusal detection, metadata propagation.
- ``_read_api_key``: file read + missing-file behavior.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from docsense.finetuning.dataset import TrainingExample
from docsense.generation.types import ChunkRef


def _load_distill_script():
    """scripts/ isn't a package; load by file path."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_training_dataset.py"
    spec = importlib.util.spec_from_file_location("build_training_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_training_dataset"] = module
    spec.loader.exec_module(module)
    return module


distill = _load_distill_script()


def _chunk(idx: int, text: str = "chunk text") -> ChunkRef:
    return ChunkRef(doc_id=f"doc{idx}.md", chunk_id=str(idx), score=1.0, text=text)


# --------------------------------------------------------------------
# JSON extraction + response parsing
# --------------------------------------------------------------------


class TestExtractJsonBlock:
    def test_clean_json_passthrough(self):
        assert distill._extract_json_block('{"answer": "x"}') == '{"answer": "x"}'

    def test_strips_markdown_fence(self):
        text = '```json\n{"answer": "x"}\n```'
        assert distill._extract_json_block(text) == '{"answer": "x"}'

    def test_strips_unlabeled_fence(self):
        text = '```\n{"answer": "x"}\n```'
        assert distill._extract_json_block(text) == '{"answer": "x"}'

    def test_extracts_from_prose_preamble(self):
        """LLMs sometimes preface output with 'Here is...'. Extract
        the JSON block out of the surrounding prose."""
        text = 'Here is the answer:\n{"answer": "x"}\nThanks!'
        assert distill._extract_json_block(text) == '{"answer": "x"}'


class TestParseDistillResponse:
    def test_clean_response(self):
        text = '{"answer": "Install with `pip install transformers` [1]."}'
        answer = distill.parse_distill_response(text)
        assert answer == "Install with `pip install transformers` [1]."

    def test_invalid_json_returns_none(self):
        assert distill.parse_distill_response("not json at all") is None

    def test_missing_answer_field_returns_none(self):
        assert distill.parse_distill_response('{"foo": "x"}') is None

    def test_non_string_answer_returns_none(self):
        """If a future LLM emits a list or dict for ``answer``, reject
        rather than coerce — the schema is string-or-fail."""
        assert distill.parse_distill_response('{"answer": ["x", "y"]}') is None
        assert distill.parse_distill_response('{"answer": 42}') is None

    def test_empty_answer_returns_none(self):
        """Empty/whitespace answer is parse-failure equivalent."""
        assert distill.parse_distill_response('{"answer": ""}') is None
        assert distill.parse_distill_response('{"answer": "   "}') is None

    def test_strips_whitespace(self):
        text = '{"answer": "  spaces around  "}'
        assert distill.parse_distill_response(text) == "spaces around"

    def test_fence_wrapped_response(self):
        text = '```json\n{"answer": "with citation [1]"}\n```'
        assert distill.parse_distill_response(text) == "with citation [1]"

    def test_plain_text_canonical_refusal_accepted(self):
        """Sonnet sometimes drops the JSON wrapper when refusing,
        returning just the canonical refusal phrase as plain text.
        Treating this as a parse failure would silently lose valid
        refusal training data — observed at ~14% of procedural queries
        in the Block 3B.2 spot-check. Accept and return as a refusal."""
        assert (
            distill.parse_distill_response("I don't have enough context to answer that.")
            == distill.CANONICAL_REFUSAL
        )

    def test_quoted_canonical_refusal_accepted(self):
        """Sonnet may also wrap the refusal in JSON-style string quotes
        (without the object wrapper). Strip those and accept."""
        assert (
            distill.parse_distill_response('"I don\'t have enough context to answer that."')
            == distill.CANONICAL_REFUSAL
        )

    def test_plain_text_canonical_refusal_with_whitespace(self):
        """Trailing/leading whitespace shouldn't defeat the match."""
        assert (
            distill.parse_distill_response("  I don't have enough context to answer that.  \n")
            == distill.CANONICAL_REFUSAL
        )

    def test_plain_text_non_refusal_still_fails(self):
        """Regression guard: only the EXACT canonical refusal phrase is
        accepted as plain text. Any other plain-text response is still
        a parse failure (we don't want to silently accept arbitrary
        non-JSON answers)."""
        assert distill.parse_distill_response("Use save_pretrained() [1].") is None
        assert distill.parse_distill_response("I don't know.") is None
        assert (
            distill.parse_distill_response("I don't have enough context to answer this.") is None
        )  # "this" instead of "that" — strict match


# --------------------------------------------------------------------
# User-message formatting
# --------------------------------------------------------------------


class TestFormatUserMessage:
    def test_chunks_numbered_with_source(self):
        chunks = [_chunk(1, "alpha"), _chunk(2, "beta")]
        msg = distill.format_user_message(query="How do I X?", chunks=chunks)
        assert "[1] (source: doc1.md)" in msg
        assert "[2] (source: doc2.md)" in msg
        assert "alpha" in msg
        assert "beta" in msg
        assert "QUESTION: How do I X?" in msg

    def test_empty_chunks_marked_as_no_chunks(self):
        """Refusal queries may have no chunks — render an explicit
        '(no chunks retrieved)' marker rather than an empty CHUNKS:
        block, so the model can see the empty-context case clearly."""
        msg = distill.format_user_message(query="off-corpus q", chunks=[])
        assert "(no chunks retrieved)" in msg
        assert "QUESTION: off-corpus q" in msg


# --------------------------------------------------------------------
# distill_one_example — full flow with a stubbed Anthropic client
# --------------------------------------------------------------------


class _StubContentBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _StubUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _StubResponse:
    def __init__(self, text: str, *, input_tokens: int = 100, output_tokens: int = 50):
        self.content = [_StubContentBlock(text)]
        self.usage = _StubUsage(input_tokens, output_tokens)


class _StubMessages:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.create_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _StubResponse:
        self.create_kwargs = kwargs
        return _StubResponse(self.response_text)


class _StubClient:
    def __init__(self, response_text: str):
        self.messages = _StubMessages(response_text)


class TestDistillOneExample:
    def test_in_corpus_returns_training_example(self):
        client = _StubClient('{"answer": "Use `pip install transformers` [1]."}')
        chunks = [_chunk(1, "Run pip install transformers.")]
        result = distill.distill_one_example(
            client,  # type: ignore[arg-type]
            query="How do I install?",
            chunks=chunks,
            model="claude-sonnet-4-5",
        )
        assert isinstance(result, TrainingExample)
        assert result.query == "How do I install?"
        assert result.ideal_answer == "Use `pip install transformers` [1]."
        assert result.is_refusal is False
        assert len(result.retrieved_chunks) == 1

    def test_refusal_detected_on_canonical_phrase(self):
        client = _StubClient('{"answer": "I don\'t have enough context to answer that."}')
        result = distill.distill_one_example(
            client,  # type: ignore[arg-type]
            query="off-corpus q?",
            chunks=[_chunk(1, "irrelevant chunk")],
            model="claude-sonnet-4-5",
        )
        assert result.is_refusal is True
        assert result.ideal_answer == distill.CANONICAL_REFUSAL

    def test_metadata_carries_provenance(self):
        client = _StubClient('{"answer": "ok [1]"}')
        result = distill.distill_one_example(
            client,  # type: ignore[arg-type]
            query="q",
            chunks=[_chunk(1)],
            model="claude-opus-4-7",
        )
        assert result.metadata["distill_model"] == "claude-opus-4-7"
        assert result.metadata["enable_thinking"] is False
        assert result.metadata["prompt_version"] == distill._DISTILL_PROMPT_VERSION
        assert "captured_at" in result.metadata
        assert result.metadata["input_tokens"] == 100
        assert result.metadata["output_tokens"] == 50

    def test_thinking_flag_in_create_kwargs(self):
        """``--enable-thinking`` translates to thinking={...} in the
        SDK call. Pin the wiring."""
        client = _StubClient('{"answer": "ok [1]"}')
        # The client.messages.create captures kwargs; we'll check after.
        distill.distill_one_example(
            client,  # type: ignore[arg-type]
            query="q",
            chunks=[_chunk(1)],
            model="claude-sonnet-4-5",
            enable_thinking=True,
        )
        kwargs = client.messages.create_kwargs
        assert kwargs.get("thinking") == {
            "type": "enabled",
            "budget_tokens": distill._THINKING_BUDGET,
        }
        # Anthropic forbids non-default temperature with thinking;
        # we drop the temperature kwarg so the SDK uses its default.
        assert "temperature" not in kwargs

    def test_no_thinking_default_passes_temperature_zero(self):
        client = _StubClient('{"answer": "ok"}')
        distill.distill_one_example(
            client,  # type: ignore[arg-type]
            query="q",
            chunks=[_chunk(1)],
            model="claude-sonnet-4-5",
        )
        kwargs = client.messages.create_kwargs
        assert kwargs.get("temperature") == 0.0
        assert "thinking" not in kwargs

    def test_opus_4_7_skips_temperature(self):
        """Opus 4.7 returns 400 'temperature is deprecated' — we
        detect by model name and omit the kwarg. Pin so a future
        refactor doesn't accidentally pass temperature to opus."""
        client = _StubClient('{"answer": "ok"}')
        distill.distill_one_example(
            client,  # type: ignore[arg-type]
            query="q",
            chunks=[_chunk(1)],
            model="claude-opus-4-7",
        )
        kwargs = client.messages.create_kwargs
        assert "temperature" not in kwargs

    def test_model_rejects_temperature_helper(self):
        """The detection helper is exact-match on the prefix so we
        don't mis-classify unrelated models."""
        assert distill._model_rejects_temperature("claude-opus-4-7") is True
        assert distill._model_rejects_temperature("claude-opus-4-7-20260101") is True
        assert distill._model_rejects_temperature("claude-sonnet-4-5") is False
        assert distill._model_rejects_temperature("claude-haiku-4-5") is False

    def test_parse_failure_raises(self):
        """Bad JSON from the model raises RuntimeError so the caller
        decides how to handle (skip + log vs abort the batch)."""
        client = _StubClient("not json, just prose")
        with pytest.raises(RuntimeError, match="parse failure"):
            distill.distill_one_example(
                client,  # type: ignore[arg-type]
                query="q",
                chunks=[_chunk(1)],
                model="claude-sonnet-4-5",
            )


# --------------------------------------------------------------------
# API key reading
# --------------------------------------------------------------------


class TestReadApiKey:
    def test_reads_from_anthropic_key_file(self, tmp_path, monkeypatch):
        """Patch HOME to a tmp path so the test's fake key file is
        what gets read."""
        fake_home = tmp_path
        monkeypatch.setenv("HOME", str(fake_home))
        # Path.home() reads HOME on Linux; verify the redirection works
        assert Path.home() == fake_home

        key_file = fake_home / ".anthropic-key"
        key_file.write_text("sk-ant-test-fake-key\n")
        key_file.chmod(0o600)

        result = distill._read_api_key()
        assert result == "sk-ant-test-fake-key"

    def test_missing_file_raises_with_setup_instructions(self, tmp_path, monkeypatch):
        """When the key file is missing, raise an error that points at
        the setup steps — not let the SDK fail with a less obvious
        'missing api key' message later."""
        monkeypatch.setenv("HOME", str(tmp_path))
        # No key file in tmp_path
        with pytest.raises(SystemExit, match="No Anthropic API key"):
            distill._read_api_key()


# --------------------------------------------------------------------
# Pilot input loading
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Resumability helpers
# --------------------------------------------------------------------


class TestRawJsonlPath:
    def test_json_output(self, tmp_path):
        out = tmp_path / "dataset.json"
        assert distill._raw_jsonl_path(out) == tmp_path / "dataset.raw.jsonl"

    def test_other_extension(self, tmp_path):
        out = tmp_path / "out.txt"
        assert distill._raw_jsonl_path(out) == tmp_path / "out.raw.jsonl"

    def test_no_extension(self, tmp_path):
        out = tmp_path / "noext"
        assert distill._raw_jsonl_path(out) == tmp_path / "noext.raw.jsonl"


class TestCacheKey:
    def test_query_and_chunk_ids_both_in_key(self):
        chunks = [_chunk(1), _chunk(2)]
        key = distill._cache_key("how do I X?", chunks)
        assert "how do I X?" in key
        assert "1" in key  # chunk_id from _chunk(1)
        assert "2" in key

    def test_same_query_different_chunks_distinct_keys(self):
        """If retrieval returns different chunks for the same query
        (e.g., re-ran Stage 1 with different sampling), the cached
        answer is correctly invalidated."""
        a = distill._cache_key("q", [_chunk(1)])
        b = distill._cache_key("q", [_chunk(2)])
        assert a != b

    def test_chunk_order_matters(self):
        """Chunk order is preserved in the key. Same chunks reordered
        → different key. This is conservative — could relax to
        order-independent if it becomes a problem, but order does
        affect the citation indices in the answer."""
        a = distill._cache_key("q", [_chunk(1), _chunk(2)])
        b = distill._cache_key("q", [_chunk(2), _chunk(1)])
        assert a != b


class TestReadDoneExamples:
    def test_returns_empty_for_missing_file(self, tmp_path):
        assert distill._read_done_examples(tmp_path / "nonexistent.jsonl") == {}

    def test_round_trip_via_jsonl(self, tmp_path):
        path = tmp_path / "raw.jsonl"
        ex = TrainingExample(
            query="how do I X?",
            retrieved_chunks=[_chunk(1)],
            ideal_answer="Use X [1].",
            is_refusal=False,
        )
        distill._append_example(path, ex)
        loaded = distill._read_done_examples(path)
        assert len(loaded) == 1
        key = distill._cache_key(ex.query, ex.retrieved_chunks)
        assert key in loaded
        assert loaded[key].query == ex.query
        assert loaded[key].ideal_answer == "Use X [1]."

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "raw.jsonl"
        ex = TrainingExample(
            query="q1",
            retrieved_chunks=[_chunk(1)],
            ideal_answer="A [1]",
            is_refusal=False,
        )
        distill._append_example(path, ex)
        # Inject a blank line in the middle.
        with path.open("a") as f:
            f.write("\n")
        ex2 = TrainingExample(
            query="q2",
            retrieved_chunks=[_chunk(2)],
            ideal_answer="B [1]",
            is_refusal=False,
        )
        distill._append_example(path, ex2)
        assert len(distill._read_done_examples(path)) == 2

    def test_skips_corrupt_lines(self, tmp_path):
        """One malformed line shouldn't lose all other examples in the
        checkpoint — partial-write tolerance for crashed runs."""
        path = tmp_path / "raw.jsonl"
        ex = TrainingExample(
            query="q1",
            retrieved_chunks=[_chunk(1)],
            ideal_answer="A [1]",
            is_refusal=False,
        )
        distill._append_example(path, ex)
        # Write a line that's NOT valid JSON.
        with path.open("a") as f:
            f.write("not-json-at-all\n")
            f.write('{"incomplete": "valid json but wrong shape"}\n')
        ex2 = TrainingExample(
            query="q2",
            retrieved_chunks=[_chunk(2)],
            ideal_answer="B [1]",
            is_refusal=False,
        )
        distill._append_example(path, ex2)
        loaded = distill._read_done_examples(path)
        assert len(loaded) == 2  # the two valid examples; corruption skipped


class TestAppendExample:
    def test_creates_file_if_absent(self, tmp_path):
        path = tmp_path / "subdir" / "raw.jsonl"
        assert not path.exists()
        ex = TrainingExample(
            query="q",
            retrieved_chunks=[_chunk(1)],
            ideal_answer="A [1]",
            is_refusal=False,
        )
        distill._append_example(path, ex)
        assert path.exists()
        assert path.read_text().strip().startswith("{")


class TestProcessOneInput:
    """Verifies the per-input wrapper catches both parse failures
    (RuntimeError) and citation hallucinations (ValueError) so a
    single bad output doesn't crash the entire run."""

    def test_returns_training_example_on_success(self):
        client = _StubClient('{"answer": "Use save_pretrained() [1]."}')
        inp = distill._PilotInput(query="how do I X?", retrieved_chunks=[_chunk(1)])
        ex = distill._process_one_input(
            client,  # type: ignore[arg-type]
            inp,
            model="claude-sonnet-4-5",
            enable_thinking=False,
        )
        assert ex is not None
        assert ex.query == "how do I X?"

    def test_parse_failure_returns_none(self):
        client = _StubClient("not json at all")
        inp = distill._PilotInput(query="q", retrieved_chunks=[_chunk(1)])
        ex = distill._process_one_input(
            client,  # type: ignore[arg-type]
            inp,
            model="claude-sonnet-4-5",
            enable_thinking=False,
        )
        assert ex is None

    def test_citation_hallucination_returns_none(self):
        """Critical regression guard: Sonnet citing [99] when only 1
        chunk exists triggers TrainingExample's citation-grounding
        validator (ValueError). Previously this would crash the entire
        run and lose all prior progress; now it's caught and skipped
        like any other bad output."""
        client = _StubClient('{"answer": "Use [1] [99]."}')  # [99] out of range
        inp = distill._PilotInput(query="q", retrieved_chunks=[_chunk(1)])
        ex = distill._process_one_input(
            client,  # type: ignore[arg-type]
            inp,
            model="claude-sonnet-4-5",
            enable_thinking=False,
        )
        assert ex is None

    def test_propagates_pool_metadata(self):
        client = _StubClient('{"answer": "Use [1]."}')
        inp = distill._PilotInput(
            query="q",
            retrieved_chunks=[_chunk(1)],
            question_type="procedural",
            query_gen_metadata={"gen_model": "claude-haiku-4-5"},
        )
        ex = distill._process_one_input(
            client,  # type: ignore[arg-type]
            inp,
            model="claude-sonnet-4-5",
            enable_thinking=False,
        )
        assert ex is not None
        assert ex.metadata["question_type"] == "procedural"
        assert ex.metadata["query_gen"]["gen_model"] == "claude-haiku-4-5"


# --------------------------------------------------------------------
# Pilot/pool input loading
# --------------------------------------------------------------------


class TestLoadInputs:
    def test_load_pilot_input_format(self, tmp_path):
        import json

        path = tmp_path / "pilot.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "query": "How do I X?",
                        "expected_refusal": False,
                        "retrieved_chunks": [
                            {"doc_id": "a.md", "chunk_id": "1", "score": 0.9, "text": "alpha"},
                            {"doc_id": "b.md", "chunk_id": "2", "score": 0.8, "text": "beta"},
                        ],
                    },
                    {
                        "query": "off-corpus?",
                        "expected_refusal": True,
                        "retrieved_chunks": [],
                        "note": "should refuse",
                    },
                ]
            )
        )
        inputs = distill._load_inputs(path)
        assert len(inputs) == 2
        assert inputs[0].query == "How do I X?"
        assert inputs[0].expected_refusal is False
        assert len(inputs[0].retrieved_chunks) == 2
        assert inputs[1].expected_refusal is True
        assert inputs[1].note == "should refuse"
        # Pilot inputs don't carry pool fields → both default None.
        assert inputs[0].question_type is None
        assert inputs[0].query_gen_metadata is None

    def test_load_pool_jsonl_format(self, tmp_path):
        """Block 3B.2.e's query_pool.jsonl format — newline-separated
        entries with extra question_type and metadata fields. Should
        load cleanly and propagate the pool-only fields onto _PilotInput."""
        import json

        path = tmp_path / "query_pool.jsonl"
        entries = [
            {
                "query": "How do I save a model?",
                "question_type": "procedural",
                "expected_refusal": False,
                "retrieved_chunks": [
                    {"doc_id": "a.md", "chunk_id": "1", "score": 0.9, "text": "alpha"},
                ],
                "seed_chunks": [],
                "seed_topic": None,
                "metadata": {
                    "gen_model": "claude-haiku-4-5",
                    "prompt_version": "v1",
                    "input_tokens": 271,
                    "output_tokens": 27,
                    "cost_usd": 0.000324,
                },
            },
            {
                "query": "How do I tune AWS Lambda cold starts?",
                "question_type": "refusal",
                "expected_refusal": True,
                "retrieved_chunks": [],
                "seed_chunks": [],
                "seed_topic": "AWS Lambda cold start optimization",
                "metadata": {"gen_model": "claude-haiku-4-5", "prompt_version": "v1"},
            },
        ]
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        inputs = distill._load_inputs(path)
        assert len(inputs) == 2
        assert inputs[0].query == "How do I save a model?"
        assert inputs[0].question_type == "procedural"
        assert inputs[0].query_gen_metadata is not None
        assert inputs[0].query_gen_metadata["gen_model"] == "claude-haiku-4-5"
        assert inputs[0].query_gen_metadata["cost_usd"] == 0.000324
        assert inputs[1].expected_refusal is True
        assert inputs[1].question_type == "refusal"

    def test_load_pool_jsonl_skips_blank_lines(self, tmp_path):
        """JSONL files sometimes acquire blank lines (manual edits,
        crashed appends). Loader should skip them rather than crash."""
        import json

        path = tmp_path / "with_blanks.jsonl"
        entry = {
            "query": "How do I X?",
            "expected_refusal": False,
            "retrieved_chunks": [
                {"doc_id": "a.md", "chunk_id": "1", "score": 0.9, "text": "alpha"},
            ],
        }
        path.write_text(f"{json.dumps(entry)}\n\n{json.dumps(entry)}\n")
        inputs = distill._load_inputs(path)
        assert len(inputs) == 2

    def test_format_dispatch_by_extension(self, tmp_path):
        """``.json`` → array dispatch; ``.jsonl`` → newline dispatch.
        Pin so a future refactor doesn't accidentally swap them."""
        import json

        # Same payload, two extensions, two format conventions.
        entry = {
            "query": "test",
            "expected_refusal": False,
            "retrieved_chunks": [{"doc_id": "a.md", "chunk_id": "1", "score": 0.9, "text": "x"}],
        }
        json_path = tmp_path / "input.json"
        json_path.write_text(json.dumps([entry]))
        jsonl_path = tmp_path / "input.jsonl"
        jsonl_path.write_text(json.dumps(entry) + "\n")

        a = distill._load_inputs(json_path)
        b = distill._load_inputs(jsonl_path)
        assert len(a) == len(b) == 1
        assert a[0].query == b[0].query


# --------------------------------------------------------------------
# Pool metadata propagation through distill_one_example
# --------------------------------------------------------------------


class TestPoolMetadataPropagation:
    def test_question_type_lands_in_metadata(self):
        client = _StubClient('{"answer": "How do I save? Use save_pretrained [1]."}')
        result = distill.distill_one_example(
            client,  # type: ignore[arg-type]
            query="how do I save?",
            chunks=[_chunk(1)],
            model="claude-sonnet-4-5",
            question_type="procedural",
        )
        assert result.metadata["question_type"] == "procedural"

    def test_query_gen_metadata_nested_under_query_gen_key(self):
        """Pool's gen-time metadata gets nested under ``query_gen`` so
        distill-time fields stay at the top level (existing test
        contracts unchanged) while gen provenance is preserved."""
        client = _StubClient('{"answer": "ok [1]"}')
        result = distill.distill_one_example(
            client,  # type: ignore[arg-type]
            query="q",
            chunks=[_chunk(1)],
            model="claude-sonnet-4-5",
            query_gen_metadata={
                "gen_model": "claude-haiku-4-5",
                "prompt_version": "v1",
                "input_tokens": 250,
                "output_tokens": 30,
            },
        )
        assert "query_gen" in result.metadata
        assert result.metadata["query_gen"]["gen_model"] == "claude-haiku-4-5"
        # Distill-time fields still at the top level.
        assert result.metadata["distill_model"] == "claude-sonnet-4-5"
        assert result.metadata["input_tokens"] != 250  # would collide if not nested

    def test_pool_fields_default_none_for_pilot_input(self):
        """When called without pool fields (legacy pilot path),
        question_type and query_gen aren't added to metadata — only
        the original distill-time fields are present."""
        client = _StubClient('{"answer": "ok [1]"}')
        result = distill.distill_one_example(
            client,  # type: ignore[arg-type]
            query="q",
            chunks=[_chunk(1)],
            model="claude-sonnet-4-5",
        )
        assert "question_type" not in result.metadata
        assert "query_gen" not in result.metadata
