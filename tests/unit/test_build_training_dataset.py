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
