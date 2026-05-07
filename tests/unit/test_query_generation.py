"""Unit tests for the type-aware query generator.

Real API calls aren't tested here — tests use a stubbed Anthropic
client to verify shape, parser robustness, and metadata propagation
without paying for actual generation.
"""

from __future__ import annotations

from typing import Any

import pytest

from docsense.finetuning.chunk_classifier import QuestionType
from docsense.finetuning.query_generation import (
    _QUERY_GEN_PROMPT_VERSION,
    GeneratedQuery,
    TypeAwareQueryGenerator,
    _extract_json_block,
    parse_query_response,
)
from docsense.generation.types import ChunkRef


def _chunk(idx: int = 1, text: str = "sample chunk text") -> ChunkRef:
    return ChunkRef(
        doc_id=f"doc{idx}.md",
        chunk_id=f"doc{idx}.md::chunk_{idx}",
        score=1.0,
        text=text,
    )


# --------------------------------------------------------------------
# Anthropic SDK response stubs (mirrors test_build_training_dataset.py).
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
    def __init__(self, text: str, *, input_tokens: int = 100, output_tokens: int = 30):
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


# --------------------------------------------------------------------
# JSON extraction + parse_query_response
# --------------------------------------------------------------------


class TestExtractJsonBlock:
    def test_clean_json_passthrough(self):
        assert _extract_json_block('{"query": "x"}') == '{"query": "x"}'

    def test_strips_markdown_fence(self):
        text = '```json\n{"query": "x"}\n```'
        assert _extract_json_block(text) == '{"query": "x"}'

    def test_strips_unlabeled_fence(self):
        text = '```\n{"query": "x"}\n```'
        assert _extract_json_block(text) == '{"query": "x"}'

    def test_extracts_from_prose_preamble(self):
        text = 'Here you go:\n{"query": "x"}\nThanks!'
        assert _extract_json_block(text) == '{"query": "x"}'

    def test_fence_with_prose_around(self):
        text = 'Sure:\n```json\n{"query": "x"}\n```\nDone.'
        assert _extract_json_block(text) == '{"query": "x"}'


class TestParseQueryResponse:
    def test_clean_response(self):
        text = '{"query": "How do I save a model?"}'
        assert parse_query_response(text) == "How do I save a model?"

    def test_invalid_json_returns_none(self):
        assert parse_query_response("not json at all") is None

    def test_missing_query_field_returns_none(self):
        assert parse_query_response('{"foo": "bar"}') is None

    def test_non_string_query_returns_none(self):
        """If a future model emits a list or dict for ``query``, reject
        rather than coerce — schema is string-or-fail."""
        assert parse_query_response('{"query": ["a", "b"]}') is None
        assert parse_query_response('{"query": 42}') is None

    def test_empty_query_returns_none(self):
        assert parse_query_response('{"query": ""}') is None
        assert parse_query_response('{"query": "   "}') is None

    def test_strips_whitespace(self):
        assert parse_query_response('{"query": "  trimmed  "}') == "trimmed"

    def test_fence_wrapped_response(self):
        text = '```json\n{"query": "How do I save a model?"}\n```'
        assert parse_query_response(text) == "How do I save a model?"


# --------------------------------------------------------------------
# GeneratedQuery model
# --------------------------------------------------------------------


class TestGeneratedQuery:
    def test_in_corpus_shape(self):
        q = GeneratedQuery(
            query="How do I save?",
            question_type=QuestionType.PROCEDURAL,
            seed_chunks=[_chunk()],
            seed_topic=None,
        )
        assert q.query == "How do I save?"
        assert q.question_type == QuestionType.PROCEDURAL
        assert len(q.seed_chunks) == 1
        assert q.seed_topic is None

    def test_refusal_shape(self):
        q = GeneratedQuery(
            query="How do I tune AWS Lambda?",
            question_type=QuestionType.REFUSAL,
            seed_topic="AWS Lambda",
        )
        assert q.question_type == QuestionType.REFUSAL
        assert q.seed_chunks == []
        assert q.seed_topic == "AWS Lambda"

    def test_empty_query_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            GeneratedQuery(
                query="",
                question_type=QuestionType.PROCEDURAL,
            )


# --------------------------------------------------------------------
# TypeAwareQueryGenerator.generate_for_chunk
# --------------------------------------------------------------------


class TestGenerateForChunk:
    def test_returns_generated_query_with_correct_type(self):
        client = _StubClient('{"query": "How do I save a model?"}')
        gen = TypeAwareQueryGenerator(client, model="claude-haiku-4-5")  # type: ignore[arg-type]
        chunk = _chunk(text="model.save_pretrained()")
        result = gen.generate_for_chunk(chunk, QuestionType.PROCEDURAL)

        assert isinstance(result, GeneratedQuery)
        assert result.query == "How do I save a model?"
        assert result.question_type == QuestionType.PROCEDURAL
        assert result.seed_chunks == [chunk]
        assert result.seed_topic is None

    def test_each_type_gets_distinct_instruction_in_prompt(self):
        """The type instruction must appear in the prompt sent to the
        model — pin so a future refactor doesn't accidentally drop the
        type-shaping signal."""
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client)  # type: ignore[arg-type]
        chunk = _chunk()

        for qt, expected_phrase in [
            (QuestionType.PROCEDURAL, "procedural question"),
            (QuestionType.COMPARISON, "comparison question"),
            (QuestionType.BEST_PRACTICE, "best practice"),
            (QuestionType.POINTER, "pointer question"),
        ]:
            gen.generate_for_chunk(chunk, qt)
            sent_prompt = client.messages.create_kwargs["messages"][0]["content"]
            assert expected_phrase in sent_prompt, f"missing '{expected_phrase}' for {qt}"

    def test_chunk_text_and_doc_id_in_prompt(self):
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client)  # type: ignore[arg-type]
        chunk = _chunk(text="UNIQUE_CHUNK_TEXT_42")
        gen.generate_for_chunk(chunk, QuestionType.PROCEDURAL)
        sent = client.messages.create_kwargs["messages"][0]["content"]
        assert "UNIQUE_CHUNK_TEXT_42" in sent
        assert chunk.doc_id in sent

    def test_metadata_includes_provenance(self):
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client, model="claude-haiku-4-5")  # type: ignore[arg-type]
        result = gen.generate_for_chunk(_chunk(), QuestionType.PROCEDURAL)

        meta = result.metadata
        assert meta["gen_model"] == "claude-haiku-4-5"
        assert meta["prompt_version"] == _QUERY_GEN_PROMPT_VERSION
        assert "captured_at" in meta
        assert meta["input_tokens"] == 100
        assert meta["output_tokens"] == 30
        assert meta["cost_usd"] is not None
        assert meta["cost_usd"] > 0

    def test_unknown_model_has_none_cost(self):
        """If the caller passes a model not in our pricing table, cost
        is None (not a crash). Graceful for ad-hoc model experimentation."""
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client, model="unknown-model-2099")  # type: ignore[arg-type]
        result = gen.generate_for_chunk(_chunk(), QuestionType.PROCEDURAL)
        assert result.metadata["cost_usd"] is None
        # Still records token counts for downstream cost analysis.
        assert result.metadata["input_tokens"] == 100

    def test_refusal_type_raises_value_error(self):
        """REFUSAL is reserved for ``generate_for_topic``. Calling
        ``generate_for_chunk`` with REFUSAL is a programmer error."""
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="REFUSAL"):
            gen.generate_for_chunk(_chunk(), QuestionType.REFUSAL)

    def test_parse_failure_raises_runtime_error(self):
        """Bad JSON from the model raises RuntimeError so the caller
        can decide how to handle (skip + log vs abort the batch)."""
        client = _StubClient("garbage prose with no json")
        gen = TypeAwareQueryGenerator(client)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="parse failure"):
            gen.generate_for_chunk(_chunk(), QuestionType.PROCEDURAL)

    def test_temperature_zero_for_chunk(self):
        """In-corpus generation is deterministic (same chunk → same query)
        so a crashed run resumed via raw_queries.jsonl produces identical
        output for completed tasks. Pin temperature=0."""
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client)  # type: ignore[arg-type]
        gen.generate_for_chunk(_chunk(), QuestionType.PROCEDURAL)
        assert client.messages.create_kwargs["temperature"] == 0.0

    def test_metadata_records_temperature_for_chunk(self):
        """Temperature is recorded in metadata for full provenance."""
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client)  # type: ignore[arg-type]
        result = gen.generate_for_chunk(_chunk(), QuestionType.PROCEDURAL)
        assert result.metadata["temperature"] == 0.0

    def test_default_model_is_haiku(self):
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client)  # type: ignore[arg-type]
        gen.generate_for_chunk(_chunk(), QuestionType.PROCEDURAL)
        assert client.messages.create_kwargs["model"] == "claude-haiku-4-5"


# --------------------------------------------------------------------
# TypeAwareQueryGenerator.generate_for_topic
# --------------------------------------------------------------------


class TestGenerateForTopic:
    def test_returns_refusal_query(self):
        client = _StubClient('{"query": "How do I tune Lambda cold starts?"}')
        gen = TypeAwareQueryGenerator(client, model="claude-haiku-4-5")  # type: ignore[arg-type]
        result = gen.generate_for_topic("AWS Lambda cold starts")

        assert result.question_type == QuestionType.REFUSAL
        assert result.seed_chunks == []
        assert result.seed_topic == "AWS Lambda cold starts"
        assert result.query == "How do I tune Lambda cold starts?"

    def test_topic_appears_in_prompt(self):
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client)  # type: ignore[arg-type]
        gen.generate_for_topic("UNIQUE_TOPIC_SEED_42")
        sent = client.messages.create_kwargs["messages"][0]["content"]
        assert "UNIQUE_TOPIC_SEED_42" in sent

    def test_topic_prompt_includes_avoid_list(self):
        """The topic prompt tells the model what HF topics to AVOID
        — pin so a future refactor doesn't drop that guardrail."""
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client)  # type: ignore[arg-type]
        gen.generate_for_topic("Linux kernels")
        sent = client.messages.create_kwargs["messages"][0]["content"]
        assert "AVOID" in sent
        assert "transformers" in sent.lower()

    def test_temperature_07_for_topic(self):
        """Refusal generation uses temperature=0.7 so repeated calls per
        seed produce diverse questions. Pin so a future change doesn't
        silently drop diversity (regression: 67% within-seed dedupe at
        temperature=0)."""
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client)  # type: ignore[arg-type]
        gen.generate_for_topic("AWS Lambda cold starts")
        assert client.messages.create_kwargs["temperature"] == 0.7

    def test_metadata_records_temperature_for_topic(self):
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client)  # type: ignore[arg-type]
        result = gen.generate_for_topic("AWS Lambda cold starts")
        assert result.metadata["temperature"] == 0.7

    def test_metadata_includes_provenance(self):
        client = _StubClient('{"query": "x"}')
        gen = TypeAwareQueryGenerator(client, model="claude-haiku-4-5")  # type: ignore[arg-type]
        result = gen.generate_for_topic("PostgreSQL VACUUM")

        meta = result.metadata
        assert meta["gen_model"] == "claude-haiku-4-5"
        assert meta["prompt_version"] == _QUERY_GEN_PROMPT_VERSION
        assert "captured_at" in meta

    def test_parse_failure_raises_runtime_error(self):
        client = _StubClient("not json")
        gen = TypeAwareQueryGenerator(client)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="parse failure"):
            gen.generate_for_topic("Linux kernel")
