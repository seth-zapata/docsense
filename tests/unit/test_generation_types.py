"""Tests for the pydantic models that define Answer/Citation/ChunkRef.

These are contract tests — they pin the schema rather than validating
LLM behavior. Once FastAPI wraps the pipeline in Phase 4, breaking these
tests is equivalent to breaking the API response shape.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from docsense.generation.types import (
    Answer,
    ChunkRef,
    Citation,
    GenerationMetadata,
)


class TestChunkRef:
    def test_round_trip(self):
        ref = ChunkRef(doc_id="d.md", chunk_id="d.md::chunk_0", score=0.87, text="hello")
        assert ChunkRef.model_validate(ref.model_dump()) == ref

    def test_score_required(self):
        with pytest.raises(ValidationError):
            ChunkRef(doc_id="d.md", chunk_id="d.md::chunk_0", text="hello")  # type: ignore[call-arg]

    def test_score_must_be_numeric(self):
        with pytest.raises(ValidationError):
            ChunkRef(doc_id="d.md", chunk_id="d.md::chunk_0", score="hi", text="hello")  # type: ignore[arg-type]


class TestCitation:
    def test_quote_defaults_to_none(self):
        c = Citation(doc_id="d.md", chunk_id="d.md::chunk_0")
        assert c.quote is None

    def test_quote_optional(self):
        c = Citation(doc_id="d.md", chunk_id="d.md::chunk_0", quote="exact span")
        assert c.quote == "exact span"


class TestGenerationMetadata:
    def test_minimal(self):
        m = GenerationMetadata(model_name="mistral-7b", latency_ms=123.4)
        assert m.prompt_tokens is None
        assert m.completion_tokens is None

    def test_full(self):
        m = GenerationMetadata(
            model_name="mistral-7b",
            latency_ms=123.4,
            prompt_tokens=512,
            completion_tokens=256,
        )
        assert m.prompt_tokens == 512


class TestAnswer:
    def test_minimal_construction(self):
        """An Answer can be constructed with just text + metadata; citations
        and retrieved_chunks default to empty."""
        ans = Answer(
            text="Hello world",
            metadata=GenerationMetadata(model_name="mistral-7b", latency_ms=100.0),
        )
        assert ans.text == "Hello world"
        assert ans.citations == []
        assert ans.retrieved_chunks == []

    def test_full_round_trip(self):
        """Construct a fully-populated Answer, serialize, validate back —
        every field round-trips. Pins the JSON schema FastAPI will expose."""
        ans = Answer(
            text="The answer is in [1].",
            citations=[Citation(doc_id="d.md", chunk_id="d.md::chunk_0", quote="excerpt")],
            retrieved_chunks=[
                ChunkRef(doc_id="d.md", chunk_id="d.md::chunk_0", score=0.9, text="hello")
            ],
            metadata=GenerationMetadata(
                model_name="mistral-7b",
                latency_ms=150.0,
                prompt_tokens=200,
                completion_tokens=50,
            ),
        )
        round_tripped = Answer.model_validate(ans.model_dump())
        assert round_tripped == ans

    def test_metadata_required(self):
        with pytest.raises(ValidationError):
            Answer(text="hi")  # type: ignore[call-arg]

    def test_extra_fields_rejected(self):
        """Pydantic default mode is to allow extras silently; the test
        documents that we accept the default. If we ever need strict mode,
        this test will need updating along with model_config."""
        # Default behavior: unknown field is ignored, not raised
        ans = Answer.model_validate(
            {
                "text": "hi",
                "metadata": {"model_name": "m", "latency_ms": 1.0},
                "extra_unknown": "tolerated",
            }
        )
        assert ans.text == "hi"
