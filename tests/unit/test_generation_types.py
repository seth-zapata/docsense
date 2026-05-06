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


class TestCitationPreservationInvariant:
    """Block D.1: pin the invariant that every Citation in
    Answer.citations must reference a chunk that's in
    Answer.retrieved_chunks. Enforced by the model_validator on Answer."""

    def _meta(self) -> GenerationMetadata:
        return GenerationMetadata(model_name="m", latency_ms=1.0)

    def test_citations_referencing_retrieved_chunks_are_valid(self):
        """Happy path: citation matches a retrieved chunk."""
        Answer(
            text="ok",
            citations=[Citation(doc_id="d.md", chunk_id="d.md::chunk_0")],
            retrieved_chunks=[
                ChunkRef(doc_id="d.md", chunk_id="d.md::chunk_0", score=0.9, text="x")
            ],
            metadata=self._meta(),
        )

    def test_empty_citations_with_retrieved_chunks_is_valid(self):
        """LLM produced text without citations — vacuously valid."""
        Answer(
            text="no citations here",
            citations=[],
            retrieved_chunks=[
                ChunkRef(doc_id="d.md", chunk_id="d.md::chunk_0", score=0.9, text="x")
            ],
            metadata=self._meta(),
        )

    def test_empty_citations_and_empty_chunks_is_valid(self):
        """Edge case: LLM short-circuited or eval-only Answer."""
        Answer(text="hi", metadata=self._meta())

    def test_citation_without_matching_chunk_rejected(self):
        """A Citation pointing at a chunk that wasn't in retrieved_chunks
        means something upstream produced a hallucinated reference. The
        invariant must reject this at construction time."""
        with pytest.raises(ValidationError, match="no such chunk is in retrieved_chunks"):
            Answer(
                text="see [1]",
                citations=[Citation(doc_id="ghost.md", chunk_id="ghost.md::chunk_0")],
                retrieved_chunks=[
                    ChunkRef(doc_id="real.md", chunk_id="real.md::chunk_0", score=0.9, text="x")
                ],
                metadata=self._meta(),
            )

    def test_citation_with_correct_doc_but_wrong_chunk_id_rejected(self):
        """Identity is (doc_id, chunk_id) — matching only on doc_id is
        not enough. Catches a regression where someone might use doc_id
        as the citation key and silently lose chunk-level precision."""
        with pytest.raises(ValidationError, match="no such chunk is in retrieved_chunks"):
            Answer(
                text="see [1]",
                citations=[Citation(doc_id="d.md", chunk_id="d.md::chunk_99")],
                retrieved_chunks=[
                    ChunkRef(doc_id="d.md", chunk_id="d.md::chunk_0", score=0.9, text="x")
                ],
                metadata=self._meta(),
            )

    def test_one_valid_one_invalid_still_rejects(self):
        """If any citation in the list is invalid, the whole Answer is
        rejected. Partial validity is not validity."""
        with pytest.raises(ValidationError, match="no such chunk is in retrieved_chunks"):
            Answer(
                text="see [1] and [2]",
                citations=[
                    Citation(doc_id="real.md", chunk_id="real.md::chunk_0"),
                    Citation(doc_id="ghost.md", chunk_id="ghost.md::chunk_0"),
                ],
                retrieved_chunks=[
                    ChunkRef(doc_id="real.md", chunk_id="real.md::chunk_0", score=0.9, text="x")
                ],
                metadata=self._meta(),
            )

    def test_citations_with_no_retrieved_chunks_rejected(self):
        """An Answer claiming citations against an empty retrieved_chunks
        list is incoherent — there's nothing to ground the citation in."""
        with pytest.raises(ValidationError, match="no such chunk is in retrieved_chunks"):
            Answer(
                text="see [1]",
                citations=[Citation(doc_id="d.md", chunk_id="d.md::chunk_0")],
                retrieved_chunks=[],
                metadata=self._meta(),
            )

    def test_validator_runs_on_model_validate_too(self):
        """The invariant must hold whether Answer is constructed directly
        or built from a dict via model_validate (the latter is what
        FastAPI will exercise on inbound JSON)."""
        with pytest.raises(ValidationError, match="no such chunk is in retrieved_chunks"):
            Answer.model_validate(
                {
                    "text": "see [1]",
                    "citations": [{"doc_id": "ghost.md", "chunk_id": "ghost.md::chunk_0"}],
                    "retrieved_chunks": [],
                    "metadata": {"model_name": "m", "latency_ms": 1.0},
                }
            )
