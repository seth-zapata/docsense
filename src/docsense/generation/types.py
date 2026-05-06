"""Typed contracts for generation pipeline output.

These pydantic models define what an `Answer` *is* — independent of how
the LLM produced it. Once Phase 4 wraps the pipeline in FastAPI,
`Answer.model_json_schema()` becomes the OpenAPI response schema for free.

The models are deliberately small. `ChunkRef` captures what the LLM saw;
`Citation` captures what it cited; `Answer` ties everything together with
generation metadata for tracing/debug.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkRef(BaseModel):
    """A snapshot of one retrieved chunk that was passed to the LLM.

    Stored on `Answer.retrieved_chunks` so consumers can audit *exactly*
    what context the LLM had — without needing to re-run retrieval.
    """

    doc_id: str
    chunk_id: str
    score: float = Field(description="Score from the final retrieval stage (RRF or cross-encoder).")
    text: str


class Citation(BaseModel):
    """A reference from the answer text to one of the retrieved chunks.

    `chunk_id` and `doc_id` together identify the source. `quote` is an
    optional verbatim span from the chunk that the answer is grounding on
    — populated when the citation parser can extract one, omitted otherwise.
    """

    doc_id: str
    chunk_id: str
    quote: str | None = None


class GenerationMetadata(BaseModel):
    """Per-call metadata for tracing / debug / cost accounting.

    Latency is wall-clock around the model's generate() call only; it does
    not include retrieval, reranking, or context assembly. Token counts are
    optional because some callers (e.g., mocked tests) won't have them.
    """

    model_name: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Answer(BaseModel):
    """The full output of the generation pipeline.

    `text` is the user-facing answer. `citations` references chunks that
    grounded the answer; the citation parser produces these from the
    LLM's output. `retrieved_chunks` is the full set of chunks the LLM
    saw — kept on the answer so callers can audit context independently
    of retrieval re-runs. `metadata` is for tracing and cost accounting.
    """

    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[ChunkRef] = Field(default_factory=list)
    metadata: GenerationMetadata
