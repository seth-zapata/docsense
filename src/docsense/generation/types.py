"""Typed contracts for generation pipeline output.

These pydantic models define what an `Answer` *is* — independent of how
the LLM produced it. Once Phase 4 wraps the pipeline in FastAPI,
`Answer.model_json_schema()` becomes the OpenAPI response schema for free.

The models are deliberately small. `ChunkRef` captures what the LLM saw;
`Citation` captures what it cited; `Answer` ties everything together with
generation metadata for tracing/debug.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


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

    ``adapter_path`` is set when a PEFT/LoRA adapter is loaded on top of
    the base model. Useful for the Phase 3 eval comparisons where we
    want every Answer to record exactly which trained checkpoint
    produced it (base vs adapter v1 vs adapter v1.1, etc.).
    """

    model_name: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    adapter_path: str | None = None


class Answer(BaseModel):
    """The full output of the generation pipeline.

    `text` is the user-facing answer. `citations` references chunks that
    grounded the answer; the citation parser produces these from the
    LLM's output. `retrieved_chunks` is the full set of chunks the LLM
    saw — kept on the answer so callers can audit context independently
    of retrieval re-runs. `metadata` is for tracing and cost accounting.

    **Citation-preservation invariant:** every entry in `citations` must
    identify a chunk that's also in `retrieved_chunks`. The pydantic
    validator below enforces this at construction; if it fires, something
    upstream produced a Citation pointing at a chunk the LLM didn't see —
    typically a bug in citation parsing or a hallucinated index. Failing
    loudly here is preferable to letting an unverifiable Answer reach
    downstream consumers.
    """

    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[ChunkRef] = Field(default_factory=list)
    metadata: GenerationMetadata

    @model_validator(mode="after")
    def _validate_citations_preserved(self) -> Answer:
        chunk_keys = {(c.doc_id, c.chunk_id) for c in self.retrieved_chunks}
        for cit in self.citations:
            if (cit.doc_id, cit.chunk_id) not in chunk_keys:
                msg = (
                    f"Citation references chunk_id={cit.chunk_id!r} "
                    f"(doc_id={cit.doc_id!r}) but no such chunk is in "
                    f"retrieved_chunks. Every citation must point at a chunk "
                    f"the LLM actually saw."
                )
                raise ValueError(msg)
        return self
