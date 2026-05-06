"""Type-aware query generator for Block 3B.2 training-data seeding.

Wraps Haiku 4.5 (validated in the mini-pilot, see
``evaluations/datasets/training/pilot/query_gen_validation.md``) with
one prompt template per question type. Given a chunk + question type,
returns a generated query grounded in the chunk and shaped to match
the type's answer pattern.

Public surface:

- ``TypeAwareQueryGenerator``: stateful generator wrapping an Anthropic
  client. Two methods — ``generate_for_chunk`` (in-corpus types) and
  ``generate_for_topic`` (off-corpus refusals).
- ``GeneratedQuery``: pydantic model with the query + provenance
  metadata. Same shape every entry, regardless of which path generated
  it, so the seeder threads them straight into the query pool.

Refusal handling: ``generate_for_topic`` is the off-corpus path
(generate a question on a topic that's NOT in HF Transformers docs).
The retrieval-failure refusal sub-type (pair a valid query with
mismatched chunks) is the seeder's job (Block 3B.2.e), not this
module's — it's not a generation task, it's a rewrite-and-pair task.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from docsense.finetuning.chunk_classifier import QuestionType

# Pydantic resolves ChunkRef at runtime for list[ChunkRef] field
# validation, so it can't live in the TYPE_CHECKING block (TC001).
from docsense.generation.types import ChunkRef  # noqa: TC001

if TYPE_CHECKING:
    import anthropic


_QUERY_GEN_PROMPT_VERSION = "v1"


# Anthropic pricing (USD per million input/output tokens). Update when
# we add or swap models. The generator computes per-call cost so
# callers don't have to know rates. Unknown models report cost=None
# rather than crashing — graceful for ad-hoc model experimentation.
_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
}


# Per-type question-shape instruction. Calibrated in the mini-pilot
# (validated 2026-05-06, see query_gen_validation.md). Each instruction
# is one sentence injected into ``_CHUNK_PROMPT`` below. REFUSAL is
# intentionally absent — refusals come from ``_TOPIC_PROMPT``, not from
# this template.
_TYPE_INSTRUCTIONS: dict[QuestionType, str] = {
    QuestionType.PROCEDURAL: "Be a 'how do I X' or 'how does X work' procedural question",
    QuestionType.COMPARISON: (
        "Be a 'what's the difference between X and Y' or "
        "'when should I use X vs Y' comparison question"
    ),
    QuestionType.BEST_PRACTICE: (
        "Be a 'what's the recommended way to X' or 'what's the best practice for X' question"
    ),
    QuestionType.POINTER: (
        "Be a 'where do I find X' or 'where do I start with X' pointer question"
    ),
}


_CHUNK_PROMPT = """You're helping build a training dataset for a documentation Q&A system over the HuggingFace Transformers documentation.

Given the following documentation chunk, generate ONE realistic question that a developer would ask whose answer is contained in (or directly inferable from) this chunk.

Constraints:
- {type_instruction}
- Be specific enough that this chunk's content is needed to answer (not a generic question)
- Phrased like a real developer's question — informal, not formal documentation language
- Do NOT quote the chunk verbatim
- 5 to 15 words
- Output JSON only, no preamble: {{"query": "your question here"}}

CHUNK:
{chunk_text}
SOURCE: {doc_id}"""


_TOPIC_PROMPT = """You're helping build a training dataset for a documentation Q&A system over the HuggingFace Transformers documentation.

Generate ONE realistic technical question on the topic "{topic}" that a developer might ask. The question should be a genuine question someone would search for, but it should NOT be answerable from the HuggingFace Transformers documentation.

Topics to AVOID (these ARE in the HF Transformers docs):
- Model loading, fine-tuning, tokenizers, attention mechanisms, training APIs
- The transformers Python library itself
- HuggingFace Hub, datasets library, accelerate library

The question should be:
- A genuine technical question (not trivia)
- Phrased like a real developer's question — informal
- 5 to 15 words
- Output JSON only, no preamble: {{"query": "your question here"}}

TOPIC: {topic}"""


_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _extract_json_block(text: str) -> str:
    """Strip markdown fences / surrounding prose, return JSON object body.

    Handles three cases the model produces:

    1. Clean JSON: ``{"query": "x"}`` → passthrough.
    2. Fenced JSON: ``\\`\\`\\`json\\n{...}\\n\\`\\`\\``` → strip fence.
    3. JSON in prose: ``Here you go:\\n{...}\\nThanks!`` → extract span.
    """
    text = text.strip()
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    obj_match = _JSON_OBJECT_RE.search(text)
    if obj_match:
        return obj_match.group(0)
    return text


def parse_query_response(text: str) -> str | None:
    """Parse the model's JSON response and return the query string.

    Returns ``None`` on any parse failure (invalid JSON, missing or
    non-string ``query`` field, empty after strip). Caller decides
    skip-vs-abort.
    """
    body = _extract_json_block(text)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    query = obj.get("query")
    if not isinstance(query, str):
        return None
    return query.strip() or None


class GeneratedQuery(BaseModel):
    """One generated query with provenance metadata.

    Same shape regardless of which path generated it. Refusal queries
    have empty ``seed_chunks`` and a non-empty ``seed_topic``;
    in-corpus queries have non-empty ``seed_chunks`` and
    ``seed_topic=None``. The seeder uses these provenance fields to
    audit how the query pool was constructed.
    """

    query: str = Field(min_length=1)
    question_type: QuestionType
    seed_chunks: list[ChunkRef] = Field(default_factory=list)
    seed_topic: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TypeAwareQueryGenerator:
    """Generates queries from chunks (or topic seeds) via Haiku 4.5.

    Stateless aside from the Anthropic client and model name. Caller
    constructs the client; the generator just uses it. Tests stub the
    client to verify shape without paying for real API calls.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        model: str = "claude-haiku-4-5",
    ) -> None:
        self.client = client
        self.model = model

    def generate_for_chunk(
        self,
        chunk: ChunkRef,
        question_type: QuestionType,
    ) -> GeneratedQuery:
        """Generate one in-corpus query for ``chunk`` targeting ``question_type``.

        Raises ``ValueError`` if ``question_type=REFUSAL`` (use
        ``generate_for_topic`` instead). Raises ``RuntimeError`` on
        parse failure so the caller can decide skip-vs-abort.
        """
        if question_type == QuestionType.REFUSAL:
            msg = (
                "generate_for_chunk does not support REFUSAL — refusals come "
                "from off-corpus topics (generate_for_topic) or by pairing "
                "valid queries with mismatched chunks (the seeder)."
            )
            raise ValueError(msg)

        prompt = _CHUNK_PROMPT.format(
            type_instruction=_TYPE_INSTRUCTIONS[question_type],
            chunk_text=chunk.text,
            doc_id=chunk.doc_id,
        )
        query, in_tokens, out_tokens = self._call_api(prompt)
        if query is None:
            msg = f"parse failure for chunk {chunk.doc_id}::{chunk.chunk_id}"
            raise RuntimeError(msg)

        return GeneratedQuery(
            query=query,
            question_type=question_type,
            seed_chunks=[chunk],
            seed_topic=None,
            metadata=self._build_metadata(in_tokens, out_tokens),
        )

    def generate_for_topic(self, topic: str) -> GeneratedQuery:
        """Generate one off-corpus refusal query for ``topic`` seed.

        Raises ``RuntimeError`` on parse failure.
        """
        prompt = _TOPIC_PROMPT.format(topic=topic)
        query, in_tokens, out_tokens = self._call_api(prompt)
        if query is None:
            msg = f"parse failure for topic seed {topic!r}"
            raise RuntimeError(msg)

        return GeneratedQuery(
            query=query,
            question_type=QuestionType.REFUSAL,
            seed_chunks=[],
            seed_topic=topic,
            metadata=self._build_metadata(in_tokens, out_tokens),
        )

    def _call_api(self, prompt: str) -> tuple[str | None, int, int]:
        """Send prompt; return (parsed_query | None, in_tokens, out_tokens)."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        raw = text_blocks[0] if text_blocks else ""
        return (
            parse_query_response(raw),
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    def _build_metadata(self, in_tokens: int, out_tokens: int) -> dict[str, Any]:
        """Compose the provenance metadata dict for a GeneratedQuery."""
        cost: float | None
        if self.model in _PRICING:
            in_rate, out_rate = _PRICING[self.model]
            cost = in_tokens * in_rate / 1_000_000 + out_tokens * out_rate / 1_000_000
        else:
            cost = None
        return {
            "gen_model": self.model,
            "prompt_version": _QUERY_GEN_PROMPT_VERSION,
            "captured_at": datetime.now(UTC).isoformat(),
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost_usd": cost,
        }
