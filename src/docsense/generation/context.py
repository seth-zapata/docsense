"""Token-budget-aware context assembly for the generation pipeline.

Takes a list of `ChunkRef` (top-k from retrieval, ordered by relevance)
and produces a prompt-ready context string that fits within a configured
token budget. Greedy fill: chunks are added in order until the next
chunk would exceed the budget, at which point assembly stops and any
remaining chunks are dropped.

Design notes:
- A chunk that *individually* exceeds the budget is dropped, not
  truncated. Partial chunks produce broken text, which the LLM tends
  to either confidently misinterpret or hedge around — both worse than
  having one less source. Document size limits should be enforced at
  chunking time, not here.
- Token counting is injectable via ``tokenize_fn`` so tests can use a
  deterministic counter and production can pass the actual LLM's
  tokenizer. The default heuristic (chars / 4) is "good enough" for
  budget sizing on English prose; off by ~10-20% on code-heavy text
  but that's within our headroom (default budget 3500 vs Mistral 32k
  context window).
- Output format is numbered with source attribution:
      [1] (source: docs/foo.md)
      {chunk text}

      [2] (source: docs/bar.md)
      {chunk text}
  This pairs with the prompt template's instruction to "cite using
  [N] notation"; the citation parser maps ``[N]`` back to the Nth
  included chunk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from docsense.generation.types import ChunkRef


def _default_tokenize(text: str) -> int:
    """Char/4 heuristic — a reasonable approximation for English prose
    when we don't want to pay a tokenizer-load cost."""
    return len(text) // 4


def _format_chunk(rank: int, chunk: ChunkRef) -> str:
    return f"[{rank}] (source: {chunk.doc_id})\n{chunk.text}"


class ContextAssembler:
    """Greedy chunk-to-context assembler with a configurable token budget."""

    def __init__(
        self,
        max_tokens: int,
        tokenize_fn: Callable[[str], int] | None = None,
    ) -> None:
        if max_tokens <= 0:
            msg = f"max_tokens must be positive, got {max_tokens}"
            raise ValueError(msg)
        self.max_tokens = max_tokens
        self._tokenize: Callable[[str], int] = tokenize_fn or _default_tokenize

    def assemble(self, chunks: list[ChunkRef]) -> tuple[str, list[ChunkRef]]:
        """Assemble chunks into a single context string under the budget.

        Returns ``(context_text, included_chunks)``. ``included_chunks``
        is the prefix of ``chunks`` that fit; downstream code (e.g.,
        ``Answer.retrieved_chunks``) should use this rather than the
        original list so the audit trail reflects what the LLM actually
        saw.
        """
        if not chunks:
            return "", []

        included: list[ChunkRef] = []
        formatted_parts: list[str] = []
        used_tokens = 0
        # The "\n\n" separator costs tokens too; account for it once per
        # chunk-after-the-first.
        separator_tokens = self._tokenize("\n\n")

        for chunk in chunks:
            rank = len(included) + 1
            formatted = _format_chunk(rank, chunk)
            chunk_tokens = self._tokenize(formatted)
            sep_cost = separator_tokens if included else 0

            if used_tokens + sep_cost + chunk_tokens > self.max_tokens:
                # Adding this chunk would exceed budget — stop assembling.
                # We drop this chunk and all subsequent ones rather than
                # truncating mid-text.
                break

            included.append(chunk)
            formatted_parts.append(formatted)
            used_tokens += sep_cost + chunk_tokens

        return "\n\n".join(formatted_parts), included
