"""Tests for PromptBuilder.

The headline test is a snapshot check against
``tests/snapshots/prompt_default.txt`` — the rendered prompt for fixed
inputs must match the committed file byte-for-byte. Accidental drift
(stray refactor drops the citation directive, system prompt subtly
edited) breaks the snapshot loudly. Intentional changes update the
snapshot in the same commit and become reviewable.
"""

from __future__ import annotations

from pathlib import Path

from docsense.generation.context import ContextAssembler
from docsense.generation.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPLATE,
    PromptBuilder,
)
from docsense.generation.types import ChunkRef

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "snapshots"


def _word_tokenizer(text: str) -> int:
    return len(text.split())


def _sample_chunks() -> list[ChunkRef]:
    return [
        ChunkRef(
            doc_id="installation.md",
            chunk_id="installation.md::chunk_0",
            score=0.91,
            text="To install transformers run: pip install transformers",
        ),
        ChunkRef(
            doc_id="quickstart.md",
            chunk_id="quickstart.md::chunk_2",
            score=0.78,
            text="Use AutoModel.from_pretrained() to load a pretrained model.",
        ),
    ]


class TestPromptSnapshot:
    def test_default_prompt_matches_snapshot(self):
        """The committed snapshot is the source of truth for the default
        prompt's exact bytes. Updating the system prompt or template
        without updating the snapshot file is a deliberate-action signal.
        """
        assembler = ContextAssembler(max_tokens=200, tokenize_fn=_word_tokenizer)
        context, _ = assembler.assemble(_sample_chunks())
        rendered = PromptBuilder().build(query="How do I install transformers?", context=context)

        snapshot_path = SNAPSHOT_DIR / "prompt_default.txt"
        # Strip ONE trailing newline that the project's end-of-file-fixer
        # pre-commit hook adds; that newline is a file-format convention,
        # not part of the prompt the LLM actually sees.
        expected = snapshot_path.read_text().rstrip("\n")
        assert rendered == expected, (
            f"Default prompt drifted from snapshot.\n"
            f"If intentional, update {snapshot_path} in the same commit.\n"
            f"---EXPECTED---\n{expected}\n---GOT---\n{rendered}"
        )


class TestPromptBuilderInvariants:
    def test_system_prompt_appears_in_output(self):
        rendered = PromptBuilder().build(query="q", context="c")
        assert DEFAULT_SYSTEM_PROMPT in rendered

    def test_query_appears_in_output(self):
        rendered = PromptBuilder().build(query="What is X?", context="c")
        assert "What is X?" in rendered

    def test_context_appears_in_output(self):
        rendered = PromptBuilder().build(query="q", context="some context here")
        assert "some context here" in rendered

    def test_citation_directive_present_in_default_system_prompt(self):
        """Pin a regression-prone invariant: the default system prompt
        must instruct the LLM to cite using [N]. If this breaks, citations
        disappear from generations even though the parser is intact."""
        assert "[N]" in DEFAULT_SYSTEM_PROMPT
        assert "cite" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_refusal_directive_present_in_default_system_prompt(self):
        """Same idea for the refusal directive. Without this, the LLM
        will tend to confabulate when context is irrelevant."""
        assert (
            "I don't have enough context" in DEFAULT_SYSTEM_PROMPT
            or "do not have enough context" in DEFAULT_SYSTEM_PROMPT.lower()
        )

    def test_custom_system_prompt_replaces_default(self):
        builder = PromptBuilder(system_prompt="CUSTOM_SYSTEM")
        rendered = builder.build(query="q", context="c")
        assert "CUSTOM_SYSTEM" in rendered
        assert DEFAULT_SYSTEM_PROMPT not in rendered

    def test_custom_template_replaces_default(self):
        builder = PromptBuilder(template="ONLY: {query} END")
        rendered = builder.build(query="hi", context="ignored")
        assert rendered == "ONLY: hi END"

    def test_default_template_has_required_placeholders(self):
        """If the default template stops referencing one of the three
        format slots, build() will silently produce a broken prompt
        rather than raising. Pin the contract."""
        assert "{system_prompt}" in DEFAULT_TEMPLATE
        assert "{context}" in DEFAULT_TEMPLATE
        assert "{query}" in DEFAULT_TEMPLATE
