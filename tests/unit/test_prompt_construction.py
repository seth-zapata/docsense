"""Tests for PromptBuilder.

The headline test is a snapshot check against
``tests/snapshots/prompt_default.json`` — the messages structure for
fixed inputs must match the committed file. Accidental drift (stray
refactor drops the citation directive, system prompt subtly edited,
user template changes) breaks the snapshot loudly. Intentional changes
update the snapshot in the same commit and become reviewable.

Snapshot format note: messages are stored as JSON (not the
chat-template-rendered output) so the snapshot is model-agnostic. The
model-specific chat formatting is the tokenizer's job and lives in
Generator._run_inference; testing that in isolation would require
loading a real tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path

from docsense.generation.context import ContextAssembler
from docsense.generation.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_TEMPLATE,
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
    def test_default_messages_match_snapshot(self):
        """The committed snapshot is the source of truth for the default
        messages structure. Updating the system prompt or user template
        without updating the snapshot is a deliberate-action signal.
        """
        assembler = ContextAssembler(max_tokens=200, tokenize_fn=_word_tokenizer)
        context, _ = assembler.assemble(_sample_chunks())
        rendered = PromptBuilder().build(query="How do I install transformers?", context=context)

        snapshot_path = SNAPSHOT_DIR / "prompt_default.json"
        expected = json.loads(snapshot_path.read_text())
        assert rendered == expected, (
            f"Default messages drifted from snapshot.\n"
            f"If intentional, update {snapshot_path} in the same commit.\n"
            f"---EXPECTED---\n{expected}\n---GOT---\n{rendered}"
        )


class TestPromptBuilderInvariants:
    def test_build_returns_two_messages(self):
        """The default split is system + user. If we ever need
        multi-turn (e.g., few-shot examples), this changes deliberately."""
        rendered = PromptBuilder().build(query="q", context="c")
        assert len(rendered) == 2

    def test_build_returns_system_then_user_in_order(self):
        rendered = PromptBuilder().build(query="q", context="c")
        assert rendered[0]["role"] == "system"
        assert rendered[1]["role"] == "user"

    def test_system_prompt_appears_in_system_message(self):
        rendered = PromptBuilder().build(query="q", context="c")
        assert DEFAULT_SYSTEM_PROMPT in rendered[0]["content"]

    def test_query_appears_in_user_message(self):
        rendered = PromptBuilder().build(query="What is X?", context="c")
        assert "What is X?" in rendered[1]["content"]

    def test_context_appears_in_user_message(self):
        rendered = PromptBuilder().build(query="q", context="some context here")
        assert "some context here" in rendered[1]["content"]

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
        assert rendered[0]["content"] == "CUSTOM_SYSTEM"
        assert DEFAULT_SYSTEM_PROMPT not in rendered[0]["content"]

    def test_custom_user_template_replaces_default(self):
        builder = PromptBuilder(user_template="ONLY: {query} END")
        rendered = builder.build(query="hi", context="ignored")
        assert rendered[1]["content"] == "ONLY: hi END"

    def test_default_user_template_has_required_placeholders(self):
        """If the default user template stops referencing one of the two
        format slots, build() will silently produce a broken user message
        rather than raising. Pin the contract."""
        assert "{context}" in DEFAULT_USER_TEMPLATE
        assert "{query}" in DEFAULT_USER_TEMPLATE
