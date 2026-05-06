"""Tests for ContextAssembler.

Verifies budget enforcement, determinism, edge cases, and that
``included_chunks`` accurately reflects what made it into the context
string. Token counting is injected via ``tokenize_fn`` so the tests are
deterministic and don't depend on a real tokenizer.
"""

from __future__ import annotations

import pytest

from docsense.generation.context import ContextAssembler
from docsense.generation.types import ChunkRef


def _ref(doc_id: str, text: str, score: float = 0.5) -> ChunkRef:
    return ChunkRef(doc_id=doc_id, chunk_id=f"{doc_id}::chunk_0", score=score, text=text)


def _word_tokenizer(text: str) -> int:
    """Deterministic word-count tokenizer for tests."""
    return len(text.split())


class TestContextAssemblerBudget:
    def test_zero_chunks_returns_empty(self):
        assembler = ContextAssembler(max_tokens=100, tokenize_fn=_word_tokenizer)
        text, included = assembler.assemble([])
        assert text == ""
        assert included == []

    def test_all_chunks_fit(self):
        assembler = ContextAssembler(max_tokens=200, tokenize_fn=_word_tokenizer)
        chunks = [
            _ref("a.md", "alpha beta gamma"),
            _ref("b.md", "delta epsilon"),
        ]
        text, included = assembler.assemble(chunks)
        assert len(included) == 2
        assert "alpha beta gamma" in text
        assert "delta epsilon" in text
        assert "[1]" in text and "[2]" in text

    def test_some_chunks_dropped_when_budget_tight(self):
        """With a small budget, only the prefix that fits gets included.
        Later chunks are silently dropped, not truncated."""
        assembler = ContextAssembler(max_tokens=10, tokenize_fn=_word_tokenizer)
        # Each chunk's formatted form is ~6 words: "[1]" "(source:" "x.md)" "<word> <word> <word>"
        chunks = [
            _ref("a.md", "alpha beta gamma"),
            _ref("b.md", "delta epsilon zeta"),
            _ref("c.md", "eta theta iota"),
        ]
        text, included = assembler.assemble(chunks)
        # Only the first chunk fits within 10 tokens
        assert len(included) == 1
        assert included[0].doc_id == "a.md"
        assert "alpha beta gamma" in text
        assert "delta" not in text

    def test_oversized_first_chunk_is_dropped(self):
        """A single chunk that exceeds the entire budget is dropped, not
        truncated. Caller gets an empty context rather than partial text."""
        assembler = ContextAssembler(max_tokens=5, tokenize_fn=_word_tokenizer)
        # The formatting prefix alone is ~5 words; chunk text pushes it past
        chunks = [_ref("a.md", "alpha beta gamma delta epsilon zeta eta theta")]
        text, included = assembler.assemble(chunks)
        assert text == ""
        assert included == []

    def test_zero_max_tokens_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ContextAssembler(max_tokens=0, tokenize_fn=_word_tokenizer)

    def test_negative_max_tokens_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ContextAssembler(max_tokens=-10, tokenize_fn=_word_tokenizer)


class TestContextAssemblerFormat:
    def test_chunks_numbered_starting_at_one(self):
        assembler = ContextAssembler(max_tokens=200, tokenize_fn=_word_tokenizer)
        chunks = [
            _ref("a.md", "first"),
            _ref("b.md", "second"),
            _ref("c.md", "third"),
        ]
        text, _ = assembler.assemble(chunks)
        assert "[1]" in text
        assert "[2]" in text
        assert "[3]" in text
        # Order is preserved
        assert text.index("[1]") < text.index("[2]") < text.index("[3]")

    def test_source_attribution_present(self):
        assembler = ContextAssembler(max_tokens=200, tokenize_fn=_word_tokenizer)
        chunks = [_ref("docs/installation.md", "pip install transformers")]
        text, _ = assembler.assemble(chunks)
        assert "source: docs/installation.md" in text

    def test_chunks_separated_by_blank_line(self):
        assembler = ContextAssembler(max_tokens=200, tokenize_fn=_word_tokenizer)
        chunks = [_ref("a.md", "alpha"), _ref("b.md", "beta")]
        text, _ = assembler.assemble(chunks)
        assert "\n\n" in text


class TestContextAssemblerDeterminism:
    def test_repeated_calls_produce_identical_output(self):
        """Determinism: same inputs → byte-for-byte same output. Catches
        accidental ordering nondeterminism (dict iteration drift, set
        construction, etc.)."""
        assembler = ContextAssembler(max_tokens=200, tokenize_fn=_word_tokenizer)
        chunks = [
            _ref("a.md", "alpha"),
            _ref("b.md", "beta"),
            _ref("c.md", "gamma"),
        ]
        first_text, first_included = assembler.assemble(chunks)
        second_text, second_included = assembler.assemble(chunks)
        assert first_text == second_text
        assert first_included == second_included


class TestContextAssemblerDefaultTokenizer:
    def test_default_char_quarter_heuristic(self):
        """Default tokenize_fn is char/4. Verify behavior without an
        explicit injection so production callers using the default get
        sane budgeting."""
        # No tokenize_fn passed — uses the default char/4 heuristic
        assembler = ContextAssembler(max_tokens=100)
        # ~400 chars budget at char/4 = 100 tokens
        chunks = [_ref("a.md", "x" * 100)]  # ~25 tokens for the text alone
        text, included = assembler.assemble(chunks)
        assert len(included) == 1
        assert "x" * 100 in text


class TestContextAssemblerIncludedChunks:
    def test_included_matches_what_appears_in_text(self):
        """The included list must reflect exactly which chunks made it
        into the context string, with the same order. This is what
        callers use to populate Answer.retrieved_chunks for auditing."""
        assembler = ContextAssembler(max_tokens=12, tokenize_fn=_word_tokenizer)
        chunks = [
            _ref("a.md", "alpha"),
            _ref("b.md", "beta"),
            _ref("c.md", "gamma"),
            _ref("d.md", "delta"),
        ]
        text, included = assembler.assemble(chunks)

        # Every included chunk's text appears in the assembled context
        for chunk in included:
            assert chunk.text in text

        # Order preserved
        included_doc_ids = [c.doc_id for c in included]
        original_doc_ids = [c.doc_id for c in chunks[: len(included)]]
        assert included_doc_ids == original_doc_ids


class TestContextAssemblerBudgetIsTokenizerAgnostic:
    """Block D.2: pin the contract that ContextAssembler respects its
    configured budget *regardless of which tokenizer is plugged in*.

    The assembler's correctness shouldn't depend on a specific tokenizer's
    behavior — only on the contract that ``tokenize_fn(text)`` returns a
    monotonically-increasing count of "tokens" for that text. Production
    will pass the LLM's actual tokenizer; tests use synthetic counters.
    These tests exercise three different tokenizers and assert the same
    invariant for each: assembled context's token count never exceeds
    the configured budget.
    """

    @pytest.mark.parametrize(
        ("name", "tokenize_fn"),
        [
            ("word_count", lambda s: len(s.split())),
            ("char_div_3", lambda s: len(s) // 3),
            ("char_div_5", lambda s: max(len(s) // 5, 1)),
        ],
    )
    def test_assembled_context_never_exceeds_budget(self, name, tokenize_fn):
        """Across multiple tokenizers, the assembled context's token count
        is always within the configured budget. Three calls per tokenizer
        with increasing budget sizes pin behavior across the typical
        range a real RAG system would use."""
        chunks = [
            _ref(f"d{i}.md", "alpha beta gamma delta epsilon zeta eta theta iota kappa")
            for i in range(8)
        ]
        for max_tokens in (5, 50, 200):
            assembler = ContextAssembler(max_tokens=max_tokens, tokenize_fn=tokenize_fn)
            text, _ = assembler.assemble(chunks)
            assert tokenize_fn(text) <= max_tokens, (
                f"Budget violation with tokenizer={name}, max_tokens={max_tokens}: "
                f"assembled context has {tokenize_fn(text)} tokens"
            )

    def test_default_tokenizer_also_respects_budget(self):
        """The production-default tokenizer (char/4 heuristic) must
        also respect the budget — verify directly without going through
        parameterization since we want the exact default path."""
        chunks = [_ref(f"d{i}.md", "x" * 200) for i in range(10)]
        for max_tokens in (10, 100, 500):
            # Default tokenize_fn (char/4)
            assembler = ContextAssembler(max_tokens=max_tokens)
            text, _ = assembler.assemble(chunks)
            # Use the same heuristic the assembler uses internally for the assertion
            assert (len(text) // 4) <= max_tokens, (
                f"Default char/4 budget violation: max_tokens={max_tokens}, "
                f"assembled context has {len(text) // 4} tokens"
            )
