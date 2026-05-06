"""Tests for the Generator wrapper.

Covers two surfaces:

1. ``parse_citations()`` — the standalone helper that extracts ``[N]``
   notation from generated text and resolves it to retrieved chunks.
2. ``Generator.generate()`` end-to-end with ``_run_inference`` stubbed.
   The actual HuggingFace model is never loaded; we verify the wiring
   that turns inference output + retrieved chunks into a typed
   ``Answer``.

Real LLM inference is exercised in Phase 3 / pre-Phase-3 LLM-judge
evals — not here. These are contract tests for the wrapper.
"""

from __future__ import annotations

import pytest

from docsense.config import GenerationConfig
from docsense.generation.generator import Generator, parse_citations
from docsense.generation.types import Answer, ChunkRef


def _ref(idx: int, doc_id: str | None = None) -> ChunkRef:
    doc = doc_id or f"doc{idx}.md"
    return ChunkRef(doc_id=doc, chunk_id=f"{doc}::chunk_0", score=0.9, text=f"text {idx}")


class TestParseCitations:
    def test_no_citations_returns_empty(self):
        chunks = [_ref(1), _ref(2)]
        assert parse_citations("This text has no citation markers.", chunks) == []

    def test_single_citation(self):
        chunks = [_ref(1, "foo.md"), _ref(2, "bar.md")]
        cits = parse_citations("Per [1] you should do X.", chunks)
        assert len(cits) == 1
        assert cits[0].doc_id == "foo.md"
        assert cits[0].chunk_id == "foo.md::chunk_0"

    def test_multiple_citations_in_order(self):
        chunks = [_ref(1, "a.md"), _ref(2, "b.md"), _ref(3, "c.md")]
        cits = parse_citations("First [2], then [1], finally [3].", chunks)
        assert [c.doc_id for c in cits] == ["b.md", "a.md", "c.md"]

    def test_duplicate_citation_deduped(self):
        """If the LLM cites the same chunk twice, we produce one
        Citation, not two. Same doc+chunk identity = one citation."""
        chunks = [_ref(1, "foo.md"), _ref(2, "bar.md")]
        cits = parse_citations("Both [1] and [1] support this; [2] adds nuance.", chunks)
        assert len(cits) == 2
        assert {c.doc_id for c in cits} == {"foo.md", "bar.md"}

    def test_out_of_range_citation_silently_dropped(self):
        """The LLM might hallucinate `[5]` when only 3 chunks exist.
        Skip silently rather than raising — the alternative is breaking
        a generation that's mostly correct."""
        chunks = [_ref(1), _ref(2)]
        cits = parse_citations("See [3] for details.", chunks)
        assert cits == []

    def test_zero_index_silently_dropped(self):
        """`[0]` is invalid since citations are 1-indexed."""
        chunks = [_ref(1), _ref(2)]
        cits = parse_citations("See [0].", chunks)
        assert cits == []

    def test_mixed_valid_and_invalid(self):
        chunks = [_ref(1, "ok.md"), _ref(2, "also-ok.md")]
        cits = parse_citations("Use [1] but not [99] — also see [2].", chunks)
        assert [c.doc_id for c in cits] == ["ok.md", "also-ok.md"]

    def test_does_not_match_log_timestamps_or_other_brackets(self):
        """`[2026-05-06 ...]` shouldn't be parsed as `[20]` etc. Anchor
        the digits-only requirement."""
        chunks = [_ref(1)]
        cits = parse_citations("Logged at [2026-05-06 12:00] — see [1].", chunks)
        assert len(cits) == 1
        assert cits[0].doc_id == "doc1.md"


class TestGeneratorGenerate:
    """Exercise the generate() flow with _run_inference stubbed.

    Subclassing Generator and overriding _run_inference is the supported
    test pattern (per the docstring on _run_inference). It avoids any
    HuggingFace import or model load while still exercising the real
    citation parser and Answer construction.
    """

    def _make_generator(self, canned_text: str, raw_meta: dict | None = None) -> Generator:
        config = GenerationConfig(
            model_name="test-model", max_new_tokens=128, temperature=0.0, device="cpu"
        )

        class _StubGenerator(Generator):
            def _run_inference(self, prompt: str) -> tuple[str, dict]:
                meta = raw_meta or {
                    "latency_ms": 42.0,
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                }
                return canned_text, meta

        return _StubGenerator(config)

    def test_generate_returns_answer_with_text(self):
        gen = self._make_generator("Per [1], install with pip.")
        chunks = [_ref(1, "install.md")]
        answer = gen.generate("how to install?", chunks)
        assert isinstance(answer, Answer)
        assert answer.text == "Per [1], install with pip."

    def test_generate_strips_whitespace_from_text(self):
        gen = self._make_generator("\n  hello world  \n")
        answer = gen.generate("?", [])
        assert answer.text == "hello world"

    def test_generate_populates_citations_from_text(self):
        gen = self._make_generator("Use [1] for setup, then [2] for examples.")
        chunks = [_ref(1, "setup.md"), _ref(2, "examples.md")]
        answer = gen.generate("how do I get started?", chunks)
        assert len(answer.citations) == 2
        assert {c.doc_id for c in answer.citations} == {"setup.md", "examples.md"}

    def test_generate_returns_empty_citations_when_text_has_none(self):
        gen = self._make_generator("Plain answer with no citations.")
        chunks = [_ref(1)]
        answer = gen.generate("?", chunks)
        assert answer.citations == []

    def test_generate_attaches_retrieved_chunks_to_answer(self):
        """The chunks passed in are preserved on the Answer for auditing —
        consumers can inspect what the LLM actually saw."""
        gen = self._make_generator("Some answer.")
        chunks = [_ref(1, "a.md"), _ref(2, "b.md")]
        answer = gen.generate("?", chunks)
        assert len(answer.retrieved_chunks) == 2
        assert {c.doc_id for c in answer.retrieved_chunks} == {"a.md", "b.md"}

    def test_generate_populates_metadata(self):
        gen = self._make_generator(
            "ok",
            raw_meta={"latency_ms": 123.4, "prompt_tokens": 256, "completion_tokens": 32},
        )
        answer = gen.generate("?", [])
        assert answer.metadata.model_name == "test-model"
        assert answer.metadata.latency_ms == pytest.approx(123.4)
        assert answer.metadata.prompt_tokens == 256
        assert answer.metadata.completion_tokens == 32

    def test_generate_handles_missing_token_counts_in_metadata(self):
        """If _run_inference omits prompt/completion token counts (e.g.,
        a tokenizer-free mock), metadata still constructs with None."""
        gen = self._make_generator("ok", raw_meta={"latency_ms": 50.0})
        answer = gen.generate("?", [])
        assert answer.metadata.prompt_tokens is None
        assert answer.metadata.completion_tokens is None

    def test_retrieved_chunks_is_a_copy_not_the_input_list(self):
        """The Answer.retrieved_chunks should be a separate list so
        caller mutations to the original don't reach into the Answer."""
        gen = self._make_generator("ok")
        chunks = [_ref(1, "a.md")]
        answer = gen.generate("?", chunks)
        chunks.clear()  # caller mutates after generate()
        assert len(answer.retrieved_chunks) == 1
        assert answer.retrieved_chunks[0].doc_id == "a.md"


class TestGeneratorLazyLoad:
    def test_construction_does_not_load_model(self):
        config = GenerationConfig(model_name="dummy", device="cpu")
        gen = Generator(config)
        assert gen._model is None
        assert gen._tokenizer is None
