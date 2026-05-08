"""Tests for the Generator wrapper.

Covers three surfaces:

1. ``parse_citations()`` — the standalone helper that extracts ``[N]``
   notation from generated text and resolves it to retrieved chunks.
2. ``Generator.generate()`` end-to-end with ``_run_inference`` stubbed
   via subclassing. The actual HuggingFace model is never loaded; we
   verify the wiring that turns inference output + retrieved chunks
   into a typed ``Answer``.
3. ``Generator.generate()`` contract via ``patch.object`` mocking —
   different test technique from #2 (subclass-based stub). Verifies
   call shape: ``_run_inference`` is invoked exactly once per
   ``generate()``, with the prompt argument, and its return value
   drives the constructed ``Answer``.

Real LLM inference is exercised in Phase 3 / pre-Phase-3 LLM-judge
evals — not here. These are contract tests for the wrapper.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

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
            def _run_inference(self, messages: list[dict[str, str]]) -> tuple[str, dict]:
                meta = raw_meta or {
                    "latency_ms": 42.0,
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                }
                return canned_text, meta

        return _StubGenerator(config)

    @staticmethod
    def _msgs(query: str = "?") -> list[dict[str, str]]:
        """Minimal valid messages list for tests. Content doesn't matter
        because _run_inference is stubbed; we just need the right type."""
        return [{"role": "user", "content": query}]

    def test_generate_returns_answer_with_text(self):
        gen = self._make_generator("Per [1], install with pip.")
        chunks = [_ref(1, "install.md")]
        answer = gen.generate(self._msgs("how to install?"), chunks)
        assert isinstance(answer, Answer)
        assert answer.text == "Per [1], install with pip."

    def test_generate_strips_whitespace_from_text(self):
        gen = self._make_generator("\n  hello world  \n")
        answer = gen.generate(self._msgs(), [])
        assert answer.text == "hello world"

    def test_generate_populates_citations_from_text(self):
        gen = self._make_generator("Use [1] for setup, then [2] for examples.")
        chunks = [_ref(1, "setup.md"), _ref(2, "examples.md")]
        answer = gen.generate(self._msgs("how do I get started?"), chunks)
        assert len(answer.citations) == 2
        assert {c.doc_id for c in answer.citations} == {"setup.md", "examples.md"}

    def test_generate_returns_empty_citations_when_text_has_none(self):
        gen = self._make_generator("Plain answer with no citations.")
        chunks = [_ref(1)]
        answer = gen.generate(self._msgs(), chunks)
        assert answer.citations == []

    def test_generate_attaches_retrieved_chunks_to_answer(self):
        """The chunks passed in are preserved on the Answer for auditing —
        consumers can inspect what the LLM actually saw."""
        gen = self._make_generator("Some answer.")
        chunks = [_ref(1, "a.md"), _ref(2, "b.md")]
        answer = gen.generate(self._msgs(), chunks)
        assert len(answer.retrieved_chunks) == 2
        assert {c.doc_id for c in answer.retrieved_chunks} == {"a.md", "b.md"}

    def test_generate_populates_metadata(self):
        gen = self._make_generator(
            "ok",
            raw_meta={"latency_ms": 123.4, "prompt_tokens": 256, "completion_tokens": 32},
        )
        answer = gen.generate(self._msgs(), [])
        assert answer.metadata.model_name == "test-model"
        assert answer.metadata.latency_ms == pytest.approx(123.4)
        assert answer.metadata.prompt_tokens == 256
        assert answer.metadata.completion_tokens == 32

    def test_generate_handles_missing_token_counts_in_metadata(self):
        """If _run_inference omits prompt/completion token counts (e.g.,
        a tokenizer-free mock), metadata still constructs with None."""
        gen = self._make_generator("ok", raw_meta={"latency_ms": 50.0})
        answer = gen.generate(self._msgs(), [])
        assert answer.metadata.prompt_tokens is None
        assert answer.metadata.completion_tokens is None

    def test_retrieved_chunks_is_a_copy_not_the_input_list(self):
        """The Answer.retrieved_chunks should be a separate list so
        caller mutations to the original don't reach into the Answer."""
        gen = self._make_generator("ok")
        chunks = [_ref(1, "a.md")]
        answer = gen.generate(self._msgs(), chunks)
        chunks.clear()  # caller mutates after generate()
        assert len(answer.retrieved_chunks) == 1
        assert answer.retrieved_chunks[0].doc_id == "a.md"


class TestGeneratorLazyLoad:
    def test_construction_does_not_load_model(self):
        config = GenerationConfig(model_name="dummy", device="cpu")
        gen = Generator(config)
        assert gen._model is None
        assert gen._tokenizer is None


class TestGeneratorQuantization:
    """Block 1A.3: pin the wiring of the NF4 4-bit quantization config.

    These tests verify the BitsAndBytesConfig that *would* be passed to
    AutoModelForCausalLM.from_pretrained, without actually loading a
    model. The full integration (does the model load, does it generate
    correctly) is exercised by the manual smoke script — pre-PR-CI
    territory because it requires a real GPU + ~5 GB download.
    """

    def test_4bit_config_uses_nf4_with_double_quant(self):
        """NF4 (NormalFloat-4) is the recommended quant type for LLMs;
        double quantization saves a further ~0.5 bit/parameter at
        negligible quality cost. Both pinned here so a well-meaning
        refactor doesn't silently switch to FP4 or drop double-quant."""
        bnb_config = Generator._build_4bit_config()

        # bitsandbytes version variations expose the config slightly
        # differently; check the public attributes that have been stable
        # across recent releases.
        assert bnb_config.load_in_4bit is True
        assert bnb_config.bnb_4bit_quant_type == "nf4"
        assert bnb_config.bnb_4bit_use_double_quant is True

    def test_use_4bit_quantization_default_true(self):
        """Default flips to True so the typical 12-GB-VRAM user gets
        working behavior out of the box; CPU-only / no-bitsandbytes
        users opt out via use_4bit_quantization=False."""
        config = GenerationConfig()
        assert config.use_4bit_quantization is True

    def test_disabling_4bit_skips_quantization_config(self):
        """When use_4bit_quantization=False, no BitsAndBytesConfig
        should reach the model loader — verified by patching the
        AutoModelForCausalLM.from_pretrained call and inspecting kwargs."""
        config = GenerationConfig(model_name="dummy", use_4bit_quantization=False)
        gen = Generator(config)

        with patch("transformers.AutoModelForCausalLM") as mock_cls:
            mock_cls.from_pretrained.return_value = MagicMock()
            _ = gen.model

        kwargs = mock_cls.from_pretrained.call_args.kwargs
        assert "quantization_config" not in kwargs

    def test_enabling_4bit_passes_quantization_config(self):
        """Mirror of the above: when the flag is on, the kwargs include
        a BitsAndBytesConfig."""
        config = GenerationConfig(model_name="dummy", use_4bit_quantization=True)
        gen = Generator(config)

        with patch("transformers.AutoModelForCausalLM") as mock_cls:
            mock_cls.from_pretrained.return_value = MagicMock()
            _ = gen.model

        kwargs = mock_cls.from_pretrained.call_args.kwargs
        assert "quantization_config" in kwargs
        # Sanity check: the config carries the NF4 marker
        assert kwargs["quantization_config"].bnb_4bit_quant_type == "nf4"


class TestGeneratorAdapterLoading:
    """Block 3C.4: PEFT/LoRA adapter is loaded on top of the base when
    ``GenerationConfig.adapter_path`` is set. Tests use mocks because
    real PEFT loading requires a trained adapter on disk + GPU."""

    def test_no_adapter_when_path_unset(self):
        """Default ``adapter_path=None`` → PEFT is never imported and
        the base model is returned unwrapped. Pre-Phase-3 baseline path."""
        config = GenerationConfig(model_name="dummy", use_4bit_quantization=False)
        gen = Generator(config)

        with patch("transformers.AutoModelForCausalLM") as mock_cls:
            mock_base = MagicMock(name="base_model")
            mock_cls.from_pretrained.return_value = mock_base
            # If PeftModel.from_pretrained gets called, the patch will fail
            # because we don't import peft at all in this branch.
            result = gen.model

        assert result is mock_base

    def test_adapter_loaded_when_path_set(self, tmp_path):
        """When ``adapter_path`` is set, PEFT is invoked to wrap the base."""
        # Pre-import peft so the patch context doesn't trigger a fresh
        # peft import (which transitively touches transformers internals
        # and was observed to undo `patch("transformers.AutoModelForCausalLM")`
        # when both patches are entered together).
        import peft  # noqa: F401

        config = GenerationConfig(
            model_name="dummy",
            use_4bit_quantization=False,
            adapter_path=tmp_path / "fake-adapter",
        )
        gen = Generator(config)

        mock_base = MagicMock(name="base_model")
        mock_wrapped = MagicMock(name="peft_model")

        # Nested rather than `with A, B:` — combining the patches via
        # parenthesized syntax was observed to leave the transformers
        # patch un-applied (likely a Python/mock interaction with how
        # the peft import resolves). Nesting is robust; the SIM117
        # combine-with hint is intentionally suppressed.
        with patch("transformers.AutoModelForCausalLM") as mock_cls:  # noqa: SIM117
            with patch("peft.PeftModel") as mock_peft:
                mock_cls.from_pretrained.return_value = mock_base
                mock_peft.from_pretrained.return_value = mock_wrapped
                result = gen.model

        # PEFT was called with the base + adapter path (as string).
        mock_peft.from_pretrained.assert_called_once()
        call_args = mock_peft.from_pretrained.call_args
        assert call_args.args[0] is mock_base
        assert call_args.args[1] == str(tmp_path / "fake-adapter")
        # The wrapped PEFT model is what Generator returns.
        assert result is mock_wrapped

    def test_adapter_path_default_is_none(self):
        """Pin: pre-Phase-3 baseline runs (no adapter) keep working
        out of the box. A future change to default-on would silently
        affect the baseline measurements."""
        config = GenerationConfig()
        assert config.adapter_path is None

    def test_adapter_path_propagates_to_generation_metadata(self):
        """Phase 3 eval needs to know which adapter (if any) produced
        each Answer. Pin: the path is stamped on metadata."""
        from docsense.generation.prompt import PromptBuilder

        config = GenerationConfig(
            model_name="dummy",
            device="cpu",
            adapter_path=Path("/some/adapter/path"),
        )

        class _StubGenerator(Generator):
            def _run_inference(self, messages):
                return (
                    "answer text",
                    {"latency_ms": 12.3, "prompt_tokens": 5, "completion_tokens": 2},
                )

        gen = _StubGenerator(config)
        chunks = [ChunkRef(doc_id="d.md", chunk_id="d.md::1", score=1.0, text="x")]
        messages = PromptBuilder().build(query="q", context="ctx")
        answer = gen.generate(messages, chunks)

        assert answer.metadata.adapter_path == "/some/adapter/path"

    def test_no_adapter_means_no_adapter_path_in_metadata(self):
        """Mirror of above: baseline path (no adapter) → metadata
        ``adapter_path`` is None, not e.g. an empty string."""
        from docsense.generation.prompt import PromptBuilder

        config = GenerationConfig(model_name="dummy", device="cpu")  # adapter_path defaults None

        class _StubGenerator(Generator):
            def _run_inference(self, messages):
                return (
                    "answer text",
                    {"latency_ms": 12.3, "prompt_tokens": 5, "completion_tokens": 2},
                )

        gen = _StubGenerator(config)
        chunks = [ChunkRef(doc_id="d.md", chunk_id="d.md::1", score=1.0, text="x")]
        messages = PromptBuilder().build(query="q", context="ctx")
        answer = gen.generate(messages, chunks)

        assert answer.metadata.adapter_path is None


class TestGeneratorContractViaPatch:
    """Block D.3: contract assertions on Generator.generate() using
    patch.object instead of subclassing. The subclass-based stub in
    TestGeneratorGenerate exercises the same code path but doesn't
    let us inspect HOW _run_inference was called. patch.object turns
    the override into a Mock object, so we can assert call count, the
    exact arguments passed, and that the mock's return value drives
    the resulting Answer.

    These tests catch a different category of regression: refactors
    that, e.g., accidentally call _run_inference twice, swap the
    prompt and chunks arguments, or ignore the inference output.
    """

    def _config(self) -> GenerationConfig:
        return GenerationConfig(model_name="contract-test", device="cpu")

    def _ref(self, doc_id: str = "d.md") -> ChunkRef:
        return ChunkRef(doc_id=doc_id, chunk_id=f"{doc_id}::chunk_0", score=0.5, text="t")

    @staticmethod
    def _msgs(content: str = "anything") -> list[dict[str, str]]:
        return [{"role": "user", "content": content}]

    def test_run_inference_called_exactly_once_per_generate(self):
        """Pin the contract: one inference call per generate(). A
        future refactor that does retry-on-empty or re-tokenizes
        would silently break latency and cost expectations."""
        gen = Generator(self._config())
        with patch.object(
            gen, "_run_inference", return_value=("text", {"latency_ms": 1.0})
        ) as mock_inf:
            gen.generate(self._msgs(), [])
        assert mock_inf.call_count == 1

    def test_run_inference_receives_messages_argument(self):
        """The messages list passed to generate() flows directly to
        _run_inference. Catches refactors that wrap, format, or
        re-build messages before passing them through."""
        gen = Generator(self._config())
        sentinel = [{"role": "user", "content": "the-exact-content"}]
        with patch.object(
            gen, "_run_inference", return_value=("text", {"latency_ms": 1.0})
        ) as mock_inf:
            gen.generate(sentinel, [self._ref()])

        # Inspect the call: messages is the first positional arg
        args, kwargs = mock_inf.call_args
        actual = args[0] if args else kwargs.get("messages")
        assert actual == sentinel

    def test_run_inference_output_drives_answer_text(self):
        """Whatever _run_inference returns becomes the Answer's text.
        The mock's return value is the only source of generated text;
        any text appearing on the Answer that isn't from the mock is
        a leak."""
        gen = Generator(self._config())
        canned = "the-exact-llm-output [1]"
        with patch.object(
            gen,
            "_run_inference",
            return_value=(canned, {"latency_ms": 5.0}),
        ):
            answer = gen.generate(self._msgs(), [self._ref("d.md")])

        assert answer.text == canned  # text passes through unchanged

    def test_run_inference_metadata_drives_answer_metadata(self):
        """Latency and token counts on Answer.metadata come from the
        raw_metadata dict _run_inference returns. Catches a regression
        where defaults silently mask real values."""
        gen = Generator(self._config())
        with patch.object(
            gen,
            "_run_inference",
            return_value=(
                "ok",
                {"latency_ms": 999.0, "prompt_tokens": 7, "completion_tokens": 3},
            ),
        ):
            answer = gen.generate(self._msgs(), [])

        assert answer.metadata.latency_ms == pytest.approx(999.0)
        assert answer.metadata.prompt_tokens == 7
        assert answer.metadata.completion_tokens == 3
        assert answer.metadata.model_name == "contract-test"

    def test_zero_chunks_and_no_citation_in_text_still_produces_valid_answer(self):
        """Edge case: model produces text without citations and no
        chunks were retrieved. The Answer is still constructable
        (vacuous citation-preservation) and minimal."""
        gen = Generator(self._config())
        with patch.object(
            gen, "_run_inference", return_value=("plain answer", {"latency_ms": 1.0})
        ):
            answer = gen.generate(self._msgs(), [])

        assert isinstance(answer, Answer)
        assert answer.citations == []
        assert answer.retrieved_chunks == []
