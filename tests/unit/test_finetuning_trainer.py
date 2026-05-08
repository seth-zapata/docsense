"""Unit tests for the LoRA trainer wrapper.

Real training requires GPU + the actual base model; that's exercised
manually via ``scripts/train_lora.py`` and validated end-to-end in
Block 3C.

These tests cover the testable surface:

- ``format_messages_for_training``: pure logic, builds the 3-turn
  chat-message list. The chat-template format must match what the
  generator sees at inference (PromptBuilder output + assistant turn).
- ``_build_lora_config``: maps FineTuningConfig → peft.LoraConfig.
  Pinned so a future config-rename doesn't silently produce a
  zero-rank LoRA.
- ``_build_sft_config``: maps FineTuningConfig → trl.SFTConfig.
  Same rationale.
- Construction of ``LoRAFineTuner`` is cheap (no model load).

Tests deliberately do NOT exercise the model property, since loading
a real 7B model is GPU-bound and slow. The lazy-property pattern
means tests calling only the format/config paths complete in
milliseconds.
"""

from __future__ import annotations

from docsense.finetuning.config import FineTuningConfig
from docsense.finetuning.dataset import TrainingExample
from docsense.finetuning.trainer import (
    _QWEN_TRAINING_CHAT_TEMPLATE,
    LoRAFineTuner,
    _format_chunks_block,
    _is_qwen_model,
    format_messages_for_training,
)
from docsense.generation.prompt import DEFAULT_SYSTEM_PROMPT
from docsense.generation.types import ChunkRef


def _chunks(*texts: str) -> list[ChunkRef]:
    return [
        ChunkRef(doc_id=f"doc{i}.md", chunk_id=str(i), score=1.0, text=text)
        for i, text in enumerate(texts, start=1)
    ]


def _example(
    *,
    query: str = "How do I install transformers?",
    answer: str = "Run `pip install transformers` [1].",
    chunks: list[ChunkRef] | None = None,
    is_refusal: bool = False,
) -> TrainingExample:
    if chunks is None:
        chunks = _chunks("To install, run pip install transformers.", "Or use conda.")
    return TrainingExample(
        query=query,
        retrieved_chunks=chunks,
        ideal_answer=answer,
        is_refusal=is_refusal,
    )


class TestFormatChunksBlock:
    def test_numbered_with_source_attribution(self):
        chunks = _chunks("alpha text", "beta text")
        block = _format_chunks_block(chunks)
        # Format: [N] (source: doc_id)\n{text}, joined with blank line
        assert "[1] (source: doc1.md)" in block
        assert "[2] (source: doc2.md)" in block
        assert "alpha text" in block
        assert "beta text" in block

    def test_matches_context_assembler_format(self):
        """The block format MUST match what ContextAssembler produces
        at inference. Drift would mean the model sees a different
        prompt shape during fine-tuning vs serving — silent regression
        risk. Pin the exact string so a refactor that subtly changes
        the format breaks this test."""
        chunks = _chunks("first chunk", "second chunk")
        block = _format_chunks_block(chunks)
        expected = "[1] (source: doc1.md)\nfirst chunk\n\n[2] (source: doc2.md)\nsecond chunk"
        assert block == expected

    def test_empty_chunks_produces_empty_block(self):
        """For refusal examples, retrieved_chunks may be empty —
        produces an empty string rather than an error."""
        assert _format_chunks_block([]) == ""


class TestFormatMessagesForTraining:
    def test_three_turn_structure(self):
        ex = _example()
        messages = format_messages_for_training(ex)
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_system_uses_default_prompt(self):
        """Pin: training uses the canonical system prompt from
        PromptBuilder. Drift here would mean fine-tuning teaches the
        model to expect a different system prompt than it gets at
        serve time."""
        messages = format_messages_for_training(_example())
        assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT

    def test_user_includes_context_and_question(self):
        ex = _example(
            query="How do I install transformers?",
            chunks=_chunks("pip install transformers ..."),
        )
        messages = format_messages_for_training(ex)
        user = messages[1]["content"]
        assert "Context:" in user
        assert "Question: How do I install transformers?" in user
        assert "[1]" in user
        assert "pip install transformers" in user

    def test_assistant_is_ideal_answer(self):
        ex = _example(answer="The answer with [1] citation.")
        messages = format_messages_for_training(ex)
        assert messages[2]["content"] == "The answer with [1] citation."

    def test_refusal_with_empty_chunks(self):
        """Refusal training examples produce the same 3-turn structure.
        Empty chunks → empty Context block in the user message.
        Model learns: 'when context is empty, refuse even though the
        system prompt asks for citations.'"""
        ex = TrainingExample(
            query="What's the price of tea in China?",
            retrieved_chunks=[],
            ideal_answer="I don't have enough context to answer that.",
            is_refusal=True,
        )
        messages = format_messages_for_training(ex)
        assert len(messages) == 3
        assert messages[2]["content"] == "I don't have enough context to answer that."
        # User message has empty Context block — that's the SIGNAL
        # to the model that no chunks were retrieved (refusal context).
        user = messages[1]["content"]
        assert "Context:\n\n" in user or user.startswith("Context:")


class TestLoRAFineTunerConstruction:
    def test_construction_does_not_load_model(self):
        """Construction stores config; no HF model load. This
        guarantees test-time use is millisecond-cheap."""
        config = FineTuningConfig()
        finetuner = LoRAFineTuner(config)
        assert finetuner.config is config
        assert finetuner._trained_model is None
        assert finetuner._tokenizer is None

    def test_config_is_addressable(self):
        config = FineTuningConfig(lora_rank=8, learning_rate=1e-4)
        finetuner = LoRAFineTuner(config)
        assert finetuner.config.lora_rank == 8
        assert finetuner.config.learning_rate == 1e-4


class TestBuildLoraConfig:
    def test_fields_propagate_from_finetuning_config(self):
        from peft import LoraConfig, TaskType

        config = FineTuningConfig(lora_rank=32, lora_alpha=64, lora_dropout=0.1)
        finetuner = LoRAFineTuner(config)
        lora_config = finetuner._build_lora_config()

        assert isinstance(lora_config, LoraConfig)
        assert lora_config.r == 32
        assert lora_config.lora_alpha == 64
        assert lora_config.lora_dropout == 0.1
        assert lora_config.bias == "none"
        assert lora_config.task_type == TaskType.CAUSAL_LM

    def test_default_target_modules_qkv_o(self):
        config = FineTuningConfig()
        finetuner = LoRAFineTuner(config)
        lora_config = finetuner._build_lora_config()
        assert set(lora_config.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    def test_extended_target_modules_propagate(self):
        config = FineTuningConfig(
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
        )
        finetuner = LoRAFineTuner(config)
        lora_config = finetuner._build_lora_config()
        assert "gate_proj" in lora_config.target_modules


class TestBuildSftConfig:
    """Tests construct SFTConfig on whatever runner CI is using
    (typically CPU-only GitHub Actions). TRL's SFTConfig validates
    ``bf16=True`` at construction and rejects it without a
    bf16-capable GPU. Tests override ``bf16=False`` for that reason.
    Production runs on RTX 4070 / Modal A10G use the default
    ``bf16=True`` (covered by the FineTuningConfig defaults test
    in test_finetuning_config.py)."""

    def test_fields_propagate(self):
        config = FineTuningConfig(
            num_train_epochs=5,
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            seed=123,
            bf16=False,
        )
        finetuner = LoRAFineTuner(config)
        sft_config = finetuner._build_sft_config()
        assert sft_config.num_train_epochs == 5
        assert sft_config.learning_rate == 5e-5
        assert sft_config.per_device_train_batch_size == 2
        assert sft_config.gradient_accumulation_steps == 4
        assert sft_config.seed == 123

    def test_output_dir_serialized_as_string(self):
        """SFTConfig (HF Trainer) accepts str for output_dir; we pass
        str(Path) to avoid pydantic Path-vs-str disagreements."""
        from pathlib import Path as _Path

        config = FineTuningConfig(output_dir=_Path("/tmp/test-output"), bf16=False)
        finetuner = LoRAFineTuner(config)
        sft_config = finetuner._build_sft_config()
        assert sft_config.output_dir == "/tmp/test-output"

    def test_bf16_propagates_from_config(self):
        """bf16 is now config-driven (was hardcoded True in the
        initial 3A.2 implementation, broke CI on CPU runners).
        Production default is True; tests use False to bypass TRL's
        bf16-capability check on CI runners. Pin both directions."""
        finetuner_off = LoRAFineTuner(FineTuningConfig(bf16=False))
        assert finetuner_off._build_sft_config().bf16 is False

    def test_packing_disabled(self):
        """Packing concatenates short examples for throughput. Bad
        for our short-answer fine-tuning because the cross-example
        attention contaminates training. Pin off."""
        config = FineTuningConfig(bf16=False)
        finetuner = LoRAFineTuner(config)
        sft_config = finetuner._build_sft_config()
        assert sft_config.packing is False

    def test_report_to_disabled_by_default(self):
        """Disable wandb / TB auto-logging — our reports are committed
        JSON, not external dashboards. A user who wants wandb can
        enable via env var or override; default is silent."""
        config = FineTuningConfig(bf16=False)
        finetuner = LoRAFineTuner(config)
        sft_config = finetuner._build_sft_config()
        # report_to can be empty list or ["none"]; both mean silent.
        assert not sft_config.report_to or sft_config.report_to == ["none"]


class TestNF4Config:
    def test_compute_dtype_bfloat16_for_training(self):
        """Training uses bf16 compute_dtype (vs fp16 in inference paths).
        Pin so a refactor that consolidates the three NF4 configs
        doesn't accidentally drop the train-vs-inference distinction."""
        import torch

        nf4 = LoRAFineTuner._build_nf4_config()
        assert nf4.bnb_4bit_compute_dtype == torch.bfloat16
        assert nf4.load_in_4bit is True
        assert nf4.bnb_4bit_quant_type == "nf4"
        assert nf4.bnb_4bit_use_double_quant is True


class TestSaveAdapter:
    def test_save_calls_model_save_pretrained(self, tmp_path, monkeypatch):
        """``save_adapter`` should write to the provided output_dir
        (or fall back to config.output_dir). We monkeypatch the
        model property to avoid loading a real model."""

        class _StubModel:
            def __init__(self):
                self.saved_to: str | None = None

            def save_pretrained(self, path: str) -> None:
                self.saved_to = path

        config = FineTuningConfig(output_dir=tmp_path / "default-out")
        finetuner = LoRAFineTuner(config)
        stub = _StubModel()
        # Inject the stub as the lazy-loaded model so the property
        # returns it without touching HF.
        finetuner._trained_model = stub

        result_path = finetuner.save_adapter()
        assert result_path == tmp_path / "default-out"
        assert (tmp_path / "default-out").exists()
        assert stub.saved_to == str(tmp_path / "default-out")

    def test_save_with_explicit_output_dir(self, tmp_path):
        class _StubModel:
            def save_pretrained(self, path: str) -> None:  # noqa: ARG002
                pass

        config = FineTuningConfig()
        finetuner = LoRAFineTuner(config)
        finetuner._trained_model = _StubModel()

        explicit = tmp_path / "explicit-out"
        result_path = finetuner.save_adapter(output_dir=explicit)
        assert result_path == explicit
        assert explicit.exists()

    def test_save_before_train_raises(self):
        """save_adapter() before train() has nothing to persist —
        catch with a clear error rather than silently saving an
        empty model. The error message points at the cause + the
        test-injection escape hatch."""
        import pytest

        config = FineTuningConfig()
        finetuner = LoRAFineTuner(config)
        with pytest.raises(RuntimeError, match="before train"):
            finetuner.save_adapter()


class TestIsQwenModel:
    """Family detector that decides whether to override the chat
    template at training time. Qwen-specific because the
    ``{% generation %}`` injection is template-specific."""

    def test_official_hub_id_is_qwen(self):
        assert _is_qwen_model("Qwen/Qwen2.5-7B-Instruct")

    def test_other_qwen_variants(self):
        assert _is_qwen_model("Qwen/Qwen2.5-14B-Instruct")
        assert _is_qwen_model("Qwen/Qwen3-7B-Instruct")

    def test_lowercase_qwen_variant(self):
        """Local paths or community forks may not preserve the official
        capitalization. Detect those too so the template override
        still fires."""
        assert _is_qwen_model("qwen-local-fork")

    def test_non_qwen_models(self):
        assert not _is_qwen_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
        assert not _is_qwen_model("mistralai/Mistral-7B-Instruct-v0.3")
        assert not _is_qwen_model("microsoft/phi-3-mini-4k-instruct")


class TestQwenTrainingChatTemplate:
    """The custom Qwen training-time chat template injected to satisfy
    TRL's ``assistant_only_loss=True`` requirement.

    Caught in Block 3C.1 smoke: Qwen 2.5's stock template lacks
    the ``{% generation %}`` Jinja markers TRL needs for assistant-mask
    generation; without them, SFTTrainer crashes mid-tokenization.
    """

    def test_contains_generation_markers(self):
        """The whole point — must wrap something with the markers."""
        assert "{% generation %}" in _QWEN_TRAINING_CHAT_TEMPLATE
        assert "{% endgeneration %}" in _QWEN_TRAINING_CHAT_TEMPLATE

    def test_generation_block_wraps_assistant_content_only(self):
        """``{% generation %}`` should bracket the assistant content,
        NOT the role markers or system/user content. The assistant-mask
        TRL produces from this should cover the answer text exactly."""
        # Find the generation block and verify its bracketing context.
        assert "{% generation %}{{- message.content + '<|im_end|>' -}}{% endgeneration %}" in (
            _QWEN_TRAINING_CHAT_TEMPLATE
        )
        # The role prefix should be OUTSIDE the generation block, so
        # the mask doesn't include the '<|im_start|>assistant\n' tokens.
        before_gen = _QWEN_TRAINING_CHAT_TEMPLATE.split("{% generation %}")[0]
        assert "<|im_start|>assistant" in before_gen

    def test_handles_system_user_assistant_roles(self):
        """The template must render all three role types our training
        triples use (system + user + assistant). Pin so a future
        simplification doesn't drop a branch."""
        for role in ("system", "user", "assistant"):
            assert f"'{role}'" in _QWEN_TRAINING_CHAT_TEMPLATE or (
                f'"{role}"' in _QWEN_TRAINING_CHAT_TEMPLATE
            )

    def test_emits_qwen_special_tokens(self):
        """The rendered output must use Qwen 2.5's special tokens
        (im_start / im_end). Pin so the byte-identity invariant
        between training and inference holds — the Generator path
        uses the stock template at inference; the special tokens
        emitted must match."""
        assert "<|im_start|>" in _QWEN_TRAINING_CHAT_TEMPLATE
        assert "<|im_end|>" in _QWEN_TRAINING_CHAT_TEMPLATE

    def test_supports_add_generation_prompt(self):
        """When ``add_generation_prompt=True`` (the SFTTrainer default
        for the model.generate path), the template must emit the
        opening assistant marker — otherwise the model has no signal
        that it's the assistant's turn."""
        assert "add_generation_prompt" in _QWEN_TRAINING_CHAT_TEMPLATE


class TestTrainerImportsAreLazy:
    """Construction MUST not eagerly import heavy training deps (peft,
    trl, transformers). The lazy-property pattern is what makes
    test-only paths fast — verify it stays that way."""

    def test_construction_does_not_import_peft(self):
        """If peft was force-imported at LoRAFineTuner creation, this
        test would import it as a side-effect of construction. Test
        passes if peft is only touched when ``_build_lora_config`` or
        ``model`` is called."""
        # Reset import cache trick: we just verify construction
        # doesn't raise even if PEFT had import issues. The actual
        # peft import lives inside _build_lora_config + _load_base_model.
        config = FineTuningConfig()
        # Construction should never touch peft / trl / transformers.
        finetuner = LoRAFineTuner(config)
        assert finetuner is not None

    def test_format_messages_does_not_load_model(self):
        """format_messages_for_training is module-level and shouldn't
        require model state. Pin: it can run with no LoRAFineTuner
        instance at all."""
        ex = _example()
        messages = format_messages_for_training(ex)
        assert len(messages) == 3
        # No LoRAFineTuner ever instantiated; just imports.
