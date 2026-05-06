"""Unit tests for the FineTuningConfig defaults + validators.

The defaults are pinned because Phase 3's per-run config snapshot
gets serialized into each training report — drifting defaults would
silently change the reproducibility story for prior runs. Tests
catch accidental edits to the defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from docsense.finetuning.config import FineTuningConfig


class TestDefaults:
    def test_base_model_is_qwen(self):
        """Locked decision (2026-05-06): Qwen 2.5 7B Instruct as the
        base for Phase 3. Match the GenerationConfig default exactly
        so fine-tuning targets the production-default model."""
        config = FineTuningConfig()
        assert config.base_model_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_4bit_quantization_default_on(self):
        config = FineTuningConfig()
        assert config.use_4bit_quantization is True

    def test_lora_rank_default_16(self):
        """Middle of the 8-32 range typical for QLoRA narrow-skill
        fine-tuning. Pin so a sweep doesn't accidentally become the
        new default."""
        config = FineTuningConfig()
        assert config.lora_rank == 16

    def test_lora_alpha_2x_rank(self):
        """Convention from the LoRA paper: alpha = 2 * rank as a
        starting point. Not strictly required but pinned so a
        deliberate change is visible in the diff."""
        config = FineTuningConfig()
        assert config.lora_alpha == 32
        assert config.lora_alpha == 2 * config.lora_rank

    def test_lora_target_modules_attention_only(self):
        """Target only attention projections by default (q/k/v/o).
        Adding MLP layers (gate/up/down_proj) would increase capacity
        + VRAM but isn't needed for citation-rate fine-tuning."""
        config = FineTuningConfig()
        assert set(config.lora_target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    def test_learning_rate_2e_minus_4(self):
        """QLoRA paper default for Llama-style models. Higher than
        full-FT lr because we're only training small adapter matrices."""
        config = FineTuningConfig()
        assert config.learning_rate == 2e-4

    def test_num_epochs_3(self):
        config = FineTuningConfig()
        assert config.num_train_epochs == 3

    def test_batch_size_1_with_grad_accum_8(self):
        """Effective batch = 8 (1 × 8 accumulation steps). Pinned for
        the 12 GB VRAM constraint; A10G could do larger but we keep
        the default conservative for local-execution reproducibility."""
        config = FineTuningConfig()
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 8

    def test_optimizer_paged_8bit(self):
        """paged_adamw_8bit pages optimizer states to CPU when VRAM
        is tight — important for the 12 GB local path. Switching to
        regular adamw_8bit or fp32 adamw should be deliberate."""
        config = FineTuningConfig()
        assert config.optim == "paged_adamw_8bit"

    def test_gradient_checkpointing_on(self):
        config = FineTuningConfig()
        assert config.gradient_checkpointing is True

    def test_max_seq_length_2048(self):
        """Covers our prompt shape: 5 chunks × ~300 tok + system +
        question ≈ 1700, plus ideal answer ≤ 200, leaves margin."""
        config = FineTuningConfig()
        assert config.max_seq_length == 2048

    def test_seed_42(self):
        """Pinned seed for reproducibility — every Phase 3 training
        run should produce the same adapter from the same dataset
        unless seed is explicitly varied in the sweep."""
        config = FineTuningConfig()
        assert config.seed == 42

    def test_val_fraction_10_percent(self):
        config = FineTuningConfig()
        assert config.val_fraction == 0.1

    def test_save_and_eval_strategies_per_epoch(self):
        config = FineTuningConfig()
        assert config.save_strategy == "epoch"
        assert config.eval_strategy == "epoch"

    def test_output_dir_v1(self):
        config = FineTuningConfig()
        assert config.output_dir == Path("models/fine-tunes/qwen-docsense-v1")

    def test_bf16_default_true(self):
        """Production default: bf16 mixed-precision on, matching the
        NF4 compute_dtype=bfloat16. RTX 4070 (sm_89) and Modal A10G
        (sm_86) both support bf16 natively. Tests override to False
        on CPU-only CI runners (TRL SFTConfig rejects bf16=True there)
        but the production default stays True — pinned here so a
        flip to False is a deliberate, reviewable change."""
        config = FineTuningConfig()
        assert config.bf16 is True

    def test_device_map_default_auto(self):
        """device_map default 'auto' lets accelerate distribute. On
        a 12 GB shared GPU (RTX 4070 with display) override to 'cuda:0'
        via CLI to force all-GPU placement; pinned default here so
        the override remains the explicit choice."""
        config = FineTuningConfig()
        assert config.device_map == "auto"


class TestFieldValidation:
    def test_lora_rank_bounds(self):
        # Lower bound
        with pytest.raises(ValidationError):
            FineTuningConfig(lora_rank=0)
        # Upper bound (sanity — > 256 is well outside any reasonable use)
        with pytest.raises(ValidationError):
            FineTuningConfig(lora_rank=300)

    def test_lora_dropout_range(self):
        """Dropout must be in [0, 0.5]. >0.5 destroys signal; <0
        is meaningless."""
        with pytest.raises(ValidationError):
            FineTuningConfig(lora_dropout=-0.1)
        with pytest.raises(ValidationError):
            FineTuningConfig(lora_dropout=0.7)
        # Boundary values valid
        FineTuningConfig(lora_dropout=0.0)
        FineTuningConfig(lora_dropout=0.5)

    def test_learning_rate_must_be_positive(self):
        with pytest.raises(ValidationError):
            FineTuningConfig(learning_rate=0.0)
        with pytest.raises(ValidationError):
            FineTuningConfig(learning_rate=-1e-4)

    def test_warmup_ratio_bounds(self):
        with pytest.raises(ValidationError):
            FineTuningConfig(warmup_ratio=-0.01)
        with pytest.raises(ValidationError):
            FineTuningConfig(warmup_ratio=0.6)

    def test_val_fraction_strict_open_interval(self):
        with pytest.raises(ValidationError):
            FineTuningConfig(val_fraction=0.0)
        with pytest.raises(ValidationError):
            FineTuningConfig(val_fraction=1.0)

    def test_max_seq_length_minimum(self):
        """Seq lengths below 512 don't cover our prompt shape."""
        with pytest.raises(ValidationError):
            FineTuningConfig(max_seq_length=128)


class TestOverrides:
    def test_per_field_override(self):
        config = FineTuningConfig(lora_rank=8, learning_rate=5e-5)
        assert config.lora_rank == 8
        assert config.learning_rate == 5e-5
        # Other defaults unchanged
        assert config.lora_alpha == 32

    def test_target_modules_extended_to_include_mlp(self):
        """Sweep target: include MLP layers if attention-only doesn't
        give enough capacity. Validate the override path works."""
        config = FineTuningConfig(
            lora_target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )
        assert "gate_proj" in config.lora_target_modules
        assert len(config.lora_target_modules) == 7

    def test_output_dir_per_run(self):
        config = FineTuningConfig(output_dir=Path("models/fine-tunes/qwen-docsense-v2"))
        assert "v2" in str(config.output_dir)
