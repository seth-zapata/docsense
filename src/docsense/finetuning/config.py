"""Pinned hyperparameter defaults for QLoRA fine-tuning.

Defaults chosen for the docsense Phase 3 baseline: Qwen 2.5 7B
Instruct as base, 12 GB VRAM constraint (RTX 4070 local) but with
24 GB headroom on Modal A10G for cloud runs. All values are
configurable for the iteration sweeps in Block 3D.

Lora hyperparameters reference:
- The Hu et al. LoRA paper (`alpha = 2 * rank` is a common starting
  point but not a hard requirement).
- For QLoRA specifically (Dettmers et al. 2023): rank=8-32 is
  typical for narrow-skill fine-tuning; we default to 16 as a
  middle ground.
- Targeting only attention projection matrices (q/k/v/o_proj) is
  the standard minimum; some recipes also adapt MLP layers
  (gate/up/down_proj) for harder skills. Citation behavior should
  be learnable from attention-only.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used at runtime as the field default type

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class FineTuningConfig(BaseSettings):
    # --- Base model -------------------------------------------------
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_4bit_quantization: bool = True

    # --- LoRA hyperparams -------------------------------------------
    # rank=16 is the middle of the typical 8-32 range for QLoRA narrow-
    # skill fine-tuning. Larger rank → more adapter capacity (and more
    # VRAM); smaller rank → faster, lighter, sometimes underfits.
    lora_rank: int = Field(default=16, ge=1, le=256)
    # alpha controls the scaling: effective_lr_scale = alpha / rank.
    # Convention: alpha = 2 * rank to roughly match the original LoRA
    # paper's effective scale. Not a hard requirement; can sweep.
    lora_alpha: int = Field(default=32, ge=1)
    # Dropout applied to LoRA layers. 0.05-0.1 is typical.
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    # Which projection matrices get LoRA adapters. Qwen2 attention is
    # ``q_proj, k_proj, v_proj, o_proj``. Adding the MLP projections
    # (``gate_proj, up_proj, down_proj``) increases capacity but also
    # VRAM; not needed for citation behavior in our experience plan.
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # --- Training hyperparams ---------------------------------------
    # 2e-4 is the QLoRA paper's default for Llama-style models;
    # generally good for adapter-only training (much higher than
    # full-FT learning rates because we're only training small matrices).
    learning_rate: float = Field(default=2e-4, gt=0.0)
    num_train_epochs: int = Field(default=3, ge=1, le=20)
    # batch=1 because 12 GB VRAM doesn't fit larger with 7B + activations.
    # Modal A10G (24 GB) could handle batch=2 or 4 — sweep target if
    # we want larger effective batches without grad accumulation.
    per_device_train_batch_size: int = Field(default=1, ge=1)
    # Effective batch = per_device_batch * gradient_accumulation.
    # Default 1 * 8 = 8 effective. Larger = smoother gradients but
    # longer per-step wall time.
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    # Linear warmup over the first 3% of steps; standard for SFT.
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    lr_scheduler_type: str = "cosine"
    # 8-bit AdamW saves ~75% of optimizer-state VRAM vs fp32 AdamW.
    # ``paged_adamw_8bit`` additionally pages optimizer states to CPU
    # if VRAM is tight — important for the 12 GB local path.
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    # Max input length (prompt + answer). Our prompts at top_k=5 with
    # ~300 tok/chunk + system + question ≈ 1700; ideal answers
    # typically ≤ 200. 2048 covers this with margin.
    max_seq_length: int = Field(default=2048, ge=512)

    # --- Output / logging -------------------------------------------
    output_dir: Path = Path("models/fine-tunes/qwen-docsense-v1")
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    logging_steps: int = Field(default=10, ge=1)
    seed: int = 42

    # --- Eval split -------------------------------------------------
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)

    # --- Device placement -------------------------------------------
    # device_map controls where the base model's weights land. The
    # default "auto" lets accelerate distribute across visible GPUs
    # (and CPU as last resort). On a 12 GB shared GPU (RTX 4070 with
    # display) "auto" can spill weights to CPU and tank training
    # throughput; force ``{"": 0}`` (everything on GPU 0) in that case
    # via the CLI ``--device cuda:0`` flag. On Modal A10G (24 GB) "auto"
    # is fine. Stored as str | dict to accept either ``"auto"``,
    # ``"cuda:0"`` (string form), or an explicit ``{"": 0}`` mapping.
    device_map: str = "auto"

    # --- Mixed precision --------------------------------------------
    # bf16 mixed-precision training. Default True matches the NF4
    # ``compute_dtype=bfloat16`` for numerical stability. SFTConfig
    # validates this at construction time and rejects True if no
    # CUDA-capable bf16 device is available — so tests running on
    # CPU-only CI override to False. RTX 4070 (sm_89) and Modal A10G
    # (sm_86) both support bf16 natively; production paths get the
    # default. Setting this to False on a GPU-capable machine isn't
    # a useful configuration but the option exists for completeness.
    bf16: bool = True

    @model_validator(mode="after")
    def _warn_if_alpha_far_from_2x_rank(self) -> FineTuningConfig:
        """Soft sanity: alpha ≪ rank or alpha ≫ 4×rank is unusual.
        We don't reject — sweeping alpha is a legitimate experiment —
        but flag if construction looks accidentally off."""
        # Pydantic validators can't easily emit warnings without
        # stdlib ``warnings`` import. Skip the warning for now;
        # callers building a config-sweep matrix may want this
        # combination intentionally. Validator stub left for future
        # if we add a constructor logger.
        return self
