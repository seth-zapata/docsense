"""Modal-aware LoRA training entry point for Phase 3 Block 3C.

Wraps the same ``LoRAFineTuner`` core that ``scripts/train_lora.py`` uses,
but invoked as a Modal function on an A10G GPU. The local script's path
remains for quick local iteration on small slices; this script is the
production training path.

Usage::

    # One-time setup (done locally, once per Modal account)
    modal token new                                       # authenticate CLI
    modal secret create hf_token HF_TOKEN=hf_xxxxxxx      # required; use a read-scoped HF token

    # Smoke: 1 grad step on 10 examples, ~30 sec compute (~6-10 min wall
    # including image build + weight download on first invocation)
    modal run scripts/train_lora_modal.py::smoke

    # Full training run: ~30-40 min wall, ~$0.55-0.75 compute
    modal run scripts/train_lora_modal.py::full

    # After the run completes, the adapter is on the Modal volume.
    # Download to local for use with the inference path:
    modal volume get docsense-adapters /v1 ./models/fine-tunes/qwen-docsense-v1

Architecture choices documented in ``docs/phase-3-block-3c-plan.md``;
key ones for this script:

- A10G (24 GB) — comfortable for Qwen 7B at NF4 + LoRA without paging.
- ``hf-cache`` volume — persistent HF cache so we download Qwen
  weights once, not every run. Saves ~5 min on every subsequent
  invocation.
- ``docsense-adapters`` volume — adapter weights are written here,
  downloaded to local after run via ``modal volume get``.
- ``hf_token`` Modal Secret — wired into the function decorator. Not
  strictly needed for Qwen 2.5 (non-gated) but included so future
  experiments with gated models work without an emergency script
  change. Use a read-scoped token (HF settings) for least privilege.
  The secret never touches the git repo; Modal injects it as an env
  var at function runtime, never persisted to image layers or logs.
- ``add_local_dir`` for the dataset — committed in the repo, mounted
  at build time. ~2 MB; trivial cost.
- ``add_local_python_source("docsense")`` — mounts the package source
  at build time. Local code changes take effect on next ``modal run``
  without re-pushing an image.
"""

from __future__ import annotations

from pathlib import Path

import modal

# --- Image -----------------------------------------------------------
# Mirror pyproject.toml's runtime + GPU deps. We pin against the same
# constraints that pip-install ``[dev,gpu]`` would resolve locally,
# so behavior matches the local trainer (the chat-template fix in
# trainer.py applies identically here — same code mounted at build).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
        "datasets>=2.18.0",
        "trl>=0.8.0",
        "bitsandbytes>=0.43.0",
        "sentence-transformers>=2.6.0",
        "pydantic>=2.6.0",
        "pydantic-settings>=2.2.0",
        "structlog>=24.1.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "huggingface-hub>=0.20.0",
    )
    # Mount the dataset committed in the repo so the Modal function
    # can read it directly. ~2 MB; cost-trivial.
    .add_local_dir(
        local_path=Path("evaluations/datasets/training"),
        remote_path="/data/training",
    )
    # Mount the local docsense package source so changes to
    # finetuning/* take effect on next modal run without rebuilding
    # the image. Critical for iteration in Block 3D.
    .add_local_python_source("docsense")
)

app = modal.App("docsense-lora-train", image=image)

# Persistent volumes. ``create_if_missing=True`` so the first invocation
# provisions them automatically — no manual setup beyond ``modal token new``.
hf_cache_vol = modal.Volume.from_name("docsense-hf-cache", create_if_missing=True)
adapters_vol = modal.Volume.from_name("docsense-adapters", create_if_missing=True)


# --- Training function -----------------------------------------------
@app.function(
    gpu="A10G",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/adapters": adapters_vol,
    },
    # HF_TOKEN secret wired in even though Qwen 2.5 (Apache 2.0,
    # non-gated) doesn't strictly require it. Reasoning:
    # - Cost of including: ~zero. The secret already exists in the
    #   Modal account regardless of whether the script references
    #   it; the reference only changes whether the value is injected
    #   into the function's runtime env.
    # - Benefit: when (not if) we switch to a gated model in some
    #   future experiment, training "just works" instead of failing
    #   with NotFoundError at exactly the wrong moment.
    # - Security: Modal's secret store keeps the value out of the
    #   git repo, image layers, and logs. The token should be
    #   read-scoped (HF Hub setting) for principle-of-least-privilege.
    # Note: Modal's `Secret.from_name` REQUIRES the secret to exist
    # by name — there is no "optional secret" mode. If you don't have
    # an HF account / token at all, comment out the `secrets=` line
    # below; the script then works for non-gated models like Qwen
    # without any auth (which is what PR #33 confirmed empirically).
    secrets=[modal.Secret.from_name("hf_token")],
    timeout=3600,
)
def train(
    *,
    limit: int | None = None,
    num_epochs: int = 3,
    output_subdir: str = "v1",
    learning_rate: float | None = None,
    lora_rank: int | None = None,
    seed: int = 42,
) -> dict:
    """Run QLoRA training on Qwen 2.5 7B Instruct.

    All hyperparameter overrides default to ``FineTuningConfig`` values
    when ``None``; pass explicit values to sweep in Block 3D.

    Returns a dict with ``training_loss``, ``metrics``, and the
    in-volume adapter path. Caller downloads the adapter from the
    ``docsense-adapters`` volume via ``modal volume get`` after the
    function completes.
    """
    import json
    import logging
    import os
    from datetime import UTC, datetime

    # Local-to-Modal-container imports. Heavy deps load only inside
    # the GPU container, not on the local entrypoint side.
    from docsense.finetuning.config import FineTuningConfig
    from docsense.finetuning.dataset import TrainingDataset
    from docsense.finetuning.trainer import LoRAFineTuner

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Make sure the HF cache lands on the persistent volume, not the
    # ephemeral container filesystem. The volume is mounted at the HF
    # default cache path, but setting HF_HOME explicitly is belt-and-
    # suspenders against future HF default changes.
    os.environ["HF_HOME"] = "/root/.cache/huggingface"

    # --- Load + slice dataset ----------------------------------------
    dataset_path = Path("/data/training/training_dataset.json")
    if not dataset_path.exists():
        msg = (
            f"Training dataset not found at {dataset_path}. "
            f"The image build should have mounted "
            f"evaluations/datasets/training/ → /data/training/. "
            f"Check the add_local_dir call in scripts/train_lora_modal.py."
        )
        raise FileNotFoundError(msg)

    full_ds = TrainingDataset.from_json(dataset_path)
    logger.info("Loaded dataset: %s", full_ds.stats())

    if limit is not None:
        examples = full_ds.examples[:limit]
        ds = TrainingDataset(
            examples=examples,
            version=f"{full_ds.version}-modal-limit{limit}",
            description=f"Modal-run subset of {full_ds.version} (first {limit} examples)",
            metadata={**full_ds.metadata, "limit_applied": limit},
        )
        logger.info("Limit applied: %d examples (was %d)", len(ds), len(full_ds))
    else:
        ds = full_ds

    # --- Build config -------------------------------------------------
    config_overrides: dict = {
        # Adapters write into the Modal volume, not the container's
        # ephemeral filesystem. ``modal volume get`` then downloads to
        # local after the run completes.
        "output_dir": Path(f"/adapters/{output_subdir}"),
        "num_train_epochs": num_epochs,
        "seed": seed,
        # A10G has 24 GB — let device_map=auto distribute. Spillover
        # to CPU is unlikely; if it happens it's a config bug worth
        # logging rather than masking.
        "device_map": "auto",
    }
    if learning_rate is not None:
        config_overrides["learning_rate"] = learning_rate
    if lora_rank is not None:
        config_overrides["lora_rank"] = lora_rank

    config = FineTuningConfig(**config_overrides)
    logger.info(
        "FineTuningConfig: rank=%d alpha=%d lr=%g epochs=%d output_dir=%s",
        config.lora_rank,
        config.lora_alpha,
        config.learning_rate,
        config.num_train_epochs,
        config.output_dir,
    )

    train_examples, val_examples = ds.stratified_train_val_split(
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    logger.info(
        "Stratified split: %d train / %d val (val_fraction=%.2f, seed=%d)",
        len(train_examples),
        len(val_examples),
        config.val_fraction,
        config.seed,
    )

    # --- Train -------------------------------------------------------
    finetuner = LoRAFineTuner(config)
    logger.info("Starting training...")
    metrics = finetuner.train(train_examples, val_examples)
    logger.info("Training complete: training_loss=%.4f", metrics["training_loss"])

    # --- Save adapter + metadata -------------------------------------
    adapter_path = finetuner.save_adapter()
    logger.info("Adapter saved to %s (volume: docsense-adapters)", adapter_path)

    # Persist the config snapshot + metrics alongside the adapter for
    # reproducibility (same shape as the local script's _save_run_metadata).
    (adapter_path / "training_config.json").write_text(config.model_dump_json(indent=2) + "\n")
    run_meta = {
        "captured_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "dataset_stats": ds.stats(),
        "metrics": metrics,
        "infrastructure": "modal-a10g",
    }
    (adapter_path / "training_metrics.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=False) + "\n"
    )

    # IMPORTANT: commit the volume so the writes are visible to
    # ``modal volume get`` from the local CLI. Without this, the
    # adapter exists in the container filesystem but isn't persisted.
    adapters_vol.commit()
    logger.info("Volume committed; adapter is now downloadable via:")
    logger.info("  modal volume get docsense-adapters /%s ./local/path", output_subdir)

    return {
        "training_loss": metrics["training_loss"],
        "metrics": metrics,
        "adapter_path": str(adapter_path),
        "output_subdir": output_subdir,
    }


# --- Local entrypoints -----------------------------------------------
@app.local_entrypoint()
def smoke(output_subdir: str = "smoke") -> None:
    """Smoke run: 1 grad step on 10 examples.

    Validates the Modal infrastructure end-to-end (image build, volume
    provisioning, secret retrieval, model download, training step,
    adapter save, volume commit) before paying for the full run. About
    30 sec of A10G compute on top of the cold-start image build (~$0.01).
    """
    result = train.remote(
        limit=10,
        num_epochs=1,
        output_subdir=output_subdir,
    )
    print("\n=== Smoke complete ===")
    print(f"  training_loss: {result['training_loss']:.4f}")
    print(f"  adapter_path (in volume): {result['adapter_path']}")
    print("\nDownload the adapter to local:")
    print(f"  modal volume get docsense-adapters /{output_subdir} ./tmp/adapter-modal-smoke")


@app.local_entrypoint()
def full(output_subdir: str = "v1") -> None:
    """Full training run on the 591-example Block 3B dataset.

    Default hyperparameters from ``FineTuningConfig`` (rank=16, alpha=32,
    lr=2e-4, 3 epochs). Use ``modal run scripts/train_lora_modal.py::sweep``
    in Block 3D for hyperparameter overrides.

    ~30-40 min wall time on A10G; ~\\$0.55-0.75 compute. Adapter saved
    to the ``docsense-adapters`` volume; download via ``modal volume get``.
    """
    result = train.remote(output_subdir=output_subdir)
    print("\n=== Full training complete ===")
    print(f"  training_loss: {result['training_loss']:.4f}")
    print(f"  metrics: {result['metrics']}")
    print(f"  adapter_path (in volume): {result['adapter_path']}")
    print("\nDownload the adapter to local:")
    print(
        f"  modal volume get docsense-adapters /{output_subdir} ./models/fine-tunes/qwen-docsense-{output_subdir}"
    )


@app.local_entrypoint()
def sweep(
    output_subdir: str,
    learning_rate: float | None = None,
    lora_rank: int | None = None,
    num_epochs: int | None = None,
) -> None:
    """Hyperparameter sweep cycle for Block 3D iterations.

    Each cycle changes ONE variable (per the Phase 3 scope's locked
    "sequential informed sweep" decision). Required: ``output_subdir``
    so each cycle writes to its own subdirectory under the adapters
    volume — runs don't overwrite each other.
    """
    if all(x is None for x in (learning_rate, lora_rank, num_epochs)):
        msg = "sweep requires at least one of --learning-rate, --lora-rank, --num-epochs"
        raise SystemExit(msg)

    kwargs: dict = {"output_subdir": output_subdir}
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    if lora_rank is not None:
        kwargs["lora_rank"] = lora_rank
    if num_epochs is not None:
        kwargs["num_epochs"] = num_epochs

    result = train.remote(**kwargs)
    print(f"\n=== Sweep run '{output_subdir}' complete ===")
    print(f"  training_loss: {result['training_loss']:.4f}")
    print(f"  adapter_path (in volume): {result['adapter_path']}")
