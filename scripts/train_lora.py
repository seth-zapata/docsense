#!/usr/bin/env python3
"""Local LoRA fine-tuning entry point for Phase 3.

Loads a TrainingDataset, splits it stratified train/val, applies the
chat template, and runs SFT with QLoRA via TRL's SFTTrainer.

Usage::

    # Full run with all defaults
    python scripts/train_lora.py --dataset evaluations/datasets/training/v1.json

    # Smoke run on a small subset (validate the loop without burning GPU time)
    python scripts/train_lora.py --dataset evaluations/datasets/training/v1.json --limit 20

    # Force all-GPU placement on a 12 GB shared GPU
    python scripts/train_lora.py --dataset ... --device cuda:0

    # One-knob hyperparam tweak (most overrides have a CLI flag for fast
    # iteration; for less common changes, edit FineTuningConfig directly)
    python scripts/train_lora.py --dataset ... --lora-rank 8 --learning-rate 5e-5

This is the **local execution path**. The Modal-aware version lands
in Block 3C; same FineTuningConfig + LoRAFineTuner core, different
entry-point shell. NOT in CI — running this kicks off real GPU work.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from docsense.finetuning.config import FineTuningConfig
from docsense.finetuning.dataset import TrainingDataset
from docsense.finetuning.trainer import LoRAFineTuner

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Path to a TrainingDataset JSON file (produced by "
            "scripts/build_training_dataset.py in Block 3B)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override FineTuningConfig.output_dir (default: models/fine-tunes/qwen-docsense-v1).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Override device_map. 'auto' lets accelerate distribute (can spill to "
            "CPU on 12 GB shared GPU); 'cuda:0' forces all-GPU placement. Pass "
            "'cpu' for a slow CPU-only smoke run."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap dataset to first N examples — for smoke runs that exercise "
            "every phase (load → split → format → train → save) without "
            "burning a full training run's GPU time."
        ),
    )
    # Most-tuned hyperparams as flags for fast iteration. For less common
    # overrides, instantiate FineTuningConfig directly in a Python REPL.
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> FineTuningConfig:
    """Apply CLI overrides on top of FineTuningConfig defaults."""
    overrides: dict[str, object] = {}
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.device is not None:
        overrides["device_map"] = args.device
    if args.lora_rank is not None:
        overrides["lora_rank"] = args.lora_rank
    if args.lora_alpha is not None:
        overrides["lora_alpha"] = args.lora_alpha
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        overrides["num_train_epochs"] = args.num_epochs
    if args.seed is not None:
        overrides["seed"] = args.seed
    return FineTuningConfig(**overrides)  # type: ignore[arg-type]


def _save_run_metadata(
    config: FineTuningConfig,
    metrics: dict[str, object],
    dataset_stats: dict[str, int],
    output_dir: Path,
) -> None:
    """Persist the per-run config snapshot + final metrics + dataset stats.

    The config snapshot is the most important reproducibility artifact:
    "exactly which hyperparameters produced the adapter at this path".
    Without it, an adapter file in models/fine-tunes/ is just a blob —
    you can't tell whether it was trained at rank=8 or rank=16, or
    whether the LR was the default or a sweep value.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "training_config.json"
    config_path.write_text(config.model_dump_json(indent=2) + "\n")

    metrics_path = output_dir / "training_metrics.json"
    payload = {
        "captured_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "dataset_stats": dataset_stats,
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    logger.info("Wrote training_config.json + training_metrics.json to %s", output_dir)


def main() -> int:
    args = _parse_args()

    # --- Load + slice dataset ----------------------------------------
    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        return 1

    full_ds = TrainingDataset.from_json(args.dataset)
    logger.info("Loaded dataset: %s", full_ds.stats())

    if args.limit is not None:
        examples = full_ds.examples[: args.limit]
        ds = TrainingDataset(
            examples=examples,
            version=f"{full_ds.version}-limit{args.limit}",
            description=f"Smoke-run subset of {full_ds.version} (first {args.limit} examples)",
            metadata={**full_ds.metadata, "limit_applied": args.limit},
        )
        logger.info("Limit applied: %d examples (was %d)", len(ds), len(full_ds))
    else:
        ds = full_ds

    # --- Build config + split ----------------------------------------
    config = _build_config(args)
    logger.info(
        "FineTuningConfig: rank=%d alpha=%d lr=%g epochs=%d device_map=%s",
        config.lora_rank,
        config.lora_alpha,
        config.learning_rate,
        config.num_train_epochs,
        config.device_map,
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
    logger.info("Starting training (this loads the base model on first iteration)...")
    metrics = finetuner.train(train_examples, val_examples)
    logger.info("Training complete: training_loss=%.4f", metrics["training_loss"])

    # --- Save adapter + metadata -------------------------------------
    adapter_path = finetuner.save_adapter()
    logger.info("Adapter saved to %s", adapter_path)

    _save_run_metadata(
        config=config,
        metrics=metrics,
        dataset_stats=ds.stats(),
        output_dir=adapter_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
