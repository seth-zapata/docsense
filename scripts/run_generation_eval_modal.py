"""Modal-aware eval driver for Phase 3.

Same eval infrastructure as ``scripts/run_generation_eval.py`` but
executed on Modal A10G. Frees the local 12 GB box during long eval
runs (which otherwise lock the machine for ~40 min) and gets us
roughly 3-4× faster wall time on top — A10G has 24 GB headroom so no
paged-AdamW thrash, no inter-process VRAM swapping.

Usage::

    # Smoke (3 queries per set, validates Modal eval infra cheaply)
    modal run scripts/run_generation_eval_modal.py::smoke

    # Modal-baseline run (no adapter — re-runs the pre-Phase-3
    # baseline shape on Modal so adapter comparisons are
    # same-hardware-fair)
    modal run scripts/run_generation_eval_modal.py::baseline

    # Adapter run (uses the trained adapter from docsense-adapters)
    modal run scripts/run_generation_eval_modal.py::adapter --subdir v1

    # Download reports to local for analysis
    modal volume get docsense-eval-reports /<run_id> ./local-path

Architecture:

- Reuses ``docsense-hf-cache`` (Qwen + Llama weights) and
  ``docsense-adapters`` (trained adapters) volumes — populated during
  training in ``train_lora_modal.py``.
- New ``docsense-eval-reports`` volume for output JSONs. The Modal
  function writes to ``/reports/<run_id>/<filename>.json``; user
  downloads via ``modal volume get`` after the run completes.
- Sequential generator → judge load pattern (same as local). A10G
  could hold both 7-8B NF4 models simultaneously, but sequential
  matches the local code path and avoids surprise OOMs on outlier
  long sequences.
- Imports the existing ``run_one_eval_set`` from the local script
  via importlib (same pattern the test suite uses) — single source
  of truth for the eval logic, no code duplication.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import modal

# Image — full eval dependency set. Heavier than train_lora_modal.py
# because eval also needs the retrieval stack (faiss-cpu, rank-bm25,
# sentence-transformers) and structlog for the eval driver's logging.
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
        "faiss-cpu>=1.8.0",
        "rank-bm25>=0.2.2",
        "pydantic>=2.6.0",
        "pydantic-settings>=2.2.0",
        "structlog>=24.1.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "huggingface-hub>=0.20.0",
    )
    # Mount the indexed corpus — required for the retrieval phase.
    # data/index is gitignored (~30 MB); mounted at build time so the
    # function has access. ~2 sec to upload on incremental builds.
    .add_local_dir("data/index", "/data/index")
    # Mount scripts/ so we can importlib-load run_generation_eval.
    # Single source of truth for the eval orchestration logic.
    .add_local_dir("scripts", "/scripts")
    # Mount docsense package for all the library imports the eval
    # driver does (Settings, Generator, judge, retrieval, etc.).
    .add_local_python_source("docsense")
)

app = modal.App("docsense-eval", image=image)

# Volumes — reuse existing training infra + add a reports volume.
hf_cache_vol = modal.Volume.from_name("docsense-hf-cache", create_if_missing=True)
adapters_vol = modal.Volume.from_name("docsense-adapters", create_if_missing=True)
reports_vol = modal.Volume.from_name("docsense-eval-reports", create_if_missing=True)


@app.function(
    gpu="A10G",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/adapters": adapters_vol,
        "/reports": reports_vol,
        # The corpus index lives in /data/index via add_local_dir at
        # image build time; the eval driver expects it at the
        # ``data/index`` path relative to PROJECT_ROOT, so we cwd into
        # /data's parent below.
    },
    secrets=[modal.Secret.from_name("hf-token")],
    # 1 hour. Full eval is ~10-15 min on A10G; budget covers cold-start
    # cache misses and any retry on transient HF Hub errors.
    timeout=3600,
)
def evaluate(
    *,
    eval_set: str = "all",
    adapter_subdir: str | None = None,
    limit: int | None = None,
    strategy: str = "recursive",
) -> dict:
    """Run the same eval driver locally uses, write reports to volume.

    Parameters mirror the local script's CLI args:

    - ``eval_set``: "curated" / "structural" / "no-answer" / "all"
    - ``adapter_subdir``: directory within ``/adapters`` (e.g., "v1");
      None → baseline run with no adapter
    - ``limit``: cap each eval set to first N queries (smoke runs)
    - ``strategy``: chunking strategy ("recursive" by default)

    Returns the run_id and the in-volume report directory; caller
    downloads reports via ``modal volume get docsense-eval-reports
    /<run_id> ./local``.
    """
    import importlib.util
    import logging
    import os
    import shutil
    import sys

    from docsense.config import GenerationConfig, Settings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Pin HF cache to the persistent volume — the default cache loc
    # would land on the ephemeral container filesystem and re-download
    # 30+ GB of model weights every cold start.
    os.environ["HF_HOME"] = "/root/.cache/huggingface"

    # The eval driver expects ``data/index/<strategy>/`` relative to
    # PROJECT_ROOT (computed from the script's __file__). Modal mounts
    # the index at /data/index, so we set up a project-root-shaped
    # directory: /workspace with /workspace/data → /data and
    # /workspace/scripts → /scripts. cwd into /workspace so the
    # script's PROJECT_ROOT computation lands on the right path.
    workspace = Path("/workspace")
    workspace.mkdir(exist_ok=True)
    if not (workspace / "data").exists():
        (workspace / "data").symlink_to("/data")
    if not (workspace / "scripts").exists():
        (workspace / "scripts").symlink_to("/scripts")
    os.chdir(workspace)
    logger.info("Working directory: %s", os.getcwd())

    # Load the eval driver via importlib — same pattern the test suite
    # uses. Module-level code runs (sets up REPORTS_DIR, etc.); we
    # then call run_one_eval_set directly with our parameters.
    spec = importlib.util.spec_from_file_location(
        "run_generation_eval", "/scripts/run_generation_eval.py"
    )
    assert spec is not None and spec.loader is not None
    eval_driver = importlib.util.module_from_spec(spec)
    sys.modules["run_generation_eval"] = eval_driver
    spec.loader.exec_module(eval_driver)

    # Build settings; inject adapter_path if requested.
    settings = Settings()
    if adapter_subdir is not None:
        adapter_path = Path(f"/adapters/{adapter_subdir}")
        if not adapter_path.exists():
            msg = (
                f"Adapter directory not found at {adapter_path}. "
                f"Available subdirs in /adapters: "
                f"{sorted(p.name for p in Path('/adapters').iterdir() if p.is_dir())}"
            )
            raise FileNotFoundError(msg)
        # Some training runs nest the adapter in a versioned subdir
        # (e.g., /adapters/v1/v1/). Auto-descend one level if the
        # config + safetensors aren't directly inside.
        if not (adapter_path / "adapter_config.json").exists():
            nested = adapter_path / adapter_subdir
            if (nested / "adapter_config.json").exists():
                adapter_path = nested
                logger.info("Auto-descended into nested subdir: %s", adapter_path)
        settings.generation = GenerationConfig(
            **{**settings.generation.model_dump(), "adapter_path": adapter_path},
        )
        logger.info("Using adapter at: %s", adapter_path)
    else:
        logger.info("No adapter — baseline run (base model only)")

    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("/reports") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run ID: %s", run_id)
    logger.info("Reports → %s", output_dir)

    eval_sets_to_run = eval_driver.ALL_EVAL_SETS if eval_set == "all" else (eval_set,)
    for es in eval_sets_to_run:
        eval_driver.run_one_eval_set(
            es,
            settings=settings,
            strategy=strategy,
            limit=limit,
            run_id=run_id,
            output_dir=output_dir,
        )

    # Volume must be committed for the writes to be visible to
    # ``modal volume get`` from the local CLI. Without this, reports
    # exist inside the container but aren't externally downloadable.
    reports_vol.commit()
    logger.info("Volume committed; reports downloadable via:")
    logger.info(
        "  modal volume get docsense-eval-reports /%s ./local-path",
        run_id,
    )

    # Mirror the eval-runs/ side artifacts (per-query intermediate
    # state) too, in case the operator wants to inspect. Optional —
    # skip if absent (e.g., if a future change removes that subdir).
    eval_runs_src = workspace / "evaluations" / "runs" / run_id
    if eval_runs_src.exists():
        shutil.copytree(eval_runs_src, output_dir / "intermediate", dirs_exist_ok=True)
        reports_vol.commit()

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "adapter_subdir": adapter_subdir,
        "eval_sets": list(eval_sets_to_run),
    }


# --- Local entrypoints ---------------------------------------------


def _print_download_hint(run_id: str, label: str) -> None:
    """Standardize the post-run output so the operator knows the next step."""
    print()
    print(f"=== {label} complete ===")
    print(f"  run_id: {run_id}")
    print()
    print("Download reports to local:")
    print(f"  modal volume get docsense-eval-reports /{run_id} ./tmp/eval-{run_id}")


@app.local_entrypoint()
def smoke() -> None:
    """3 queries per eval set, validates Modal eval infra. ~2-3 min compute.

    Uses the v1 adapter so we exercise the full pipeline end-to-end
    (base model load + adapter load + judge model load). Cheap insurance
    before paying for full baseline + adapter runs.
    """
    result = evaluate.remote(eval_set="all", limit=3, adapter_subdir="v1")
    _print_download_hint(result["run_id"], "Smoke")


@app.local_entrypoint()
def baseline() -> None:
    """Full eval, no adapter. Re-runs the pre-Phase-3 baseline shape on
    Modal A10G so adapter comparisons are same-hardware-fair.

    ~10-15 min wall, ~$0.20-0.25 compute.
    """
    result = evaluate.remote(eval_set="all", adapter_subdir=None)
    _print_download_hint(result["run_id"], "Baseline")


@app.local_entrypoint()
def adapter(subdir: str = "v1") -> None:
    """Full eval with adapter loaded.

    ``subdir`` selects from ``/adapters/<subdir>/`` on the Modal volume
    (default "v1" matches the Block 3C.3 training output). Future
    Block 3D iterations would use "v1.1", "v1.2", etc.
    """
    result = evaluate.remote(eval_set="all", adapter_subdir=subdir)
    _print_download_hint(result["run_id"], f"Adapter eval ({subdir})")
