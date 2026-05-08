# Block 3C — Training run + eval

> Builds on the [Phase 3 scope](phase-3-scope.md). Block 3B closed
> 2026-05-08; we have `evaluations/datasets/training/training_dataset.json`
> with 591 examples (349 in-corpus + 242 refusal). This doc is the
> next-step plan: take the dataset, fine-tune a QLoRA adapter on Modal
> A10G, evaluate against the pre-Phase-3 baseline, and decide whether
> to ship or iterate.

## Goal

Hit the metric targets locked in [`phase-3-scope.md`](phase-3-scope.md):

| Metric | Pre-FT baseline | Phase 3 target | Constraint type |
|---|---:|---:|---|
| Citation rate (`[N]` markers) | ~48% | **≥ 80%** | headline win |
| Refusal correctness (cross-validated, n=25) | 100% | **100%** | hard floor |
| Faithfulness (claim-level mean) | 0.89 | ≥ 0.85 | tolerance band |
| Relevance (5-anchor mean) | 0.76 | ≥ 0.70 | tolerance band |
| Generate p50 latency | 12-16 s | within +5% | LoRA adapter overhead |
| `agreement_rate` (refusal judge ↔ rule) | 1.000 | ≥ 0.95 | drift canary |

Citation rate is the load-bearing target. Everything else is a
guardrail — fine-tuning that improves citations at the cost of
weakening refusal discipline or hallucinating facts is a regression
in disguise.

## Sub-block plan

### 3C.1 — Local trainer smoke test

Verify `LoRAFineTuner` can ingest `training_dataset.json` and produce
a valid adapter end-to-end before paying for Modal compute.

- Run on a tiny slice (~10 examples, 1 epoch) on the local RTX 4070
- Catches: dataset format mismatches with TRL's SFTTrainer, tokenizer
  / chat-template surprises, OOM patterns
- Cost: $0 (local). Wall time: ~5 min.

The 12 GB VRAM local box is tighter than Modal's 24 GB A10G; if it
runs locally on a slice, Modal at scale is comfortable.

### 3C.2 — Modal infrastructure

Add `scripts/train_lora_modal.py` plus a Modal app definition.

- A10G GPU (24 GB), `gpu="a10g"`
- Persistent volume for HF model cache to avoid re-downloading the
  ~15 GB Qwen weights on every cold-start invocation
- HF_TOKEN via Modal Secrets (paralleling the Anthropic-key flow)
- Wrap `LoRAFineTuner.train()` as a Modal function, stream logs back
  to local for visibility
- Save adapter to a Modal volume; download to local
  `models/fine-tunes/qwen-docsense-v1/` after run completes

First-time Modal setup is the friction point — auth, image build,
secret management. Allocate ~30-60 min for it; bake into 3C.2 timing.

### 3C.3 — Training run #1 (default hyperparameters)

Execute on Modal A10G with the locked defaults from `FineTuningConfig`:

- `lora_rank=16`, `lora_alpha=32`, `lora_dropout=0.05`
- `learning_rate=2e-4`, `num_train_epochs=3`, `warmup_ratio=0.03`
- `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`
  → effective batch 8
- `bf16=True`, `optim="paged_adamw_8bit"`, NF4 4-bit quantization

~30-45 min wall time on A10G, ~$0.55-0.85 compute cost. Save adapter
to `models/fine-tunes/qwen-docsense-v1/` (committed for reproducibility
— ~50 MB at rank=16).

**Capture:** training loss curve (per-step), final eval loss on the
held-out 10% val split, training metrics dict from `LoRAFineTuner.train()`.

### 3C.4 — Adapter loading in inference path

Wire the trained adapter into `Generator`:

- Add `GenerationConfig.adapter_path: Path | None`
- After base model load, call PEFT's `model.load_adapter()` if
  `adapter_path` is set
- Smoke test: generate one answer with adapter loaded, confirm output
  shape matches `Answer` schema

Adapter-loading order with NF4 quantization has gotcha potential — PEFT
+ bitsandbytes can be order-sensitive. Following PEFT docs precisely;
the smoke test catches any quantization-vs-adapter ordering issues
before 3C.5 spends compute on the full eval.

### 3C.5 — Full eval against pre-Phase-3 baseline

Re-run `scripts/run_generation_eval.py` with `--adapter-path` pointing
at the trained adapter. Use IDENTICAL eval queries as the baseline
(curated + structural + no-answer) — the comparison is meaningless
otherwise.

**Validity guard:** assertion in the eval script that the loaded
query set's count + first-3-queries identity match the baseline's.
Catches accidental query-set drift before metrics are reported.

Compare per-metric deltas vs `evaluations/baselines/pre_phase3_generation_base.json`.
Commit the report at `evaluations/analyses/2026-05-XX-phase3-v1-finetune-eval.md`.

**Decision point — did we hit the success criterion?** Per Decision 4
in [`phase-3-scope.md`](phase-3-scope.md):

| Outcome | Action |
|---|---|
| Citation rate ≥ 80% AND refusal = 100% AND faithfulness ≥ 0.85 | Ship as `v_final`, skip Block 3D |
| Citation rate moves but misses target | Block 3D first cycle: tune one variable (lr, rank, epochs) |
| Refusal correctness drops below 95% | Block 3D first cycle: catastrophic-forgetting recovery (more in-corpus signal, lower lr) |
| All metrics flat | **See "dataset-size pain point" below** — possibly need more data, not hyperparameter changes |

## Validated decisions (locked from phase-3-scope.md)

| Decision | Choice | Rationale |
|---|---|---|
| Training infrastructure | Modal A10G | Local RTX 4070 works (~5-8 hr/run) but locks dev machine; Modal cuts to ~30-45 min, $30 credit covers it |
| Hyperparameter approach | Sequential informed sweep (Block 3D, max 3 cycles) | Loss curves are highly informative; small compute, more understanding |
| Success criterion | Hit citation target with best-of-3 fallback | Avoids "perfect the fine-tune indefinitely" trap |
| Adapter committed to repo | YES | ~50 MB; full reproducibility for portfolio |
| Eval set | Curated + structural + no-answer (same as baseline) | Comparison validity demands identity |

## Cost & timeline

| Step | Cost | Wall time |
|---|---:|---:|
| 3C.1 (local smoke) | $0 | ~5 min |
| 3C.2 (Modal setup, including first-time auth + image build) | $0 (image build is free) | ~30-60 min |
| 3C.3 (training run #1, A10G) | ~$0.55-0.85 | ~30-45 min |
| 3C.4 (adapter loading + smoke) | $0 | ~10 min |
| 3C.5 (eval — uses local Llama judge) | $0 | ~30 min |
| **Block 3C total** | **~$0.85** | **~2-2.5 hr active** |

Cumulative across the entire 3B + 3C series:

| Series | Cost |
|---|---:|
| 3B (distillation) | ~$5.00 |
| 3C (this) | ~$0.85 |
| **Cumulative** | **~$5.85** |

Well under the $10 distillation budget AND the $30 Modal credit pool.

## Implementation surface

```
scripts/
  train_lora_modal.py          # NEW — Modal-aware training driver

src/docsense/
  generation/config.py         # EXTEND — add adapter_path field
  generation/generator.py      # EXTEND — adapter loading after base
  finetuning/trainer.py        # exists; verify its load path matches
                               # Modal's expected file layout

models/fine-tunes/
  qwen-docsense-v1/            # NEW — adapter weights + config
    adapter_config.json
    adapter_model.safetensors

evaluations/
  analyses/2026-05-XX-phase3-v1-finetune-eval.md  # NEW — eval writeup

tests/unit/
  test_generation_config.py    # EXTEND — adapter_path field
  test_generator_adapter.py    # NEW — adapter loading smoke test
```

## Risks (ordered by likelihood × impact)

1. **Adapter-loading interaction with NF4 quantization.** PEFT +
   bitsandbytes ordering matters. Mitigation: 3C.4 smoke test isolates
   this before the full eval pays for it.

2. **Modal first-time setup friction.** Could eat 30-60 min on auth,
   image build, secret config. Not a technical risk so much as a
   timing one — bake into 3C.2 budget.

3. **Eval comparison invalidity.** If the eval script accidentally
   loads a different query set than the baseline, the comparison
   is meaningless. Mitigation: explicit identity assertion (count
   + first-3-queries match) in the eval script.

4. **Catastrophic forgetting on refusals.** Risk #1 from the Phase 3
   scope. Mitigation: refusal correctness measured directly; if it
   drops below 95%, Block 3D's first cycle is the recovery path.

5. **Training instability on a small dataset.** 591 examples is
   on the smaller side for SFT — see "dataset-size pain point" below.

## Open questions / pain points

### Dataset size — 591 enough for narrow LoRA on an instructed base?

The empirical sweet spot for narrow-skill LoRA on already-instructed
bases is 500-1000 examples. We're at 591 — defensible but on the
lower edge. Reasoning for proceeding without adding more data first:

- Qwen 2.5 7B Instruct already has the RAG-style answering circuits
  from its base instruction tuning. We're nudging existing capability
  toward our preferred output format (cite with `[N]`) and decision
  rule (refuse on coverage gaps) — not teaching a new capability.
- LoRA's low-rank constraint (rank=16, ~50 MB adapter on 15 GB base)
  is itself a regularizer. The bigger risk is *underfitting* (no
  behavior change), not overfitting.
- Training is cheap (~$0.85, ~45 min). The information value of "train
  and check" exceeds "speculate and add data preemptively" — the
  former either confirms 591 works (saving the data-gen cost) or
  tells us exactly what to add.

**Detection signals at 3C.5 that would indicate 591 wasn't enough:**

| Signal | Likely diagnosis | Block 3D first-cycle action |
|---|---|---|
| Training loss plateaus high or oscillates | Convergence issue — could be data OR hyperparameters | Try lr=1e-4 or 5 epochs first; data-volume only if hyperparameter sweep is flat |
| Citation rate moves < 10pp (e.g., 48% → 55%) | Likely hyperparameter issue | Sweep lr / rank / epochs |
| Citation rate moves < 5pp | Likely data-volume issue | Generate ~400 more examples, retrain |
| Refusal correctness drops below 95% | Catastrophic forgetting | Lower lr, more in-corpus signal in next dataset cycle |
| All metrics dead flat from baseline | Something fundamental broken | Check adapter-loading wiring, training config |

If 3D's first cycle adds data: the cost is ~$5 (re-run Stage 1 with
a different seed → fresh queries from different chunks → distill).
~30-60 min wall time. Within the budget if it's needed.

The deliberate choice here is to spend $0.85 of compute to find out,
rather than $5 of distillation as a hedge against a problem that may
not exist.

### Adapter precision — fp16 or bf16 for inference?

Training uses bf16 (NF4 compute_dtype matches A10G/4070 native
support). Inference Generator currently loads at fp16. Adapter weights
saved in bf16 might need an explicit cast at load time. Test in 3C.4
smoke; if bf16 inference works on the inference hardware, leave it
matching training. If not, fp16 cast at load.

### Modal volume layout for HF cache

The HF cache organizes by model_id; persistent volume should mount at
`/root/.cache/huggingface` (the default). One detail to verify: the
volume must be writable on first download and read-only on subsequent
runs to avoid race conditions if multiple training runs ever execute
concurrently. Single-run usage in 3C is fine; flag for Block 3D if
parallel sweeps come up.

### Adapter version naming

Naming convention: `qwen-docsense-v1` for the first run. If 3D iterates
and produces multiple adapters, increment minor: `v1.1`, `v1.2`, etc.
Block 3E picks the best as `v_final` and that becomes the production
default in `GenerationConfig.adapter_name`.
