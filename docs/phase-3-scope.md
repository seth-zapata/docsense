# Phase 3 — QLoRA fine-tuning of Qwen 2.5 7B Instruct

> **Pre-Phase-3 baseline** is at
> [`evaluations/baselines/pre_phase3_generation_base.json`](../evaluations/baselines/pre_phase3_generation_base.json).
> Methodology log:
> [`evaluations/analyses/2026-05-06-baseline-generation-eval.md`](../evaluations/analyses/2026-05-06-baseline-generation-eval.md).
> The numbers below are the targets Phase 3 is designed to move.

## Goal — single objective with multi-objective guardrails

Fine-tune Qwen 2.5 7B Instruct on the docsense corpus to measurably
improve **citation behavior** while preserving the strong baseline
behaviors (faithfulness, refusal correctness).

| Metric | Pre-FT baseline | Phase 3 target | Constraint type |
|---|---:|---:|---|
| **Citation rate** (`[N]` markers in answer text) | ~48% | **≥ 80%** | headline win |
| **Refusal correctness** (cross-validated, n=25) | 100% | **100%** | hard floor |
| **Faithfulness** (claim-level, mean) | 0.89 | ≥ 0.85 | tolerance band |
| **Relevance** (5-anchor, mean) | 0.76 | ≥ 0.70 | tolerance band |
| **Generate p50 latency** | 12-16 s | within +5% | LoRA adapter overhead |
| **`agreement_rate`** (refusal judge ↔ rule) | 1.000 | ≥ 0.95 | drift canary |

The objective ranking is intentional. Citation rate is the headline
target the pre-Phase-3 baseline surfaced (~48% vs the 100% the prompt
directive requests). Refusal correctness is the hard floor —
fine-tuning that improves citations at the cost of weakening the
"answer-only-from-context" discipline is a regression in disguise.
Faithfulness and relevance are "don't materially regress" guardrails
with explicit tolerance bands. The `agreement_rate` between refusal
judge and rule is the canary the cross-validation methodology
(established in PR C.1) was built for — a drop indicates the fine-tune
shifted phrasing in a way one of the methods missed.

## Why QLoRA, why this stack

**QLoRA** = Quantized LoRA: load base model at NF4 4-bit, attach low-rank
adapter matrices to attention projections, freeze base weights, train
only the adapters. Final artifact is a ~50 MB adapter file you load on
top of the 15 GB base.

Three reasons it's the right tool:

1. **Hardware fit.** RTX 4070 has 12 GB VRAM. Base model NF4
   (~5-6 GB) + LoRA adapters (~50 MB) + 8-bit AdamW optimizer states
   (~100 MB) + activations (with gradient checkpointing, ~1-2 GB at
   batch=1, seq=2048) ≈ ~9-10 GB peak. Native fp16 full-FT of 7B
   needs ~14 GB just for weights. **QLoRA is the only realistic
   local path.**
2. **Adapter portability.** ~50 MB artifact loadable on top of base
   weights. Versioning is trivial; comparing fine-tunes is just
   loading different adapter files. No 15 GB model copies.
3. **Industry standard for narrow-skill fine-tuning.** Citation
   behavior is a relatively narrow skill — model needs to learn
   "after each factual sentence, output `[N]`". LoRA at rank 8-32
   is plenty.

Stack:
- **`peft`** (HuggingFace PEFT) for LoRA mechanics
- **`trl.SFTTrainer`** for the supervised fine-tuning loop
- **`bitsandbytes`** for NF4 quantization + 8-bit paged AdamW
- **`accelerate`** under the hood for device management
- **Modal** for cloud-GPU execution (A10G, 24 GB, ~$1.10/hr)
- **Anthropic Claude (Sonnet)** for distilled training-data generation

## Block structure (5 blocks, ~3-5 PRs)

### Block 3A — Training infrastructure scaffolding (closed 2026-05-06)

Built the moving parts, tested with synthetic dataset + mocked model loads.

**3A.1 — Dataset shape + config** ✅
- `src/docsense/finetuning/dataset.py`: `TrainingExample` typed
  contract, `TrainingDataset` with stratified train/val split,
  citation-grounding model_validator, JSON round-trip via
  `from_json`/`to_json`.
- `src/docsense/finetuning/config.py`: `FineTuningConfig` with
  pinned defaults (rank=16, alpha=32, lr=2e-4, num_epochs=3,
  grad_accumulation=8 to compensate for batch=1).

**3A.2 — Trainer wrapper + CLI** ✅
- `src/docsense/finetuning/trainer.py`: `LoRAFineTuner` wrapping
  PEFT model prep + LoRA config + TRL SFTTrainer with
  `assistant_only_loss=True` and NF4 4-bit quantization defaults.
- `scripts/train_lora.py`: local CLI entry point. Modal-aware
  driver lives in 3C.

### Block 3B — Training dataset construction (closed 2026-05-08)

The original two-bullet decomposition was correct in spirit; the
realized work decomposed further once the pipeline complexity became
clear. Final shape:

**3B.1 — Distillation pilot** ✅ (4-way model comparison)
- `scripts/build_training_dataset.py` with `--limit N` and
  `--enable-thinking` flags.
- Pilot: 6 queries × 4 model variants (Haiku 4.5 / Sonnet 4.5 /
  Sonnet 4.5 + thinking / Opus 4.7) at $0.45 total. Picked
  **Sonnet 4.5 with prompt v2** based on citation density (26
  markers, highest), correctness on prerequisite-rich procedurals,
  and absence of URL hallucination.
- Comparison artifact: `evaluations/datasets/training/pilot/comparison.md`.

**3B.2 plan doc** ✅ → [`phase-3-block-3b2-plan.md`](phase-3-block-3b2-plan.md)
- Two-stage architecture (corpus → query pool → distilled answers)
- 5 validated decisions including a mini-pilot for Haiku-vs-Sonnet
  on query generation (Haiku won, quality wash at 3.5× cheaper)

**3B.2.a — Heuristic chunk-affinity classifier** ✅
- `src/docsense/finetuning/chunk_classifier.py`: classifies chunks
  as procedural / comparison / best-practice / pointer (multi-label).
- 53 unit tests covering positive/negative examples per detector,
  multi-label combinations, real-chunk parity tests.

**3B.2.b — Type-aware query generator** ✅
- `src/docsense/finetuning/query_generation.py`:
  `TypeAwareQueryGenerator` wrapping Haiku 4.5 with one prompt
  template per type, `GeneratedQuery` pydantic model.
- 29 unit tests with stubbed Anthropic client.

**3B.2.c — Query pool quality filters** ✅
- `src/docsense/finetuning/query_filters.py`:
  `filter_by_length`, `filter_duplicates` (type-stratified embedding
  dedupe), `filter_eval_contamination` (cross-type, vs eval queries).
- 21 unit tests with deterministic mock embedder.

**3B.2.d — Refusal topic seeds** ✅
- `evaluations/datasets/training/refusal_topic_seeds.json`: 30
  hand-curated topics (10 adjacent_ml, 12 general_cs, 8 unrelated).
- `src/docsense/finetuning/refusal_seeds.py`: typed loader with
  `Literal` category validation. 16 tests.

**3B.2.e — Stage 1 query-pool seeder** ✅
- `scripts/build_query_pool.py`: end-to-end Stage 1 driver tying
  3B.2.a-d together. Resumable via per-task JSONL append.
  Synthesizes retrieval-failure refusals by pairing in-corpus
  queries with chunks from a different doc family.
- 27 unit tests.

**3B.2.f — Distillation script JSONL adapter** ✅
- `scripts/build_training_dataset.py` extended to consume both pilot
  JSON arrays (3B.1 hand-curated) and pool JSONL entries (3B.2.e
  output). Pool-only fields (`question_type`, gen metadata) propagate
  to `TrainingExample.metadata` for end-to-end provenance.
- 6 new tests, 24 existing preserved.

**Quality fixes** (post-first-run findings) ✅
- Refusal generation at `temperature=0.7` for diversity (in-corpus
  stays at 0 for resumability). Eliminated 67% within-seed dedupe
  observed at temperature=0.
- `filter_by_type_shape` step: dropped 16.8% of generated queries
  whose phrasing didn't match their assigned type. Pointer was the
  worst offender (53 dropped) — confirms chunk-classifier
  pointer-affinity heuristic is over-eager.
- Distill parser plain-text refusal escape hatch: Sonnet sometimes
  drops the JSON wrapper when refusing. Recovered ~14% of procedural
  queries that were silently being lost as "parse failures".

**Pre-flight audit (PR A)** ✅
- `_process_one_input` wrapper catches both `RuntimeError` (parse
  failure) and `ValueError` (citation hallucination from
  `TrainingExample` validator). Without this, a single
  hallucinated citation would crash the run and lose all prior
  progress.
- Per-example resumability: each successful TrainingExample appends
  to `<output>.raw.jsonl` immediately. On restart, cache key is
  `(query, chunk_ids)` so changed inputs correctly invalidate.
- Caught 8 of 599 distillation failures cleanly during the actual
  run (5 citation hallucinations, 3 plain-text non-refusal answers).

**Stage 1 actual run** ✅ — `evaluations/datasets/training/query_pool/`
- 599 query-pool entries (508 in-corpus + 92 refusal pool entries:
  53 off-corpus + 39 retrieval-failure)
- $0.31 actual cost (v1 + v2 regen for refusal temperature fix)
- Filter pass-through: length 0 dropped, type-shape 136 dropped,
  dedupe 66 dropped, eval-contamination 9 dropped (810 → 599)

**Stage 2 actual run** ✅ — `evaluations/datasets/training/training_dataset.json`
- 591 distilled training examples (349 in-corpus + 242 refusal)
- 41% natural refusal rate after distillation (Sonnet correctly
  refuses when retrieved chunks don't fully cover the question;
  this is the production failure mode we want Qwen to learn)
- $3.87 actual cost vs $4.16 projection
- 30 min wall time on Modal-free local Anthropic API path
- 8 losses (1.3%) all caught cleanly by PR A's safety net

**Total Block 3B spend:** ~$5.00 of the $10 distillation budget.
Full arc and lessons learned:
[`docs/journal/2026-05-08-block-3b-distillation-close.md`](journal/2026-05-08-block-3b-distillation-close.md).

### Block 3C — First training run + eval (Modal cloud)

> Detailed sub-block plan: [`phase-3-block-3c-plan.md`](phase-3-block-3c-plan.md)
> (5 sub-blocks 3C.1 through 3C.5, including the dataset-size pain
> point captured in "Open questions / followups").

- Add Modal-aware training entry point (`scripts/train_lora_modal.py`)
- Modal config: A10G GPU, persistent volume for HF model cache
  (avoids re-downloading 15 GB Qwen weights per run), HF token via
  Modal Secrets.
- First training run: ~30-45 min wall on A10G (~$0.55-0.85).
- Save adapter to `models/fine-tunes/qwen-docsense-v1/` (committed).
- Add `GenerationConfig.adapter_path` field; eval driver loads
  adapter on top of base for the fine-tuned eval.
- Run full eval (curated + structural + no-answer) against fine-tuned
  model. Compare against pre-Phase-3 baseline.
- Commit fine-tune report + analysis under
  `evaluations/analyses/2026-05-XX-phase3-v1-finetune-eval.md`.

### Block 3D — Iteration (sequential informed sweep)

Likely 1-3 cycles. Per cycle:
- Inspect loss curves + eval deltas from prior run
- Tune one variable (LoRA rank, learning rate, num_epochs, dataset
  size)
- Train + eval
- Commit results

Stop conditions per Decision 4 (success criterion):
- Hit citation rate ≥ 80% with refusal=100% and faithfulness ≥ 0.85
  (ship as v_final), OR
- 3 attempts without hitting target → ship best-of and document gap
  as Phase 4 follow-up.

### Block 3E — Phase 3 closeout

- Pick best fine-tune as production default. Update
  `GenerationConfig.adapter_name` to point at it.
- Update `CLAUDE.md`, `docs/architecture.md`,
  `evaluations/performance.md` with closed-phase status + new
  baseline numbers.
- Add Phase 3 journal entry capturing the arc.

## Locked decisions

All ten decisions confirmed before scoping. Captured here so the
rationale survives the discussion thread.

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Distillation source | **Anthropic API (Claude Sonnet)**, ~$6-10 one-time | Industry-standard distillation pattern; reproducible script artifact for portfolio |
| 2 | Training infrastructure | **Modal cloud** (with $30 free credits + $10 user-added; A10G at $1.10/hr) | Local works but locks dev machine for 5-8 hr per run; cloud cuts to 30-45 min |
| 3 | Dataset size | **~500-800 in-corpus + 50-100 refusal** | Citation is a narrow skill; bigger doesn't help much for LoRA |
| 4 | Success criterion | **A (hit citation target) with B (best-of-3) fallback** | Avoids "perfect the fine-tune indefinitely" trap |
| 5 | Hyperparameter approach | **B (sequential informed sweep)** | Loss curves are highly informative; spend less compute, more on understanding |
| 6 | OUT_OF_RANGE issue | **Defer to Phase 4** | Judge-side infrastructure, not generator fine-tuning |
| 7 | `--adapter` flag in eval driver | **Part of Block 3C** | No value before we have a real adapter |
| 8 | Refusal examples in training | **Yes**, ~50-100 off-corpus + canonical refusal | Guardrail against catastrophic forgetting; cheap insurance |
| 9 | Training query source | **Re-run `generate_structural_queries.py` with new seed + new doc-heading sample** | Disjoint from eval queries (critical for measurement integrity); reproducible |
| 10 | Anthropic key paste | **Done** (file at `~/.anthropic-key`, mode 600) | Paralleled the HF token flow |

## Cost & infra budget

| Component | Estimated spend | Available credit | Net cost |
|---|---:|---:|---:|
| Anthropic distillation (one-time) | ~$6-10 | $10 user-added | **~$0** |
| Modal cloud GPU (3-5 training runs) | ~$3-5 | $30 free | **~$0** |
| **Phase 3 total** | ~$10-15 | $40 | **effectively $0 net** |

The credit pools comfortably cover Phase 3 with margin. The portfolio
narrative remains "consumer-hardware fine-tuning" — the only
non-local pieces are the API distillation step (script artifact in
the repo, reproducible) and cloud training execution (Modal config
in the repo, runnable by anyone with their own Modal account).

## Risks

Honest list, ordered by likelihood × impact:

1. **Catastrophic forgetting / refusal regression.** Fine-tuning to
   add citations might weaken Qwen's "answer-only-from-context"
   discipline, dropping refusal rate below the 100% baseline.
   - **Mitigation**: include ~50-100 refusal examples in training
     (Decision 8); watch `agreement_rate` between refusal judge and
     rule as the early-warning canary; refusal=100% is a hard floor
     for what we'll ship.
2. **Distilled-answer quality risk.** If Claude's "ideal" answers
   contain subtle errors or stylistic biases, the fine-tune
   inherits them.
   - **Mitigation**: pilot 5-10 examples first, manual review, prompt
     iteration before scaling (Block 3B.1); spot-check ~20 random
     samples post-batch.
3. **Hardware limit on training-batch size.** 12 GB at NF4 + LoRA
   limits us to batch_size=1 with `gradient_accumulation_steps=8`.
   Effective batch=8 is fine but raw step count is high.
   - **Mitigation**: cloud A10G has 24 GB, supports batch_size=2 or
     4. Decision 2 already addresses this.
4. **Eval-judge dependence.** All Phase 3 conclusions depend on
   Llama 3.1 8B as judge. If the judge has subtle biases that differ
   pre-FT vs post-FT (e.g., recognizes its own generation style
   differently), Δ-comparisons could mislead.
   - **Mitigation**: cross-validation methodology (refusal: judge +
     rule + agreement_rate). AnthropicJudge calibration follow-up if
     this becomes load-bearing.
5. **Methodology lock-in.** Once Phase 3 ships, the eval methodology
   is effectively frozen for the duration of the comparison. Any
   further methodology changes invalidate the pre/post-FT diff.
   - **Mitigation**: PR C.2 already closed the methodology cleanup.
     We're shipping Phase 3 against a defensible baseline.

## Reproducibility — committed artifacts

Everything material to the Phase 3 result lives in the repo:

- **Scope doc**: this file
- **Training dataset**: `evaluations/datasets/training/v1.json` —
  full queries + chunks + ideal answers (~5-10 MB)
- **Training queries source**: regenerated structural queries with
  pinned seed; commitment in the dataset manifest
- **Distillation script**: `scripts/build_training_dataset.py`
  (Anthropic API key read from `~/.anthropic-key`, not committed)
- **Training config**: `FineTuningConfig` defaults pinned in code +
  per-run config snapshot serialized into the report JSON
- **Training script**: `scripts/train_lora.py` (local) +
  `scripts/train_lora_modal.py` (cloud)
- **Adapter weights**: `models/fine-tunes/qwen-docsense-v1/` —
  committed (LoRA adapters are ~50 MB, well under git LFS threshold
  for typical repos but we'll use git-lfs if needed)
- **Eval reports**: `evaluations/reports/generation-<run-id>-<set>-finetune-v<N>.json`
- **Analysis**: `evaluations/analyses/2026-05-XX-phase3-v<N>-finetune-eval.md`
- **Modal config**: in `scripts/train_lora_modal.py` (image, volume,
  GPU type all declared in code)

What's *not* committed (intentionally):
- Anthropic API key (`~/.anthropic-key`, gitignored location)
- HF model cache (`/mnt/e/hf-cache`, ~31 GB)
- Modal account credentials (`~/.modal.toml`)
- Per-developer training run logs (`logs/`)

## What this is NOT (scope guard)

The conservative bias, made visible:

- **Not RLHF / DPO / PPO.** Reward modeling and preference-based
  fine-tuning are Phase 4 territory if at all. SFT is the right
  starting point for a single-skill behavior fix.
- **Not full-precision fine-tuning.** LoRA adapters only. A
  full-FT comparison would be interesting but doesn't fit on the
  hardware and isn't necessary for the citation skill.
- **Not domain pre-training or continued pre-training.** We're
  fine-tuning chat behavior, not adding new knowledge. Qwen
  already has strong baseline knowledge of HF Transformers.
- **Not multi-task fine-tuning.** Single skill: citation rate.
  Trying to simultaneously improve citation + faithfulness +
  relevance + refusal would be unfocused. Refusal preservation is a
  *constraint*, not an *objective*.
- **Not constrained-decoding for the judge** (the OUT_OF_RANGE
  remediation). That's judge-side infrastructure, Phase 4 work.
- **Not multi-judge calibration** (AnthropicJudge follow-up). Still
  deferred to "if needed in Phase 3 measurement and Llama-judge
  bias becomes load-bearing".

## Resuming from this doc

When picking up Phase 3 work in a future session:

1. Read this file first.
2. Check `git log --oneline` for which Block 3A-E commits already
   landed; the last completed block is your starting line.
3. Look at the unstaged tree — there should be no work in progress
   from a prior session unless an in-flight commit was abandoned.
4. Confirm with the user before starting the next block. The
   expectation is **explicit approval per block**, not autonomy
   across the full plan. Same as Phase 2.
5. Update this scope doc only if the user changes the agreed scope.
   Routine implementation does not modify it.

When closing Phase 3: update `CLAUDE.md` and `docs/architecture.md`
in a single commit (per the rule in `CLAUDE.md`'s Roadmap section)
and add a journal entry. This scope doc remains as-is until Phase 4
planning starts, at which point it gets superseded by
`docs/phase-4-scope.md` or appended.
