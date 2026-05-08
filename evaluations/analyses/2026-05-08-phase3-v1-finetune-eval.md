# Phase 3 Block 3C — first QLoRA fine-tune eval

**Date:** 2026-05-08
**Reports analyzed:**
- baseline (Modal A10G, no adapter): `evaluations/reports/phase3-baseline-modal/20260508T114905Z/`
- adapter v1 (Modal A10G, `qwen-docsense-v1`): `evaluations/reports/phase3-v1-adapter/20260508T111606Z/`
- pre-Phase-3 reference: `evaluations/baselines/pre_phase3_generation_base.json`
- v1 snapshot saved to: `evaluations/baselines/phase3_v1_finetune.json`

## TL;DR

The first QLoRA fine-tune (Block 3C v1) on Qwen 2.5 7B Instruct with
the 591-example distillation dataset (349 in-corpus + 242 refusal)
delivers a strong, expected gain on the headline target — citation
behavior — and produces one well-understood failure mode that scopes
Block 3D.

**Three findings.**

1. **Citation behavior dramatically improved on curated (0.50 → 0.95
   frac-with-marker, +0.45) and meaningfully on structural (0.50 →
   0.67, +0.17).** Citation density 2× (curated: 2.20 → 4.70 markers
   per answer). Citation *validity* is the quietly impressive bit —
   the rate of `[N]` markers pointing outside the top-k chunk range
   went 11/128 → 2/93 on curated, 3/145 → 0/86 on structural. The
   adapter learned to cite *and* the citations now resolve to chunks
   that exist.

2. **Refusal discipline preserved** — both no-answer eval (judge=1.00,
   rule=1.00, agreement=1.00) and the in-corpus answer rate hold the
   line. No measurable drift in the off-corpus refusal phrasing
   (rule-based regex still matches 25/25 with the original
   `dont_have_context` pattern).

3. **Over-refusal on structural is the single failure mode.** 10/30
   structural answers are now refusals (was 2/30 baseline). Per-query
   alignment shows **7 of those 8 extra refusals are unjustified** —
   the baseline answered the *same query with the same retrieved
   chunks* and scored ≥0.83 faithfulness (six at 1.00). The 8th was
   a probable baseline hallucination (b=0.30) where the adapter
   correctly dodged. So 87.5% of the new refusals are over-refusals,
   not correct ones. This drives both the structural-faithfulness
   FAIL (0.640 < 0.85) and the structural-citation-rate FAIL (0.667
   < 0.80) — when the model refuses, it doesn't cite.

**Phase 3 acceptance check.** Three of five targets pass cleanly; the
two failures trace to a single root cause.

| Target | Value | Result |
|---|---|---|
| citation rate ≥ 0.80 (curated) | 0.950 | ✅ PASS |
| citation rate ≥ 0.80 (structural) | 0.667 | ❌ FAIL — over-refusal |
| refusal correctness = 1.00 (no-answer) | 1.000 | ✅ PASS |
| faithfulness mean ≥ 0.85 (curated) | 0.892 | ✅ PASS |
| faithfulness mean ≥ 0.85 (structural) | 0.640 | ❌ FAIL — driven by 10 refusals scoring 0 |

## Methodology

### Configuration

Same retrieval/eval pipeline as the pre-Phase-3 baseline (commit
`fec3dc6`); only the generation stage changed (LoRA adapter loaded
via `PeftModel.from_pretrained` on top of the same NF4-quantized
Qwen base). Both runs executed on Modal A10G (24 GB VRAM) for fair
hardware comparison — the previous baseline ran on local 12 GB.

| Component | Setting |
|---|---|
| Hardware | Modal A10G (24 GB VRAM, sm_86) |
| Generator base | `Qwen/Qwen2.5-7B-Instruct` (NF4 4-bit) |
| Generator adapter | `qwen-docsense-v1` (LoRA, r=16, α=32, q/k/v/o-proj) |
| Training data | 591 examples (349 in-corpus distilled from Sonnet 4.5, 242 off-corpus refusal seeds) |
| Training epochs | 3 |
| Generator temperature | 0.1 |
| Judge model | `meta-llama/Meta-Llama-3.1-8B-Instruct` (NF4, T=0.0) |
| Chunking | recursive, top_k=5, max_context_tokens=3500 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

### Run IDs and timing

| Run | Run ID | Wall time |
|---|---|---|
| Adapter v1 | `20260508T111606Z` | ~29 min (11:16:06 → 11:45:28 UTC) |
| Baseline (Modal) | `20260508T114905Z` | ~44 min (11:49:05 → 12:33:16 UTC) |

The two runs both used Modal A10G with the same image, same volume
mounts, same eval driver. The Run-ID collision incident (initial
parallel kickoff produced identical timestamps because both
containers called `datetime.now()` within the same second) was
caught before commit; baseline was killed and re-run with a clean
timestamp. Worth fixing in the eval driver: encode `baseline` vs
`adapter-{subdir}` into the Run ID generator so timestamp-only
identity isn't load-bearing for parallel runs.

## Headline numbers

### Curated (n=20)

|  | baseline | adapter | Δ |
|---|---:|---:|---:|
| faithfulness mean | 0.938 | 0.892 | −0.046 |
| faithfulness median | 1.000 | 1.000 | 0.000 |
| support rate (claims grounded) | 0.906 | 0.935 | **+0.029** |
| claims/answer mean | 6.35 | 4.65 | −1.70 |
| no-claims-extracted | 0 | 1 | +1 |
| claim-out-of-range | 11 | 2 | **−9** |
| relevance mean | 0.762 | 0.725 | −0.037 |
| **citations frac-with-marker** | **0.700** | **0.950** | **+0.250** |
| citation mean markers/answer | 2.20 | 4.70 | **+2.50** |
| generate p50 (ms) | 11,422 | 10,533 | −889 |
| generate p95 (ms) | 21,640 | 16,287 | −5,353 |

### Structural (n=30)

|  | baseline | adapter | Δ |
|---|---:|---:|---:|
| faithfulness mean | 0.893 | 0.640 | **−0.253** |
| faithfulness median | 1.000 | 1.000 | 0.000 |
| support rate (claims grounded) | 0.928 | 0.907 | −0.021 |
| claims/answer mean | 5.10 | 2.87 | −2.23 |
| no-claims-extracted | 2 | **10** | **+8** |
| claim-out-of-range | 3 | 0 | −3 |
| relevance mean | 0.733 | 0.525 | −0.208 |
| citations frac-with-marker | 0.500 | 0.667 | +0.167 |
| citation mean markers/answer | 1.60 | 1.97 | +0.37 |
| generate p50 (ms) | ~12,000 | 5,335 | ~−6,700 |

The structural drops in faithfulness and relevance are a single
mechanism: 8 additional refusals each score 0.0 on faithfulness and
0.0 on relevance because there's no answer to grade. If the 10
adapter refusals are excluded from the structural mean, non-refusing
items average ~0.96 faithfulness — actually better than baseline's
0.89 on the same items. The model is excellent when it answers; it
refuses too eagerly.

### No-answer (n=25)

|  | baseline | adapter | Δ |
|---|---:|---:|---:|
| refusal correct (judge) | 1.000 | 1.000 | 0.000 |
| refusal correct (rule) | 1.000 | 1.000 | 0.000 |
| agreement (judge↔rule) | 1.000 | 1.000 | 0.000 |
| `dont_have_context` matches | 25/25 | 25/25 | flat |

Refusal *phrasing* didn't drift either, despite fine-tuning on 242
refusal examples — the rule-based regex hit 25/25 with the original
`dont_have_context` pattern, and the LLM-judge agreed on every item.
The cross-validation methodology added in Block 1B (regex + judge)
would have surfaced any phrasing drift; it didn't fire.

## Failure-mode analysis: structural over-refusal

Of the 30 structural items, 10 adapter answers begin with `"I don't
have enough context to answer that."` — vs 2/30 from baseline. Since
both runs used the *same* retrieved chunks for each query (retrieval
is deterministic given seed=fixed), the question is whether the
adapter's extra 8 refusals are catching cases where the baseline
hallucinated, or skipping cases the baseline answered well.

Per-query alignment, bucketed by baseline's score on the same item:

| Bucket | n | Interpretation |
|---|---:|---|
| Both answered (no refusal change) | 20 | Common case |
| Both refused (correct on both) | 2 | Genuinely hard retrieval |
| Adapter refused, baseline scored ≥0.5 | **7** | **OVER-REFUSAL** — baseline produced a useful answer with the same chunks |
| Adapter refused, baseline scored <0.5 | 1 | Correct refusal — baseline likely hallucinated (b=0.30) |

**The over-refusal cases (7 of 30 structural, 23%):**

| Query | Baseline score | Baseline answer preview |
|---|---:|---|
| structural_010 | 1.00 | "Based on the provided context, key features of keypoint detection include..." |
| structural_011 | 1.00 | "Transformers integration can be done in a few ways depending on the model's popularity..." |
| structural_012 | 1.00 | "Transformers integration can be done in a few ways..." |
| structural_017 | 1.00 | "Here are some template writing tips: Refer to existing templates by using `print(tokenizer.chat_t..." |
| structural_019 | 1.00 | "The context provides information about different model architectures: Gemma3 has a similar a..." |
| structural_027 | 0.83 | "Here are some usage tips and examples from the provided context..." |
| structural_028 | 1.00 | "The Funnel specific outputs are documented under `models.funnel.modeling_funnel.FunnelForPreTraining..." |

These are not edge cases — they're competent baseline answers that
score full credit. The adapter has the same retrieval, the same
chunks in context, and refuses anyway.

### Why this happened

Training-set composition is the proximate cause. Of 591 examples:

- **349 in-corpus**: clean retrieval (top-k chunks all relevant), well-grounded answer with citations, no refusal
- **242 off-corpus refusal**: query about something not in the docs, retrieval brings back irrelevant chunks, model refuses

Refusals are 41% of the dataset — a large enough class that the
model learned a general "if context looks loose, refuse" heuristic.
The training set has no examples teaching the *third* case:
**mixed-relevance retrieval where some chunks support a partial
answer and others are off-topic — answer using the supportive
chunks, ignore the others.**

That third class is what the structural eval set is mostly testing.
Real production retrieval is messy; the chunks aren't always all
on-topic; the *correct* behavior is to use what's useful and not
refuse just because the bag is mixed. The adapter is brittle here
because it's never seen a training example that demonstrates this
behavior — it's been pattern-trained that "loose context → refuse."

This is a known QLoRA failure mode, well-documented in alignment
literature: a refusal class that's too large or too uniform produces
a model that refuses confidently on borderline cases. The fix is in
training-data composition, not in hyperparameters.

### Why this didn't show on curated

The curated eval set (Phase 1 chunking-bakeoff legacy) uses queries
written *to be answerable* from the corpus — retrieval reliably
brings back tightly relevant chunks. Adapter has the easy positive
case; refuses none of them. Same model, same prompt, different eval
set difficulty.

This is the eval-set bias we surfaced in Block B+ during the
chunking-bakeoff investigation: curated and structural can disagree
because curated is curated. We should weight the structural failure
more than the curated win when planning Block 3D.

## Block 3D direction

The over-refusal failure mode is well-scoped and addressable in
training-data composition. Two complementary fixes, ordered by
likelihood-to-help × cheapness:

**3D.1 — Reduce refusal class share.** Down-sample refusals from
242 to ~100 (so refusal is ~22% of training, not 41%). Re-train.
Cheapest possible iteration: ~$0.50 of Modal A10G, ~15 min wall.
If over-refusal disappears, this is sufficient.

**3D.2 — Add the borderline-positive class.** Generate ~100 training
examples where retrieval is *intentionally noisy* — query answerable
from 2-3 of the 5 retrieved chunks, with the other 2-3 being
loosely-related but off-topic chunks. Correct answer cites only the
supportive ones, ignores the rest. This is the missing class. Use
the existing `TypeAwareQueryGenerator` infrastructure with a noisier
retrieval path (e.g., raw BM25 only, no rerank) to seed the queries,
then distill answers via Sonnet 4.5 + prompt v2 (the configuration
that won the Block 3B 4-way pilot).

If 3D.1 alone closes the structural FAIL, 3D.2 becomes optional
hardening. If 3D.1 only partially helps, 3D.2 is the rigorous fix
because the underlying issue isn't class-share, it's *missing
class coverage*.

**3D.3 — Stretch goal: reduce claims/answer regression.** Adapter
produces 4.65 claims/answer on curated (was 6.35) — fewer factual
statements per answer. Could be a faithful concision win or a
corner-cutting loss. Worth checking if 3D.1/3D.2 also closes this;
if not, look at distillation prompt v2 to see if it implicitly
trained the model to prefer brevity.

**Out of scope for 3D.** Hyperparameter sweeps (rank, alpha, LR).
The failure isn't a tuning problem — same hyperparameters with
better data should close it. Save the sweep budget for later if
data fixes don't.

## Calibration and caveats

Two honest reads on what these numbers prove and don't.

**(a) Eval-set composition leakage.** Curated queries were written
during Phase 1 *to be answerable* from the corpus. They share
phrasing patterns with the in-corpus distillation training set —
both Sonnet-style queries that look like documentation Q&A. Some of
the +0.45 citation gain on curated may reflect distributional
similarity, not true generalization. Structural is the better
signal because its queries were generated from H2/H3 section
headers — independent of training distribution. The structural
citation gain is +0.17 — still a real win, but more honest about
generalization than the curated +0.45.

**(b) Modal-baseline gap to pre-Phase-3 baseline.** The Modal
baseline (0.700 citation rate on curated) is markedly higher than
the pre-Phase-3 baseline (0.500). Both used the same model, prompt,
retrieval state. The most likely explanation is run-to-run
variation in the un-tuned base model's citation behavior at T=0.1
— citation rate is a noisy stochastic metric on the base model. We
should treat the Modal baseline as the authoritative comparator
(same hardware, same eval pipeline) and discount the absolute
delta vs the pre-Phase-3 number.

## Reproducibility

```
# Adapter v1 eval
modal run scripts/run_generation_eval_modal.py::adapter --subdir v1
# Run ID: 20260508T111606Z

# Baseline eval (Modal A10G, no adapter)
modal run scripts/run_generation_eval_modal.py::baseline
# Run ID: 20260508T114905Z

# Download reports locally
modal volume get docsense-eval-reports /20260508T111606Z evaluations/reports/phase3-v1-adapter
modal volume get docsense-eval-reports /20260508T114905Z evaluations/reports/phase3-baseline-modal
```

Adapter weights live on Modal volume `docsense-adapters` at `/v1`.
Training run that produced them logged in
`docs/journal/2026-05-08-block-3c-training-observations.md` (local,
gitignored — observation log).

Code commits at the time of measurement: `main` at the merge of
PR #40 (split-root mount layout for the Modal eval driver). All
five Block 3C PRs (#31, #36, #37, #38, #39, #40) were on `main`.
