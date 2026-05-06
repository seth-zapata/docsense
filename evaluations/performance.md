# docsense — performance report card

A consolidated, at-a-glance view of measured performance across the
docsense system. Per-experiment JSON in `reports/`, narrative
interpretations in `analyses/`, raw smoke artifacts in `manual-runs/` —
this file is the dashboard that points into them.

**Last updated:** 2026-05-06 (closing pre-Phase-3 Block 1B — first
LLM-judged generation eval landed; pre-Phase-3 baseline at
[`baselines/pre_phase3_generation_base.json`](baselines/pre_phase3_generation_base.json)).

**Status legend:**

- ✅ measured and meeting expectations
- ⚠️ measured, with a known caveat or gap
- ❌ measured, regression or failure
- ◻️ not yet measured (next eval slot named in the row)
- 🔒 enforced by CI (test or contract; not a measurement)

---

## At a glance

| Subsystem | Quality | Latency | Notes |
|---|---|---|---|
| Retrieval (hybrid+rerank, structural) | ✅ MRR 0.685, Recall@10 0.800 | ✅ retrieve p50 ~320 ms (n=58) | recursive chunker; production default |
| Retrieval (hybrid+rerank, curated) | ⚠️ MRR 0.599 — see analysis | (same path) | curated MRR caveat documented |
| Reranking (cross-encoder lift) | ✅ +0.05 to +0.20 Recall@10 | ⚠️ ~1.3 s/chunk; first-call ~10 s warmup | dominates retrieve+rerank time |
| Generation contracts | 🔒 4 invariants pinned in CI | — | citation, budget, prompt, generator |
| Generation faithfulness (Llama judge) | ⚠️ mean 0.75 — judge anchor saturation | — | 49/50 queries scored exactly 0.75 |
| Generation answer relevance (Llama judge) | ✅ mean 0.74; healthy spread on structural | — | 3 low-relevance outliers identified |
| Generation citation rate | ⚠️ 54% (cross-set agreement) | — | Qwen cites half the time despite directive |
| Generation refusal on off-corpus | ✅ 100% (8/8) | ✅ p50 2.4 s (refusals are short) | post-pattern-fix; see analysis |
| End-to-end (in-corpus) | ✅ measured n=50 | ✅ generate p50 10–15 s, p90 19–24 s | RTX 4070 NF4 |
| System fit on RTX 4070 12 GB | ✅ Qwen 7B + Llama 8B at NF4 ≈ 5–6 GB each | — | sequential load; both verified |

---

## Retrieval

### Phase 1 baseline — dense-only, curated 20q, eval_k=10

Reference numbers Phase 2 measurements diff against. Immutable.

| Strategy | MRR | P@1 | Recall@10 | nDCG@10 | hit@5 |
|---|---:|---:|---:|---:|---:|
| **recursive** | **0.692** | **0.550** | **0.863** | **0.684** | 0.80 |
| header        | 0.623 | 0.450 | 0.743 | 0.620 | 0.85 |
| fixed         | 0.551 | 0.350 | 0.764 | 0.565 | 0.80 |

Source: [`baselines/phase1_chunking.json`](baselines/phase1_chunking.json)

### Phase 2 production — hybrid+rerank, eval_k=10

Headline: **recursive wins on the unbiased structural eval set**;
fixed's MRR spike on the curated set was a curated-distribution
artifact. Production default is unchanged from Phase 1: `recursive`.

#### Curated (n=20)
| Strategy | MRR | P@1 | Recall@10 | nDCG@10 | hit@5 |
|---|---:|---:|---:|---:|---:|
| fixed         | **0.701** | **0.600** | 0.814 | **0.678** | **0.85** |
| recursive     | 0.599 | 0.400 | **0.864** | 0.643 | 0.85 |
| header        | 0.604 | 0.450 | 0.863 | 0.645 | 0.85 |

#### Structural (n=30)
| Strategy | MRR | P@1 | Recall@10 | nDCG@10 | hit@5 |
|---|---:|---:|---:|---:|---:|
| **recursive** | **0.685** | **0.600** | **0.800** | **0.713** | **0.80** |
| header        | 0.649 | 0.567 | 0.800 | 0.686 | 0.767 |
| fixed         | 0.631 | 0.567 | 0.733 | 0.656 | 0.733 |

Sources: [`reports/bakeoff-20260506-hybrid-rerank-curated.json`](reports/bakeoff-20260506-hybrid-rerank-curated.json),
[`reports/bakeoff-20260506-hybrid-rerank-structural.json`](reports/bakeoff-20260506-hybrid-rerank-structural.json) ·
Analysis: [`analyses/2026-05-06-bakeoff-investigation.md`](analyses/2026-05-06-bakeoff-investigation.md)

### Production default

`hybrid+rerank` over the recursive-chunked corpus
(`chunking.strategy=recursive`, `dense_weight=0.6`, `sparse_weight=0.4`,
`rerank_candidates=20`, `top_k=5`). See the analysis for *why* this is
the right default despite the curated MRR exception.

### Retrieval latency

| Stage | p50 | p90 | max | n | Source |
|---|---:|---:|---:|---:|---|
| Hybrid retrieve + rerank (curated, top_k=5) | 315 ms | 446 ms | 10,573 ms | 20 | [`baselines/pre_phase3_generation_base.json`](baselines/pre_phase3_generation_base.json) |
| Hybrid retrieve + rerank (structural, top_k=5) | 322 ms | 388 ms | 3,417 ms | 30 | same |
| Hybrid retrieve + rerank (no-answer, top_k=5) | 393 ms | 1,401 ms | 3,100 ms | 8 | same |
| Per-stage breakdown (dense / sparse / RRF / rerank) | ◻️ followup | | | | |

The `max` outlier on the curated set (10.6 s on the very first query)
is a one-time cross-encoder warm-up — the model loads lazily on first
`predict()`. After that, retrieve + rerank is consistently sub-second.
Per-substage timing (dense vs. sparse vs. RRF vs. rerank) is still a
single bucket; instrumenting that is a small follow-up.

---

## Reranking

### Recall lift across the 3×2 grid

The cross-encoder's contribution is the most stable signal in the
bakeoff. Recall@10 improves with reranking for every cell.

| Eval set | Avg ΔRecall@10 (dense → hybrid+rerank) | Range |
|---|---:|---|
| Curated (n=20)     | +0.063 | +0.000 to +0.120 |
| Structural (n=30)  | +0.117 | +0.100 to +0.200 |

Source: [`analyses/2026-05-06-bakeoff-investigation.md`](analyses/2026-05-06-bakeoff-investigation.md) §4

### Reranking latency

~1.3 s for `rerank_candidates=20` on CPU
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, batch_size=16). Single data
point; needs Block 1B's per-query instrumentation to publish a
distribution.

---

## Generation

### Pipeline contracts (CI-enforced)

These aren't measurements — they're invariants the test suite
guarantees on every PR. They're listed here because "what does the
system promise to never violate" is the first thing an interviewer
should be able to see.

| Invariant | Pinned by | Source |
|---|---|---|
| 🔒 Citations on `Answer` always reference retrieved chunks | `Answer` pydantic model_validator | `src/docsense/generation/types.py` |
| 🔒 Context assembly never exceeds the token budget | strict tokenizer-based test | `tests/unit/test_context_assembly.py` |
| 🔒 Prompt template structure is stable | JSON snapshot | `tests/snapshots/prompt_default.json` |
| 🔒 Generator messages-API contract | unit-test override of `_run_inference` | `tests/unit/test_generator.py` |

### LLM behavior — pre-Phase-3 baseline (Qwen 2.5 7B Instruct, NF4)

Measured across **58 queries** (curated 20 + structural 30 + no-answer 8)
running the production retrieval+rerank stack with recursive chunking.
LLM judge: Llama 3.1 8B Instruct (NF4, temperature=0). Source data:
[`baselines/pre_phase3_generation_base.json`](baselines/pre_phase3_generation_base.json).

| Aspect | Status | Result | Source |
|---|---|---|---|
| Loads at NF4 in 12 GB VRAM | ✅ | Qwen ~5–6 GB; Llama ~5–6 GB; sequential load works on shared 12 GB GPU | [smoke.md](manual-runs/2026-05-06-smoke.md) + eval handoff (9 MB residual after each unload) |
| Inference throughput | ✅ | 14.6 tok/s (smoke); generate p50 9.7–15 s (eval) | smoke.md, eval reports |
| **Faithfulness (Llama judge)** | ⚠️ | mean **0.750** curated / **0.758** structural | judge anchor saturation — 49/50 in-corpus answers scored exactly 0.75; flagged for cross-judge calibration |
| **Answer relevance (Llama judge)** | ✅ | mean **0.762** curated / **0.733** structural | structural shows 3×0.25, 23×0.75, 4×1.0 — healthier spread than faithfulness |
| **Citation rate** (`[N]` markers in text) | ⚠️ | **54%** of in-corpus answers (55% curated, 53% structural) | cross-set agreement → real, not noise; smoke 0% was unrepresentative |
| **Refusal on off-corpus** (rule-based) | ✅ | **100%** (8/8) | Qwen reliably refuses with "I don't have enough context"; pattern coverage gap caught + fixed mid-eval |
| Generated answer quality (eyeball) | ✅ | technically correct on the smoke run | smoke.md §Generated answer |
| Prompt-tuning sweep for citation rate | ◻️ followup | — | optional; the citation gap is documented and Phase 3 fine-tune may target it |

### Generation latency (Qwen 2.5 7B, NF4, RTX 4070)

| Stage | p50 | p90 | max | n | Source |
|---|---:|---:|---:|---:|---|
| Generate (curated in-corpus) | 15,125 ms | 24,311 ms | 26,566 ms | 20 | [`baselines/pre_phase3_generation_base.json`](baselines/pre_phase3_generation_base.json) |
| Generate (structural in-corpus) | 9,733 ms | 18,822 ms | 33,779 ms | 30 | same |
| Generate (no-answer / refusal) | 2,438 ms | 3,994 ms | 4,286 ms | 8 | same |
| Model load (cold, one-time per process) | ~163 s | — | — | n=1 | smoke.md |

Refusals are short (≤ 50 tokens) so generation is much faster — ~2.4 s
p50 vs ~10–15 s for in-corpus answers. Curated questions tend to elicit
longer expository answers ("How do I install transformers?" gets a
multi-paragraph response); structural queries lean terminology-focused
("Albert specific outputs") and produce shorter answers.

---

## End-to-end

| Aspect | Status | Source / Plan |
|---|---|---|
| Single-query smoke completes | ✅ end-to-end success | [smoke.md](manual-runs/2026-05-06-smoke.md) |
| Per-query latency distribution (n=58) | ✅ measured across 3 eval sets | [`baselines/pre_phase3_generation_base.json`](baselines/pre_phase3_generation_base.json) |
| Per-query quality scores (n=50 in-corpus) | ✅ faithfulness, relevance, citation grounding | same |
| Off-corpus refusal robustness (n=8) | ✅ 100% with patched heuristic | same; methodology iteration documented in analysis |
| Adversarial / prompt-injection eval | ◻️ Phase 4 territory | small adversarial set when serving exists |
| Streaming generation | ◻️ Phase 4 (serving) | not currently wired |

---

## System characteristics

### Hardware envelope

| Constraint | Value | Verified |
|---|---|---|
| Target VRAM (vanilla RTX 4070) | 12 GB total, ~5 GB free with display | ✅ smoke run |
| Generation model fit (Qwen 2.5 7B, NF4) | ~5–6 GB resident | ✅ smoke run + eval |
| Judge model fit (Llama 3.1 8B, NF4, sequential load) | ~5–6 GB resident, post-`del` of generator | ✅ eval (9 MB residual after generator unload, 9 MB after judge unload) |
| Required device flag | `device=cuda:0` (not `device=auto`) | ⚠️ followup: change default — see smoke.md |
| HF cache location | `/mnt/e/hf-cache` (E: drive) | ✅ ~31 GB used (Qwen 15 GB + Llama 15 GB + small models ~80 MB each) |
| HF auth (Llama 3.1 is a gated model) | required for judge model | ✅ user has accepted license + HF_TOKEN configured |

### Code & test footprint

| Metric | Value | Source |
|---|---:|---|
| Tests | 266 | `pytest --collect-only` |
| Coverage gate (CI-enforced) | ≥ 90% | `.github/workflows/ci.yml` |
| Lint / format | ruff (pinned 0.15.8) | `pyproject.toml` |
| Type check | mypy (pinned 1.19.1, strict) | `pyproject.toml` |
| Pre-commit hooks | ruff + mypy | `.pre-commit-config.yaml` |
| Pre-push hook | full pytest | `.pre-commit-config.yaml` |

### Corpus

| Metric | Value |
|---|---:|
| Source | `huggingface/transformers` docs (en), git sparse-checkout |
| Documents | 656 |
| Chunks (recursive, prod default) | 12,601 |
| Chunks (fixed) | 10,681 |
| Chunks (header) | 14,218 |

---

## How this document is maintained

- **One row per measurement, with a link.** Every metric here points
  back at a JSON in `reports/`, an analysis in `analyses/`, or a
  manual artifact in `manual-runs/`. No claim without a source.
- **Updated in the same PR as new evaluation data.** When a
  measurement lands (a new bakeoff report, a Block 1B eval run), the
  PR that adds the report also updates the relevant rows here.
- **Gaps are visible, not silent.** ◻️ rows are explicit pointers to
  *what we haven't measured yet and which block will measure it*.
  This is intentional — gaps with named owners are healthier than
  invisible gaps.
- **Old rows survive re-baselines.** When a measurement is rerun and
  numbers change materially, the new value replaces the old, but the
  source link remains so the diff is traceable in git history.
- **No auto-generation, yet.** This is a hand-maintained markdown.
  If maintenance becomes painful (≥ ~20 reports), revisit and
  consider a generator script that reads the JSON reports and emits
  the tables.

---

## Pointers

- **Architecture & narrative status:** [`docs/architecture.md`](../docs/architecture.md) §Status snapshot
- **Eval methodology:** [`docs/eval-methodology.md`](../docs/eval-methodology.md)
- **Phase 2/3/4 scope:** [`docs/phase-2-4-scope.md`](../docs/phase-2-4-scope.md)
- **Engineering journal:** [`docs/journal/`](../docs/journal/)
- **Eval directory conventions:** [`README.md`](README.md)
