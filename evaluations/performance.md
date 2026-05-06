# docsense — performance report card

A consolidated, at-a-glance view of measured performance across the
docsense system. Per-experiment JSON in `reports/`, narrative
interpretations in `analyses/`, raw smoke artifacts in `manual-runs/` —
this file is the dashboard that points into them.

**Last updated:** 2026-05-06 (closing Block 1A; Block 1B will populate
the generation-quality and end-to-end-quality sections.)

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
| Retrieval (hybrid+rerank, structural) | ✅ MRR 0.685, Recall@10 0.800 | ✅ ~6.5 s incl. rerank (n=1) | recursive chunker; production default |
| Retrieval (hybrid+rerank, curated) | ⚠️ MRR 0.599 — see analysis | (same path) | curated MRR caveat documented |
| Reranking (cross-encoder lift) | ✅ +0.05 to +0.20 Recall@10 | ⚠️ ~1.3 s/chunk | dominates retrieve+rerank time |
| Generation contracts | 🔒 4 invariants pinned in CI | — | citation, budget, prompt, generator |
| Generation LLM behavior | ⚠️ citations not honored on n=1 | ✅ 14.6 tok/s on RTX 4070 NF4 | flagged for Block 1B |
| Generation quality (faithfulness, etc.) | ◻️ Block 1B | ◻️ Block 1B | LLM-judge scaffolding |
| End-to-end | ⚠️ single smoke point (n=1) | ⚠️ 21.8 s/query steady-state | systematic measurement: Block 1B |
| System fit on RTX 4070 12 GB | ✅ Qwen 7B at NF4 ≈ 5–6 GB | — | requires `device=cuda:0` not `auto` |

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

| Stage | Wall time | Sample | Source |
|---|---:|---|---|
| Index + retriever setup (cold) | ~10 s | n=1 | [`manual-runs/2026-05-06-smoke.md`](manual-runs/2026-05-06-smoke.md) |
| Hybrid retrieve + rerank (5 chunks) | 6.5 s | n=1 | same |
| Per-stage breakdown (dense / sparse / RRF / rerank) | ◻️ Block 1B will instrument | | |

The 6.5-second figure is dominated by the cross-encoder (~1.3 s/chunk
× 5). Dense + sparse retrieval is sub-second; RRF and assembly are
negligible. Block 1B's `scripts/run_generation_eval.py` will record
per-stage latencies into each report's `timing` field so we can
publish quartiles instead of n=1 anecdotes.

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

### LLM behavior — Block 1A smoke (n=1)

Single end-to-end execution against Qwen 2.5 7B Instruct (NF4) on the
user's RTX 4070. Real query, real corpus, real model. Aspect-by-aspect:

| Aspect | Status | Notes / Source |
|---|---|---|
| Loads at NF4 in 12 GB VRAM | ✅ ~5–6 GB resident | [smoke.md](manual-runs/2026-05-06-smoke.md) |
| Inference throughput | ✅ 14.6 tok/s (223 tok / 15.3 s) | smoke.md §Timing |
| Generated answer quality (eyeball) | ✅ technically correct, no hallucinated APIs | smoke.md §Generated answer |
| Citation honoring (`[N]` markers) | ⚠️ ignored despite system-prompt directive | smoke.md §Citations — n=1 |
| Refusal / no-answer behavior | ◻️ Block 1B `check_no_answer_behavior` | |
| Faithfulness (LLM-judge) | ◻️ Block 1B `LlamaJudge` | |
| Answer relevance (LLM-judge) | ◻️ Block 1B `LlamaJudge` | |
| Citation grounding (LLM-judge) | ◻️ Block 1B `check_citations_grounded` | |
| Prompt-tuning sweep for citation rate | ◻️ Block 1B+ if Block 1B confirms the gap | |

### Generation latency (Qwen 2.5 7B, NF4, RTX 4070)

| Stage | Wall time | Sample | Notes |
|---|---:|---|---|
| Model load (cold, one-time) | ~163 s | n=1 | per-process; warm sessions skip |
| Inference (223 tokens) | 15.3 s | n=1 | 14.6 tok/s |
| Per-query steady-state (retrieve + rerank + assemble + generate) | ~21.8 s | n=1, derived | excludes one-time loads |

Block 1B will publish quartile latencies over a meaningful query
distribution — the n=1 figures here establish the order of magnitude
only.

---

## End-to-end

| Aspect | Status | Source / Plan |
|---|---|---|
| Single-query smoke completes | ✅ end-to-end success | [smoke.md](manual-runs/2026-05-06-smoke.md) |
| Per-query latency distribution | ◻️ Block 1B `run_generation_eval.py` | will publish p50 / p90 / max |
| Per-query quality scores | ◻️ Block 1B LLM-judge | faithfulness, relevance, citations |
| Robustness (no-answer queries, adversarial) | ◻️ Block 1B+ | rule-based + LLM-judge mix |
| Streaming generation | ◻️ Phase 4 (serving) | not currently wired |

---

## System characteristics

### Hardware envelope

| Constraint | Value | Verified |
|---|---|---|
| Target VRAM (vanilla RTX 4070) | 12 GB total, ~5 GB free with display | ✅ smoke run |
| Generation model fit (Qwen 2.5 7B, NF4) | ~5–6 GB resident | ✅ smoke run |
| Judge model fit (Llama 3.1 8B, NF4, sequential load) | ~6 GB resident, post-`del` of generator | ◻️ Block 1B will verify |
| Required device flag | `device=cuda:0` (not `device=auto`) | ⚠️ followup: change default — see smoke.md |
| HF cache location | `/mnt/e/hf-cache` (E: drive, ~1 TB free) | ✅ documented in CLAUDE.md |

### Code & test footprint

| Metric | Value | Source |
|---|---:|---|
| Tests | 203 | `pytest --collect-only` |
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
