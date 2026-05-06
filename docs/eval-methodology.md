# Evaluation methodology

A reference for what docsense measures, where the data lives, and what
runs in PR CI vs nightly vs manual. Updated as new metrics or eval
sets land.

> **Sister docs:**
> - [`docs/architecture.md`](architecture.md) — system overview.
> - [`docs/phase-2-4-scope.md`](phase-2-4-scope.md) — phased scope.
> - [`evaluations/README.md`](../evaluations/README.md) — directory
>   conventions for the committed eval artifacts.

## What we measure

### Retrieval (Phase 1+)

Standard rank-sensitive IR metrics computed at chunk granularity
deduped to document granularity:

- **P@k** — fraction of top-k retrieved docs that are relevant.
- **Recall@k** — fraction of relevant docs that appear in top-k.
- **MRR** — reciprocal of the rank of the first relevant doc.
- **nDCG@k** — rank-discounted relevance.
- **hit_rate@5** — fraction of queries with ≥1 relevant doc in top-5.

Implementation: [`src/docsense/evaluation/retrieval_metrics.py`](../src/docsense/evaluation/retrieval_metrics.py).
The `deduplicate_preserving_order` helper is non-optional — chunk-level
retrieval with doc-level relevance labels inflates recall and nDCG
without dedup.

### Generation contracts (Phase 2 — what runs in PR CI)

These are **rule-based invariants** that pin the wrapper's behavior
independent of LLM quality. They're cheap, deterministic, and run on
every PR:

- **Citation preservation** — every `Citation` on an `Answer` must
  reference a chunk in `Answer.retrieved_chunks`. Enforced by a
  pydantic `model_validator` on `Answer`; tested in
  `tests/unit/test_generation_types.py::TestCitationPreservationInvariant`.
- **Token-budget enforcement** — `ContextAssembler` guarantees
  `tokenize(final_text) <= max_tokens` for any monotonic tokenizer.
  Tested across multiple synthetic tokenizers in
  `tests/unit/test_context_assembly.py::TestContextAssemblerBudgetIsTokenizerAgnostic`.
- **Prompt drift** — the rendered default prompt must match
  `tests/snapshots/prompt_default.txt` byte-for-byte (modulo a
  trailing-newline file convention).
- **Generator contract** — `_run_inference` is invoked exactly once
  per `generate()` call, with the prompt argument; its return drives
  `Answer.text` and `Answer.metadata`. Tested via both subclassing
  and `patch.object` mocking.

### Generation behavior (Phase 3+ — LLM-judged, NOT in PR CI)

These require actual LLM calls and are **not appropriate for PR
gating** (cost, latency, score noise). They run manually or on a
nightly schedule against committed baselines:

- **Faithfulness** — does the answer follow from the retrieved
  chunks? LLM-judged.
- **Answer relevance** — does the answer address the question?
  LLM-judged.
- **Citation grounding** — are the cited chunks actually the source
  of the asserted facts? LLM-judged.
- **Real no-answer behavior** — when retrieval returns irrelevant
  chunks, does the model produce a refusal? Pattern-matched against
  the prompt template's refusal directive on real LLM output.

Implementation deferred to pre-Phase-3 — see
[`docs/phase-2-4-scope.md`](phase-2-4-scope.md) section C1.

### Operational metrics (Phase 4)

When the FastAPI service exists: latency percentiles (p50/p95/p99),
error rate, request volume, downstream query traces. These are
service-level metrics, not quality metrics, and don't belong in this
document — they belong in operational dashboards.

## Eval sets

Eval sets are **versioned by filename**. Different distributions get
different filenames; reports cite the eval set used in their
`eval_set` field.

| Set | Source | Size | Bias | Status |
|-----|--------|-----:|------|--------|
| Curated | hand-authored in `src/docsense/evaluation/eval_queries.py` | 20 | Selection-biased toward docs the curator knew existed. | Live |
| Structural | programmatic from `##` doc headings via `scripts/generate_structural_queries.py` | 30 | Programmatically derived; some heading-as-query noise (e.g., generic "Configuration" headings) | Live |
| LLM-generated (5c) | Anthropic SDK against held-out doc subset | TBD | Different bias profile from the other two | Pre-Phase-3, deferred |

The single biggest finding from Block B+ was that **single-eval-set
conclusions are unreliable** for this corpus — the same pipeline
produced different strategy rankings on curated vs structural. Going
forward, retrieval bakeoffs should run both sets and report both.

## What runs where

### Every PR (`.github/workflows/ci.yml`)

- `lint` — ruff check + format check, 3.12.
- `typecheck` — mypy with `check_untyped_defs`, `warn_unused_ignores`,
  `warn_redundant_casts`, `no_implicit_optional`.
- `test` — pytest excluding `slow`/`gpu`/`integration`-marked tests.
  Coverage gate: `--cov-fail-under=90`. Runs all unit tests and the
  unmarked integration tests.

Total budget: ~3 minutes wall-clock. Catches lint/type/test failures
before merge to `main` (branch protection blocks force-push and
direct merge without these checks).

### Pre-push hook (`.pre-commit-config.yaml`)

`pytest -m "not slow and not gpu and not integration"` runs locally
on every `git push`. Catches the bulk of CI failures before they hit
GitHub. ~5 seconds.

### Manual (dev workflow)

- `python scripts/run_bakeoff.py [--pipeline ...] [--eval-set ...]`
  — chunking + retrieval ablations. Can produce 6 reports (3 pipelines
  × 2 eval sets) in ~3 minutes total.
- Notebook re-execution — `jupyter nbconvert --to notebook --execute
  --inplace --ExecutePreprocessor.kernel_name=docsense
  notebooks/chunking_comparison.ipynb` to refresh the rendered Phase 1
  bakeoff.
- Loading the actual Mistral 7B / Llama 3 8B base model and running
  end-to-end inference. Requires ~14 GB of weights downloaded;
  appropriate for local exploration, not CI.

### Nightly / scheduled (Phase 3+, not yet implemented)

- `pip-audit` against `pyproject.toml`. Add when dep set stabilizes.
- LLM-judge evals against the committed Phase 2 baseline.
- Full bakeoff with all three pipelines × all three eval sets
  (including 5c LLM-generated).

## Where artifacts live

```
evaluations/
├── README.md                      conventions for this directory
├── baselines/                     immutable reference numbers — committed
│                                  once per phase, treated as canonical
│                                  unless formally re-baselined
├── eval_sets/                     versioned query distributions
├── reports/                       raw bakeoff output, committed every run
└── analyses/                      markdown interpretations of reports —
                                   committed when a report set surfaces
                                   something worth interpreting (not for
                                   routine runs that match expectations)
```

The hand-curated eval queries live in
[`src/docsense/evaluation/eval_queries.py`](../src/docsense/evaluation/eval_queries.py)
rather than under `evaluations/` because they're imported by code; the
structural set is JSON because it's a generated artifact.

## How to add a new eval

- **A new metric:** add to
  [`src/docsense/evaluation/retrieval_metrics.py`](../src/docsense/evaluation/retrieval_metrics.py)
  with unit tests in `tests/unit/test_retrieval_metrics.py`. If the
  metric assumes deduplicated input, document the assumption in the
  module docstring (the existing module has a precedent).
- **A new eval set:** write a generator script in `scripts/`, output
  to `evaluations/eval_sets/<name>.json` with the same
  `[{"query": ..., "relevant": [...]}, ...]` shape as
  `structural.json`. Update `_load_eval_set` in `scripts/run_bakeoff.py`
  to recognize the new name.
- **An LLM-judge eval:** structure TBD as part of pre-Phase-3 work
  (see scope doc C1). Will live under
  `src/docsense/training/judge.py` and produce reports under
  `evaluations/llm_judge_eval/`.

## Why this split exists

The PR-CI evals validate **our code's contracts** — that data flows
correctly, types compose, and the deterministic pieces of the system
behave as specified. They run in seconds, on every change, with no
external dependencies and no cost.

The LLM-judge evals validate **the system's quality** — that
generated answers are actually faithful, relevant, and grounded.
They take minutes per run, cost money per call, and have noisy
scores. Gating PRs on them produces false positives that train
contributors to ignore the gate.

Both are important; mixing them dilutes both.
