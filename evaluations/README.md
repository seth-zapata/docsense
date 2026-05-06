# Evaluations

This directory holds the committed eval artifacts for docsense:
baselines, query sets, raw bakeoff reports, and analysis docs that
interpret them.

```
evaluations/
├── README.md                this file
├── performance.md           consolidated, dashboard-style report card
│                            across all subsystems. Updated whenever a new
│                            measurement lands. Start here.
├── baselines/               committed-once reference numbers
│   └── phase1_chunking.json     Phase 1 corrected dense-only baseline,
│                                eval_k=10. Phase 2+ runs diff against it.
├── eval_sets/               committed query distributions
│   └── structural.json          30 queries derived programmatically from
│                                doc h2 headings (seeded; reproducible via
│                                scripts/generate_structural_queries.py).
├── reports/                 raw output from scripts/run_bakeoff.py
│   └── bakeoff-<YYYYMMDD>-<pipeline>-<eval-set>.json
├── analyses/                committed markdown interpretations of reports
│   └── <YYYY-MM-DD>-<topic>.md
└── manual-runs/             one-off smoke / debugging artifacts
    └── <YYYY-MM-DD>-<topic>.md
```

## Conventions

- **`performance.md` is the dashboard.** When you add or update a
  measurement (new report, new analysis, new smoke run), update the
  matching row in `performance.md` in the same PR. Per-experiment
  files are the system of record; `performance.md` is the at-a-glance
  consolidated view that points into them.
- **Baselines are committed once** and treated as immutable until a
  formally documented re-baseline. Phase 1 baseline at eval_k=10 is the
  reference point for all Phase 2 retrieval reports.
- **Reports are committed every time you run** `scripts/run_bakeoff.py`
  with non-default flags (or any time you want a checkpoint). They are
  small (~2 KB) and the diff history of a series of reports tells the
  experimental story by itself.
- **Analyses are committed when a report or set of reports surfaces
  something worth interpreting** — not for routine runs that match
  expectations. Each analysis cross-references the specific reports
  it interprets by filename so the data and the interpretation stay
  linked.
- **Eval sets are versioned** (filename = the eval set's identity).
  When a query distribution changes meaningfully, commit a new file
  rather than overwriting; downstream reports cite the eval-set
  filename in their `eval_set` field.
- The hand-curated eval queries live in
  [`src/docsense/evaluation/eval_queries.py`](../src/docsense/evaluation/eval_queries.py)
  rather than under `evaluations/` because they're imported by code;
  the structural set is JSON because it's generated artifact, not code.

## Running the bakeoff

```bash
# Phase 1 parity (dense-only, curated eval, eval_k=10)
python scripts/run_bakeoff.py

# Production pipeline on the unbiased structural eval set
python scripts/run_bakeoff.py --pipeline hybrid-rerank --eval-set structural

# Ablate one addition at a time
for p in dense hybrid hybrid-rerank; do
  for e in curated structural; do
    python scripts/run_bakeoff.py --pipeline $p --eval-set $e
  done
done
```
