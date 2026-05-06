# docsense — System architecture

A reference document for understanding what docsense is, how the
pieces fit together, and where each phase plugs in. Updated as phases
close.

> Companion docs:
> - [`README.md`](../README.md) — project description, install, commands.
> - [`CLAUDE.md`](../CLAUDE.md) — dev/Claude instructions, current phase,
>   roadmap with deferred items.
> - [`notebooks/chunking_comparison.ipynb`](../notebooks/chunking_comparison.ipynb)
>   — the chunking bakeoff with rendered results.

## What docsense is

A production-grade Retrieval-Augmented Generation (RAG) system for
technical documentation Q&A, built on the Hugging Face Transformers
docs as its target corpus. Two halves:

1. **Retrieval pipeline.** Given a natural-language question, find the
   relevant passages from the corpus.
2. **Generation pipeline.** Given those passages and the question,
   produce an accurate, cited answer using an LLM — with an optional
   QLoRA fine-tune on the same corpus to test whether domain
   adaptation improves answer quality.

Wrapped in a FastAPI service with structured query tracing.

The project deliberately builds the retrieval, context assembly, prompt
construction, and generation glue **from scratch** rather than using
LangChain or LlamaIndex. The framework versions ship faster but
obscure the parts most worth being able to explain in detail.

## Pipeline architecture

A RAG system is a four-stage pipeline. Each stage answers a different
question, has different metrics, and benefits from different
optimizations.

```
                                    Phase 1 ✅       Phase 2 ✅      Phase 2 ✅        Phase 4
                                  ┌──────────┐   ┌──────────┐    ┌────────────┐   ┌─────────┐
   User query  ─────────────────► │ Retrieve │ ──►│ Re-rank  │ ──►│  Generate  │ ──► Answer
                                  └──────────┘   └──────────┘    └────────────┘   └─────────┘
                                       ▲              ▲                ▲
                                       │              │                │
                                  BM25+FAISS+RRF  cross-encoder  LLM (Mistral 7B
                                  hybrid          re-ranking      or Llama 3 8B;
                                                                  scaffolded,
                                                                  fine-tune in P3)
```

| Stage | Question it answers | Status |
|-------|---------------------|--------|
| **Retrieve** | Of all chunks in the corpus, which are *plausibly* relevant to this query? | Phase 1 ✅ |
| **Re-rank** | Of the candidates retrieval found, which is *most* relevant for *this specific* query? | Phase 2 ✅ |
| **Generate** | Given the relevant chunks, what's the actual answer? | Phase 2 ✅ (scaffold, mockable) · Phase 3 (real LLM + fine-tune) |
| **Serve** | How do users actually interact with this system? | Phase 4 |

The reason this is a pipeline of separate stages — rather than one
giant model that does everything — is that **each stage scales
differently and benefits from different model architectures**.
Bi-encoder embeddings are cheap and run over millions of chunks;
cross-encoders are 10–100× more expensive but score query-doc pairs
more accurately, so they only run on a handful of candidates;
generative LLMs are most expensive of all and only see a few chunks
in their context window. Cascading from cheap-and-broad to
expensive-and-precise is the standard design pattern.

## Phase 1 — retrieval (closed)

### What got built

```
src/docsense/
├── ingestion/      Documents in: HF Transformers docs (656 markdown files
│                   pulled via git sparse-checkout), parsed for frontmatter
│                   and titles.
├── chunking/       Three strategies — fixed-size, recursive (paragraph→
│                   line→sentence→word), markdown-header-based — sharing a
│                   common ChunkingStrategy ABC so any pipeline can swap
│                   strategies via config.
├── embedding/      sentence-transformers wrapper. Default model:
│                   all-MiniLM-L6-v2 (384-d, fast, good baseline).
│                   Lazy-loaded so test setup doesn't pay the model
│                   download cost.
├── retrieval/      Hybrid search:
│                   - BM25 (sparse, keyword-sensitive)
│                   - FAISS IndexFlatIP (dense, semantic; cosine via
│                     normalized embeddings)
│                   - Reciprocal Rank Fusion combining their rankings,
│                     weighted 0.6 dense / 0.4 sparse by default
├── reranking/      CrossEncoderReranker class scaffolded but NOT yet
│                   wired into HybridRetriever. Wiring is Phase 2 task #1.
└── evaluation/     Retrieval metrics (P@k, Recall@k, MRR, nDCG) +
                    deduplication helper for chunk→doc granularity
                    conversion. Two query sets: hand-curated (20 queries)
                    and structural (programmatically generated from doc
                    headings) for unbiased complement.
```

### The chunking bakeoff

The headline experiment of Phase 1 compared the three chunking strategies
on a 20-query hand-curated eval set. Same corpus, same embedder, same
retrieval pipeline — only the chunking varied.

**Final corrected results** (after a metrics-bug fix that originally
inflated recall):

| Metric | Fixed | Recursive | Header |
|--------|-------|-----------|--------|
| MRR | 0.551 | **0.692** | 0.623 |
| P@1 | 0.350 | **0.550** | 0.450 |
| Recall@10 | 0.764 | **0.863** | 0.743 |
| nDCG@10 | 0.565 | **0.684** | 0.620 |
| Hit rate (top 5) | 16/20 | 16/20 | **17/20** |

**Recursive wins on rank-sensitive metrics**, header is competitive
on hit rate. The reasoning: header chunks are larger, which forces
the bi-encoder embedding to compress more content into a single
384-d vector, hurting precision against specific queries. Recursive
splits at natural paragraph/sentence boundaries, producing chunks
that are semantically focused.

This result feeds directly into Phase 2 — see "Why the bakeoff result
matters for Phase 2" below.

### Infrastructure that landed alongside

- 115 tests, 98% coverage with an enforced 90% CI floor.
- mypy gating with strict-ish settings; ruff lint+format; pre-commit
  hooks for both.
- Pinned tool versions across pyproject + CI to eliminate local-vs-CI
  drift.
- Docs/architecture.md (this file), engineering journal at
  `docs/journal/` (gitignored).

## Phase 2 — re-ranking + generation (closed)

### What got built

```
src/docsense/
├── retrieval/hybrid.py        CHANGE — HybridRetriever now accepts
│                              an optional CrossEncoderReranker; over-
│                              retrieves rerank_candidates from the
│                              fused list and trims to top_k after
│                              cross-encoder scoring.
├── reranking/reranker.py      CHANGE — rerank() takes explicit top_k
│                              (required); RerankingConfig.batch_size
│                              is now exclusively the cross-encoder
│                              inference batch size.
├── generation/
│   ├── types.py               NEW — Answer, Citation, ChunkRef,
│   │                          GenerationMetadata. Pydantic models with
│   │                          a model_validator enforcing that every
│   │                          Citation references a chunk in
│   │                          retrieved_chunks.
│   ├── context.py             NEW — ContextAssembler. Greedy fill under
│   │                          a strict token budget. Tokenize_fn is
│   │                          injectable; budget check tokenizes the
│   │                          candidate joined string each iteration
│   │                          to handle non-additive (BPE) tokenizers.
│   ├── prompt.py              NEW — PromptBuilder with default system
│   │                          prompt (cite-by-[N], refusal directive)
│   │                          and template. Snapshot-tested against
│   │                          tests/snapshots/prompt_default.txt.
│   └── generator.py           NEW — Generator wrapping HuggingFace
│                              AutoModelForCausalLM. _run_inference
│                              is the override point for tests.
│                              parse_citations() resolves [N] notation
│                              against retrieved chunks.
configs/                       NEW — YAML preset files for the bakeoff:
                               default, no-rerank, header-strategy.
evaluations/
├── baselines/phase1_chunking.json  NEW — corrected Phase 1 numbers
├── eval_sets/structural.json       NEW — 30 programmatic queries
├── reports/bakeoff-*.json          NEW — committed reports per run
└── analyses/2026-05-06-bakeoff-investigation.md  NEW — Block B+ writeup
scripts/run_bakeoff.py         NEW — reproducible bakeoff runner with
                               --pipeline {dense,hybrid,hybrid-rerank}
                               and --eval-set {curated,structural}.
```

### The Block B+ ablation revealed eval-set bias

Phase 2's headline experimental finding wasn't about chunking — it was
about *eval methodology*. The full hybrid+rerank pipeline gave
contradictory results across two eval sets:

| pipeline | curated MRR winner | structural MRR winner |
|---|---|---|
| dense (Phase 1) | recursive (0.692) | header (0.518) |
| hybrid (no rerank) | recursive (0.667) | header (0.611) |
| hybrid+rerank | **fixed (0.701)** | **recursive (0.685)** |

The "fixed-size chunking wins under hybrid+rerank" result that surfaced
on the curated 20-query set **didn't replicate** on the unbiased
30-query structural set, where recursive wins. The Block B
hypothesis-confirming finding was a curated-eval artifact.

The full ablation analysis lives at
[`evaluations/analyses/2026-05-06-bakeoff-investigation.md`](../evaluations/analyses/2026-05-06-bakeoff-investigation.md).

**Takeaways carried into Phase 3+:**
- Always run both eval sets going forward; single-set conclusions are
  unreliable.
- Recall@10 improves universally with the cross-encoder (every
  cell of the 3×2 grid). The reranker is doing real work.
- Production default: hybrid+rerank pipeline with `chunking.strategy=recursive`.
  Recursive wins on structural under hybrid+rerank; isn't worst on
  curated either. Phase 1's choice still holds — for refined reasons.

### Generation surface scaffolded, real LLM deferred

Block C built the generation layer end-to-end:

- **Types** define what an `Answer` is — text, citations, retrieved
  chunks (audit trail), and metadata. Citation-preservation is a
  pydantic invariant, not just a parser-side property: an Answer
  with a Citation pointing at a non-retrieved chunk fails validation
  at construction time.
- **Context assembly** budgets retrieved chunks under
  `GenerationConfig.max_context_tokens`. Greedy fill with a strict
  invariant: `tokenize(final_text) <= max_tokens` for any monotonic
  tokenizer. Block D's parameterized test caught an additive-token-
  drift bug here that would have manifested as silent budget
  violations in production with real BPE tokenizers; fix landed
  alongside the test.
- **Prompt construction** is template-based with a snapshot test.
  Default system prompt instructs the LLM to cite by `[N]` and to
  refuse when context is insufficient.
- **Generator** wraps HuggingFace `AutoModelForCausalLM` with the
  same lazy-load pattern as `Embedder` and `CrossEncoderReranker`.
  `_run_inference` is the override point for tests — subclassing or
  `patch.object` both work without touching the model load path.
- **Citation parser** turns `[N]` notation in generated text into
  typed `Citation` objects. Hallucinated indices (`[99]` when only
  3 chunks exist) and zero-indices are silently dropped, not raised.

**What's not done in Phase 2:** loading a real LLM and running real
inference end-to-end. The wiring exists; behavioral evaluation
(faithfulness, answer relevance, real no-answer behavior, citation
grounding) is **deliberately deferred** to pre-Phase-3 LLM-judge
evals. See [`docs/eval-methodology.md`](eval-methodology.md) for the
full breakdown of what runs where.

### Infrastructure landed alongside

- **PR-based workflow** — branch protection on `main`, CI gates
  (lint, typecheck, test), auto-merge with rebase on `gh pr merge`.
- **Pre-push pytest hook** — `pytest -m "not slow and not gpu and
  not integration"` runs locally before every push, catches the
  bulk of CI failures before they hit GitHub.
- **194 tests, 90% CI coverage gate.** Two of the new tests
  (D.2, D.3) caught real bugs in production-relevant code paths.

## Phase 3 — QLoRA fine-tuning

Take the base model from Phase 2, train a QLoRA adapter on a
synthetic dataset built from the corpus (probably "given this
section, write a question someone might ask, and the answer"), and
compare fine-tuned vs base on the same end-to-end eval.

Constraints:
- Target hardware: RTX 5080 with 16 GB VRAM. QLoRA's 4-bit
  quantization plus low-rank adapter weights make this fit; full
  fine-tuning wouldn't.
- Dataset construction is its own subproject — synthetic Q&A pairs
  via an LLM, validated against the source docs to avoid
  hallucinated training data.

Interview narrative target: "I built the fine-tuning pipeline, ran a
real training job within hardware constraints, and showed measurable
improvement (or lack thereof — also informative) on domain-specific
questions versus the off-the-shelf base."

## Phase 4 — serving + observability

- **FastAPI** wrapping the full pipeline behind a `/query` endpoint.
- **Tracing** via the (currently empty) `src/docsense/tracing/`
  module — structured logs per query with retrieval results, rerank
  scores, prompt, generation latency. structlog + correlation IDs.
- **Docker** packaging.
- **Optional:** CD pipeline — build container → push to GHCR → deploy
  to staging → smoke test → manual gate → prod.
- **End-to-end generation eval** — faithfulness (answer is grounded
  in retrieved chunks), answer relevance, citation accuracy.

## Design decisions worth knowing

These are the choices that shape the system; each has a journal
entry going deeper.

- **No LangChain/LlamaIndex for core logic.** Build the glue
  ourselves so every stage is explicable. Trade-off: more code, but
  total visibility into what's happening.
- **Hybrid retrieval (BM25 + dense).** BM25 for keyword/identifier
  matches (`AutoModel.from_pretrained`), dense for semantic
  ("how do I make my model faster"). Reciprocal rank fusion combines
  them rank-only, sidestepping the score-normalization problem.
- **Three chunking strategies, compared empirically.** Wisdom in the
  RAG community is contradictory; the only honest answer is "it
  depends" with data attached.
- **Document-level relevance, chunk-level retrieval.** A chunk is
  relevant if its source doc is. Dedupe by doc_id before computing
  metrics so multiple chunks from the same doc don't inflate recall.
- **Lazy-loaded models.** `Embedder` and `CrossEncoderReranker` only
  download weights on first use, not on construction. Tests run fast
  by mocking; FastAPI startup stays cheap.
- **mypy strict settings + warn_unused_ignores.** Type discipline
  enforced in CI. Real bugs caught (e.g., the `list` invariance issue
  on cross-encoder predict).
- **90% coverage gate, with documented omits.** Bullshit tests just
  to hit a number degrade the suite. Each omit (`fetcher.py`,
  `eval_queries.py`) has an inline justification.

## Repo map

```
src/docsense/                  Source code (per-stage modules)
tests/unit/                    Unit tests (~190, 90% CI gate)
tests/integration/             End-to-end tests with mocked LLM
tests/snapshots/               prompt_default.txt — pinned default prompt
notebooks/                     chunking_comparison.ipynb (rendered Phase 1)
scripts/                       Pipeline scripts:
                                 fetch_docs.py     — pull HF docs
                                 build_index.py    — build a FAISS index
                                                     for one chunking strategy
                                 search.py         — CLI search interface
                                 run_bakeoff.py    — reproducible bakeoff runner
                                                     (--pipeline / --eval-set)
                                 generate_structural_queries.py
                                                   — produce unbiased
                                                     structural eval set
configs/                       YAML presets for run_bakeoff.py:
                                 default.yaml, no-rerank.yaml,
                                 header-strategy.yaml
evaluations/                   Committed eval artifacts:
├── README.md                   conventions for the directory
├── baselines/                  immutable reference numbers
├── eval_sets/                  versioned query distributions
├── reports/                    bakeoff results per run
└── analyses/                   markdown interpretations of reports
data/                          (gitignored) raw docs, embeddings, indices
docs/journal/                  (gitignored) personal engineering journal —
                               decisions, surprises, debug logs
docs/architecture.md           This file
docs/phase-2-4-scope.md        Phase 2-4 scope + block-paced plan
docs/eval-methodology.md       What's measured where; PR vs nightly vs manual
CLAUDE.md                      Dev/Claude instructions + active roadmap
.claude/settings.json          Project Claude permissions (workflow allowlist)
.github/workflows/ci.yml       CI: lint, typecheck, test+coverage
.pre-commit-config.yaml        Pre-commit hooks (commit + pre-push)
pyproject.toml                 Build + tool config (ruff, mypy, pytest,
                               coverage)
```

## Evaluation methodology

The full breakdown — what's measured, on which eval sets, where it
runs (PR / nightly / manual), and where artifacts live — is in
[`docs/eval-methodology.md`](eval-methodology.md). Quick orientation:

- **Retrieval (Phase 1+):** P@k, Recall@k, MRR, nDCG against curated
  and structural eval sets. Run via `scripts/run_bakeoff.py`.
- **Generation contracts (Phase 2):** citation-preservation invariant,
  strict token-budget enforcement, prompt snapshot, generator-flow
  contract tests. All run on every PR.
- **Generation behavior (Phase 3+):** faithfulness, answer relevance,
  citation grounding, real no-answer behavior. LLM-judge evals,
  manual or nightly. Not in PR CI.
- **Serving (Phase 4):** latency percentiles, error rate, query
  traces. Operational metrics, not quality metrics.

## Status snapshot

| | |
|---|---|
| **Current phase** | Pre-Phase-3 Block 1B closed; Phase 3 (QLoRA fine-tuning) up next |
| **Phase 1 finding** | Recursive chunking wins under dense-only retrieval (MRR 0.692). Established the eval set + corrected metrics. |
| **Phase 2 finding** | The "fixed wins under hybrid+rerank" result on the curated set didn't replicate on the structural set — exposed eval-set bias as a real issue. Production default: hybrid+rerank with `chunking.strategy=recursive`. |
| **Pre-Phase-3 finding** | Qwen 2.5 7B Instruct cites correctly on ~54% of in-corpus answers (cross-set agreement) and refuses 100% of off-corpus queries. Llama 3.1 8B as judge anchors faithfulness at 0.75 — calibration prerequisite for absolute-faithfulness claims. **Citation rate is the highest-leverage Phase 3 fine-tune target.** Full analysis: [`evaluations/analyses/2026-05-06-baseline-generation-eval.md`](../evaluations/analyses/2026-05-06-baseline-generation-eval.md). |
| **Next experiment** | Phase 3 QLoRA fine-tune targeting the citation-rate gap; AnthropicJudge calibration if absolute faithfulness becomes load-bearing. |
| **Test coverage** | 266 tests, 90% CI gate enforced |
| **Workflow** | PR-based, branch-protected `main`, auto-merge with rebase on passing CI; pre-push pytest hook locally. |
| **Recent commit** | See `git log` for the canonical state |
| **Metric-level view** | See [`evaluations/performance.md`](../evaluations/performance.md) for the per-subsystem report card with measurement provenance and explicit gaps |
