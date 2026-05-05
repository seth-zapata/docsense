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
                                    Phase 1 ✅       Phase 2          Phase 2-3        Phase 4
                                  ┌──────────┐   ┌──────────┐    ┌────────────┐   ┌─────────┐
   User query  ─────────────────► │ Retrieve │ ──►│ Re-rank  │ ──►│  Generate  │ ──► Answer
                                  └──────────┘   └──────────┘    └────────────┘   └─────────┘
                                       ▲              ▲                ▲
                                       │              │                │
                                  BM25+FAISS+RRF  cross-encoder  LLM (Mistral 7B
                                  hybrid          re-ranking      or Llama 3 8B,
                                                                  optional QLoRA)
```

| Stage | Question it answers | Status |
|-------|---------------------|--------|
| **Retrieve** | Of all chunks in the corpus, which are *plausibly* relevant to this query? | Phase 1 ✅ |
| **Re-rank** | Of the candidates retrieval found, which is *most* relevant for *this specific* query? | Phase 2 |
| **Generate** | Given the relevant chunks, what's the actual answer? | Phase 2 (base) + Phase 3 (fine-tuned) |
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

## Phase 2 — re-ranking + generation

### Why the bakeoff result matters for Phase 2

The bakeoff revealed that header-based chunking *was* held back by the
bi-encoder's compression problem — but cross-encoders are
**fundamentally different**. A cross-encoder takes the query and the
chunk text *together* and computes a similarity score directly, no
single-vector summary in between. The compression handicap goes away.

So Phase 2's first experiment is to re-run the chunking bakeoff with
cross-encoder re-ranking enabled. The hypothesis: header recovers
significantly, possibly overtakes recursive. The interview-quality
finding either way: "the optimal chunking strategy depends on whether
you have a re-ranker downstream, and here's the empirical evidence."

### Phase 2 work, in order

1. **Wire the cross-encoder re-ranker into HybridRetriever.** The
   class already exists at `src/docsense/reranking/reranker.py` with
   tests; it just needs to be invoked after RRF fusion. Hybrid takes
   the top-N (default 20) candidates → cross-encoder scores each
   → trim to top-k (default 5).
2. **Re-run the chunking bakeoff with re-ranking on.** Update the
   notebook, capture the corrected numbers, write a journal coda.
   This is *the* interesting Phase 2 experiment.
3. **Context assembly.** Take the reranked top-k chunks and format
   them into LLM-ready context: token-budget management, source
   attribution, deduplication. Lives in `src/docsense/generation/`.
4. **Prompt construction.** System prompt + assembled context + user
   query. This is where "answer only from these sources" instructions
   and citation directives live.
5. **Base-model generation.** Wire up Mistral 7B Instruct or Llama 3
   8B (decision pending; depends on QLoRA fit on 16 GB VRAM). End-to-end
   query → answer working with the off-the-shelf model.

By end of Phase 2, the system answers questions about HF Transformers
docs using a base LLM — no fine-tuning yet, no API yet.

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
tests/unit/                    Unit tests (115 currently, 98% coverage)
tests/integration/             Reserved for end-to-end tests in Phase 2+
notebooks/                     Experiments — chunking_comparison.ipynb
                               is the headline Phase 1 artifact
scripts/                       Pipeline scripts:
                                 fetch_docs.py    — pull HF docs
                                 build_index.py   — build a FAISS index
                                                    for one chunking strategy
                                 search.py        — CLI search interface
                                 generate_structural_queries.py
                                                  — produce unbiased
                                                    eval set from doc
                                                    headings
configs/                       (Reserved) per-environment YAML configs
data/                          (gitignored) raw docs, embeddings, indices
docs/journal/                  (gitignored) personal engineering journal —
                               decisions, surprises, debug logs
docs/architecture.md           This file
CLAUDE.md                      Dev/Claude instructions + active roadmap
.github/workflows/ci.yml       CI: lint, typecheck, test+coverage
.pre-commit-config.yaml        Pre-commit hooks
pyproject.toml                 Build + tool config (ruff, mypy, pytest,
                               coverage)
```

## Evaluation methodology

What's measured at each phase, and how:

- **Retrieval (Phase 1):** P@k, Recall@k, MRR, nDCG against a
  20-query hand-curated set. Will add a structural eval set
  (`evaluation/structural_queries.py`, generated from doc headings)
  in Phase 2 for an unbiased complement. LLM-generated eval set
  ("5c") deferred to Phase 3.
- **Reranking (Phase 2):** Same metrics as retrieval, run on the
  *reranked* top-k. Re-runs the bakeoff to compare strategies in the
  presence of re-ranking.
- **Generation (Phase 2 base, Phase 3 fine-tuned):** Faithfulness
  (does the answer follow from the retrieved chunks?), answer
  relevance (does it actually address the question?), citation
  accuracy. End-to-end eval on a held-out question set.
- **Serving (Phase 4):** Latency percentiles, error rate, downstream
  query traces. SLA-style metrics, not quality metrics.

## Status snapshot

| | |
|---|---|
| **Current phase** | Phase 2 (just starting) |
| **Headline Phase 1 result** | Recursive chunking wins on rank-sensitive retrieval metrics; header is competitive enough that re-ranking might flip the order |
| **Next experiment** | Wire cross-encoder re-ranker into HybridRetriever and re-run the bakeoff |
| **Test coverage** | 98% with a 90% CI gate |
| **Recent commit** | See `git log` for the canonical state |
