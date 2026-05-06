# docsense

Production-grade RAG system with fine-tuned LLM for technical documentation Q&A.
Built on Hugging Face Transformers docs as the target corpus.

## Architecture

```
Query → Embed → Hybrid Retrieval (BM25 + Dense) → Score Fusion → Cross-Encoder Re-rank → Context Assembly → LLM Generation → Response
```

## Key Design Decisions

- **No LangChain/LlamaIndex for core logic.** Retrieval, context assembly, prompt construction, and generation are built from scratch.
- **Hybrid retrieval:** BM25 (sparse) + sentence-transformers (dense) with reciprocal rank fusion.
- **Cross-encoder re-ranking** after initial retrieval for precision.
- **Three chunking strategies** compared quantitatively: fixed-size, recursive/semantic, section-header-based.
- **QLoRA fine-tuning** on Mistral 7B or Llama 3 8B, targeting 16GB VRAM (RTX 5080).
- **FAISS** for vector storage.

## Project Layout

```
src/docsense/
├── ingestion/     # Document fetching and parsing (HF docs)
├── chunking/      # Chunking strategies (fixed, semantic, header-based)
├── embedding/     # Embedding pipeline (sentence-transformers)
├── retrieval/     # Dense retrieval (FAISS), sparse (BM25), hybrid fusion
├── reranking/     # Cross-encoder re-ranking
├── evaluation/    # Retrieval metrics (P@k, MRR, nDCG) + generation metrics
├── generation/    # LLM inference, context assembly, prompt construction
├── api/           # FastAPI serving layer
├── tracing/       # Structured observability and query tracing
└── config.py      # Centralized configuration
```

## Commands

```bash
# Install (dev — no real model loads, all tests use mocks)
pip install -e ".[dev]"
pre-commit install                                   # commit hooks: ruff, mypy
pre-commit install --hook-type pre-push              # pre-push hook: pytest

# Install (dev + GPU — adds bitsandbytes for NF4 quantization).
# Required when actually loading the LLM via Generator on a CUDA host.
pip install -e ".[dev,gpu]"

# Lint and type-check
ruff check src/ tests/
ruff format src/ tests/
mypy                                                 # uses files/config from pyproject.toml

# Test
pytest                                               # all tests
pytest -m "not slow and not gpu"                     # fast tests only
pytest --cov=docsense --cov-report=term              # with coverage
```

### Model cache location (`HF_HOME`)

The HuggingFace transformers stack caches model weights at
`~/.cache/huggingface/hub/` by default — which on WSL lives inside
the WSL2 ext4 virtual disk on the C: drive. The 7-8B Instruct models
docsense uses (Qwen 2.5 7B for generation, Llama 3.1 8B for the
LLM-judge) are ~15-16 GB each on disk at BF16, plus ~80 MB each for
the cross-encoder and embedder. Total ~31 GB just for the LLMs.

If C: is constrained, redirect the cache to a roomier drive **before**
any model loads:

```bash
mkdir -p /mnt/e/hf-cache
echo 'export HF_HOME=/mnt/e/hf-cache' >> ~/.bashrc   # persists across sessions
source ~/.bashrc
```

`HF_HOME` covers `transformers`, `sentence-transformers`, and the
HuggingFace CLI uniformly. Trade-off: WSL2 → Windows-mounted-drive
I/O is slower than native ext4, so cold model loads (per-session,
~once) take ~30-60 s longer. Inference latency is unaffected — once
weights are in VRAM, where they came from doesn't matter.

## Workflow

`main` is branch-protected: direct push is blocked, every change goes through
a PR with `lint`, `typecheck`, and `test` checks required. Auto-merge with
rebase + branch deletion is enabled, so the typical cycle is:

```bash
git checkout -b <branch-name>
# ...commits...
git push -u origin <branch-name>           # pre-push hook runs pytest locally
gh pr create --fill --title "..." --body "..."
gh pr merge --auto --rebase --delete-branch  # merges when CI passes
```

## Tech Stack

- Python 3.12, PyTorch, Hugging Face (Transformers, PEFT, Accelerate, Datasets, TRL)
- sentence-transformers for embeddings + cross-encoder re-ranking
- FAISS for vector store, rank_bm25 for sparse retrieval
- FastAPI + uvicorn for serving
- structlog for structured logging, W&B for experiment tracking
- Docker for containerization

## Conventions

- Source code in `src/docsense/`, tests in `tests/`
- Config via pydantic-settings (env vars + config files)
- Type hints throughout, ruff for linting/formatting, mypy for type checking
- Tests use pytest with markers: `slow`, `gpu`, `integration`

## Roadmap

> The full system overview lives in [`docs/architecture.md`](docs/architecture.md).
> This section is the active worklist. **When closing a phase, update both
> this Phases list and the corresponding section in `docs/architecture.md`** so
> the architecture doc stays in sync with reality.

### Phases

- **Phase 1 — closed.** Ingestion, three chunking strategies, hybrid retrieval
  (BM25 + FAISS + RRF), retrieval metrics, chunking bakeoff, eval scaffold,
  CI fortification (test matrix, mypy, pre-commit).
- **Phase 2 — closed (2026-05-06).** Cross-encoder re-ranker wired into
  `HybridRetriever`. Bakeoff investigation (Block B+) ablated BM25/RRF/rerank
  separately and added a structural eval set; ablation revealed the original
  "fixed wins" finding was a curated-eval artifact and surfaced eval-set bias
  as a real concern. Generation surface scaffolded — `Answer`/`Citation`/
  `ChunkRef` types, `ContextAssembler` (strict token budget), `PromptBuilder`
  (snapshot-tested), `Generator` (mockable, with citation parser). Rule-based
  invariants pinned: citation-preservation as a pydantic model_validator,
  tokenizer-agnostic budget enforcement (caught and fixed a real
  non-additive-tokenizer drift bug). 194 tests, 90% CI coverage gate. **The
  full pipeline is end-to-end runnable; real LLM behavior is exercised in
  Phase 3 / pre-Phase-3 LLM-judge evals, not yet.**
- **Pre-Phase-3 Block 1B — closed (2026-05-06).** LLM-judge framework
  (`LLMJudge` ABC, `LlamaJudge` concrete impl with NF4 4-bit), rule-based
  evals (`check_no_answer_behavior`, `check_citations_grounded`),
  `scripts/run_generation_eval.py` (sequential generator → judge load
  on 12 GB GPU), and the first end-to-end LLM-judged baseline at
  `evaluations/baselines/pre_phase3_generation_base.json`. Findings:
  citation rate ~54% (cross-set agreement; highest-leverage fine-tune
  target), 100% refusal on off-corpus, faithfulness mean 0.75 with
  judge anchor saturation flagged. Full analysis:
  `evaluations/analyses/2026-05-06-baseline-generation-eval.md`.
- **Phase 3.** QLoRA fine-tuning track — supervised dataset construction,
  training script, eval against the pre-Phase-3 baseline. Pre-Phase-3
  Block 1B's deferred items still in scope: 5c LLM-generated eval set
  for cross-validation, AnthropicJudge calibration if absolute
  faithfulness becomes load-bearing.
- **Phase 4.** FastAPI serving + `tracing/` (structured query logs), end-to-end
  generation eval (faithfulness, answer relevance), Docker packaging.

### Deferred CI fortification

Items left out of the initial CI fortification (test matrix + mypy + pre-commit
shipped 2026-05-04, coverage gate at 90% added same day). Pick up as the
relevant code or context lands:

- **Notebook smoke test.** `nbmake` pytest plugin runs notebooks as tests.
  Worth adding once notebooks become user-facing artifacts in Phase 2/3.
- **Dependency audit.** `pip-audit` action against `pyproject.toml` once the
  dep set stabilizes (post-Phase 3, after fine-tuning settles).
- **CD pipeline (deploy stages).** Build container → push to GHCR → deploy
  to staging → smoke test → manual gate → prod. Phase 4 work — only kicks in
  when the FastAPI service is ready to deploy.

### Eval methodology — additional sets

- **5c (LLM-generated queries).** Use the Anthropic SDK to generate
  "what would a user ask about this section?" queries against a held-out doc
  set. Add when re-bakeoff happens after the re-ranker, so three eval sets
  exist (curated, structural, LLM-generated) and we can compare agreement.
