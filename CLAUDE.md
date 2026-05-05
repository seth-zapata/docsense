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
# Install (dev)
pip install -e ".[dev]"
pre-commit install                        # one-time, enables git-commit hooks

# Lint and type-check
ruff check src/ tests/
ruff format src/ tests/
mypy                                      # uses files/config from pyproject.toml

# Test
pytest                                    # all tests
pytest -m "not slow and not gpu"          # fast tests only
pytest --cov=docsense --cov-report=term   # with coverage
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
- **Phase 2 — in progress.** Cross-encoder re-ranker wired into
  `HybridRetriever`, re-bakeoff with re-ranking, context assembly, prompt
  construction, base-model generation (Mistral 7B / Llama 3 8B).
- **Phase 3.** QLoRA fine-tuning track — supervised dataset construction,
  training script, eval against base-model baseline.
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
