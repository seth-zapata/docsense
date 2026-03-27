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

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Test
pytest                                    # all tests
pytest -m "not slow and not gpu"          # fast tests only
pytest --cov=docsense --cov-report=term   # with coverage
```

## Tech Stack

- Python 3.11+, PyTorch, Hugging Face (Transformers, PEFT, Accelerate, Datasets, TRL)
- sentence-transformers for embeddings + cross-encoder re-ranking
- FAISS for vector store, rank_bm25 for sparse retrieval
- FastAPI + uvicorn for serving
- structlog for structured logging, W&B for experiment tracking
- Docker for containerization

## Conventions

- Source code in `src/docsense/`, tests in `tests/`
- Config via pydantic-settings (env vars + config files)
- Type hints throughout, ruff for linting/formatting
- Tests use pytest with markers: `slow`, `gpu`, `integration`
