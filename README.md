# docsense

Production-grade RAG system with fine-tuned LLM for technical
documentation Q&A. Built on the Hugging Face Transformers docs as the
target corpus.

> Development assisted by Claude (Anthropic) as an AI coding assistant.

## Architecture

```
Query → Embed → Hybrid Retrieval (BM25 + Dense) → RRF Fusion
      → Cross-Encoder Re-rank → Context Assembly
      → LLM Generation → Response
```

For the full system overview — design rationale, phase progress, and
how the pieces fit together — see
[**`docs/architecture.md`**](docs/architecture.md).

For dev setup and commands, see [`CLAUDE.md`](CLAUDE.md).

## Tech stack

Python 3.12 · PyTorch · Hugging Face Transformers / PEFT / Accelerate /
TRL · sentence-transformers · FAISS · rank_bm25 · FastAPI · structlog ·
W&B
