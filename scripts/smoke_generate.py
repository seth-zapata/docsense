#!/usr/bin/env python3
"""Smoke test: load the configured LLM and run one query end-to-end.

Manual verification step for Block 1A. Confirms that:
- The configured generation model (Qwen 2.5 7B Instruct by default)
  loads at NF4 4-bit and fits within 12 GB VRAM.
- The chat-template refactor in PromptBuilder + Generator produces
  text the model honors (no garbled output, no obvious truncation).
- The full pipeline composes — retrieval → reranking → context
  assembly → prompt → generate → typed Answer.
- Citations parse correctly out of the generated text.

Usage::

    python scripts/smoke_generate.py
    python scripts/smoke_generate.py --query "How do I install transformers?"
    python scripts/smoke_generate.py --strategy header --no-rerank

NOT in CI. The first run will download ~5 GB of weights for the system
model (and the cross-encoder, if not already cached). Run once on the
target hardware, capture the stdout, and commit it under
``evaluations/manual-runs/`` as a portfolio artifact.

Prerequisites:
- Indices built: ``python scripts/build_index.py --strategy {fixed,recursive,header}``
- ``[gpu]`` extras installed if running on GPU with NF4 quantization:
  ``pip install -e ".[dev,gpu]"`` — gives you bitsandbytes.
- Set ``GenerationConfig.use_4bit_quantization=False`` (or pass
  ``--no-4bit``) for CPU-only runs.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time

import faiss

from docsense.config import DATA_DIR, GenerationConfig, Settings
from docsense.embedding.embedder import Embedder
from docsense.generation.context import ContextAssembler
from docsense.generation.generator import Generator
from docsense.generation.prompt import PromptBuilder
from docsense.generation.types import ChunkRef
from docsense.reranking.reranker import CrossEncoderReranker
from docsense.retrieval.dense import DenseRetriever
from docsense.retrieval.hybrid import HybridRetriever
from docsense.retrieval.sparse import SparseRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_QUERY = "How do I use AutoModel.from_pretrained() to load a pretrained model?"
INDEX_DIR = DATA_DIR / "index"


def _load_chunks_and_index(strategy: str) -> tuple[faiss.Index, list]:
    idx_dir = INDEX_DIR / strategy
    if not idx_dir.exists():
        msg = (
            f"No index at {idx_dir}. Run "
            f"`python scripts/build_index.py --strategy {strategy}` first."
        )
        raise FileNotFoundError(msg)
    index = faiss.read_index(str(idx_dir / "index.faiss"))
    with open(idx_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)  # noqa: S301
    return index, chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="The natural-language question.")
    parser.add_argument(
        "--strategy",
        choices=("fixed", "recursive", "header"),
        default="recursive",
        help="Chunking strategy whose index to load (default: recursive).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the cross-encoder reranker (dense+sparse RRF only).",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help=(
            "Disable NF4 4-bit quantization. Use this on CPU or environments "
            "without bitsandbytes installed."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Override GenerationConfig.device. Default 'auto' lets accelerate "
            "distribute weights across devices (slow if VRAM is tight and "
            "weights spill to CPU). Pass 'cuda:0' to force all-GPU placement "
            "(fast but OOMs if model doesn't fit). Pass 'cpu' for CPU-only."
        ),
    )
    args = parser.parse_args()

    settings = Settings()
    overrides: dict = {}
    if args.no_4bit:
        overrides["use_4bit_quantization"] = False
    if args.device is not None:
        overrides["device"] = args.device
    if overrides:
        settings.generation = GenerationConfig(**{**settings.generation.model_dump(), **overrides})

    logger.info("=== Smoke generate ===")
    logger.info("System model:   %s", settings.generation.model_name)
    logger.info("4-bit:          %s", settings.generation.use_4bit_quantization)
    logger.info("Chunking:       %s", args.strategy)
    logger.info("Reranker:       %s", "off" if args.no_rerank else "on")
    logger.info("Query:          %s", args.query)

    # --- Retrieval setup ----------------------------------------------------
    logger.info("\nLoading index + chunks...")
    index, chunks = _load_chunks_and_index(args.strategy)
    logger.info("  %d chunks loaded", len(chunks))

    embedder = Embedder(settings.embedding)
    dense = DenseRetriever(dimension=index.d)
    dense.index = index
    dense.chunks = chunks

    sparse = SparseRetriever()
    sparse.add(chunks)

    reranker = None
    if not args.no_rerank:
        reranker = CrossEncoderReranker(settings.reranking)

    hybrid = HybridRetriever(dense, sparse, embedder, settings.retrieval, reranker=reranker)

    # --- Retrieve + assemble + prompt --------------------------------------
    t0 = time.perf_counter()
    results = hybrid.search(args.query)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    logger.info("\nRetrieved %d results in %.0f ms", len(results), retrieve_ms)

    chunk_refs = [
        ChunkRef(
            doc_id=r.chunk.doc_id,
            chunk_id=r.chunk.chunk_id,
            score=r.score,
            text=r.chunk.text,
        )
        for r in results
    ]

    assembler = ContextAssembler(max_tokens=settings.generation.max_context_tokens)
    context, included = assembler.assemble(chunk_refs)
    logger.info("Assembled context with %d chunks", len(included))

    prompt_builder = PromptBuilder()
    messages = prompt_builder.build(query=args.query, context=context)

    # --- Generate -----------------------------------------------------------
    logger.info("\nLoading generation model (this may take a minute on first run)...")
    generator = Generator(settings.generation)

    logger.info("Running inference...")
    t0 = time.perf_counter()
    answer = generator.generate(messages, included)
    generate_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "  generated in %.0f ms (%.0f ms per token)",
        generate_ms,
        generate_ms / max(answer.metadata.completion_tokens or 1, 1),
    )

    # --- Output -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"QUERY: {args.query}")
    print("=" * 80)
    print(f"\nANSWER:\n{answer.text}\n")

    if answer.citations:
        print(f"CITATIONS ({len(answer.citations)}):")
        for cit in answer.citations:
            print(f"  - {cit.doc_id} (chunk: {cit.chunk_id})")
    else:
        print("CITATIONS: (none)")

    print("\nRETRIEVED CHUNKS PASSED TO LLM:")
    for i, c in enumerate(included, 1):
        preview = c.text[:120].replace("\n", " ")
        print(f"  [{i}] {c.doc_id} (score={c.score:.3f})")
        print(f"      {preview}{'...' if len(c.text) > 120 else ''}")

    print("\nMETADATA:")
    print(f"  model:            {answer.metadata.model_name}")
    print(f"  prompt tokens:    {answer.metadata.prompt_tokens}")
    print(f"  completion tokens: {answer.metadata.completion_tokens}")
    print(f"  inference latency: {answer.metadata.latency_ms:.0f} ms")
    print(f"  total wall:       {(retrieve_ms + generate_ms):.0f} ms")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
