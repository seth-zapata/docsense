#!/usr/bin/env python3
"""Search the FAISS index with a natural language query.

Usage:
    python scripts/search.py "How do I use AutoModel?"
    python scripts/search.py "What is gradient checkpointing?" --top-k 5
    python scripts/search.py "tokenizer padding" --strategy fixed
"""

from __future__ import annotations

import argparse
import logging
import pickle

import faiss

from docsense.config import DATA_DIR, Settings
from docsense.embedding.embedder import Embedder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INDEX_DIR = DATA_DIR / "index"


def search(query: str, strategy: str, top_k: int) -> None:
    config = Settings()
    idx_dir = INDEX_DIR / strategy

    if not idx_dir.exists():
        logger.error("No index found at %s", idx_dir)
        logger.error("Run `python scripts/build_index.py --strategy %s` first.", strategy)
        raise SystemExit(1)

    # Load index and chunks
    index = faiss.read_index(str(idx_dir / "index.faiss"))
    with open(idx_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)  # noqa: S301

    logger.info("Loaded index: %d vectors, %d chunks", index.ntotal, len(chunks))

    # Embed query
    embedder = Embedder(config.embedding)
    query_emb = embedder.embed_texts([query])

    # Search
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_emb, k)

    # Display results
    print(f"\n{'=' * 80}")
    print(f"Query: {query}")
    print(f"Strategy: {strategy} | Top {top_k} results")
    print(f"{'=' * 80}\n")

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=True), 1):
        if idx == -1:
            break
        chunk = chunks[idx]
        print(f"--- Result {rank} (score: {score:.4f}) ---")
        print(f"Source: {chunk.doc_id} | Chunk {chunk.chunk_index}")
        # Show first 300 chars of the chunk
        preview = chunk.text[:300]
        if len(chunk.text) > 300:
            preview += "..."
        print(preview)
        print()


def main():
    parser = argparse.ArgumentParser(description="Search the document index")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "header"],
        default="header",
        help="Which index to search (default: header)",
    )
    args = parser.parse_args()

    search(args.query, args.strategy, args.top_k)


if __name__ == "__main__":
    main()
