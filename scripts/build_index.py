#!/usr/bin/env python3
"""Build a FAISS index from ingested documents.

Usage:
    python scripts/build_index.py                    # uses default data/raw/transformers
    python scripts/build_index.py --docs-dir path/   # custom docs directory
    python scripts/build_index.py --strategy header   # choose chunking strategy
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from docsense.chunking.fixed import FixedSizeChunker
from docsense.chunking.header import HeaderChunker
from docsense.chunking.recursive import RecursiveChunker
from docsense.config import DATA_DIR, Settings
from docsense.embedding.embedder import Embedder
from docsense.ingestion.loader import load_markdown_directory
from docsense.retrieval.dense import DenseRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHUNKERS = {
    "fixed": lambda cfg: FixedSizeChunker(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    ),
    "recursive": lambda cfg: RecursiveChunker(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    ),
    "header": lambda cfg: HeaderChunker(
        max_chunk_size=cfg.chunking.chunk_size,
    ),
}

INDEX_DIR = DATA_DIR / "index"


def build_index(docs_dir: Path, strategy: str) -> None:
    config = Settings()

    # 1. Load documents
    logger.info("Loading documents from %s...", docs_dir)
    documents = load_markdown_directory(docs_dir)
    logger.info("Loaded %d documents", len(documents))

    # 2. Chunk
    chunker = CHUNKERS[strategy](config)
    logger.info("Chunking with strategy=%s (size=%d)...", strategy, config.chunking.chunk_size)
    t0 = time.time()
    chunks = chunker.chunk_many(documents)
    logger.info("Created %d chunks in %.1fs", len(chunks), time.time() - t0)

    # 3. Embed
    logger.info("Embedding chunks (model=%s)...", config.embedding.model_name)
    embedder = Embedder(config.embedding)
    t0 = time.time()
    embeddings = embedder.embed_chunks(chunks)
    logger.info("Embedded in %.1fs", time.time() - t0)

    # 4. Index
    dim = embeddings.shape[1]
    retriever = DenseRetriever(dimension=dim)
    retriever.add(chunks, embeddings)
    logger.info("FAISS index built: %d vectors, dim=%d", retriever.index.ntotal, dim)

    # 5. Save
    out_dir = INDEX_DIR / strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    import faiss

    faiss.write_index(retriever.index, str(out_dir / "index.faiss"))
    with open(out_dir / "chunks.pkl", "wb") as f:
        pickle.dump(retriever.chunks, f)
    with open(out_dir / "embeddings.npy", "wb") as f:
        np.save(f, embeddings)

    logger.info("Saved index to %s", out_dir)
    logger.info("Done! Run `python scripts/search.py --strategy %s` to query.", strategy)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from documents")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DATA_DIR / "raw" / "transformers",
        help="Directory containing markdown files",
    )
    parser.add_argument(
        "--strategy",
        choices=list(CHUNKERS),
        default="header",
        help="Chunking strategy (default: header)",
    )
    args = parser.parse_args()

    if not args.docs_dir.exists():
        logger.error("Docs directory not found: %s", args.docs_dir)
        logger.error("Run `python scripts/fetch_docs.py` first.")
        raise SystemExit(1)

    build_index(args.docs_dir, args.strategy)


if __name__ == "__main__":
    main()
