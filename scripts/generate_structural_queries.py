#!/usr/bin/env python3
"""Generate a structural eval-query set from the ingested HF docs corpus.

Output is JSON with the same `(query, [doc_id_prefix])` shape as the
hand-curated eval set, ready to be loaded by the bakeoff notebook and
treated as a complementary, unbiased eval signal.

Usage:
    python scripts/generate_structural_queries.py                       # defaults
    python scripts/generate_structural_queries.py --n 50 --seed 0       # custom
    python scripts/generate_structural_queries.py --docs-dir path/      # custom docs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from docsense.config import DATA_DIR
from docsense.evaluation.structural_queries import generate_structural_queries
from docsense.ingestion.loader import load_markdown_directory

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluations" / "eval_sets" / "structural.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DATA_DIR / "raw" / "transformers",
        help="Directory of ingested markdown docs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path. Default lives under evaluations/ so the eval set is committed and reproducible.",
    )
    parser.add_argument("--n", type=int, default=30, help="Number of queries to sample.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    documents = load_markdown_directory(args.docs_dir)
    logger.info("Loaded %d documents", len(documents))

    queries = generate_structural_queries(documents, n_queries=args.n, seed=args.seed)
    logger.info("Generated %d structural queries", len(queries))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump([{"query": q, "relevant": rel} for q, rel in queries], f, indent=2)

    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
