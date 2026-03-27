#!/usr/bin/env python3
"""Fetch HF Transformers documentation to data/raw/."""

from docsense.ingestion.fetcher import fetch_hf_docs


def main():
    output = fetch_hf_docs()
    count = len(list(output.rglob("*.md")))
    print(f"Done. {count} markdown files in {output}")


if __name__ == "__main__":
    main()
