"""Fetch HF Transformers documentation source files."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import TYPE_CHECKING

from docsense.config import DATA_DIR

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

TRANSFORMERS_REPO = "https://github.com/huggingface/transformers.git"
DOCS_SUBPATH = "docs/source/en"
DEFAULT_OUTPUT_DIR = DATA_DIR / "raw" / "transformers"


def fetch_hf_docs(
    output_dir: Path | None = None,
    repo_url: str = TRANSFORMERS_REPO,
    docs_path: str = DOCS_SUBPATH,
) -> Path:
    """Download HF Transformers markdown docs via git sparse checkout.

    Clones only the docs directory (not the full repo) and copies
    the markdown files to the output directory.

    Returns the path to the output directory containing .md files.
    """
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_clone = output_dir.parent / "_clone_tmp"
    if tmp_clone.exists():
        shutil.rmtree(tmp_clone)

    try:
        logger.info("Cloning docs from %s (sparse checkout)...", repo_url)

        # Shallow clone with sparse checkout — only downloads the docs directory.
        # We avoid --filter=blob:none because sparse-checkout needs actual blob
        # content for the matched paths, and the blobless filter creates stubs.
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--no-checkout",
                repo_url,
                str(tmp_clone),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        subprocess.run(
            ["git", "sparse-checkout", "init", "--cone"],
            cwd=tmp_clone,
            check=True,
            capture_output=True,
            text=True,
        )

        subprocess.run(
            ["git", "sparse-checkout", "set", docs_path],
            cwd=tmp_clone,
            check=True,
            capture_output=True,
            text=True,
        )

        subprocess.run(
            ["git", "checkout"],
            cwd=tmp_clone,
            check=True,
            capture_output=True,
            text=True,
        )

        source_dir = tmp_clone / docs_path
        if not source_dir.exists():
            msg = f"Docs path {docs_path} not found in cloned repo"
            raise FileNotFoundError(msg)

        md_files = [f for f in source_dir.rglob("*.md") if f.is_file()]
        if not md_files:
            msg = f"No .md files found in {source_dir}"
            raise FileNotFoundError(msg)

        for md_file in md_files:
            rel = md_file.relative_to(source_dir)
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(md_file, dest)

        logger.info("Fetched %d markdown files to %s", len(md_files), output_dir)
        return output_dir

    finally:
        if tmp_clone.exists():
            shutil.rmtree(tmp_clone)
