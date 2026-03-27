"""Document loaders for fetching and parsing HF Transformers docs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single ingested document with metadata."""

    content: str
    source: str
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", self.source)


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?\n)---\s*\n", re.DOTALL)


def _strip_frontmatter(text: str) -> tuple[str, dict]:
    """Remove YAML frontmatter and extract key-value pairs."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return text, {}

    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    return text[match.end() :], meta


def _extract_title(content: str) -> str | None:
    """Extract the first markdown heading as the document title."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return None


def load_markdown_directory(directory: Path) -> list[Document]:
    """Load all markdown files from a directory into Documents."""
    directory = Path(directory)
    if not directory.is_dir():
        msg = f"Directory not found: {directory}"
        raise FileNotFoundError(msg)

    documents = []
    md_files = sorted(directory.rglob("*.md"))

    for path in md_files:
        raw = path.read_text(encoding="utf-8")
        content, frontmatter = _strip_frontmatter(raw)
        content = content.strip()

        if not content:
            logger.debug("Skipping empty file: %s", path)
            continue

        rel_path = str(path.relative_to(directory))
        title = frontmatter.get("title") or _extract_title(content) or rel_path

        documents.append(
            Document(
                content=content,
                source=rel_path,
                metadata={
                    "doc_id": rel_path,
                    "title": title,
                    **frontmatter,
                },
            )
        )

    logger.info("Loaded %d documents from %s", len(documents), directory)
    return documents
