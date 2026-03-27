"""Markdown header-based chunking strategy."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from docsense.chunking.base import Chunk, ChunkingStrategy

if TYPE_CHECKING:
    from docsense.ingestion.loader import Document

# Matches markdown headers: # Title, ## Section, ### Subsection, etc.
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class HeaderChunker(ChunkingStrategy):
    """Split documents on markdown headers, preserving section structure.

    Each section (header + body until the next same-or-higher-level header)
    becomes a chunk. Sections exceeding max_chunk_size are split at
    paragraph boundaries as a fallback.
    """

    def __init__(
        self,
        max_chunk_size: int = 1024,
        min_header_level: int = 2,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.min_header_level = min_header_level

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        sections = self._split_by_headers(text)
        chunks = []

        for header, body in sections:
            section_text = f"{header}\n\n{body}".strip() if header else body.strip()
            if not section_text:
                continue

            if len(section_text) <= self.max_chunk_size:
                chunks.append(section_text)
            else:
                chunks.extend(self._split_large_section(header, body))

        return [
            Chunk(
                text=t,
                doc_id=document.doc_id,
                chunk_index=i,
                metadata={"strategy": "header"},
            )
            for i, t in enumerate(chunks)
        ]

    def _split_by_headers(self, text: str) -> list[tuple[str, str]]:
        """Split text into (header_line, body) pairs at header boundaries."""
        matches = list(_HEADER_RE.finditer(text))

        if not matches:
            return [("", text)]

        sections = []

        # Content before the first header (preamble)
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(("", preamble))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            if level < self.min_header_level:
                # Skip headers above our threshold (e.g., skip h1 page titles)
                # but still include their content in the next section
                pass

            header_line = match.group(0)
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()

            sections.append((header_line, body))

        return sections

    def _split_large_section(self, header: str, body: str) -> list[str]:
        """Split an oversized section at paragraph boundaries."""
        paragraphs = re.split(r"\n\n+", body)
        chunks = []
        current = header + "\n\n" if header else ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            candidate = (current + "\n\n" + para).strip() if current.strip() else para
            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If a single paragraph exceeds max size, include it as-is
                # rather than losing content
                current = para

        if current.strip():
            chunks.append(current.strip())

        return chunks
