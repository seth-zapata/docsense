"""Document loaders for fetching and parsing HF Transformers docs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """A single ingested document with metadata."""

    content: str
    source: str
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", self.source)
