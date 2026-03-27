"""Tests for header-based chunking strategy."""

from docsense.chunking.header import HeaderChunker
from docsense.ingestion.loader import Document


def _make_doc(content: str) -> Document:
    return Document(content=content, source="test.md", metadata={"doc_id": "test"})


SAMPLE_MD = """\
# Page Title

Introduction paragraph.

## First Section

Content of the first section. This explains how things work.

## Second Section

Content of the second section with more details.

### Subsection

Subsection content goes here.

## Third Section

Final section content.
"""


class TestHeaderChunker:
    def test_splits_on_headers(self):
        chunker = HeaderChunker(max_chunk_size=2000)
        chunks = chunker.chunk(_make_doc(SAMPLE_MD))
        # Should have multiple chunks based on headers
        assert len(chunks) >= 3
        # First section content should be in one chunk
        texts = [c.text for c in chunks]
        assert any("First Section" in t for t in texts)
        assert any("Second Section" in t for t in texts)

    def test_preserves_header_in_chunk(self):
        chunker = HeaderChunker(max_chunk_size=2000)
        chunks = chunker.chunk(_make_doc(SAMPLE_MD))
        # Each chunk that came from a header should include the header
        section_chunks = [c for c in chunks if "Section" in c.text]
        for c in section_chunks:
            assert c.text.startswith("#")

    def test_splits_large_sections(self):
        long_body = ("This is a paragraph. " * 50 + "\n\n") * 5
        text = f"## Big Section\n\n{long_body}"
        chunker = HeaderChunker(max_chunk_size=200)
        chunks = chunker.chunk(_make_doc(text))
        assert len(chunks) > 1
        # First chunk should have the header
        assert chunks[0].text.startswith("## Big Section")

    def test_empty_document(self):
        chunker = HeaderChunker()
        assert chunker.chunk(_make_doc("")) == []

    def test_no_headers(self):
        chunker = HeaderChunker(max_chunk_size=2000)
        text = "Just plain text without any headers."
        chunks = chunker.chunk(_make_doc(text))
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_preamble_before_first_header(self):
        text = "Some intro text.\n\n## Section\n\nSection content."
        chunker = HeaderChunker(max_chunk_size=2000)
        chunks = chunker.chunk(_make_doc(text))
        assert chunks[0].text == "Some intro text."

    def test_metadata_includes_strategy(self):
        chunker = HeaderChunker(max_chunk_size=2000)
        chunks = chunker.chunk(_make_doc("## Section\n\nContent."))
        assert chunks[0].metadata["strategy"] == "header"

    def test_chunk_indices_sequential(self):
        chunker = HeaderChunker(max_chunk_size=2000)
        chunks = chunker.chunk(_make_doc(SAMPLE_MD))
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_code_blocks_preserved(self):
        text = "## API\n\n```python\nmodel = AutoModel.from_pretrained('bert')\n```\n\nUsage above."
        chunker = HeaderChunker(max_chunk_size=2000)
        chunks = chunker.chunk(_make_doc(text))
        combined = "\n".join(c.text for c in chunks)
        assert "AutoModel.from_pretrained" in combined

    def test_min_header_level(self):
        text = "# Title\n\nIntro.\n\n## Section\n\nContent."
        chunker = HeaderChunker(max_chunk_size=2000, min_header_level=2)
        chunks = chunker.chunk(_make_doc(text))
        # h1 title should still appear but section-level splitting at h2
        assert len(chunks) >= 2
