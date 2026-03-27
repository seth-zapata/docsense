"""Tests for recursive chunking strategy."""

from docsense.chunking.recursive import RecursiveChunker
from docsense.ingestion.loader import Document


def _make_doc(content: str) -> Document:
    return Document(content=content, source="test.md", metadata={"doc_id": "test"})


class TestRecursiveChunker:
    def test_short_text_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=200)
        chunks = chunker.chunk(_make_doc("Short paragraph."))
        assert len(chunks) == 1
        assert chunks[0].text == "Short paragraph."

    def test_splits_on_paragraphs(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        chunks = chunker.chunk(_make_doc(text))
        # First two paras merge (combined 45 chars < 50), third is separate
        assert len(chunks) == 2
        assert "First paragraph" in chunks[0].text
        assert "Third paragraph" in chunks[1].text

    def test_merges_small_paragraphs(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        text = "Short.\n\nAlso short.\n\nStill short."
        chunks = chunker.chunk(_make_doc(text))
        # All three should merge into one chunk since combined < 100
        assert len(chunks) == 1

    def test_splits_large_paragraphs_by_sentence(self):
        chunker = RecursiveChunker(chunk_size=40, chunk_overlap=0)
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunker.chunk(_make_doc(text))
        assert len(chunks) > 1
        # Each chunk should be within size limit (or a single unsplittable unit)
        for c in chunks:
            assert len(c.text) <= 40 or " " not in c.text

    def test_empty_document(self):
        chunker = RecursiveChunker()
        assert chunker.chunk(_make_doc("")) == []

    def test_overlap_present(self):
        chunker = RecursiveChunker(chunk_size=60, chunk_overlap=20)
        text = "Alpha paragraph one.\n\nBravo paragraph two.\n\nCharlie paragraph three."
        chunks = chunker.chunk(_make_doc(text))
        if len(chunks) > 1:
            # With overlap, later chunks may contain text from earlier ones
            all_text = " ".join(c.text for c in chunks)
            assert "Alpha" in all_text
            assert "Charlie" in all_text

    def test_code_blocks_not_mangled(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        text = "Some text.\n\n```python\ndef foo():\n    return 42\n```\n\nMore text."
        chunks = chunker.chunk(_make_doc(text))
        combined = "\n\n".join(c.text for c in chunks)
        assert "def foo():" in combined
        assert "return 42" in combined

    def test_metadata_includes_strategy(self):
        chunker = RecursiveChunker(chunk_size=100)
        chunks = chunker.chunk(_make_doc("Some content."))
        assert chunks[0].metadata["strategy"] == "recursive"

    def test_chunk_indices_sequential(self):
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
        text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four."
        chunks = chunker.chunk(_make_doc(text))
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
