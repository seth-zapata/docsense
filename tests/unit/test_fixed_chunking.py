"""Tests for fixed-size chunking strategy."""

from docsense.chunking.fixed import FixedSizeChunker
from docsense.ingestion.loader import Document


def _make_doc(content: str) -> Document:
    return Document(content=content, source="test.md", metadata={"doc_id": "test"})


class TestFixedSizeChunker:
    def test_short_text_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk(_make_doc("Short text."))
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_splits_at_whitespace(self):
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=0)
        text = "one two three four five six seven eight"
        chunks = chunker.chunk(_make_doc(text))
        # No chunk should end mid-word
        for c in chunks:
            assert not c.text.startswith(" ")
            assert not c.text.endswith(" ")

    def test_overlap_produces_repeated_content(self):
        chunker = FixedSizeChunker(chunk_size=30, chunk_overlap=10)
        text = "alpha bravo charlie delta echo foxtrot golf hotel"
        chunks = chunker.chunk(_make_doc(text))
        assert len(chunks) > 1
        # Consecutive chunks should share some text due to overlap
        texts = [c.text for c in chunks]
        # At least one word should appear in adjacent chunks
        words_0 = set(texts[0].split())
        words_1 = set(texts[1].split())
        assert words_0 & words_1, "Expected overlapping words between chunks"

    def test_empty_document(self):
        chunker = FixedSizeChunker()
        assert chunker.chunk(_make_doc("")) == []
        assert chunker.chunk(_make_doc("   ")) == []

    def test_chunk_ids_are_sequential(self):
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk(_make_doc("one two three four five six seven eight"))
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_metadata_includes_strategy(self):
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk(_make_doc("Some content."))
        assert chunks[0].metadata["strategy"] == "fixed"

    def test_no_content_lost(self):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        text = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = chunker.chunk(_make_doc(text))
        # Every word in original should appear in at least one chunk
        original_words = set(text.split())
        chunk_words = set()
        for c in chunks:
            chunk_words.update(c.text.split())
        assert original_words <= chunk_words

    def test_chunk_many_delegates(self):
        chunker = FixedSizeChunker(chunk_size=100)
        docs = [_make_doc("Doc one content."), _make_doc("Doc two content.")]
        chunks = chunker.chunk_many(docs)
        assert len(chunks) == 2
