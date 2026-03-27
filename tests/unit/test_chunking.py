"""Tests for chunking base classes."""

from docsense.chunking.base import Chunk, ChunkingStrategy
from docsense.ingestion.loader import Document


class TestChunk:
    def test_chunk_id(self):
        chunk = Chunk(text="hello", doc_id="doc_001", chunk_index=3)
        assert chunk.chunk_id == "doc_001::chunk_3"

    def test_chunk_metadata(self):
        chunk = Chunk(text="test", doc_id="doc_001", chunk_index=0, metadata={"page": 1})
        assert chunk.metadata["page"] == 1


class TestDocument:
    def test_doc_id_from_metadata(self):
        doc = Document(content="text", source="file.md", metadata={"doc_id": "custom_id"})
        assert doc.doc_id == "custom_id"

    def test_doc_id_falls_back_to_source(self):
        doc = Document(content="text", source="file.md")
        assert doc.doc_id == "file.md"


class TestChunkingStrategy:
    def test_chunk_many(self):
        class DoubleChunker(ChunkingStrategy):
            def chunk(self, document: Document) -> list[Chunk]:
                return [
                    Chunk(text=document.content[:10], doc_id=document.doc_id, chunk_index=0),
                    Chunk(text=document.content[10:], doc_id=document.doc_id, chunk_index=1),
                ]

        chunker = DoubleChunker()
        docs = [
            Document(content="abcdefghij1234567890", source="a.md", metadata={"doc_id": "a"}),
            Document(content="klmnopqrst0987654321", source="b.md", metadata={"doc_id": "b"}),
        ]
        chunks = chunker.chunk_many(docs)
        assert len(chunks) == 4
        assert chunks[0].doc_id == "a"
        assert chunks[2].doc_id == "b"
