"""Tests for BM25 sparse retrieval."""

from docsense.chunking.base import Chunk
from docsense.retrieval.sparse import SparseRetriever


class TestSparseRetriever:
    def test_add_and_search(self):
        retriever = SparseRetriever()
        chunks = [
            Chunk(
                text="machine learning is a subset of artificial intelligence",
                doc_id="d1",
                chunk_index=0,
            ),
            Chunk(text="python programming language for data science", doc_id="d2", chunk_index=0),
            Chunk(
                text="neural networks learn representations from data", doc_id="d3", chunk_index=0
            ),
        ]
        retriever.add(chunks)
        assert retriever.size == 3

        results = retriever.search("machine learning", top_k=2)
        assert len(results) > 0
        assert results[0].chunk.doc_id == "d1"

    def test_empty_retriever(self):
        retriever = SparseRetriever()
        results = retriever.search("anything")
        assert results == []

    def test_no_match(self):
        retriever = SparseRetriever()
        chunks = [Chunk(text="alpha beta gamma", doc_id="d1", chunk_index=0)]
        retriever.add(chunks)
        results = retriever.search("xyzzy foobar")
        assert results == []
