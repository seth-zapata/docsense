"""Tests for FAISS dense retrieval."""

import numpy as np

from docsense.chunking.base import Chunk
from docsense.retrieval.dense import DenseRetriever


class TestDenseRetriever:
    def test_add_and_search(self):
        retriever = DenseRetriever(dimension=4)
        chunks = [
            Chunk(text="doc a", doc_id="d1", chunk_index=0),
            Chunk(text="doc b", doc_id="d2", chunk_index=0),
        ]
        embeddings = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        retriever.add(chunks, embeddings)
        assert retriever.size == 2

        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = retriever.search(query, top_k=1)
        assert len(results) == 1
        assert results[0].chunk.doc_id == "d1"

    def test_empty_index(self):
        retriever = DenseRetriever(dimension=4)
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = retriever.search(query, top_k=5)
        assert results == []

    def test_top_k_larger_than_index(self):
        retriever = DenseRetriever(dimension=2)
        chunks = [Chunk(text="only one", doc_id="d1", chunk_index=0)]
        embeddings = np.array([[1, 0]], dtype=np.float32)
        retriever.add(chunks, embeddings)

        results = retriever.search(np.array([1, 0], dtype=np.float32), top_k=10)
        assert len(results) == 1
