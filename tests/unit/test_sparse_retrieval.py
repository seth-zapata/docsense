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

    def test_incremental_add_preserves_existing(self):
        """Calling add() multiple times should keep all chunks searchable
        and not re-tokenize the existing corpus — the token cache is
        extended, not rebuilt from scratch.

        Uses a 3-chunk corpus because BM25Okapi's IDF math returns zero
        for tiny corpora (N=2) where each term appears in exactly one
        document, which would mask whether retrieval is actually working.
        """
        retriever = SparseRetriever()
        retriever.add(
            [
                Chunk(text="machine learning models for inference", doc_id="d1", chunk_index=0),
                Chunk(text="neural networks learn from training data", doc_id="d2", chunk_index=0),
            ]
        )
        # Capture the cached tokens after first add to verify extension, not rebuild
        first_tokens_id = id(retriever._tokenized[0])

        retriever.add(
            [Chunk(text="python programming basics for beginners", doc_id="d3", chunk_index=0)]
        )

        assert retriever.size == 3
        assert len(retriever._tokenized) == 3
        # First-chunk tokens are the same object — not re-tokenized on second add
        assert id(retriever._tokenized[0]) == first_tokens_id

        # Both original and newly-added chunks remain searchable
        results_old = retriever.search("machine learning")
        results_new = retriever.search("python")
        assert results_old and results_old[0].chunk.doc_id == "d1"
        assert results_new and results_new[0].chunk.doc_id == "d3"
