"""Tests for hybrid retrieval and reciprocal rank fusion."""

from docsense.chunking.base import Chunk
from docsense.retrieval.dense import RetrievalResult
from docsense.retrieval.hybrid import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def _make_result(self, doc_id: str, score: float) -> RetrievalResult:
        chunk = Chunk(text=f"text for {doc_id}", doc_id=doc_id, chunk_index=0)
        return RetrievalResult(chunk=chunk, score=score)

    def test_single_list(self):
        results = [self._make_result("d1", 0.9), self._make_result("d2", 0.5)]
        fused = reciprocal_rank_fusion([results], weights=[1.0])
        assert len(fused) == 2
        assert fused[0].chunk.doc_id == "d1"

    def test_two_lists_boost_overlap(self):
        list_a = [self._make_result("d1", 0.9), self._make_result("d2", 0.5)]
        list_b = [self._make_result("d2", 0.8), self._make_result("d3", 0.4)]

        fused = reciprocal_rank_fusion([list_a, list_b], weights=[1.0, 1.0])
        # d2 appears in both lists, so it should get boosted
        ids = [r.chunk.doc_id for r in fused]
        assert "d2" in ids

    def test_weights_matter(self):
        list_a = [self._make_result("d1", 0.9)]
        list_b = [self._make_result("d2", 0.9)]

        # Heavily weight list_a
        fused = reciprocal_rank_fusion([list_a, list_b], weights=[10.0, 0.1])
        assert fused[0].chunk.doc_id == "d1"

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([], weights=[])
        assert fused == []
