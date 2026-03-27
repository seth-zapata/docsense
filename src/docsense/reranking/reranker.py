"""Cross-encoder re-ranking for retrieval results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docsense.retrieval.dense import RetrievalResult

if TYPE_CHECKING:
    from docsense.config import RerankingConfig


class CrossEncoderReranker:
    """Re-ranks retrieval results using a cross-encoder model."""

    def __init__(self, config: RerankingConfig) -> None:
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.config.model_name, device=self.config.device)
        return self._model

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int | None = None
    ) -> list[RetrievalResult]:
        if not results:
            return []

        top_k = top_k or self.config.batch_size
        pairs = [(query, r.chunk.text) for r in results]
        scores = self.model.predict(pairs, batch_size=self.config.batch_size)

        reranked = [
            RetrievalResult(chunk=r.chunk, score=float(s))
            for r, s in zip(results, scores, strict=True)
        ]
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
