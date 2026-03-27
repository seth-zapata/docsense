"""Dense embedding using sentence-transformers models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from docsense.chunking.base import Chunk
    from docsense.config import EmbeddingConfig


class Embedder:
    """Wraps a sentence-transformers model for dense embedding."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
        return self._model

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )

    def embed_chunks(self, chunks: list[Chunk]) -> NDArray[np.float32]:
        texts = [c.text for c in chunks]
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        return self.embed_texts([query])[0]
