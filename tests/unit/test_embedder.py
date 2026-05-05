"""Tests for the Embedder wrapper.

The actual sentence-transformers model is mocked. We're verifying *our*
glue around it (config plumbing, batch handling, lazy loading), not
sentence-transformers itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from docsense.chunking.base import Chunk
from docsense.config import EmbeddingConfig
from docsense.embedding.embedder import Embedder


def _make_mock_model(dim: int = 4) -> MagicMock:
    """Mock SentenceTransformer that returns deterministic float32 vectors."""
    model = MagicMock()

    def encode(texts: list[str], **_: object) -> np.ndarray:
        # One row per input text; deterministic so tests can assert shape/values
        return np.zeros((len(texts), dim), dtype=np.float32)

    model.encode.side_effect = encode
    return model


class TestEmbedder:
    def test_lazy_model_load_does_not_construct_until_used(self):
        """Constructing an Embedder must not download/load weights."""
        config = EmbeddingConfig(model_name="dummy-model", device="cpu")
        embedder = Embedder(config)
        assert embedder._model is None

    def test_embed_texts_returns_correct_shape(self):
        config = EmbeddingConfig(model_name="dummy-model", device="cpu", batch_size=32)
        embedder = Embedder(config)
        embedder._model = _make_mock_model(dim=4)

        result = embedder.embed_texts(["hello", "world", "foo"])

        assert result.shape == (3, 4)
        assert result.dtype == np.float32

    def test_embed_texts_passes_config_to_encode(self):
        """batch_size and normalize must reach the underlying model."""
        config = EmbeddingConfig(
            model_name="dummy-model", device="cpu", batch_size=16, normalize=True
        )
        embedder = Embedder(config)
        mock_model = _make_mock_model()
        embedder._model = mock_model

        embedder.embed_texts(["a", "b"])

        call_kwargs = mock_model.encode.call_args.kwargs
        assert call_kwargs["batch_size"] == 16
        assert call_kwargs["normalize_embeddings"] is True

    def test_embed_texts_progress_bar_only_for_large_batches(self):
        """show_progress_bar should only fire above the 100-text threshold."""
        config = EmbeddingConfig(model_name="dummy-model", device="cpu")
        embedder = Embedder(config)
        mock_model = _make_mock_model()
        embedder._model = mock_model

        embedder.embed_texts(["x"] * 50)
        assert mock_model.encode.call_args.kwargs["show_progress_bar"] is False

        embedder.embed_texts(["x"] * 150)
        assert mock_model.encode.call_args.kwargs["show_progress_bar"] is True

    def test_embed_chunks_extracts_text_field(self):
        """embed_chunks should pass chunk.text values, not the Chunk objects."""
        config = EmbeddingConfig(model_name="dummy-model", device="cpu")
        embedder = Embedder(config)
        mock_model = _make_mock_model()
        embedder._model = mock_model

        chunks = [
            Chunk(text="alpha", doc_id="d1", chunk_index=0),
            Chunk(text="beta", doc_id="d1", chunk_index=1),
        ]
        embedder.embed_chunks(chunks)

        passed_texts = mock_model.encode.call_args.args[0]
        assert passed_texts == ["alpha", "beta"]

    def test_embed_query_returns_single_vector(self):
        """embed_query should return a 1-D vector, not a 2-D batch."""
        config = EmbeddingConfig(model_name="dummy-model", device="cpu")
        embedder = Embedder(config)
        embedder._model = _make_mock_model(dim=8)

        result = embedder.embed_query("a query")

        assert result.shape == (8,)
        assert result.dtype == np.float32

    def test_model_property_constructs_with_config_values(self):
        """First access to .model should pass model_name and device through."""
        config = EmbeddingConfig(model_name="some/model", device="cuda")
        embedder = Embedder(config)

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = _make_mock_model()
            _ = embedder.model

            mock_st.assert_called_once_with("some/model", device="cuda")

    def test_model_property_caches_after_first_access(self):
        """Second access to .model should not reconstruct the model."""
        config = EmbeddingConfig(model_name="some/model", device="cpu")
        embedder = Embedder(config)

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = _make_mock_model()
            _ = embedder.model
            _ = embedder.model

            assert mock_st.call_count == 1
