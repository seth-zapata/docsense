"""Tests for configuration."""

from docsense.config import (
    EmbeddingConfig,
    RetrievalConfig,
    Settings,
)


class TestSettings:
    def test_defaults(self):
        settings = Settings()
        assert settings.log_level == "INFO"
        assert settings.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert settings.retrieval.top_k == 20

    def test_retrieval_weights_sum(self):
        config = RetrievalConfig()
        assert config.dense_weight + config.sparse_weight == 1.0

    def test_embedding_defaults(self):
        config = EmbeddingConfig()
        assert config.batch_size == 64
        assert config.normalize is True
