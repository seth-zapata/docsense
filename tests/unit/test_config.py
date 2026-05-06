"""Tests for configuration."""

from docsense.config import (
    EmbeddingConfig,
    GenerationConfig,
    RetrievalConfig,
    Settings,
)


class TestSettings:
    def test_defaults(self):
        settings = Settings()
        assert settings.log_level == "INFO"
        assert settings.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert settings.retrieval.top_k == 5
        assert settings.retrieval.rerank_candidates == 20

    def test_retrieval_weights_sum(self):
        config = RetrievalConfig()
        assert config.dense_weight + config.sparse_weight == 1.0

    def test_rerank_candidates_at_least_top_k(self):
        """A meaningful reranker pipeline needs at least top_k candidates to
        choose from — usually substantially more. Codifying the relationship
        so the defaults can't accidentally invert."""
        config = RetrievalConfig()
        assert config.rerank_candidates >= config.top_k

    def test_embedding_defaults(self):
        config = EmbeddingConfig()
        assert config.batch_size == 64
        assert config.normalize is True


class TestGenerationConfig:
    def test_default_model_is_qwen(self):
        """Pin the system-model decision (locked 2026-05-06) so an
        accidental revert flips a regression-prone test, not a silent
        config change. See docs/journal/2026-05-06-pre-phase-3-model-decisions.md."""
        config = GenerationConfig()
        assert config.model_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_4bit_quantization_default_on(self):
        """Hardware constraint: vanilla RTX 4070 has 12 GB. A 7-8B model
        at full precision is too big; NF4 fits at ~5-6 GB. Default-on
        means the typical user gets working behavior; CPU users opt
        out via use_4bit_quantization=False."""
        config = GenerationConfig()
        assert config.use_4bit_quantization is True

    def test_max_context_tokens_under_typical_window(self):
        """Sanity check: max_context_tokens + max_new_tokens + system
        prompt overhead must comfortably fit any 7-8B model's context
        window. Qwen 2.5 7B has 32K, Llama 3.1 8B has 128K. Default
        budget of ~4000 leaves substantial headroom for both."""
        config = GenerationConfig()
        assert config.max_context_tokens + config.max_new_tokens < 8192
