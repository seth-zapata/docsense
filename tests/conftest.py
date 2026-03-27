"""Shared test fixtures."""

import pytest

from docsense.chunking.base import Chunk
from docsense.config import (
    ChunkingConfig,
    EmbeddingConfig,
    RerankingConfig,
    RetrievalConfig,
    Settings,
)
from docsense.ingestion.loader import Document


@pytest.fixture
def sample_document():
    return Document(
        content="This is a test document about transformers. "
        "The AutoModel class provides a simple API for loading pre-trained models. "
        "You can use AutoTokenizer to load the corresponding tokenizer.",
        source="test_doc.md",
        metadata={"doc_id": "test_001"},
    )


@pytest.fixture
def sample_documents():
    return [
        Document(
            content="AutoModel provides a unified API for loading models. "
            "It automatically detects the model type and loads the appropriate class.",
            source="automodel.md",
            metadata={"doc_id": "doc_001"},
        ),
        Document(
            content="The Trainer class provides an API for feature-complete training. "
            "It supports distributed training, mixed precision, and gradient accumulation.",
            source="trainer.md",
            metadata={"doc_id": "doc_002"},
        ),
        Document(
            content="Tokenizers convert text into token IDs that models can process. "
            "Use AutoTokenizer.from_pretrained() to load a tokenizer.",
            source="tokenizer.md",
            metadata={"doc_id": "doc_003"},
        ),
    ]


@pytest.fixture
def sample_chunks():
    return [
        Chunk(text="AutoModel provides a unified API.", doc_id="doc_001", chunk_index=0),
        Chunk(text="Trainer supports distributed training.", doc_id="doc_002", chunk_index=0),
        Chunk(text="Tokenizers convert text to IDs.", doc_id="doc_003", chunk_index=0),
    ]


@pytest.fixture
def settings():
    return Settings(
        embedding=EmbeddingConfig(device="cpu"),
        retrieval=RetrievalConfig(),
        reranking=RerankingConfig(device="cpu"),
        chunking=ChunkingConfig(),
    )
