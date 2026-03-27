"""Centralized configuration for docsense."""

from pathlib import Path

from pydantic_settings import BaseSettings

# Resolve project root relative to this file: src/docsense/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class EmbeddingConfig(BaseSettings):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    device: str = "cpu"
    normalize: bool = True


class RetrievalConfig(BaseSettings):
    top_k: int = 20
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rerank_top_k: int = 5


class RerankingConfig(BaseSettings):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 16
    device: str = "cpu"


class GenerationConfig(BaseSettings):
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    device: str = "auto"


class ChunkingConfig(BaseSettings):
    strategy: str = "recursive"  # fixed, recursive, header
    chunk_size: int = 512
    chunk_overlap: int = 64


class Settings(BaseSettings):
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranking: RerankingConfig = RerankingConfig()
    generation: GenerationConfig = GenerationConfig()
    chunking: ChunkingConfig = ChunkingConfig()

    data_dir: Path = DATA_DIR
    log_level: str = "INFO"
