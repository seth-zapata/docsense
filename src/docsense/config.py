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
    # Final number of results returned to the caller (e.g., chunks fed to the LLM).
    top_k: int = 5
    # How many fused candidates are sent to the cross-encoder reranker. Only
    # consumed when a reranker is wired into HybridRetriever; without one,
    # search() returns top_k directly from the fusion stage.
    rerank_candidates: int = 20
    dense_weight: float = 0.6
    sparse_weight: float = 0.4


class RerankingConfig(BaseSettings):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 16
    device: str = "cpu"


class GenerationConfig(BaseSettings):
    # Default chosen 2026-05-06 over Mistral 7B Instruct v0.3 for the
    # HF Transformers (Python) corpus: Qwen 2.5 7B Instruct's HumanEval
    # ~85 vs Mistral's ~40 directly reflects code-comprehension quality.
    # Apache 2.0 license. See docs/journal/2026-05-06-pre-phase-3-model-decisions.md
    # for the full rationale.
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 512
    # Token budget reserved for retrieved context inside the prompt.
    # Independent of max_new_tokens; the LLM's full context window must
    # accommodate system prompt + this + query + max_new_tokens.
    max_context_tokens: int = 3500
    temperature: float = 0.1
    top_p: float = 0.9
    device: str = "auto"
    # Load the model at NF4 4-bit precision via bitsandbytes. Required to
    # fit a 7-8B model in 12 GB VRAM (vanilla RTX 4070). Set False for CPU
    # inference or environments without bitsandbytes installed; the model
    # then loads at default precision (float16/bfloat16 depending on dtype).
    # Bitsandbytes ships in the `gpu` extras (`pip install -e ".[gpu]"`).
    use_4bit_quantization: bool = True


class ChunkingConfig(BaseSettings):
    strategy: str = "recursive"  # fixed, recursive, header
    chunk_size: int = 512
    chunk_overlap: int = 64


class JudgeConfig(BaseSettings):
    # Llama 3.1 8B Instruct as the local LLM-judge. Picked 2026-05-06
    # for permissive license, strong instruction-following, and
    # conveniently different lineage from Qwen so judge ↔ generator
    # disagreement signal isn't swamped by family resemblance.
    # See docs/journal/2026-05-06-pre-phase-3-model-decisions.md.
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # Short generations: judge output is SCORE + 1-2 sentence
    # rationale, not free-form prose.
    max_new_tokens: int = 256
    # Deterministic by default — eval reproducibility matters more than
    # creative variation when grading the same Answer.
    temperature: float = 0.0
    top_p: float = 1.0
    device: str = "auto"
    # NF4 4-bit quantization, same constraint as the generator: 8B at
    # full precision doesn't fit alongside other resident processes on
    # 12 GB. The eval driver loads judge and generator sequentially,
    # but each must individually fit.
    use_4bit_quantization: bool = True


class Settings(BaseSettings):
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranking: RerankingConfig = RerankingConfig()
    generation: GenerationConfig = GenerationConfig()
    judge: JudgeConfig = JudgeConfig()
    chunking: ChunkingConfig = ChunkingConfig()

    data_dir: Path = DATA_DIR
    log_level: str = "INFO"
