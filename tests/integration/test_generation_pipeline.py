"""End-to-end test of the full generation pipeline with mocked LLM.

Composes real components — DenseRetriever, SparseRetriever, HybridRetriever,
CrossEncoderReranker, ContextAssembler, PromptBuilder, Generator — wired
together exactly as production runs them. The cross-encoder model and the
LLM are mocked; everything else is the real implementation.

The test verifies the *wiring*: chunks flow through retrieval → reranking
→ context assembly → prompt → generator and end up in a typed Answer with
populated citations, retrieved_chunks, and metadata. Real LLM behavioral
quality is for Phase 3 / pre-Phase-3 LLM-judge evals, not here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from docsense.chunking.base import Chunk
from docsense.config import GenerationConfig, RerankingConfig, RetrievalConfig
from docsense.generation.context import ContextAssembler
from docsense.generation.generator import Generator
from docsense.generation.prompt import PromptBuilder
from docsense.generation.types import Answer, ChunkRef
from docsense.reranking.reranker import CrossEncoderReranker
from docsense.retrieval.dense import DenseRetriever
from docsense.retrieval.hybrid import HybridRetriever
from docsense.retrieval.sparse import SparseRetriever


def _make_chunks() -> list[Chunk]:
    """Three chunks with realistic-ish content covering the install topic."""
    return [
        Chunk(
            text="To install transformers run: pip install transformers",
            doc_id="installation.md",
            chunk_index=0,
        ),
        Chunk(
            text="Use AutoModel.from_pretrained() to load a pretrained model.",
            doc_id="quickstart.md",
            chunk_index=0,
        ),
        Chunk(
            text="The Trainer class supports distributed training and mixed precision.",
            doc_id="trainer.md",
            chunk_index=0,
        ),
    ]


def _stub_embedder(dim: int = 4) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_query.return_value = np.zeros(dim, dtype=np.float32)
    return embedder


def _build_dense_retriever(chunks: list[Chunk], dim: int = 4) -> DenseRetriever:
    retriever = DenseRetriever(dimension=dim)
    embeddings = np.zeros((len(chunks), dim), dtype=np.float32)
    retriever.add(chunks, embeddings)
    return retriever


def _make_reranker(scores: list[float]) -> CrossEncoderReranker:
    config = RerankingConfig(model_name="dummy", device="cpu", batch_size=10)
    reranker = CrossEncoderReranker(config)
    mock_model = MagicMock()

    def predict(pairs: list, **_: object) -> np.ndarray:
        return np.array(scores[: len(pairs)], dtype=np.float32)

    mock_model.predict.side_effect = predict
    reranker._model = mock_model
    return reranker


def _word_tokenizer(text: str) -> int:
    return len(text.split())


def _make_stub_generator(canned_text: str) -> Generator:
    """Generator subclass with _run_inference stubbed to return canned text."""
    config = GenerationConfig(model_name="test-model", device="cpu")

    class _StubGenerator(Generator):
        def _run_inference(self, messages: list[dict[str, str]]) -> tuple[str, dict]:
            # Capture the messages for assertions in the test below.
            self._last_messages = messages  # type: ignore[attr-defined]
            return canned_text, {
                "latency_ms": 100.0,
                "prompt_tokens": 50,
                "completion_tokens": 20,
            }

    return _StubGenerator(config)


class TestGenerationPipelineEndToEnd:
    """Wire the full pipeline together and verify the data flow.

    Stages:
        query
          → HybridRetriever.search (real dense + real sparse + RRF + mocked CE rerank)
          → list[RetrievalResult]
          → list[ChunkRef]
          → ContextAssembler.assemble (real)
          → context string + included chunks
          → PromptBuilder.build (real)
          → list[dict] messages
          → Generator.generate (stubbed _run_inference)
          → typed Answer
    """

    def _build_pipeline(
        self, canned_llm_text: str, rerank_scores: list[float]
    ) -> tuple[HybridRetriever, ContextAssembler, PromptBuilder, Generator]:
        chunks = _make_chunks()
        dense = _build_dense_retriever(chunks)
        sparse = SparseRetriever()
        sparse.add(chunks)
        embedder = _stub_embedder()
        retrieval_config = RetrievalConfig(top_k=2, rerank_candidates=3)
        reranker = _make_reranker(rerank_scores)
        hybrid = HybridRetriever(dense, sparse, embedder, retrieval_config, reranker=reranker)

        assembler = ContextAssembler(max_tokens=500, tokenize_fn=_word_tokenizer)
        prompt_builder = PromptBuilder()
        generator = _make_stub_generator(canned_llm_text)

        return hybrid, assembler, prompt_builder, generator

    def test_end_to_end_produces_typed_answer(self):
        """The minimal happy path: every stage executes, types compose,
        an Answer comes out."""
        hybrid, assembler, prompt_builder, generator = self._build_pipeline(
            canned_llm_text="To install, run pip install transformers [1].",
            rerank_scores=[0.9, 0.5, 0.1],
        )

        # Stage 1: retrieve
        retrieval_results = hybrid.search("how do I install transformers?")
        assert len(retrieval_results) > 0

        # Stage 2: convert to ChunkRefs
        chunk_refs = [
            ChunkRef(
                doc_id=r.chunk.doc_id,
                chunk_id=r.chunk.chunk_id,
                score=r.score,
                text=r.chunk.text,
            )
            for r in retrieval_results
        ]

        # Stage 3: assemble context
        context, included = assembler.assemble(chunk_refs)
        assert len(included) > 0
        assert context  # non-empty string

        # Stage 4: build messages (chat-format) — the user's query and
        # the assembled context land inside the user message.
        messages = prompt_builder.build(query="how do I install transformers?", context=context)
        # System + user, in order
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "how do I install transformers?" in messages[1]["content"]
        assert context in messages[1]["content"]

        # Stage 5: generate
        answer = generator.generate(messages, included)
        assert isinstance(answer, Answer)
        assert answer.text == "To install, run pip install transformers [1]."

    def test_citations_reference_chunks_actually_passed_to_llm(self):
        """The chunks passed to assemble() are what the LLM sees, and what
        citations resolve against. If only 2 chunks made it through assembly,
        a `[1]` citation must reference the 1st of those, not the 1st
        retrieval result."""
        hybrid, assembler, prompt_builder, generator = self._build_pipeline(
            canned_llm_text="Per [1], the install command is `pip install transformers`.",
            rerank_scores=[0.9, 0.7, 0.3],
        )

        retrieval_results = hybrid.search("install")
        chunk_refs = [
            ChunkRef(
                doc_id=r.chunk.doc_id,
                chunk_id=r.chunk.chunk_id,
                score=r.score,
                text=r.chunk.text,
            )
            for r in retrieval_results
        ]
        context, included = assembler.assemble(chunk_refs)
        messages = prompt_builder.build(query="install", context=context)
        answer = generator.generate(messages, included)

        # The single citation [1] resolves to included[0] — i.e., the
        # first chunk that survived assembly, not necessarily the first
        # retrieval result.
        assert len(answer.citations) == 1
        assert answer.citations[0].doc_id == included[0].doc_id
        assert answer.citations[0].chunk_id == included[0].chunk_id

    def test_answer_retrieved_chunks_match_what_was_assembled(self):
        """Answer.retrieved_chunks captures the post-assembly context for
        auditing — distinct from the raw retrieval output. With top_k=2
        and small chunks fitting easily, both reranked chunks make it
        through assembly."""
        hybrid, assembler, prompt_builder, generator = self._build_pipeline(
            canned_llm_text="Some answer.",
            rerank_scores=[0.9, 0.5, 0.1],
        )

        results = hybrid.search("query")
        chunk_refs = [
            ChunkRef(
                doc_id=r.chunk.doc_id,
                chunk_id=r.chunk.chunk_id,
                score=r.score,
                text=r.chunk.text,
            )
            for r in results
        ]
        context, included = assembler.assemble(chunk_refs)
        messages = prompt_builder.build(query="query", context=context)
        answer = generator.generate(messages, included)

        # retrieved_chunks on the Answer is exactly the included list,
        # not the broader retrieval output.
        assert len(answer.retrieved_chunks) == len(included)
        for ans_chunk, included_chunk in zip(answer.retrieved_chunks, included, strict=True):
            assert ans_chunk.chunk_id == included_chunk.chunk_id

    def test_metadata_is_populated_from_inference(self):
        """GenerationMetadata flows from _run_inference through the
        generator into the Answer. Stub returns canned latency / token
        counts; assert they reach the Answer unchanged."""
        hybrid, assembler, prompt_builder, generator = self._build_pipeline(
            canned_llm_text="ok",
            rerank_scores=[0.5, 0.5, 0.5],
        )

        results = hybrid.search("query")
        chunk_refs = [
            ChunkRef(
                doc_id=r.chunk.doc_id,
                chunk_id=r.chunk.chunk_id,
                score=r.score,
                text=r.chunk.text,
            )
            for r in results
        ]
        context, included = assembler.assemble(chunk_refs)
        messages = prompt_builder.build(query="query", context=context)
        answer = generator.generate(messages, included)

        assert answer.metadata.model_name == "test-model"
        assert answer.metadata.latency_ms == 100.0
        assert answer.metadata.prompt_tokens == 50
        assert answer.metadata.completion_tokens == 20

    def test_messages_passed_to_llm_include_assembled_context(self):
        """The messages list the generator runs inference on must contain
        the assembled context inside the user message. Verifies the
        wiring between PromptBuilder output and Generator input — easy
        to break with a refactor that swaps args or drops the user
        message."""
        hybrid, assembler, prompt_builder, generator = self._build_pipeline(
            canned_llm_text="ok",
            rerank_scores=[0.9, 0.5, 0.1],
        )

        results = hybrid.search("query")
        chunk_refs = [
            ChunkRef(
                doc_id=r.chunk.doc_id,
                chunk_id=r.chunk.chunk_id,
                score=r.score,
                text=r.chunk.text,
            )
            for r in results
        ]
        context, included = assembler.assemble(chunk_refs)
        messages = prompt_builder.build(query="query", context=context)
        generator.generate(messages, included)

        # The stub captures the last messages list it saw. Concatenate
        # the user-message contents and verify all included chunk texts
        # appear there.
        last_messages = generator._last_messages  # type: ignore[attr-defined]
        user_content = next(m["content"] for m in last_messages if m["role"] == "user")
        for chunk in included:
            assert chunk.text in user_content
        assert "Question: query" in user_content
