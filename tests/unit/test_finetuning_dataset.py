"""Unit tests for the fine-tuning dataset shape + stratified split.

The stratification logic is the highest-stakes piece — without it,
small val splits can land with zero refusal examples and the
"did fine-tuning preserve refusal behavior?" signal silently
disappears during in-training eval. Tests pin the contract that
every split preserves the in-corpus / refusal ratio.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

from docsense.finetuning.dataset import (
    TrainingDataset,
    TrainingExample,
    count_strata,
)
from docsense.generation.types import ChunkRef


def _chunks(*texts: str) -> list[ChunkRef]:
    return [
        ChunkRef(doc_id=f"doc{i}.md", chunk_id=str(i), score=1.0, text=text)
        for i, text in enumerate(texts, start=1)
    ]


def _example(
    *,
    query: str = "How do I X?",
    answer: str = "Do X with Y [1].",
    chunks: list[ChunkRef] | None = None,
    is_refusal: bool = False,
) -> TrainingExample:
    if chunks is None:
        chunks = _chunks("alpha", "beta")
    return TrainingExample(
        query=query,
        retrieved_chunks=chunks,
        ideal_answer=answer,
        is_refusal=is_refusal,
    )


class TestTrainingExample:
    def test_in_corpus_with_valid_citations(self):
        ex = _example(answer="Step one [1] then step two [2].", chunks=_chunks("a", "b"))
        assert ex.is_refusal is False
        assert len(ex.retrieved_chunks) == 2

    def test_refusal_with_no_chunks_allowed(self):
        """Refusals can have empty retrieved_chunks — the eval set
        contains queries where retrieval returned irrelevant or no
        chunks. The model still needs to refuse cleanly."""
        ex = TrainingExample(
            query="What's the price of tea in China?",
            retrieved_chunks=[],
            ideal_answer="I don't have enough context to answer that.",
            is_refusal=True,
        )
        assert ex.is_refusal is True

    def test_in_corpus_with_zero_chunks_rejected(self):
        """An in-corpus example with no chunks has nothing to cite —
        catch at construction so corrupt data fails fast."""
        with pytest.raises(ValidationError, match="zero retrieved_chunks"):
            TrainingExample(
                query="How do I X?",
                retrieved_chunks=[],
                ideal_answer="Do X.",
                is_refusal=False,
            )

    def test_out_of_range_citation_rejected(self):
        """Catch citations that reference chunks outside the
        retrieved set. Same invariant as Answer.citations."""
        with pytest.raises(ValidationError, match=r"\[7\]"):
            _example(answer="Use chunk [7] for this.", chunks=_chunks("a", "b"))

    def test_refusal_with_orphan_citation_allowed(self):
        """Refusals don't have meaningful citations by definition.
        Skip the citation grounding check for refusals — even if the
        refusal text accidentally contains [N], it's not a real
        citation reference."""
        ex = TrainingExample(
            query="off-corpus q?",
            retrieved_chunks=[],
            ideal_answer="I don't have context. (note: [99] not a real citation)",
            is_refusal=True,
        )
        assert ex.is_refusal is True

    def test_empty_answer_rejected(self):
        with pytest.raises(ValidationError):
            _example(answer="", chunks=_chunks("a"))

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            _example(query="", chunks=_chunks("a"))

    def test_metadata_passthrough(self):
        ex = TrainingExample(
            query="q",
            retrieved_chunks=_chunks("a"),
            ideal_answer="ans [1]",
            is_refusal=False,
            metadata={"distill_model": "claude-sonnet-4-5", "ts": "2026-05-06T12:00:00Z"},
        )
        assert ex.metadata["distill_model"] == "claude-sonnet-4-5"


class TestTrainingDataset:
    def test_round_trip_through_json(self, tmp_path: Path):
        ds = TrainingDataset(
            examples=[_example(query=f"q{i}") for i in range(3)],
            version="v_test",
            description="round-trip test",
        )
        path = tmp_path / "ds.json"
        ds.to_json(path)
        loaded = TrainingDataset.from_json(path)
        assert len(loaded) == 3
        assert loaded.version == "v_test"
        assert loaded.examples[0].query == "q0"

    def test_stats(self):
        examples = [_example() for _ in range(5)] + [
            TrainingExample(
                query=f"refusal_{i}",
                retrieved_chunks=[],
                ideal_answer="I don't know.",
                is_refusal=True,
            )
            for i in range(2)
        ]
        ds = TrainingDataset(examples=examples)
        stats = ds.stats()
        assert stats["total"] == 7
        assert stats["in_corpus"] == 5
        assert stats["refusal"] == 2

    def test_count_strata_helper(self):
        examples = [_example() for _ in range(3)] + [
            TrainingExample(
                query=f"r{i}",
                retrieved_chunks=[],
                ideal_answer="No.",
                is_refusal=True,
            )
            for i in range(4)
        ]
        n_in_corpus, n_refusal = count_strata(examples)
        assert n_in_corpus == 3
        assert n_refusal == 4


class TestStratifiedTrainValSplit:
    def _build(self, n_in_corpus: int, n_refusal: int) -> TrainingDataset:
        examples = [_example(query=f"q_{i}") for i in range(n_in_corpus)] + [
            TrainingExample(
                query=f"r_{i}",
                retrieved_chunks=[],
                ideal_answer="I don't have enough context.",
                is_refusal=True,
            )
            for i in range(n_refusal)
        ]
        return TrainingDataset(examples=examples)

    def test_split_preserves_total(self):
        ds = self._build(n_in_corpus=80, n_refusal=20)
        train, val = ds.stratified_train_val_split(val_fraction=0.1, seed=42)
        assert len(train) + len(val) == 100

    def test_split_preserves_strata_ratio(self):
        """The crux: train and val should both have ~10% refusal
        examples (matching input). Without stratification a 10% val
        on 100 examples could land with 0 refusals by chance."""
        ds = self._build(n_in_corpus=80, n_refusal=20)
        train, val = ds.stratified_train_val_split(val_fraction=0.1, seed=42)
        train_ic, train_ref = count_strata(train)
        val_ic, val_ref = count_strata(val)
        # Val has 10% of each stratum (rounded): 8 in-corpus, 2 refusal
        assert val_ic == 8
        assert val_ref == 2
        # Train gets the remainder
        assert train_ic == 72
        assert train_ref == 18

    def test_small_set_minimum_one_in_each(self):
        """With small sets, ensure val gets at least one of each
        stratum if available — the stratification's whole point."""
        ds = self._build(n_in_corpus=10, n_refusal=3)
        train, val = ds.stratified_train_val_split(val_fraction=0.1, seed=42)
        val_ic, val_ref = count_strata(val)
        assert val_ic >= 1
        assert val_ref >= 1  # would be 0 without the max(1, ...) guard

    def test_seed_reproducibility(self):
        ds = self._build(n_in_corpus=50, n_refusal=10)
        train_a, val_a = ds.stratified_train_val_split(val_fraction=0.1, seed=42)
        train_b, val_b = ds.stratified_train_val_split(val_fraction=0.1, seed=42)
        # Same seed, same split — list contents identical (in same order
        # after the post-shuffle).
        assert [ex.query for ex in train_a] == [ex.query for ex in train_b]
        assert [ex.query for ex in val_a] == [ex.query for ex in val_b]

    def test_different_seed_produces_different_split(self):
        ds = self._build(n_in_corpus=50, n_refusal=10)
        _, val_42 = ds.stratified_train_val_split(val_fraction=0.1, seed=42)
        _, val_7 = ds.stratified_train_val_split(val_fraction=0.1, seed=7)
        # Same size split (deterministic from val_fraction) but
        # different example identities chosen.
        assert len(val_42) == len(val_7)
        assert [ex.query for ex in val_42] != [ex.query for ex in val_7]

    def test_invalid_val_fraction_rejected(self):
        ds = self._build(n_in_corpus=10, n_refusal=2)
        for bad in (-0.1, 0.0, 1.0, 1.5):
            with pytest.raises(ValueError, match="val_fraction"):
                ds.stratified_train_val_split(val_fraction=bad)

    def test_no_refusals_no_crash(self):
        """Edge case: dataset has zero refusal examples (e.g.,
        in-corpus-only ablation). Split shouldn't crash; val should
        contain only in-corpus examples."""
        ds = self._build(n_in_corpus=20, n_refusal=0)
        train, val = ds.stratified_train_val_split(val_fraction=0.1, seed=42)
        assert len(val) > 0
        assert all(not ex.is_refusal for ex in val)

    def test_no_in_corpus_no_crash(self):
        """Symmetric edge case — refusals only. Shouldn't happen in
        practice but verify no crash."""
        ds = self._build(n_in_corpus=0, n_refusal=10)
        train, val = ds.stratified_train_val_split(val_fraction=0.1, seed=42)
        assert all(ex.is_refusal for ex in val)


class TestJsonRoundTrip:
    def test_full_field_preservation(self, tmp_path: Path):
        """Pin: every field on every example survives to-disk →
        from-disk round trip. Adding a new field without updating
        from_json/to_json would silently lose data."""
        ex = TrainingExample(
            query="How do I install transformers?",
            retrieved_chunks=_chunks("pip install ...", "or via conda ..."),
            ideal_answer="Run `pip install transformers` [1] or use conda [2].",
            is_refusal=False,
            metadata={"distill_model": "claude-sonnet-4-5", "ts": "2026-05-06T13:00:00Z"},
        )
        ds = TrainingDataset(
            examples=[ex],
            version="v_test_round_trip",
            description="ensure all fields survive",
            metadata={"generation_run": "test_run_001"},
        )
        path = tmp_path / "ds.json"
        ds.to_json(path)

        loaded = TrainingDataset.from_json(path)
        loaded_ex = loaded.examples[0]
        assert loaded_ex.query == ex.query
        assert loaded_ex.ideal_answer == ex.ideal_answer
        assert loaded_ex.is_refusal == ex.is_refusal
        assert loaded_ex.metadata == ex.metadata
        assert len(loaded_ex.retrieved_chunks) == 2
        assert loaded_ex.retrieved_chunks[0].text == "pip install ..."
        assert loaded.version == "v_test_round_trip"
        assert loaded.metadata == {"generation_run": "test_run_001"}

    def test_json_format_is_diff_friendly(self, tmp_path: Path):
        """to_json should produce indented output so dataset diffs
        are reviewable — not a single-line blob."""
        ds = TrainingDataset(examples=[_example()])
        path = tmp_path / "ds.json"
        ds.to_json(path)
        text = path.read_text()
        assert "\n" in text
        assert text.endswith("\n")
        # Sanity: indented JSON, not minified
        assert '"examples"' in text
        # Validate it's still valid JSON
        json.loads(text)
