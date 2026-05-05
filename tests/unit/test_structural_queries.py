"""Tests for structural-eval query generation."""

from docsense.evaluation.structural_queries import (
    extract_meaningful_headings,
    generate_structural_queries,
)
from docsense.ingestion.loader import Document


def _doc(content: str, doc_id: str = "test.md") -> Document:
    return Document(content=content, source=doc_id, metadata={"doc_id": doc_id})


class TestExtractMeaningfulHeadings:
    def test_extracts_h2_only(self):
        doc = _doc(
            "# Title\n\n## Configuring mixed precision\n\nbody\n\n"
            "### Subsection\n\n## Memory budgeting strategies"
        )
        headings = extract_meaningful_headings(doc)
        assert headings == ["Configuring mixed precision", "Memory budgeting strategies"]

    def test_strips_anchor_syntax(self):
        doc = _doc("## Gradient checkpointing [[gradient-checkpointing]]")
        assert extract_meaningful_headings(doc) == ["Gradient checkpointing"]

    def test_filters_generic_headings(self):
        doc = _doc("## Overview\n\nbody\n\n## Resources\n\n## How to fine-tune with DeepSpeed")
        assert extract_meaningful_headings(doc) == ["How to fine-tune with DeepSpeed"]

    def test_filters_too_short(self):
        # "Train" is one word, below the 2-word minimum
        # "API ref" is short on chars
        doc = _doc("## Train\n\n## API ref\n\n## Multi-GPU training with FSDP")
        assert extract_meaningful_headings(doc) == ["Multi-GPU training with FSDP"]

    def test_filters_annotation_prefixes(self):
        doc = _doc("## Note: this is important\n\n## How memory budgeting works")
        assert extract_meaningful_headings(doc) == ["How memory budgeting works"]

    def test_no_h2_returns_empty(self):
        doc = _doc("# Just a title\n\nsome body text")
        assert extract_meaningful_headings(doc) == []


class TestGenerateStructuralQueries:
    def test_picks_longest_heading_per_doc(self):
        doc = _doc(
            "## Short heading here\n\nbody\n\n## A much longer and more specific heading\n",
            doc_id="d.md",
        )
        queries = generate_structural_queries([doc], n_queries=10)
        assert len(queries) == 1
        query, relevant = queries[0]
        assert query == "A much longer and more specific heading"
        assert relevant == ["d.md"]

    def test_skips_docs_without_meaningful_headings(self):
        good = _doc("## How to configure mixed precision training", doc_id="good.md")
        bad = _doc("## Overview\n\n## Resources", doc_id="bad.md")
        queries = generate_structural_queries([good, bad], n_queries=10)
        assert len(queries) == 1
        assert queries[0][1] == ["good.md"]

    def test_seeded_sampling_is_reproducible(self):
        docs = [
            _doc(f"## How to use feature {i} in transformers", doc_id=f"d{i}.md") for i in range(20)
        ]
        first = generate_structural_queries(docs, n_queries=5, seed=42)
        second = generate_structural_queries(docs, n_queries=5, seed=42)
        assert first == second

    def test_different_seeds_produce_different_samples(self):
        docs = [
            _doc(f"## How to use feature {i} in transformers", doc_id=f"d{i}.md") for i in range(20)
        ]
        first = generate_structural_queries(docs, n_queries=5, seed=42)
        second = generate_structural_queries(docs, n_queries=5, seed=99)
        assert first != second

    def test_sample_size_capped_by_pool(self):
        docs = [_doc("## How to use this feature properly", doc_id="d.md")]
        queries = generate_structural_queries(docs, n_queries=100)
        assert len(queries) == 1
