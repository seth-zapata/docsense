"""Typed contracts for fine-tuning training data.

Each training example is a (query, retrieved_chunks, ideal_answer)
triple — the same shape the generator sees at inference time, plus
the gold-standard answer we want it to learn to produce.

Two kinds of examples:

- **In-corpus** (``is_refusal=False``): the question can be answered
  from the retrieved chunks; the ideal answer cites them with
  ``[N]`` markers grounded in the chunk indices.
- **Refusal** (``is_refusal=True``): the question is off-corpus;
  the ideal answer is a refusal in the canonical Qwen phrasing
  ("I don't have enough context..."). Included as a guardrail
  against catastrophic forgetting during fine-tuning — without
  refusal examples, the model may learn to always cite, even
  when the context doesn't support the question.

The split is stratified on ``is_refusal`` so train and val sets
preserve the same in-corpus / refusal ratio. Without stratification,
a small val set can end up with zero refusal examples and we lose
the ability to validate refusal preservation during training.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

# Pydantic resolves ChunkRef at runtime for list[ChunkRef] field
# validation, so it can't live in the TYPE_CHECKING block (TC001).
from docsense.generation.types import ChunkRef  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Iterable


# Matches `[N]` where N is one or more digits; same shape as the
# generator's citation parser. Used by the citation-grounding
# validator below.
_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


class TrainingExample(BaseModel):
    """One supervised fine-tuning example.

    Validated at construction so dataset-quality issues fail fast
    rather than silently corrupting the training run:

    - ``ideal_answer`` non-empty
    - For in-corpus examples (``is_refusal=False``): every ``[N]``
      marker in ``ideal_answer`` must reference a valid chunk
      index in ``retrieved_chunks`` (1-indexed, same convention as
      the generator's ``parse_citations``)
    - For refusal examples (``is_refusal=True``):
      ``retrieved_chunks`` may be empty or contain irrelevant
      chunks; citation grounding is not enforced (refusals don't
      cite by definition)
    """

    query: str = Field(min_length=1)
    retrieved_chunks: list[ChunkRef] = Field(default_factory=list)
    ideal_answer: str = Field(min_length=1)
    is_refusal: bool = False
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Generation provenance — distill model name, timestamp, "
            "prompt-template version, etc. Carried through to reports "
            "for reproducibility audits but not load-bearing for training."
        ),
    )

    @model_validator(mode="after")
    def _validate_citations_grounded(self) -> TrainingExample:
        """Every ``[N]`` in ideal_answer must reference a valid chunk
        index for in-corpus examples. Skipped for refusals."""
        if self.is_refusal:
            return self
        markers = _CITATION_MARKER_RE.findall(self.ideal_answer)
        n_chunks = len(self.retrieved_chunks)
        for marker in markers:
            idx = int(marker)
            if idx < 1 or idx > n_chunks:
                msg = (
                    f"Citation [{idx}] in ideal_answer references chunk index "
                    f"outside [1, {n_chunks}]. Either the answer cites a "
                    f"chunk we didn't pass, or the chunk-list is incomplete. "
                    f"Query: {self.query[:80]!r}"
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_refusal_chunks_optional(self) -> TrainingExample:
        """Soft check: refusal examples may have any number of chunks
        (including zero — "no relevant context retrieved"). In-corpus
        examples should have at least one chunk; otherwise there's
        nothing to cite."""
        if not self.is_refusal and len(self.retrieved_chunks) == 0:
            msg = (
                f"In-corpus example has zero retrieved_chunks. The model "
                f"can't learn to cite from nothing. "
                f"Query: {self.query[:80]!r}"
            )
            raise ValueError(msg)
        return self


class TrainingDataset(BaseModel):
    """A collection of TrainingExamples + a stratified train/val split.

    Persisted as JSON in ``evaluations/datasets/training/`` for full
    reproducibility — the actual ideal_answer text is committed (not
    just a manifest) so anyone can reconstruct the training run from
    the repo alone.
    """

    examples: list[TrainingExample]
    version: str = "v1"
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path | str) -> TrainingDataset:
        """Load a dataset JSON file written by
        ``scripts/build_training_dataset.py`` or hand-curated."""
        raw = json.loads(Path(path).read_text())
        return cls.model_validate(raw)

    def to_json(self, path: Path | str, *, indent: int = 2) -> None:
        """Persist to disk in a stable, diff-friendly format."""
        Path(path).write_text(self.model_dump_json(indent=indent) + "\n")

    def __len__(self) -> int:
        return len(self.examples)

    def stratified_train_val_split(
        self,
        val_fraction: float = 0.1,
        *,
        seed: int = 42,
    ) -> tuple[list[TrainingExample], list[TrainingExample]]:
        """Split examples into train/val, stratified on ``is_refusal``.

        Without stratification, a 10% val split on a dataset with ~10%
        refusal examples can land with zero refusals in val by chance.
        That breaks the "did fine-tuning preserve refusal behavior?"
        signal during in-training eval. Stratifying preserves the
        in-corpus/refusal ratio in both splits.

        Returns ``(train, val)`` as plain lists. The underlying
        examples list is not mutated; both returned lists are new
        sequences.
        """
        if not 0.0 < val_fraction < 1.0:
            msg = f"val_fraction must be in (0, 1); got {val_fraction}"
            raise ValueError(msg)

        in_corpus = [ex for ex in self.examples if not ex.is_refusal]
        refusals = [ex for ex in self.examples if ex.is_refusal]

        rng = random.Random(seed)
        rng.shuffle(in_corpus)
        rng.shuffle(refusals)

        n_val_in_corpus = max(1, int(round(len(in_corpus) * val_fraction))) if in_corpus else 0
        n_val_refusals = max(1, int(round(len(refusals) * val_fraction))) if refusals else 0

        val = in_corpus[:n_val_in_corpus] + refusals[:n_val_refusals]
        train = in_corpus[n_val_in_corpus:] + refusals[n_val_refusals:]

        # Shuffle the combined splits so refusal examples aren't
        # clumped at the end of each list. Same seed-derived RNG so
        # the split is fully reproducible.
        rng.shuffle(train)
        rng.shuffle(val)
        return train, val

    def stats(self) -> dict[str, int]:
        """Quick summary numbers for logging / report headers."""
        n_in_corpus = sum(1 for ex in self.examples if not ex.is_refusal)
        n_refusal = sum(1 for ex in self.examples if ex.is_refusal)
        return {
            "total": len(self.examples),
            "in_corpus": n_in_corpus,
            "refusal": n_refusal,
        }


def count_strata(examples: Iterable[TrainingExample]) -> tuple[int, int]:
    """Helper for tests / logging: ``(n_in_corpus, n_refusal)`` counts."""
    n_in_corpus = sum(1 for ex in examples if not ex.is_refusal)
    n_refusal = sum(1 for ex in examples if ex.is_refusal)
    return n_in_corpus, n_refusal
