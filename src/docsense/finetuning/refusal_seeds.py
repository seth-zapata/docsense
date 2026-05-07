"""Loader + typed contracts for off-corpus refusal topic seeds.

The Block 3B.2 plan calls for ~30 hand-curated topic seeds covering
topics that are NOT in the HuggingFace Transformers documentation
(Linux kernel, AWS, JAX internals, etc.). The seeder script generates
~3 questions per seed via Haiku 4.5; each generated query, paired with
retrieved chunks (which won't actually answer it), becomes a training
example teaching the canonical refusal phrase.

Why hand-curated instead of LLM-generated topics: taste matters more
than scale at this size (~30 entries), and LLM-generated topics risk
accidentally overlapping with HF content — defeating the "off-corpus"
property we're trying to teach.

The canonical seed file lives at the path constant ``DEFAULT_SEEDS_PATH``
and is committed to the repo for reproducibility. Loading is a simple
``RefusalSeeds.from_json`` round-trip; pydantic enforces the schema,
including ``Literal`` validation on the category field so a typo in
the JSON fails at parse time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Repo-root-relative path to the canonical seeds file. Computed from
# ``__file__`` so it works regardless of cwd.
# parents[0]=finetuning/  parents[1]=docsense/  parents[2]=src/  parents[3]=repo root.
DEFAULT_SEEDS_PATH = (
    Path(__file__).resolve().parents[3]
    / "evaluations"
    / "datasets"
    / "training"
    / "refusal_topic_seeds.json"
)


SeedCategory = Literal["adjacent_ml", "general_cs", "unrelated"]


class RefusalSeed(BaseModel):
    """One hand-curated off-corpus topic seed.

    ``topic`` is the noun-phrase string injected into the topic prompt
    in ``query_generation.TypeAwareQueryGenerator.generate_for_topic``.
    ``id`` is a kebab-case stable identifier so dedupe / re-run logic
    can reference seeds across runs. ``notes`` is human-only — flags
    overlap risk with HF docs where curation needed extra care.
    """

    id: str = Field(min_length=1)
    category: SeedCategory
    topic: str = Field(min_length=1)
    notes: str = ""


class RefusalSeeds(BaseModel):
    """The full seed file — schema for ``refusal_topic_seeds.json``."""

    version: str
    description: str = ""
    seeds: list[RefusalSeed]

    @classmethod
    def from_json(cls, path: Path | str) -> RefusalSeeds:
        """Load and validate a seeds file from disk."""
        return cls.model_validate(json.loads(Path(path).read_text()))

    def by_category(self) -> dict[str, list[RefusalSeed]]:
        """Group seeds by category. Useful for the seeder when it needs
        to sample proportionally or report per-category counts."""
        result: dict[str, list[RefusalSeed]] = {}
        for s in self.seeds:
            result.setdefault(s.category, []).append(s)
        return result


def load_default_seeds() -> RefusalSeeds:
    """Load the canonical seeds file from ``DEFAULT_SEEDS_PATH``.

    Convenience for the seeder script — most callers want the canonical
    file. Tests that need a custom seed set use ``RefusalSeeds.from_json``
    with an explicit path.
    """
    return RefusalSeeds.from_json(DEFAULT_SEEDS_PATH)
