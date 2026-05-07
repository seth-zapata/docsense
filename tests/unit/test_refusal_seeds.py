"""Tests for the refusal topic-seeds loader and the canonical seeds file.

Two layers:

1. Schema/loader tests — synthetic data, exercise the pydantic model
   and ``RefusalSeeds.from_json`` round-trip. Pin the contract so
   future schema changes break loudly.

2. Canonical-file tests — load the real ``refusal_topic_seeds.json``
   committed in the repo and assert content invariants (unique ids,
   all three categories present, count in expected range). Guards
   against accidental deletions or category drift in future edits.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from docsense.finetuning.refusal_seeds import (
    DEFAULT_SEEDS_PATH,
    RefusalSeed,
    RefusalSeeds,
    load_default_seeds,
)

# --------------------------------------------------------------------
# RefusalSeed model
# --------------------------------------------------------------------


class TestRefusalSeed:
    def test_valid_minimal(self):
        seed = RefusalSeed(
            id="aws-lambda",
            category="general_cs",
            topic="AWS Lambda cold start optimization",
        )
        assert seed.id == "aws-lambda"
        assert seed.category == "general_cs"
        assert seed.notes == ""

    def test_notes_optional(self):
        seed = RefusalSeed(
            id="x",
            category="adjacent_ml",
            topic="x",
            notes="HF docs may overlap on Y",
        )
        assert seed.notes == "HF docs may overlap on Y"

    def test_id_required_nonempty(self):
        with pytest.raises(ValidationError):
            RefusalSeed(id="", category="general_cs", topic="x")

    def test_topic_required_nonempty(self):
        with pytest.raises(ValidationError):
            RefusalSeed(id="x", category="general_cs", topic="")

    def test_invalid_category_rejected(self):
        """``category`` is a Literal; a typo at parse time fails fast."""
        with pytest.raises(ValidationError):
            RefusalSeed(id="x", category="ml_adjacent", topic="x")  # type: ignore[arg-type]


# --------------------------------------------------------------------
# RefusalSeeds.from_json — schema/loader contract
# --------------------------------------------------------------------


class TestRefusalSeedsFromJson:
    def test_loads_minimal_valid_file(self, tmp_path):
        path = tmp_path / "seeds.json"
        path.write_text(
            json.dumps(
                {
                    "version": "v1",
                    "description": "test seeds",
                    "seeds": [
                        {
                            "id": "test-seed",
                            "category": "unrelated",
                            "topic": "cooking",
                        }
                    ],
                }
            )
        )
        loaded = RefusalSeeds.from_json(path)
        assert loaded.version == "v1"
        assert len(loaded.seeds) == 1
        assert loaded.seeds[0].id == "test-seed"

    def test_invalid_category_fails_at_load(self, tmp_path):
        """A typo in a JSON ``category`` field fails at parse time
        rather than silently producing an invalid seed."""
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {
                    "version": "v1",
                    "seeds": [
                        {
                            "id": "x",
                            "category": "INVALID",
                            "topic": "x",
                        }
                    ],
                }
            )
        )
        with pytest.raises(ValidationError):
            RefusalSeeds.from_json(path)

    def test_missing_seeds_field_fails(self, tmp_path):
        path = tmp_path / "missing.json"
        path.write_text(json.dumps({"version": "v1"}))
        with pytest.raises(ValidationError):
            RefusalSeeds.from_json(path)


# --------------------------------------------------------------------
# RefusalSeeds.by_category
# --------------------------------------------------------------------


class TestByCategory:
    def test_groups_seeds_by_category(self):
        seeds = RefusalSeeds(
            version="v1",
            seeds=[
                RefusalSeed(id="a", category="adjacent_ml", topic="a"),
                RefusalSeed(id="b", category="general_cs", topic="b"),
                RefusalSeed(id="c", category="adjacent_ml", topic="c"),
                RefusalSeed(id="d", category="unrelated", topic="d"),
            ],
        )
        groups = seeds.by_category()
        assert len(groups["adjacent_ml"]) == 2
        assert len(groups["general_cs"]) == 1
        assert len(groups["unrelated"]) == 1
        assert {s.id for s in groups["adjacent_ml"]} == {"a", "c"}

    def test_empty_seeds_returns_empty_dict(self):
        seeds = RefusalSeeds(version="v1", seeds=[])
        assert seeds.by_category() == {}


# --------------------------------------------------------------------
# Canonical-file content — loaded from the real committed JSON.
# --------------------------------------------------------------------


class TestCanonicalSeedsFile:
    def test_default_seeds_path_exists(self):
        """The canonical seeds file must exist at the documented path —
        the seeder script depends on this."""
        assert DEFAULT_SEEDS_PATH.exists(), (
            f"Canonical seeds file missing at {DEFAULT_SEEDS_PATH}. "
            "Did the file get moved or deleted?"
        )

    def test_load_default_seeds_succeeds(self):
        seeds = load_default_seeds()
        assert isinstance(seeds, RefusalSeeds)
        assert seeds.version

    def test_seed_count_in_expected_range(self):
        """~30 seeds is the design target. Tolerance band 25-35 catches
        accidental deletions or runaway additions while leaving room
        for deliberate calibration."""
        seeds = load_default_seeds()
        n = len(seeds.seeds)
        assert 25 <= n <= 35, f"unexpected seed count: {n}"

    def test_all_three_categories_present(self):
        """Each category needs enough seeds for diversity. The plan
        doc specifies all three (adjacent_ml, general_cs, unrelated)
        with ≥5 in each."""
        seeds = load_default_seeds()
        groups = seeds.by_category()
        assert set(groups) == {"adjacent_ml", "general_cs", "unrelated"}
        for cat, items in groups.items():
            assert len(items) >= 5, f"category {cat!r} has only {len(items)} seeds"

    def test_seed_ids_are_unique(self):
        """Duplicate ids would break dedupe / re-run logic that keys
        on ``seed.id``."""
        seeds = load_default_seeds()
        ids = [s.id for s in seeds.seeds]
        assert len(ids) == len(set(ids))

    def test_topics_are_nonempty_and_distinct(self):
        seeds = load_default_seeds()
        topics = [s.topic for s in seeds.seeds]
        assert all(t for t in topics)
        assert len(topics) == len(set(topics)), "duplicate topics found"
