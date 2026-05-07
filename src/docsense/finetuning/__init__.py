"""QLoRA fine-tuning scaffolding for Phase 3.

Modules:

- ``dataset``: ``TrainingExample`` typed contract + ``TrainingDataset``
  with stratified train/val split. The dataset shape that
  ``scripts/build_training_dataset.py`` (Block 3B) writes to disk and
  ``scripts/train_lora.py`` (Block 3A.2) reads back.
- ``config``: ``FineTuningConfig`` pydantic-settings with pinned
  hyperparameter defaults for QLoRA on Qwen 2.5 7B Instruct.
- ``trainer``: (Block 3A.2) ``LoRAFineTuner`` wrapping PEFT model
  prep + LoRA config + TRL SFTTrainer construction.
- ``chunk_classifier``: (Block 3B.2.a) heuristic chunk-affinity
  classifier — given a chunk's text, returns the set of
  ``QuestionType``s it has affinity for. Foundation for the
  type-stratified query seeder in Block 3B.2.
- ``query_generation``: (Block 3B.2.b) ``TypeAwareQueryGenerator``
  wrapping Haiku 4.5 — given a chunk + question type, generates a
  realistic query grounded in the chunk. Also handles off-corpus
  refusal queries from topic seeds.
- ``query_filters``: (Block 3B.2.c) quality filters for the query
  pool — length floor, type-stratified embedding dedupe, eval-set
  contamination filter. Each returns a ``FilterReport`` with
  explicit ``kept``/``dropped`` lists for the seeder's audit trail.
- ``refusal_seeds``: (Block 3B.2.d) loader + typed contracts for the
  canonical off-corpus topic seeds file. The seeder feeds each seed's
  ``topic`` field into ``TypeAwareQueryGenerator.generate_for_topic``
  to produce off-corpus refusal queries.

See ``docs/phase-3-scope.md`` and ``docs/phase-3-block-3b2-plan.md``
for the full Phase 3 plan.
"""
