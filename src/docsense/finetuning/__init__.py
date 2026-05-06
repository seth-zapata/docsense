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

See ``docs/phase-3-scope.md`` and ``docs/phase-3-block-3b2-plan.md``
for the full Phase 3 plan.
"""
