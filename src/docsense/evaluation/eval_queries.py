"""Hand-curated evaluation queries for retrieval bakeoffs.

Each entry is ``(query, relevant_doc_id_prefixes)``. A retrieved chunk's
``doc_id`` is considered relevant if it starts with any of the prefixes.

**Selection bias caveat:** these queries were authored by hand by someone
who already knew which HF Transformers docs exist. As a result, the eval
rewards retrieval that mirrors the curator's mental model of the corpus.
This is a useful eval set — the queries are realistic and the relevance
judgments are reliable — but it's *not* an unbiased measure of retrieval
quality. For an unbiased complement, see
:func:`docsense.evaluation.structural_queries.generate_structural_queries`,
which builds queries programmatically from document headings.
"""

from __future__ import annotations

EvalQuery = tuple[str, list[str]]

CURATED_QUERIES: list[EvalQuery] = [
    # Core concepts
    ("How do I install transformers?", ["installation.md"]),
    (
        "What are the different types of tokenizers?",
        ["tokenizer_summary.md", "fast_tokenizers.md"],
    ),
    ("How does the pipeline API work?", ["pipeline_tutorial.md"]),
    (
        "What is gradient checkpointing and when should I use it?",
        ["perf_train_gpu_one.md", "deepspeed.md"],
    ),
    (
        "How do I use AutoModel to load a pretrained model?",
        ["model_doc/auto.md", "autoclass_tutorial.md"],
    ),
    # Training
    ("How do I fine-tune a model with Trainer?", ["trainer.md", "training.md"]),
    (
        "How to train on multiple GPUs with data parallelism?",
        ["perf_train_gpu_many.md", "fsdp.md"],
    ),
    ("What is DeepSpeed and how do I use it with transformers?", ["deepspeed.md"]),
    (
        "How do I customize the training loop with callbacks?",
        ["trainer_callbacks.md", "trainer_customize.md"],
    ),
    ("How to do hyperparameter search?", ["hpo_train.md"]),
    # Generation
    (
        "How does text generation work in transformers?",
        ["generation_strategies.md", "generation_features.md"],
    ),
    (
        "What are the different decoding strategies like beam search?",
        ["generation_strategies.md"],
    ),
    ("How do I use chat templates?", ["chat_templating.md"]),
    # Performance
    (
        "How to optimize inference on GPU?",
        ["perf_infer_gpu_multi.md", "perf_torch_compile.md"],
    ),
    ("How to reduce memory usage during training?", ["perf_train_gpu_one.md"]),
    (
        "How to train on CPU efficiently?",
        ["perf_train_cpu.md", "perf_train_cpu_many.md"],
    ),
    # Architecture / advanced
    ("How do I add a new model to transformers?", ["add_new_model.md"]),
    ("What is GGUF and how to use quantized models?", ["gguf.md", "quantization"]),
    ("How to use PEFT and LoRA with transformers?", ["peft.md", "trainer.md"]),
    ("How do I create a custom tokenizer?", ["custom_tokenizers.md"]),
]
