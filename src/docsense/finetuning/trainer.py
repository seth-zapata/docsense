"""LoRA fine-tuning wrapper for QLoRA training.

Wraps PEFT model preparation + LoRA config + TRL SFTTrainer
construction with sensible defaults. Mirrors the lazy-load structure
of ``Generator`` and ``LlamaJudge`` for consistency.

Public surface:

- ``LoRAFineTuner.train(train_examples, val_examples)``: one-call
  training. Loads base + applies LoRA + builds dataset + runs the
  SFTTrainer loop. Returns final training metrics.
- ``LoRAFineTuner.save_adapter(output_dir)``: persist adapter
  weights to disk (~50 MB for 7B + rank 16). Base model weights
  are NOT saved — adapters apply on top of the original base model.

**Test boundary:** real training requires GPU + the actual base
model. Unit tests cover the pure-logic pieces (message formatting,
LoRA config construction, SFT config construction) and stub the
model-load path. The full ``train()`` loop is exercised manually
via ``scripts/train_lora.py`` against a real model in Block 3C+.

**NF4 config note:** the same NF4 quantization shape is now used
in three places (``Generator``, ``LlamaJudge``, ``LoRAFineTuner``).
This file duplicates it locally with ``compute_dtype=bfloat16`` for
training stability — the inference paths use ``float16`` and
consolidating into one helper would require a parameterized
``compute_dtype`` argument. Worth refactoring as a small follow-up
once Phase 3 is past Block 3C; right now the duplication is one
4-line config dict, low cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from docsense.generation.prompt import PromptBuilder

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from docsense.finetuning.config import FineTuningConfig
    from docsense.finetuning.dataset import TrainingExample
    from docsense.generation.types import ChunkRef


# TRL 0.29 introduced ``assistant_only_loss=True`` on SFTConfig, which
# uses the tokenizer's chat template to identify the assistant turn
# and mask prompt tokens from the loss automatically. No manual
# ``DataCollatorForCompletionOnlyLM`` + per-model response template
# needed. If we ever swap to a TRL version that doesn't have this,
# fall back to the older approach with ``response_template`` markers
# specific to the base model family (Qwen: ``<|im_start|>assistant\n``,
# Llama: ``<|start_header_id|>assistant<|end_header_id|>\n\n``,
# Mistral: ``[/INST]``).
#
# Caught in Block 3C.1 smoke: Qwen 2.5's stock chat template lacks the
# ``{% generation %}...{% endgeneration %}`` Jinja markers that TRL
# 0.29's assistant-mask machinery needs. Without them, SFTTrainer
# crashes mid-tokenization with "at least one example has no assistant
# tokens." We patch the tokenizer's template at training-time only
# (Generator's inference path keeps the stock template — the special
# tokens emitted are byte-identical, so train/serve consistency holds).
#
# Strategy: replace Qwen 2.5's complex multi-branch assistant block
# (which mixes user/system/assistant/tool-call rendering) with a
# minimal Qwen-compatible template that explicitly wraps assistant
# content in {% generation %} markers. Our training data has no tool
# calls and exactly one system message, so the simplification is safe
# for our use case.

_QWEN_TRAINING_CHAT_TEMPLATE = """{%- if messages[0].role == 'system' -%}
{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' -}}
{%- endif -%}
{%- for message in messages -%}
{%- if message.role == 'user' -%}
{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' -}}
{%- elif message.role == 'assistant' -%}
{{- '<|im_start|>assistant\\n' -}}
{% generation %}{{- message.content + '<|im_end|>' -}}{% endgeneration %}
{{- '\\n' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\\n' -}}
{%- endif -%}"""


def _is_qwen_model(model_name: str) -> bool:
    """Whether ``model_name`` is a Qwen-family model needing our chat template."""
    return model_name.startswith("Qwen/") or model_name.lower().startswith("qwen")


def _format_chunks_block(chunks: list[ChunkRef]) -> str:
    """Format retrieved chunks as numbered context blocks.

    Mirrors ``ContextAssembler._format_chunk`` exactly so the training
    distribution matches what the generator sees at inference time.
    Drift between train and inference formatting would mean the model
    is fine-tuned on a slightly different prompt shape than it gets
    at serve time — easy way to silently regress.

    Format: ``[N] (source: doc_id)\\n{text}`` joined with ``\\n\\n``.
    """
    return "\n\n".join(
        f"[{i}] (source: {c.doc_id})\n{c.text}" for i, c in enumerate(chunks, start=1)
    )


def format_messages_for_training(example: TrainingExample) -> list[dict[str, str]]:
    """Build the 3-turn chat-message list for one training example.

    System (citation directive) + user (numbered chunks + question) +
    assistant (ideal cited answer). The system and user turns come
    from the canonical ``PromptBuilder`` so train and inference share
    the exact same prompt prefix; we just append the assistant turn
    with the gold-standard answer for SFT loss computation.

    Refusal examples produce the same 3-turn structure with empty
    chunks (so the user message has an empty Context block) and the
    canonical refusal answer. The model learns: "even when the
    citation directive is in the system prompt, refusals are still
    appropriate when context is empty/irrelevant."
    """
    context = _format_chunks_block(example.retrieved_chunks)
    builder = PromptBuilder()
    messages = builder.build(query=example.query, context=context)
    messages.append({"role": "assistant", "content": example.ideal_answer})
    return messages


class LoRAFineTuner:
    """QLoRA fine-tuner for Qwen 2.5 7B Instruct (default base).

    Construction is cheap — just stores the config. Real model
    loading happens lazily on first ``train()`` call so tests that
    only exercise config and message-formatting paths don't pay the
    HF-load cost.

    Tests can override the ``model`` property or the
    ``_load_base_model`` method to inject a stub model for shape
    tests; ``train()`` itself isn't covered in unit tests because
    the SFT loop requires a real model.
    """

    def __init__(self, config: FineTuningConfig) -> None:
        self.config = config
        # ``_trained_model`` is the LoRA-wrapped model after train()
        # completes. Set by train(); ``save_adapter()`` reads from it.
        # Tests can also inject a stub model directly to exercise
        # save_adapter without running real training.
        self._trained_model: Any | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Lazy-loaded tokenizer for the base model.

        Sets ``pad_token = eos_token`` if the model lacks a pad token
        — Qwen 2.5 has one but other models in this size class
        (Mistral, some Llama variants) don't, and SFTTrainer needs
        a pad token to right-pad batched inputs. Doing this once at
        load time avoids a hard-to-spot SFTTrainer crash on first
        batch.

        For Qwen-family bases, also overrides ``chat_template`` with
        a custom training-time template that includes the
        ``{% generation %}`` markers TRL's ``assistant_only_loss=True``
        needs. The override is training-side only — Generator's
        inference tokenizer keeps the stock template, and the rendered
        special tokens are byte-identical between the two.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.config.base_model_name)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            if _is_qwen_model(self.config.base_model_name):
                tok.chat_template = _QWEN_TRAINING_CHAT_TEMPLATE
            self._tokenizer = tok
        return self._tokenizer

    @staticmethod
    def _build_nf4_config() -> Any:
        """NF4 quantization config for QLoRA training.

        ``compute_dtype=bfloat16`` for training stability — bf16's
        wider dynamic range avoids the gradient-scaler dance fp16
        requires. RTX 4070 (Ada, sm_89) and Modal A10G (Ampere,
        sm_86) both support bf16 natively. The inference-side configs
        in Generator and LlamaJudge use fp16 because they don't need
        gradient stability; same shape, different compute_dtype.
        """
        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def _load_base_model(self) -> PreTrainedModel:
        """Load the base model with NF4 quantization + k-bit prep.

        ``prepare_model_for_kbit_training`` from PEFT does the
        boring-but-load-bearing work: enables gradient on input
        embeddings, casts layer norms to fp32 for stability, sets up
        gradient checkpointing if requested. Without it, training
        either crashes on ``backward()`` or silently produces NaN
        gradients.
        """
        from peft import prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {"device_map": self.config.device_map}
        if self.config.use_4bit_quantization:
            kwargs["quantization_config"] = self._build_nf4_config()

        model = AutoModelForCausalLM.from_pretrained(self.config.base_model_name, **kwargs)
        if self.config.use_4bit_quantization:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
            )
        return model

    def _build_lora_config(self) -> Any:
        """Build the PEFT LoraConfig from our FineTuningConfig.

        ``bias="none"`` is the QLoRA-paper default — adapting bias
        terms adds parameters and rarely helps for narrow-skill
        fine-tuning. ``task_type="CAUSAL_LM"`` tells PEFT to wire up
        the right forward-pass shape for autoregressive training.
        """
        from peft import LoraConfig, TaskType

        return LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    def _build_sft_config(self) -> Any:
        """Build TRL SFTConfig from our FineTuningConfig.

        Maps our config fields onto TRL/HF Trainer field names. The
        notable choices:
        - ``bf16=True`` matches the NF4 compute_dtype.
        - ``report_to=[]`` disables wandb / TB auto-logging by default
          — Phase 3 logs are committed JSON, not external dashboards.
        - ``packing=False`` so each example is its own sequence;
          packing concatenates examples for throughput at the cost
          of cross-example attention contamination, which is bad for
          short-answer fine-tuning like ours.
        - ``assistant_only_loss=True`` masks prompt tokens from the
          loss using the tokenizer's chat template. New in TRL 0.29+;
          replaces the older DataCollatorForCompletionOnlyLM pattern.
        """
        from trl import SFTConfig

        return SFTConfig(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            save_strategy=self.config.save_strategy,
            eval_strategy=self.config.eval_strategy,
            logging_steps=self.config.logging_steps,
            seed=self.config.seed,
            bf16=self.config.bf16,
            report_to=[],
            packing=False,
            assistant_only_loss=True,
            # TRL 0.29+ renamed ``max_seq_length`` → ``max_length`` in
            # SFTConfig. Our FineTuningConfig field stays
            # ``max_seq_length`` for naming clarity (it's the *sequence*
            # length cap); we map to TRL's name here.
            max_length=self.config.max_seq_length,
        )

    def _format_dataset(self, examples: list[TrainingExample]) -> Any:
        """Convert TrainingExamples to a TRL-friendly HF Dataset.

        Each example becomes a ``{"messages": [...]}`` row in the
        conversational format SFTTrainer recognizes. Combined with
        ``assistant_only_loss=True`` on SFTConfig, TRL handles the
        chat-template application and prompt-masking automatically.
        """
        from datasets import Dataset

        rows: list[dict[str, Any]] = []
        for ex in examples:
            rows.append({"messages": format_messages_for_training(ex)})
        return Dataset.from_list(rows)

    def train(
        self,
        train_examples: list[TrainingExample],
        val_examples: list[TrainingExample] | None = None,
    ) -> dict[str, Any]:
        """Run the full SFT loop. Returns final training metrics.

        Builds train + (optional) val datasets in TRL's conversational
        format. Loads the base model with NF4 + k-bit prep, then hands
        the base + LoRA config to SFTTrainer (which wraps with PEFT
        internally). Runs the SFT loop with assistant-only loss
        masking. Returns ``{"training_loss": float, "metrics": {...}}``.

        After training, ``self._trained_model`` holds the LoRA-wrapped
        trained model so ``save_adapter()`` can persist it. The trainer
        also saves checkpoints to ``config.output_dir`` per
        ``save_strategy`` (default: end of each epoch); the explicit
        ``save_adapter()`` call writes the final adapter to a
        well-known path the caller knows.
        """
        from trl import SFTTrainer

        train_dataset = self._format_dataset(train_examples)
        eval_dataset = self._format_dataset(val_examples) if val_examples is not None else None

        trainer = SFTTrainer(
            model=self._load_base_model(),
            args=self._build_sft_config(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            peft_config=self._build_lora_config(),
        )

        result = trainer.train()
        self._trained_model = trainer.model
        return {
            "training_loss": float(result.training_loss),
            "metrics": dict(result.metrics) if result.metrics else {},
        }

    def save_adapter(self, output_dir: Path | None = None) -> Path:
        """Save the trained LoRA adapter weights to disk.

        Writes only the adapter parameters (~50 MB at rank=16) plus
        the LoRA config metadata. The base model weights are not
        saved — at inference time the adapter is loaded on top of the
        original base model file from the HF cache.

        ``output_dir`` defaults to ``self.config.output_dir`` if not
        provided. Must be called after ``train()`` so there's a
        trained adapter to save (or a stub model injected via
        ``_trained_model`` for testing).
        """
        from pathlib import Path as _Path

        if self._trained_model is None:
            msg = (
                "save_adapter() called before train(). The LoRA-wrapped "
                "trained model is set by train(); without it there's "
                "nothing to save. Call train() first, or inject a stub "
                "via self._trained_model = ... for testing."
            )
            raise RuntimeError(msg)

        target = _Path(output_dir) if output_dir is not None else self.config.output_dir
        target.mkdir(parents=True, exist_ok=True)
        # PEFT's save_pretrained writes adapter_config.json +
        # adapter_model.safetensors (or .bin on older versions).
        self._trained_model.save_pretrained(str(target))
        return target
