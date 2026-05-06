"""LLM wrapper that produces typed Answer objects.

`Generator` ties model loading, inference, and citation parsing together.
The structure is deliberately split so tests can mock the inference call
without dragging the model into the test process:

- ``model`` and ``tokenizer`` properties lazy-load the underlying
  HuggingFace components on first use. Tests that don't need real
  inference can avoid the cost entirely by overriding ``_run_inference``
  directly.
- ``_run_inference(messages) -> (text, raw_metadata)`` is the override
  point. Production runs through HuggingFace via
  ``tokenizer.apply_chat_template`` (so the chat formatting is
  model-specific but the input shape stays the same — Qwen, Llama,
  Mistral all work via the same code path); tests stub a fixed string
  and a fake latency.
- ``parse_citations()`` turns ``[N]`` notation in the generated text into
  ``Citation`` objects pointing back at the chunks the LLM saw.

The full ``generate()`` flow assembles a typed ``Answer`` with text,
citations, the retrieved chunks the LLM saw, and generation metadata
suitable for tracing and cost accounting.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any, cast

from docsense.generation.types import Answer, ChunkRef, Citation, GenerationMetadata

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from docsense.config import GenerationConfig


# Matches `[N]` where N is one or more digits. Anchored to a brace to
# avoid catching things like log timestamps `[2026-05-06 ...]`. The
# parser then validates N against the number of retrieved chunks.
_CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_citations(text: str, retrieved_chunks: list[ChunkRef]) -> list[Citation]:
    """Extract `[N]` citations from generated text and resolve them to chunks.

    A citation is included only if its numeric index falls within
    ``retrieved_chunks`` (1-indexed). Out-of-range indices and any
    duplicate `[N]` references in the text are quietly de-duplicated;
    the LLM citing the same chunk twice produces one Citation entry,
    not two.
    """
    seen: set[tuple[str, str]] = set()
    citations: list[Citation] = []
    for match in _CITATION_RE.finditer(text):
        idx = int(match.group(1))
        if idx < 1 or idx > len(retrieved_chunks):
            continue
        chunk = retrieved_chunks[idx - 1]
        key = (chunk.doc_id, chunk.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(doc_id=chunk.doc_id, chunk_id=chunk.chunk_id))
    return citations


class Generator:
    """Wraps a causal LM for the docsense generation pipeline.

    Lazy-loads the model and tokenizer on first use. Tests that don't
    need real inference can override ``_run_inference`` directly and
    avoid the model load entirely.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            from transformers import AutoModelForCausalLM

            kwargs: dict[str, Any] = {"device_map": self.config.device}
            if self.config.use_4bit_quantization:
                kwargs["quantization_config"] = self._build_4bit_config()

            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **kwargs,
            )
        return self._model

    @staticmethod
    def _build_4bit_config() -> Any:
        """Construct the BitsAndBytesConfig for NF4 4-bit loading.

        Lazy-imports both ``transformers.BitsAndBytesConfig`` and
        ``torch`` so installs without the ``[gpu]`` extras (no
        bitsandbytes) can still import the rest of this module — the
        caller will hit a clear ImportError only when they try to load
        the model with 4-bit on. This is what we want; module import
        should never fail on optional deps.
        """
        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return self._tokenizer

    def _run_inference(self, messages: list[dict[str, str]]) -> tuple[str, dict]:
        """Run the LLM and return ``(generated_text, raw_metadata)``.

        ``messages`` is the chat-format list produced by
        ``PromptBuilder.build()`` — ``[{"role": "system", ...},
        {"role": "user", ...}]``. The tokenizer's
        ``apply_chat_template(..., add_generation_prompt=True)`` renders
        them into the model-specific format (Qwen's ``<|im_start|>``,
        Llama's ``<|begin_of_text|>``, Mistral's ``[INST]``) before
        tokenization. This means the same Generator works against any
        Instruct-tuned model whose tokenizer ships a chat template.

        Tests override this method to inject a canned response without
        loading a real model. ``raw_metadata`` carries latency and token
        counts that flow into ``GenerationMetadata``.
        """
        # apply_chat_template returns a BatchEncoding when return_dict=True,
        # which is what we want so model.generate gets attention_mask in
        # addition to input_ids. cast: transformers' stub overloads return
        # too broad a union to narrow statically.
        inputs = cast(
            "Any",
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ),
        )
        input_token_count = int(inputs["input_ids"].shape[1])

        start = time.perf_counter()
        # cast: transformers' stub types AutoModelForCausalLM.from_pretrained()
        # as something whose .generate() resolves to Tensor; the runtime
        # behavior is a callable that returns a token-id tensor.
        output_ids = cast("Any", self.model).generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.temperature > 0,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # Slice off the prompt tokens to get just the generation
        completion_ids = output_ids[0][input_token_count:]
        completion_token_count = int(completion_ids.shape[0])
        # cast: tokenizer.decode() returns str | list[str] depending on input
        # shape; we pass a single sequence so the runtime type is str.
        text = cast("str", self.tokenizer.decode(completion_ids, skip_special_tokens=True))

        return text, {
            "latency_ms": latency_ms,
            "prompt_tokens": input_token_count,
            "completion_tokens": completion_token_count,
        }

    def generate(self, messages: list[dict[str, str]], retrieved_chunks: list[ChunkRef]) -> Answer:
        """Run inference, parse citations, and return a typed Answer.

        ``messages`` is the chat-format list from ``PromptBuilder.build()``.
        ``retrieved_chunks`` is the list the context-assembly stage chose
        to actually pass to the LLM (i.e., the prefix that fit within the
        token budget). Citations are resolved against this list, and the
        chunks themselves are stored on the returned ``Answer`` so the
        caller can audit context independently of retrieval.
        """
        text, raw = self._run_inference(messages)
        citations = parse_citations(text, retrieved_chunks)
        metadata = GenerationMetadata(
            model_name=self.config.model_name,
            latency_ms=float(raw["latency_ms"]),
            prompt_tokens=raw.get("prompt_tokens"),
            completion_tokens=raw.get("completion_tokens"),
        )
        return Answer(
            text=text.strip(),
            citations=citations,
            retrieved_chunks=list(retrieved_chunks),
            metadata=metadata,
        )
