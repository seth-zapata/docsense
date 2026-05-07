#!/usr/bin/env python3
"""Anthropic-distilled training-data generator for Phase 3 fine-tuning.

Block 3B of Phase 3 (per ``docs/phase-3-scope.md``). Reads (query,
retrieved_chunks) pairs from an input JSON file, distills "ideal cited
answer" for each via the Anthropic API, and writes a TrainingDataset.

Block 3B.1 — pilot run: small batch (5-7 examples) used to compare
distillation models (Sonnet 4.5 / Opus 4.7 / Haiku 4.5, with optional
extended thinking) on identical inputs. We pick the winning variant
based on citation accuracy + style match before scaling to ~600-900
examples in Block 3B.2.

Usage::

    # Variant A: Sonnet 4.5, no thinking (default)
    python scripts/build_training_dataset.py \\
        --input evaluations/datasets/training/pilot_input.json \\
        --output evaluations/datasets/training/pilot/sonnet.json

    # Variant B: Opus 4.7
    python scripts/build_training_dataset.py \\
        --input ... --output ... --model claude-opus-4-7

    # Variant C: Haiku 4.5
    python scripts/build_training_dataset.py \\
        --input ... --output ... --model claude-haiku-4-5

    # Variant D (conditional): Sonnet + extended thinking
    python scripts/build_training_dataset.py \\
        --input ... --output ... --enable-thinking

API key is read from ``~/.anthropic-key`` (mode 600 file). Same flow
as the HF token setup — paste once, scripts read on demand.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docsense.finetuning.dataset import TrainingDataset, TrainingExample
from docsense.generation.types import ChunkRef

if TYPE_CHECKING:
    from anthropic import Anthropic

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


CANONICAL_REFUSAL = "I don't have enough context to answer that."

# Default model. CLI ``--model`` overrides for the pilot comparison.
DEFAULT_MODEL = "claude-sonnet-4-5"

# Token budgets for the API call. ``max_tokens`` is the hard cap on
# the response; we keep it generous (1024) since most ideal answers
# are 50-200 tokens but occasional longer ones shouldn't get truncated.
# ``thinking_budget_tokens`` is for variant D (extended thinking) —
# 2000 is a reasonable starting point per the Anthropic docs.
_MAX_TOKENS = 1024
_THINKING_BUDGET = 2000

# Per-call wait — gentle backoff to avoid hammering the API. Anthropic's
# rate limits are generous; we add a small delay mainly so KeyboardInterrupt
# stops us cleanly between calls.
_INTER_CALL_DELAY_SEC = 0.3


# Models that have deprecated the ``temperature`` parameter — calling
# them with ``temperature=...`` returns 400 "deprecated for this model".
# Exact model-name match (versioned) so a future model unrelated to
# Opus 4.7 doesn't get the same treatment by accident.
_MODELS_WITHOUT_TEMPERATURE: frozenset[str] = frozenset(
    {
        "claude-opus-4-7",
        "claude-opus-4-7-20260101",  # date-stamped variant if Anthropic adds one
    }
)


def _model_rejects_temperature(model: str) -> bool:
    """True if the model's API endpoint rejects the temperature kwarg."""
    return any(model.startswith(name) for name in _MODELS_WITHOUT_TEMPERATURE)


# --- Distillation prompt --------------------------------------------
# The system prompt that frames Claude's role: produce ideal cited
# answers in the style we want Qwen to learn. Style choices:
# - Concise + direct (no marketing fluff, no hedging)
# - Numbered citations after factual claims
# - ONLY use information present in chunks (no parametric memory leak)
# - Canonical refusal phrase when chunks don't support the question
#
# We embed two few-shot examples (one good answer + the canonical
# refusal phrasing) so Claude calibrates to our exact format. Pin
# the prompt verbatim as a module constant so the per-run config
# snapshot can record which prompt version produced which dataset.

_DISTILL_PROMPT_VERSION = "v2"

_DISTILL_SYSTEM_PROMPT = """You are a distillation assistant generating ideal cited answers for fine-tuning a smaller documentation Q&A model.

Given a question and retrieved context chunks, produce the ideal answer the smaller model should learn to produce. Your answers must:

1. Match length to what the answer actually requires. The bar: a developer reading your answer should be able to ACT on it without looking elsewhere for prerequisites or basic configuration. Don't skip install steps, configuration toggles, or other necessary prerequisites mentioned in the chunks. Don't include filler ("great question", "as you can see"), redundant paraphrases, or tangential information about adjacent features. A 2-sentence answer is right for a simple comparison; a 4-5 sentence answer is right for a multi-step procedure with prerequisites — both are correct lengths for their question type.
2. Cite chunks with [N] markers — append [N] after each factual claim that comes from chunk N. Multiple citations after one claim are fine ([1][3]) when the claim draws from multiple chunks.
3. Use direct, technical prose. No marketing language, no preamble like "Based on the provided context...". Get to the answer.
4. ONLY use information present in the chunks. Do NOT draw from your training data, even if you "know" the right answer — pretend the chunks are your only source. Inventing facts beyond the chunks is a hard failure.
5. If the chunks don't contain enough information to answer the question, respond with EXACTLY this phrase and nothing else: "I don't have enough context to answer that."
6. Do NOT generate URLs. When documentation references a file or commit, mention the filename or path in plain text (e.g., `modeling_bert.py`) without linking. Specific URLs may not be stable and shouldn't appear in training data.

Format your response as a single JSON object with one key:

{"answer": "<your cited answer here>"}

Do not include markdown code fences. Do not include any other text. Just the JSON object.

EXAMPLE 1 (procedural with prerequisites — appropriately longer):
QUESTION: How do I use Flash Attention 2 with transformers models?
CHUNKS:
[1] (source: flash_attn.md)
Install the flash-attn package via `pip install flash-attn --no-build-isolation` before using Flash Attention 2.
[2] (source: flash_attn.md)
Pass `attn_implementation="flash_attention_2"` to `.from_pretrained()` to enable it.
[3] (source: flash_attn.md)
Models should be loaded in half-precision (`torch.float16` or `torch.bfloat16`) for the memory and speed benefits.

GOOD: {"answer": "Install the flash-attn package first with `pip install flash-attn --no-build-isolation` [1]. Then pass `attn_implementation=\\"flash_attention_2\\"` to `.from_pretrained()` when loading the model [2]. Load in half-precision (`torch.float16` or `torch.bfloat16`) to get the memory and speed benefits [3]."}

(Note: the install step is included even though the question said "use Flash Attention 2", not "install". A user trying to use Flash Attention 2 without the package will hit ImportError. Skipping the prerequisite would be a correctness gap.)

EXAMPLE 2 (simple comparison — appropriately short):
QUESTION: What's the difference between AutoModel and AutoModelForCausalLM?
CHUNKS:
[1] (source: model_doc/auto.md)
AutoModel is a barebones class that outputs hidden states without a task-specific head.
[2] (source: model_doc/auto.md)
AutoModelForCausalLM adds a causal language modeling head for next-token prediction tasks.

GOOD: {"answer": "AutoModel is a barebones class that outputs hidden states without a task-specific head [1], while AutoModelForCausalLM adds a causal language modeling head for next-token prediction tasks [2]."}

(Note: 2 sentences, no padding. Don't add a third sentence saying "the same architecture can be used with different heads" — that's redundant context, not new information.)

EXAMPLE 3 (pointer answer — single sentence is enough):
QUESTION: How do I convert a model from TensorFlow to PyTorch?
CHUNKS:
[1] (source: add_new_model.md)
The BERT conversion script in `modeling_bert.py` is a good starting point — copy, adapt, and reuse it for your model.

GOOD: {"answer": "Use the BERT conversion script in `modeling_bert.py` as a starting point — copy, adapt, and reuse it for your model [1]."}

(Note: filename in plain text, no URL. The chunk content is straightforward; one sentence covers it.)

EXAMPLE 4 (off-corpus refusal):
QUESTION: What's the latest stable Linux kernel version?
CHUNKS:
[1] (source: trainer.md)
The Trainer class supports distributed training via DDP and FSDP.
[2] (source: trainer.md)
Use `Trainer.train()` to start a fine-tuning run.

GOOD: {"answer": "I don't have enough context to answer that."}"""


# --- I/O helpers ----------------------------------------------------


@dataclass
class _PilotInput:
    """One (query, chunks) pair from an input file.

    Two file formats are supported:

    - **Pilot JSON** (``pilot_input.json``): hand-curated array of
      ``{query, retrieved_chunks, expected_refusal, note}`` entries
      from Block 3B.1.
    - **Pool JSONL** (``query_pool.jsonl``): newline-separated entries
      from Block 3B.2.e's ``build_query_pool.py``, with extra fields
      ``question_type``, ``seed_chunks``, ``seed_topic``, ``metadata``.

    Pool-only fields default to ``None`` / empty so the same dataclass
    serves both inputs. ``query_gen_metadata`` carries the generation
    provenance (gen_model, prompt_version, captured_at, tokens, cost)
    forward into the distilled ``TrainingExample.metadata`` so the
    final dataset has full end-to-end provenance.
    """

    query: str
    retrieved_chunks: list[ChunkRef]
    expected_refusal: bool = False
    note: str = ""
    question_type: str | None = None
    query_gen_metadata: dict[str, Any] | None = None


def _input_to_pilot(item: dict[str, Any]) -> _PilotInput:
    """Build a _PilotInput from one dict (pilot or pool entry).

    Common fields come from the same keys; pool-specific fields
    (``question_type``, ``metadata``) are picked up if present.
    """
    chunks = [ChunkRef.model_validate(c) for c in item["retrieved_chunks"]]
    return _PilotInput(
        query=item["query"],
        retrieved_chunks=chunks,
        expected_refusal=item.get("expected_refusal", False),
        note=item.get("note", ""),
        question_type=item.get("question_type"),
        query_gen_metadata=item.get("metadata"),
    )


def _load_inputs(path: Path) -> list[_PilotInput]:
    """Parse the input file into PilotInput records.

    Dispatches by file extension: ``.jsonl`` → newline-separated
    (pool format), anything else → JSON array (pilot format). The
    pilot-input dispatch is kept as the default so existing
    ``pilot_input.json`` callers don't break.
    """
    if path.suffix == ".jsonl":
        items: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    else:
        items = json.loads(path.read_text())
    return [_input_to_pilot(item) for item in items]


def _read_api_key() -> str:
    """Read the Anthropic API key from ``~/.anthropic-key``.

    Mirrors the HF token flow — file is mode 600 so only the owner
    can read it; never committed; the script reads it on demand.
    Raises a clear error pointing at the setup instructions if the
    file is missing rather than letting the SDK fail with a less
    obvious "missing API key" later.
    """
    key_path = Path.home() / ".anthropic-key"
    if not key_path.exists():
        msg = (
            f"No Anthropic API key at {key_path}. Set up with:\n"
            '  /home/sethz/projects/docsense/.venv/bin/python -c """\n'
            "  from getpass import getpass; from pathlib import Path\n"
            "  key = getpass('Anthropic API key: ').strip()\n"
            "  if not key.startswith('sk-ant-'): raise SystemExit('not an Anthropic key')\n"
            "  p = Path.home() / '.anthropic-key'\n"
            "  p.write_text(key + chr(10)); p.chmod(0o600)\n"
            '  print("written to", p)\n'
            '"""'
        )
        raise SystemExit(msg)
    return key_path.read_text().strip()


# --- Resumability ---------------------------------------------------
#
# A 30-min, ~$4 distillation run loses everything on a single uncaught
# exception (network blip, citation-grounding ValueError, OOM). The
# fix is per-example incremental persistence: each successful
# TrainingExample is appended to a sibling JSONL file immediately
# after generation. On restart, we read that file, build a set of
# already-done queries (keyed by query text + chunk_ids so changes
# to either invalidate the cache), and skip them in the main loop.
#
# Mirrors the same pattern used by build_query_pool.py for Stage 1.


def _raw_jsonl_path(output_path: Path) -> Path:
    """Sibling JSONL file for incremental persistence + resume.

    For ``output=/tmp/dataset.json``, returns
    ``/tmp/dataset.raw.jsonl``. The final TrainingDataset JSON is
    assembled from this raw file at the end of the run.
    """
    return output_path.parent / (output_path.stem + ".raw.jsonl")


def _cache_key(query: str, chunks: list[ChunkRef]) -> str:
    """Resume key: (query, sorted chunk_ids).

    Both fields matter — a query alone could repeat with different
    retrieved_chunks, and the answer depends on both. If either
    changes between runs (user re-runs Stage 1, the seeder picks
    different chunks), the cached answer is correctly invalidated.
    """
    chunks_key = ",".join(c.chunk_id for c in chunks)
    return f"{query}::{chunks_key}"


def _read_done_examples(raw_path: Path) -> dict[str, TrainingExample]:
    """Read existing checkpoint. Returns ``{cache_key: TrainingExample}``.

    Tolerates blank lines and individually corrupt lines so a partial-
    write or manual edit doesn't lose ALL prior progress.
    """
    if not raw_path.exists():
        return {}
    done: dict[str, TrainingExample] = {}
    for line in raw_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ex = TrainingExample.model_validate_json(line)
        except (ValueError, json.JSONDecodeError):
            # Corrupt line; skip. One bad line shouldn't lose the rest.
            continue
        done[_cache_key(ex.query, ex.retrieved_chunks)] = ex
    return done


def _append_example(raw_path: Path, example: TrainingExample) -> None:
    """Append one TrainingExample to the JSONL checkpoint."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a") as f:
        f.write(example.model_dump_json() + "\n")


def _process_one_input(
    client: Anthropic,
    inp: _PilotInput,
    *,
    model: str,
    enable_thinking: bool,
) -> TrainingExample | None:
    """Distill one input. Returns ``None`` on any recoverable error.

    Two recoverable failure modes are caught here so the caller can
    skip-and-continue rather than crashing the entire run:

    - ``RuntimeError``: parser couldn't extract an answer from the
      model's response (bad JSON, missing field, etc.).
    - ``ValueError``: ``TrainingExample``'s citation-grounding
      validator rejected the answer (Sonnet hallucinated a chunk
      index outside ``[1, len(chunks)]``).

    Both are "bad output, skip this one" cases. Without catching
    ``ValueError``, a single hallucinated citation in the middle of
    a 600-query run would crash the script and lose all prior
    progress. Reproduced once in pre-flight stress tests; would
    have lost ~$4 of compute mid-run.
    """
    try:
        return distill_one_example(
            client,
            query=inp.query,
            chunks=inp.retrieved_chunks,
            model=model,
            enable_thinking=enable_thinking,
            question_type=inp.question_type,
            query_gen_metadata=inp.query_gen_metadata,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "Skipped (%s): %s — query: %r",
            type(exc).__name__,
            exc,
            inp.query[:80],
        )
        return None


# --- JSON response parsing ------------------------------------------

# Same JSON-extraction pattern we use for the Llama judge in
# ``llama_judge.py``. Strips markdown fences if present, then locates
# the first ``{...}`` block. Tolerant of LLM prose around the JSON.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_block(text: str) -> str:
    text = text.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match is not None:
        return fence_match.group(1).strip()
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match is not None:
        return brace_match.group(0)
    return text


def parse_distill_response(raw_text: str) -> str | None:
    """Extract the ``answer`` string from the model's JSON output.

    Returns ``None`` on parse failure — caller decides whether to
    retry or fall back. Same conservative pattern as the eval-pipeline
    parsers: typed result on every input, never raise.

    **Plain-text refusal escape hatch:** Sonnet 4.5 sometimes drops
    the JSON wrapper specifically when refusing — returning just
    ``"I don't have enough context to answer that."`` (sometimes
    quoted, sometimes bare) instead of ``{"answer": "..."}``.
    Observed at ~14% of procedural queries in the Block 3B.2 spot-check.
    Treating this as a parse failure would silently lose valid
    refusal training data, so we detect the canonical phrase first
    and return it as a refusal answer before falling through to the
    JSON path.
    """
    # Plain-text refusal: strip surrounding whitespace + optional
    # JSON-string quotes ("..."), then exact-match against the
    # canonical refusal. Catches both bare and string-quoted forms
    # the model produces.
    stripped = raw_text.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        stripped = stripped[1:-1]
    if stripped == CANONICAL_REFUSAL:
        return CANONICAL_REFUSAL

    block = _extract_json_block(raw_text)
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    return answer.strip()


# --- Per-call distillation ------------------------------------------


def format_user_message(query: str, chunks: list[ChunkRef]) -> str:
    """Build the user message: numbered chunks + the question.

    Format mirrors what the Llama judge sees during faithfulness
    attribution and what ContextAssembler produces at inference time
    — same numbered ``[N] (source: doc_id)\\n{text}`` shape. We're
    consistent across all "the LLM reads chunks" surfaces in the
    codebase so prompt-shape drift can't cause subtle distribution
    mismatch.
    """
    if not chunks:
        chunks_block = "(no chunks retrieved)"
    else:
        chunks_block = "\n\n".join(
            f"[{i}] (source: {c.doc_id})\n{c.text}" for i, c in enumerate(chunks, start=1)
        )
    return f"CHUNKS:\n{chunks_block}\n\nQUESTION: {query}"


def distill_one_example(
    client: Anthropic,
    *,
    query: str,
    chunks: list[ChunkRef],
    model: str,
    enable_thinking: bool = False,
    question_type: str | None = None,
    query_gen_metadata: dict[str, Any] | None = None,
) -> TrainingExample:
    """Produce one TrainingExample by calling the Anthropic API.

    Returns a TrainingExample with metadata recording the model,
    thinking flag, prompt version, token usage, and timestamp — full
    provenance for portfolio reproducibility. ``is_refusal`` is set
    automatically when the answer matches the canonical refusal phrase.

    On JSON parse failure, raises RuntimeError so the caller can
    decide whether to abort the batch or skip the example. Pilot
    runs are small enough that aborting is fine; the full Block 3B.2
    run would log + skip + report parse failures at the end.
    """
    user_message = format_user_message(query=query, chunks=chunks)

    # Anthropic constraint: when extended thinking is enabled,
    # max_tokens must be strictly greater than thinking.budget_tokens
    # (the budget is the cap on thinking-block tokens; max_tokens caps
    # the entire response including thinking + visible text). Add the
    # thinking budget on top of the regular max so we leave room for
    # both the thought trace and the actual answer.
    max_tokens = _MAX_TOKENS + _THINKING_BUDGET if enable_thinking else _MAX_TOKENS

    create_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": _DISTILL_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message}],
    }
    # Temperature handling — different Anthropic models accept
    # different temperature configurations:
    # - Opus 4.7 deprecated the parameter entirely (rejects with 400
    #   "`temperature` is deprecated for this model")
    # - Sonnet/Haiku accept any value in [0, 1]; we use 0.0 for
    #   determinism (same input → same output across reruns)
    # - Extended thinking incompatible with non-default temperature
    #   on any model that supports it
    # Skip explicit temperature when the model rejects it OR when
    # thinking is on; otherwise pass 0.0 for reproducibility.
    if not enable_thinking and not _model_rejects_temperature(model):
        create_kwargs["temperature"] = 0.0
    if enable_thinking:
        create_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": _THINKING_BUDGET,
        }

    response = client.messages.create(**create_kwargs)

    # Response.content is a list of content blocks (thinking + text
    # when extended thinking is enabled). We only train on the text
    # — the thinking block is internal, used for Claude's reasoning,
    # not for downstream consumption.
    text_blocks = [block.text for block in response.content if block.type == "text"]
    raw_text = "\n".join(text_blocks)

    answer = parse_distill_response(raw_text)
    if answer is None:
        msg = (
            f"Distillation parse failure for query {query[:60]!r}. "
            f"Model {model!r} produced unparseable JSON output. Raw "
            f"response (first 300 chars): {raw_text[:300]!r}"
        )
        raise RuntimeError(msg)

    is_refusal = answer == CANONICAL_REFUSAL

    metadata: dict[str, Any] = {
        "distill_model": model,
        "enable_thinking": enable_thinking,
        "prompt_version": _DISTILL_PROMPT_VERSION,
        "captured_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    # Pool-only provenance: nest under a sub-key so distill fields
    # remain at the top level (existing test contracts unchanged) while
    # query-gen provenance is preserved end-to-end.
    if question_type is not None:
        metadata["question_type"] = question_type
    if query_gen_metadata is not None:
        metadata["query_gen"] = query_gen_metadata

    return TrainingExample(
        query=query,
        retrieved_chunks=chunks,
        ideal_answer=answer,
        is_refusal=is_refusal,
        metadata=metadata,
    )


# --- Main -----------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to a JSON or JSONL input file. Accepts two formats:\n"
            "  - .json: pilot-input array (Block 3B.1 hand-curated).\n"
            "  - .jsonl: query_pool entries from build_query_pool.py "
            "(Block 3B.2). Pool-only fields (question_type, gen "
            "metadata) are propagated into TrainingExample.metadata."
        ),
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write the TrainingDataset JSON."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model name. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable extended thinking (Variant D in the Block 3B.1 pilot). "
            "Anthropic charges thinking tokens as output; expect ~3-4× output "
            "cost when on. Forces temperature=1 internally (Anthropic constraint)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap input examples to first N (smoke runs).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Delete the existing per-example JSONL checkpoint and start "
            "fresh. Default behavior reads <output>.raw.jsonl and skips "
            "queries already done — safe to re-run after a crash without "
            "re-paying for completed calls."
        ),
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    # --- Load inputs ----------------------------------------------
    inputs = _load_inputs(args.input)
    if args.limit is not None:
        inputs = inputs[: args.limit]
        logger.info(
            "Limit applied: %d examples (was %d)", len(inputs), len(_load_inputs(args.input))
        )
    logger.info(
        "Loaded %d pilot inputs (%d in-corpus + %d refusal)",
        len(inputs),
        sum(1 for x in inputs if not x.expected_refusal),
        sum(1 for x in inputs if x.expected_refusal),
    )

    # --- Connect ---------------------------------------------------
    import anthropic  # imported here so test paths don't load the SDK

    api_key = _read_api_key()
    client = anthropic.Anthropic(api_key=api_key)

    logger.info(
        "Distilling with model=%s thinking=%s",
        args.model,
        "on" if args.enable_thinking else "off",
    )

    # --- Resume bookkeeping ---------------------------------------
    raw_path = _raw_jsonl_path(args.output)
    if args.no_resume and raw_path.exists():
        logger.info("--no-resume: deleting existing %s", raw_path)
        raw_path.unlink()
    done_examples = _read_done_examples(raw_path)
    if done_examples:
        logger.info("Resuming: %d examples already done in %s", len(done_examples), raw_path)

    # --- Run -------------------------------------------------------
    examples: list[TrainingExample] = []
    total_input_tokens = 0
    total_output_tokens = 0
    n_skipped_done = 0
    n_skipped_failure = 0
    for i, inp in enumerate(inputs, start=1):
        # Resume: if we've already distilled this exact (query, chunks)
        # pair, reuse the cached TrainingExample and skip the API call.
        cached = done_examples.get(_cache_key(inp.query, inp.retrieved_chunks))
        if cached is not None:
            examples.append(cached)
            n_skipped_done += 1
            continue

        logger.info(
            "[%d/%d] %s%s — %s",
            i,
            len(inputs),
            args.model,
            " +thinking" if args.enable_thinking else "",
            inp.query[:80],
        )
        ex = _process_one_input(
            client,
            inp,
            model=args.model,
            enable_thinking=args.enable_thinking,
        )
        if ex is None:
            n_skipped_failure += 1
            continue
        examples.append(ex)
        # Append IMMEDIATELY so a later crash doesn't lose this example.
        _append_example(raw_path, ex)
        if i < len(inputs):
            time.sleep(_INTER_CALL_DELAY_SEC)

    # Token totals are computed across ALL examples (fresh + cached) so
    # a resumed run reports the true cumulative token spend, not just
    # this run's incremental cost.
    for ex in examples:
        total_input_tokens += int(ex.metadata.get("input_tokens", 0))
        total_output_tokens += int(ex.metadata.get("output_tokens", 0))

    if n_skipped_done:
        logger.info("Reused %d cached examples from prior run(s)", n_skipped_done)
    if n_skipped_failure:
        logger.info("Skipped %d examples due to parse/citation failures", n_skipped_failure)

    # --- Persist --------------------------------------------------
    n_in_corpus = sum(1 for ex in examples if not ex.is_refusal)
    n_refusal = sum(1 for ex in examples if ex.is_refusal)

    variant_tag = f"{args.model}{'_thinking' if args.enable_thinking else ''}"
    dataset = TrainingDataset(
        examples=examples,
        version=f"pilot-{variant_tag}-v1",
        description=(
            f"Pilot distillation run (Block 3B.1). "
            f"Model={args.model}, enable_thinking={args.enable_thinking}, "
            f"prompt_version={_DISTILL_PROMPT_VERSION}, "
            f"input_file={args.input.name}."
        ),
        metadata={
            "model": args.model,
            "enable_thinking": args.enable_thinking,
            "prompt_version": _DISTILL_PROMPT_VERSION,
            "input_file": str(args.input),
            "n_inputs": len(inputs),
            "n_examples": len(examples),
            "n_in_corpus": n_in_corpus,
            "n_refusal": n_refusal,
            "n_resumed_from_cache": n_skipped_done,
            "n_parse_failures": n_skipped_failure,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "captured_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(args.output)

    logger.info(
        "Wrote %d examples (%d in-corpus, %d refusal) to %s",
        len(examples),
        n_in_corpus,
        n_refusal,
        args.output,
    )
    logger.info(
        "Token usage: %d input / %d output (model=%s)",
        total_input_tokens,
        total_output_tokens,
        args.model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
