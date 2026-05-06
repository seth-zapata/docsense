"""Concrete LLMJudge using Llama 3.1 8B Instruct at NF4 4-bit.

Mirrors ``Generator``'s structure on purpose so the codebase has one
shape for "lazy-loaded NF4 causal LM with a chat-template inference
hook":

- ``model`` and ``tokenizer`` are lazy properties; first use triggers
  the HF load. Tests that override ``_run_inference`` avoid the
  load entirely.
- ``_run_inference(messages) -> str`` is the override point. Production
  goes through ``tokenizer.apply_chat_template(...)`` so the prompt
  format is whatever Llama 3.1 ships with; tests stub a canned string.

**Faithfulness** uses RAGAS-style claim-level decomposition with
per-chunk attribution (replaced the absolute-scale anchor approach
2026-05-06 after the baseline showed 49 of 50 in-corpus answers
clustering at 0.75). Two LLM calls per query:

1. ``extract_claims(answer)`` — decompose into atomic factual claims.
2. ``attribute_claims_to_chunks(claims, chunks)`` — for each claim,
   identify which chunk supports it (or ``None``).

Aggregate: ``score = n_supported / n_total`` in ``[0, 1]``,
continuous (no anchor snapping). Per-claim attributions are
preserved on the JudgeScore so reports carry grounded evidence.

**Relevance** still uses the absolute-scale anchor approach
(0.0 / 0.25 / 0.5 / 0.75 / 1.0). Relevance had healthier spread in
the baseline (3 queries at 0.25, 4 at 1.0 on structural) and
"does the answer address the question?" doesn't naturally
decompose into claims the way faithfulness does.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

from docsense.evaluation.judge import (
    ClaimAttribution,
    JudgeMetric,
    JudgeScore,
    LLMJudge,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from docsense.config import JudgeConfig
    from docsense.generation.types import ChunkRef


_RELEVANCE_ANCHORS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

# Tolerant of the model dropping the colon, putting the number inline
# with text, or emitting a leading-dot decimal like ".75". Anchored to
# avoid catching numbers inside the rationale.
_SCORE_RE = re.compile(
    r"SCORE\s*:?\s*([01](?:\.\d+)?|\.\d+)",
    re.IGNORECASE,
)
# Captures everything from "RATIONALE:" to the next blank line or end
# of string. DOTALL so multi-line rationales are kept.
_RATIONALE_RE = re.compile(
    r"RATIONALE\s*:?\s*(.+?)(?=\n\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


# --- Faithfulness: claim extraction -----------------------------------

_EXTRACT_CLAIMS_SYSTEM = (
    "You are decomposing an LLM-generated answer into atomic factual "
    "claims. A claim is a single, verifiable factual statement small "
    "enough to be checked against a single source. Skip rhetorical "
    "content, conversational framing, opinions, and code/markdown "
    "formatting — only extract factual content.\n\n"
    "Examples of well-formed claims:\n"
    "- AutoModel.from_pretrained accepts a model name from the "
    "HuggingFace Hub.\n"
    "- The default cache directory is ~/.cache/huggingface.\n"
    "- Setting trust_remote_code=True allows execution of custom "
    "model code.\n\n"
    "Output one claim per line as a numbered list:\n"
    "1. <claim 1>\n"
    "2. <claim 2>\n"
    "...\n\n"
    "If the answer contains no factual claims (e.g., it's a refusal, "
    "or purely a code example with no surrounding statements), "
    "output exactly: NO_CLAIMS"
)
_EXTRACT_CLAIMS_USER_TEMPLATE = "QUESTION:\n{question}\n\nANSWER:\n{answer}"

# Numbered-list line: "<num>. <claim>". Multiline mode so ^ matches
# each line. The (?:.|\n(?!\d+\.|\Z))* trick lets a claim span lines
# until the next numbered item or end of string — handles claims that
# wrap mid-sentence.
_CLAIM_LINE_RE = re.compile(
    r"^\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|\Z)",
    re.MULTILINE | re.DOTALL,
)


def parse_claims(text: str) -> list[str]:
    """Extract claims from the LLM's claim-extraction output.

    Returns an empty list if the model emitted ``NO_CLAIMS`` (its
    sentinel for "answer has no factual content"). Otherwise returns
    a list of claim texts in numerical order. The numbered prefixes
    are stripped; whitespace is trimmed.

    Robust to:
    - Multi-line claims (whitespace between numbered items)
    - Out-of-order numbering (still returns claims in document order)
    - Missing trailing period or other punctuation
    - Leading/trailing whitespace per claim
    """
    if re.search(r"\bNO_CLAIMS\b", text, re.IGNORECASE):
        return []
    matches = _CLAIM_LINE_RE.findall(text)
    return [claim.strip() for _, claim in matches if claim.strip()]


# --- Faithfulness: claim attribution ----------------------------------

_ATTRIBUTE_CLAIMS_SYSTEM = (
    "You are attributing claims to retrieved source chunks. For each "
    "claim, identify which chunk (1-indexed) supports it. A claim is "
    '"supported by chunk N" if the chunk\'s content provides evidence '
    "for the claim via reasonable inference. Direct quotation is not "
    "required — paraphrasing and reasonable inference are fine. If no "
    'chunk supports the claim, output "none".\n\n'
    "Output ONE LINE PER CLAIM in this exact format:\n"
    "CLAIM <i>: chunk <chunk_idx> | RATIONALE: <one short sentence>\n"
    "or\n"
    "CLAIM <i>: none | RATIONALE: <one short sentence>\n\n"
    "Output the claims in numerical order. Do not output anything else."
)

_ATTRIBUTE_CLAIMS_USER_TEMPLATE = "CHUNKS:\n{chunks_block}\n\nCLAIMS:\n{claims_block}"

# "CLAIM 3: chunk 2 | RATIONALE: ..." — captures claim index, the
# chunk index (or "none"), and the optional rationale that follows
# the pipe. Tolerant of the rationale being missing.
_ATTRIBUTION_LINE_RE = re.compile(
    r"^\s*CLAIM\s+(\d+)\s*:?\s*(?:chunk\s+(\d+)|(none))\s*"
    r"(?:\|\s*RATIONALE\s*:?\s*(.+?))?\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def parse_claim_attributions(text: str, claims: list[str], n_chunks: int) -> list[ClaimAttribution]:
    """Parse the LLM's per-claim attribution output into ClaimAttribution objects.

    Returns one ClaimAttribution per claim in ``claims``, in order.
    For each claim:
    - If the LLM output a parseable line with a chunk index in
      ``[1, n_chunks]``, ``supporting_chunk_idx`` is populated.
    - If the LLM said "none" or the chunk index was out of range,
      ``supporting_chunk_idx`` is ``None``.
    - If the LLM omitted the line for this claim entirely (parse
      failure), ``supporting_chunk_idx`` is ``None`` and the
      rationale carries a ``PARSE_FAILED`` marker.

    Returning a typed result for every input claim — rather than
    raising — keeps the eval loop robust. Parse failures show up in
    the per-claim data on the report so a reviewer can spot which
    queries to recheck.
    """
    # Index the LLM's output by claim_idx for direct lookup.
    by_idx: dict[int, tuple[str | None, str | None]] = {}
    for match in _ATTRIBUTION_LINE_RE.finditer(text):
        claim_idx = int(match.group(1))
        chunk_idx_str = match.group(2)
        none_marker = match.group(3)
        rationale = match.group(4)
        rationale_clean = rationale.strip() if rationale else None
        if none_marker is not None:
            by_idx[claim_idx] = (None, rationale_clean)
        elif chunk_idx_str is not None:
            by_idx[claim_idx] = (chunk_idx_str, rationale_clean)

    results: list[ClaimAttribution] = []
    for i, claim_text in enumerate(claims, start=1):
        if i not in by_idx:
            results.append(
                ClaimAttribution(
                    claim_idx=i,
                    claim_text=claim_text,
                    supporting_chunk_idx=None,
                    rationale=f"PARSE_FAILED: no attribution line for claim {i}",
                )
            )
            continue

        chunk_str, rationale = by_idx[i]
        if chunk_str is None:
            # Explicit "none" from the model.
            results.append(
                ClaimAttribution(
                    claim_idx=i,
                    claim_text=claim_text,
                    supporting_chunk_idx=None,
                    rationale=rationale,
                )
            )
            continue

        chunk_idx = int(chunk_str)
        if chunk_idx < 1 or chunk_idx > n_chunks:
            # Out-of-range index — model hallucinated a chunk number
            # that doesn't exist. Treat as unsupported and flag it.
            note = f"OUT_OF_RANGE: chunk {chunk_idx} not in [1, {n_chunks}]"
            results.append(
                ClaimAttribution(
                    claim_idx=i,
                    claim_text=claim_text,
                    supporting_chunk_idx=None,
                    rationale=f"{note}. {rationale}" if rationale else note,
                )
            )
            continue

        results.append(
            ClaimAttribution(
                claim_idx=i,
                claim_text=claim_text,
                supporting_chunk_idx=chunk_idx,
                rationale=rationale,
            )
        )
    return results


# --- Relevance (unchanged absolute-scale approach) --------------------

_RELEVANCE_SYSTEM = (
    "You are an evaluator of LLM-generated answers. Score how RELEVANT "
    "an answer is to the question that was asked. A relevant answer "
    "directly addresses what was asked. Off-topic, evasive, or "
    "tangential answers are not relevant — even if their content is "
    "true.\n\n"
    "Score on this five-point scale:\n"
    "- 1.0: Directly answers the question, on point.\n"
    "- 0.75: Answers the question with some extraneous content.\n"
    "- 0.5: Partially answers; misses key parts of the question.\n"
    "- 0.25: Mostly off-topic; only minor relevance.\n"
    "- 0.0: Does not address the question.\n\n"
    "Respond in EXACTLY this format and nothing else:\n"
    "SCORE: <one of 0.0, 0.25, 0.5, 0.75, 1.0>\n"
    "RATIONALE: <one or two sentences explaining the score>"
)

_RELEVANCE_USER_TEMPLATE = "QUESTION:\n{question}\n\nANSWER:\n{answer}"


def _snap_to_anchor(raw: float) -> float:
    """Snap an arbitrary float in [0, 1] to the nearest 5-anchor value.

    Used for relevance, which still uses the absolute-scale approach.
    The score parser receives whatever the LLM produced — sometimes
    exactly an anchor, sometimes off by a hair (0.7 instead of 0.75),
    occasionally a wildly out-of-band number. We clamp to the [0, 1]
    range first, then pick the closest anchor by absolute distance.
    """
    clamped = max(0.0, min(1.0, raw))
    return min(_RELEVANCE_ANCHORS, key=lambda a: abs(a - clamped))


def parse_relevance_response(text: str, metric: JudgeMetric = "relevance") -> JudgeScore:
    """Extract SCORE and RATIONALE from the relevance judge's response.

    Robust to: missing colon, lowercase keys, leading-dot decimals,
    multi-line rationales. Out-of-range scores (e.g., "SCORE: 5.0")
    fall through to the parse-failure path rather than being clamped
    silently — an LLM that ignored the [0, 1] scale should be flagged
    for review, not have its bad output coerced.
    """
    score_match = _SCORE_RE.search(text)
    if score_match is None:
        return JudgeScore(
            metric=metric,
            score=0.0,
            rationale=f"PARSE_FAILED: no SCORE in response. Raw: {text[:200]!r}",
        )

    raw_score = float(score_match.group(1))
    score = _snap_to_anchor(raw_score)

    rationale_match = _RATIONALE_RE.search(text)
    rationale = (
        rationale_match.group(1).strip()
        if rationale_match is not None
        else "(no RATIONALE produced)"
    )

    return JudgeScore(metric=metric, score=score, rationale=rationale)


def _format_chunks_for_attribution(chunks: list[ChunkRef]) -> str:
    """Render chunks as `[N] <text>` blocks separated by blank lines.

    Same numbered-bracket convention the generator's prompt uses, so
    the judge sees the same chunk identifiers as the model that
    produced the answer. Drift here would make per-chunk attribution
    indices not match the indices used elsewhere on the eval report.
    """
    pieces: list[str] = []
    for i, c in enumerate(chunks, start=1):
        pieces.append(f"[{i}] {c.text}")
    return "\n\n".join(pieces)


def _format_claims_for_attribution(claims: list[str]) -> str:
    """Render claims as a numbered list matching the extraction output."""
    return "\n".join(f"{i}. {claim}" for i, claim in enumerate(claims, start=1))


class LlamaJudge(LLMJudge):
    """LLM-judge backed by Llama 3.1 8B Instruct (NF4 4-bit by default).

    See module docstring for the structural mirror to ``Generator``
    and the methodology choice for faithfulness vs relevance. Tests
    should override ``_run_inference`` directly to avoid model
    loading; production loads the model on first ``judge_*`` call.
    """

    def __init__(self, config: JudgeConfig) -> None:
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

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return self._tokenizer

    @staticmethod
    def _build_4bit_config() -> Any:
        """Same NF4 config as Generator. Duplicated rather than shared
        because two callers isn't enough to justify a helper module yet;
        revisit if a third NF4-loaded model class lands."""
        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # Per-call max_new_tokens overrides. The default ``JudgeConfig.max_new_tokens=256``
    # is sized for the relevance flow (SCORE + 1-2 sentence rationale, typically
    # ~50 tokens). Faithfulness flows have very different output sizes:
    #
    # - Extraction: a numbered list of N claims (avg ~30 tokens/claim).
    #   For a long answer with 15 claims, that's ~450 tokens.
    # - Attribution: a numbered list of N attribution lines
    #   (avg ~75 tokens/line including "CLAIM <i>: chunk <X> | RATIONALE: ..."),
    #   so a 16-claim answer needs ~1200 tokens.
    #
    # The original 256-token cap silently truncated attribution output for
    # answers with > 4 claims, producing PARSE_FAILED on every claim past the
    # cap (curated_001/008/010 in the first PR-B run). Sized 4× larger than
    # the worst observed case so the cap doesn't gate us at the next
    # eval-set growth step.
    _EXTRACTION_MAX_TOKENS = 768
    _ATTRIBUTION_MAX_TOKENS = 2048

    def _run_inference(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        """Run the judge model and return the generated text.

        ``max_new_tokens`` overrides ``self.config.max_new_tokens`` for
        a single call. Used by the faithfulness flow to give extraction
        and attribution more headroom than relevance (see class-level
        constants for the rationale).
        """
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
        inputs = inputs.to(self.model.device)
        input_token_count = int(inputs["input_ids"].shape[1])

        output_ids = cast("Any", self.model).generate(
            **inputs,
            max_new_tokens=max_new_tokens
            if max_new_tokens is not None
            else self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.temperature > 0,
        )
        completion_ids = output_ids[0][input_token_count:]
        return cast("str", self.tokenizer.decode(completion_ids, skip_special_tokens=True))

    def extract_claims(self, question: str, answer: str) -> list[str]:
        """LLM call 1 of faithfulness: decompose the answer into claims.

        Single LLM call. Returns the claims as a list of strings; an
        empty list signals the model emitted ``NO_CLAIMS`` (e.g., the
        answer was a refusal or pure code with no factual statements).
        Public so eval drivers and tests can call it directly without
        going through the full faithfulness flow.
        """
        messages = [
            {"role": "system", "content": _EXTRACT_CLAIMS_SYSTEM},
            {
                "role": "user",
                "content": _EXTRACT_CLAIMS_USER_TEMPLATE.format(question=question, answer=answer),
            },
        ]
        text = self._run_inference(messages, max_new_tokens=self._EXTRACTION_MAX_TOKENS)
        return parse_claims(text)

    def attribute_claims_to_chunks(
        self, claims: list[str], chunks: list[ChunkRef]
    ) -> list[ClaimAttribution]:
        """LLM call 2 of faithfulness: attribute each claim to a chunk.

        Single batched LLM call covering all claims at once — N×K
        verification scaling collapses to a single inference. Returns
        one ClaimAttribution per claim, in input order. Out-of-range
        chunk indices and missing attribution lines are surfaced as
        ``supporting_chunk_idx=None`` with a marker in the rationale.

        Uses a much larger ``max_new_tokens`` than the default config —
        attribution output scales linearly with claim count (~75 tokens
        per attribution line), and the default 256-token cap was
        silently truncating output on long answers. See the class-level
        ``_ATTRIBUTION_MAX_TOKENS`` constant for sizing rationale.

        Empty ``claims`` returns an empty list without invoking the
        LLM (no work to do).
        """
        if not claims:
            return []
        messages = [
            {"role": "system", "content": _ATTRIBUTE_CLAIMS_SYSTEM},
            {
                "role": "user",
                "content": _ATTRIBUTE_CLAIMS_USER_TEMPLATE.format(
                    chunks_block=_format_chunks_for_attribution(chunks),
                    claims_block=_format_claims_for_attribution(claims),
                ),
            },
        ]
        text = self._run_inference(messages, max_new_tokens=self._ATTRIBUTION_MAX_TOKENS)
        return parse_claim_attributions(text, claims=claims, n_chunks=len(chunks))

    def judge_faithfulness(self, question: str, chunks: list[ChunkRef], answer: str) -> JudgeScore:
        """RAGAS-style claim-level faithfulness with per-chunk attribution.

        Two LLM calls: extract atomic claims from the answer, then
        attribute each claim to a specific chunk (or mark unsupported).
        Score is ``n_supported / n_total`` — continuous in ``[0, 1]``,
        no anchor snapping. Per-claim attributions are preserved on
        the JudgeScore so the eval report carries grounded evidence.

        Edge cases:
        - **No claims extracted** (model emitted ``NO_CLAIMS`` or the
          extraction parsed empty): returns ``score=0.0`` with a
          rationale flagging the parse failure. The eval driver
          should ideally not call this on refusal answers — that's
          why no-answer queries skip faithfulness in the driver.
        - **All unsupported**: ``score=0.0``, real signal that the
          model hallucinated.
        - **Mix of supported and unsupported**: ``score`` is the
          supported fraction.
        """
        claims = self.extract_claims(question=question, answer=answer)
        if not claims:
            return JudgeScore(
                metric="faithfulness",
                score=0.0,
                rationale=(
                    "NO_CLAIMS_EXTRACTED: claim-extraction step returned no "
                    "atomic claims. Answer may be a refusal, pure code, or "
                    "the extraction parser may have failed."
                ),
                claim_attributions=[],
            )

        attributions = self.attribute_claims_to_chunks(claims=claims, chunks=chunks)
        n_supported = sum(1 for a in attributions if a.supporting_chunk_idx is not None)
        n_total = len(attributions)
        score = n_supported / n_total if n_total > 0 else 0.0

        rationale = f"{n_supported} of {n_total} claims supported by retrieved chunks."
        return JudgeScore(
            metric="faithfulness",
            score=score,
            rationale=rationale,
            claim_attributions=attributions,
        )

    def judge_relevance(self, question: str, answer: str) -> JudgeScore:
        messages = [
            {"role": "system", "content": _RELEVANCE_SYSTEM},
            {
                "role": "user",
                "content": _RELEVANCE_USER_TEMPLATE.format(question=question, answer=answer),
            },
        ]
        text = self._run_inference(messages)
        return parse_relevance_response(text, "relevance")
