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

**Output parsing (added 2026-05-06 in PR C.2):** every judge prompt
asks for JSON output, and parsing goes through pydantic schemas
(``_ClaimsResponse``, ``_AttributionsResponse``, ``_RelevanceResponse``,
``_RefusalResponse``). The ``_call_with_json_retry`` helper does one
retry on parse failure with a clarifying message. Replaces the prior
regex-based parsing — eliminates the ``curated_001`` 8/8 PARSE_FAILED
edge case that appeared when LLM output drifted from the regex format.
The judge is *our* tool; we own the format; structured output is the
right tool when we own the format.

**Faithfulness** uses RAGAS-style claim-level decomposition with
per-chunk attribution. Two LLM calls per query:

1. ``extract_claims(question, answer)`` — decompose into atomic
   factual claims via ``_ClaimsResponse``.
2. ``attribute_claims_to_chunks(claims, chunks)`` — single batched
   call. For each claim, the judge returns a chunk index or ``null``
   in ``_AttributionsResponse.attributions``.

Aggregate: ``score = n_supported / n_total`` in ``[0, 1]``,
continuous (no anchor snapping). Per-claim attributions are
preserved on the JudgeScore so reports carry grounded evidence.

**Relevance** uses the absolute-scale anchor approach
(0.0 / 0.25 / 0.5 / 0.75 / 1.0). The score field on
``_RelevanceResponse`` is constrained to ``[0, 1]``; the
post-processor snaps to the nearest anchor (so an LLM-emitted 0.7
becomes 0.75 silently rather than triggering retry).

**Refusal** ( ``judge_refusal``) returns a ``RefusalJudgment(refused:
bool, rationale: str)`` from the ``_RefusalResponse`` schema. Used
on off-corpus queries; runs alongside the rule-based regex check
(see ``run_judging_phase``) for cross-validation.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field, ValidationError

from docsense.evaluation.judge import (
    ClaimAttribution,
    JudgeMetric,
    JudgeScore,
    LLMJudge,
    RefusalJudgment,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from docsense.config import JudgeConfig
    from docsense.generation.types import ChunkRef


_RELEVANCE_ANCHORS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


# --- JSON output helpers (added 2026-05-06 for PR C.2) ---------------
#
# Replaces regex-based parsing of LLM output. The judge is *our* tool,
# we own the prompt and the output format — so we should ask for a
# parseable shape (JSON) rather than parse free-form text with regex.
# Eliminates the ``curated_001`` 8/8 PARSE_FAILED edge case that
# motivated this PR (LLM output drifted from our regex format in ways
# the regex couldn't accommodate; JSON has much more forgiving shape
# tolerance — extra whitespace, key casing, fence wrapping all become
# easy to handle).
#
# Each judge method has its own pydantic response schema below; the
# ``_parse_json_response`` helper does the parse + validate; the
# ``_call_with_json_retry`` method on LlamaJudge handles one retry on
# parse failure with a clarifying instruction. Permanent failure
# (after retry) returns None — the calling method handles the
# fallback the same way the prior PARSE_FAILED markers worked.


# Strips ```json ... ``` markdown fences if present, then locates the
# first top-level JSON object. Tolerant of LLM prose before/after the
# JSON (occasional "Here is my response:" preambles).
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_block(text: str) -> str:
    """Return the JSON-looking substring from raw LLM output.

    Tries, in order: a markdown code fence's contents, then the first
    ``{...}`` block in the text, then the original text. Doesn't
    validate JSON syntax — that's ``_parse_json_response``'s job.
    """
    text = text.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match is not None:
        return fence_match.group(1).strip()
    # Find the first balanced-ish JSON object. We use a simple greedy
    # match here — if the LLM emits multiple JSON blocks we take the
    # outermost; if it emits prose around a JSON block we take the
    # JSON. json.loads will reject anything truly malformed.
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match is not None:
        return brace_match.group(0)
    return text


def _parse_json_response[T: BaseModel](text: str, schema: type[T]) -> T | None:
    """Extract + parse + validate ``text`` against ``schema``.

    Returns ``None`` on any failure (json decode error, pydantic
    validation error). The caller decides what to do — typically
    one retry then a fallback to a parse-failure marker.
    """
    block = _extract_json_block(text)
    try:
        raw = json.loads(block)
    except json.JSONDecodeError:
        return None
    try:
        return schema.model_validate(raw)
    except ValidationError:
        return None


# --- Response schemas, one per judge call ----------------------------


class _ClaimsResponse(BaseModel):
    """Schema for ``extract_claims`` output. Empty ``claims`` list is
    the explicit "no factual claims" signal (replaces the prior
    NO_CLAIMS sentinel)."""

    claims: list[str]


class _AttributionEntry(BaseModel):
    """One per-claim attribution as the judge emits it. The eval
    driver post-processes these into ``ClaimAttribution`` objects;
    this schema is just the wire format."""

    claim_idx: int = Field(ge=1)
    supporting_chunk_idx: int | None = None
    rationale: str | None = None


class _AttributionsResponse(BaseModel):
    """Schema for ``attribute_claims_to_chunks`` output."""

    attributions: list[_AttributionEntry]


class _RelevanceResponse(BaseModel):
    """Schema for ``judge_relevance`` output. ``score`` is constrained
    to ``[0, 1]`` here; the snap-to-anchor happens in post-processing
    so 0.7 → 0.75 doesn't trip pydantic validation."""

    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class _RefusalResponse(BaseModel):
    """Schema for ``judge_refusal`` output."""

    refused: bool
    rationale: str


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
    "Respond with ONLY a JSON object matching this schema (no other "
    "text, no markdown fences):\n"
    '{"claims": ["<claim 1>", "<claim 2>", ...]}\n\n'
    "If the answer has no factual claims (e.g., it's a refusal or "
    'purely a code example), respond with: {"claims": []}'
)
_EXTRACT_CLAIMS_USER_TEMPLATE = "QUESTION:\n{question}\n\nANSWER:\n{answer}"


# --- Faithfulness: claim attribution ----------------------------------

_ATTRIBUTE_CLAIMS_SYSTEM = (
    "You are attributing claims to retrieved source chunks. For each "
    "claim, identify which chunk (1-indexed) supports it. A claim is "
    '"supported by chunk N" if the chunk\'s content provides evidence '
    "for the claim via reasonable inference. Direct quotation is not "
    "required — paraphrasing and reasonable inference are fine. If no "
    "chunk supports the claim, set supporting_chunk_idx to null.\n\n"
    "Respond with ONLY a JSON object matching this schema (no other "
    "text, no markdown fences):\n"
    "{\n"
    '  "attributions": [\n'
    '    {"claim_idx": <int, 1-indexed>, '
    '"supporting_chunk_idx": <int 1..K or null>, '
    '"rationale": "<one short sentence>"},\n'
    "    ...\n"
    "  ]\n"
    "}\n\n"
    "Output one entry per claim, in numerical order. claim_idx values "
    "must match the numbered claims you were given."
)

_ATTRIBUTE_CLAIMS_USER_TEMPLATE = "CHUNKS:\n{chunks_block}\n\nCLAIMS:\n{claims_block}"


def _post_process_attributions(
    response: _AttributionsResponse | None,
    claims: list[str],
    n_chunks: int,
) -> list[ClaimAttribution]:
    """Convert a parsed _AttributionsResponse into ClaimAttribution per claim.

    Validates that:
    - Every input claim has an attribution. Missing attributions get
      a ``PARSE_FAILED`` marker on the rationale.
    - ``supporting_chunk_idx`` is in ``[1, n_chunks]``. Out-of-range
      values get an ``OUT_OF_RANGE`` marker (and supporting_chunk_idx
      is set to None — the LLM hallucinated a chunk that doesn't
      exist).
    - Extra attributions for claim_idx values we didn't ask about
      are silently ignored.

    If ``response`` is ``None`` (parse failure after retry), every
    claim gets a ``PARSE_FAILED`` marker — same end-state as the
    prior regex parser.
    """
    if response is None:
        return [
            ClaimAttribution(
                claim_idx=i,
                claim_text=claim_text,
                supporting_chunk_idx=None,
                rationale="PARSE_FAILED: judge response did not produce valid JSON",
            )
            for i, claim_text in enumerate(claims, start=1)
        ]

    by_idx: dict[int, _AttributionEntry] = {a.claim_idx: a for a in response.attributions}
    results: list[ClaimAttribution] = []
    for i, claim_text in enumerate(claims, start=1):
        entry = by_idx.get(i)
        if entry is None:
            results.append(
                ClaimAttribution(
                    claim_idx=i,
                    claim_text=claim_text,
                    supporting_chunk_idx=None,
                    rationale=f"PARSE_FAILED: no attribution for claim {i}",
                )
            )
            continue

        chunk_idx = entry.supporting_chunk_idx
        if chunk_idx is not None and (chunk_idx < 1 or chunk_idx > n_chunks):
            note = f"OUT_OF_RANGE: chunk {chunk_idx} not in [1, {n_chunks}]"
            results.append(
                ClaimAttribution(
                    claim_idx=i,
                    claim_text=claim_text,
                    supporting_chunk_idx=None,
                    rationale=f"{note}. {entry.rationale}" if entry.rationale else note,
                )
            )
            continue

        results.append(
            ClaimAttribution(
                claim_idx=i,
                claim_text=claim_text,
                supporting_chunk_idx=chunk_idx,
                rationale=entry.rationale,
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
    "Respond with ONLY a JSON object matching this schema (no other "
    "text, no markdown fences):\n"
    '{"score": <one of 0.0, 0.25, 0.5, 0.75, 1.0>, '
    '"rationale": "<one or two sentences explaining the score>"}'
)

_RELEVANCE_USER_TEMPLATE = "QUESTION:\n{question}\n\nANSWER:\n{answer}"


# --- Refusal detection (added 2026-05-06) -----------------------------

_REFUSAL_SYSTEM = (
    "You are evaluating whether an answer indicates the model could "
    "NOT answer the question because the necessary information wasn't "
    "in the retrieved context. This is *acknowledged inability*, not "
    "answer quality. We're detecting whether the model recognized the "
    "context didn't have what it needed and said so — not whether the "
    "answer is correct or wrong.\n\n"
    "Important distinctions:\n"
    "- A refusal is acknowledged inability: 'I don't have enough "
    "context.' / 'The provided context doesn't cover this.' / "
    "'I'm unable to answer based on the documentation.'\n"
    "- A *hedged but attempted* answer is NOT a refusal: 'I'm not "
    "sure, but I think it's around 1 GB.' / 'Based on the docs, this "
    "appears to be...' — these attempt an answer with uncertainty, "
    "not refuse.\n"
    "- A wrong or off-topic answer is NOT a refusal — that's a "
    "quality failure, separate from refusal.\n"
    "- A safety/policy refusal ('I can't help with that') is also "
    "NOT what we're measuring; we're specifically about "
    "context-availability refusals.\n\n"
    "Examples:\n"
    "- 'I don't have enough context to answer that.' → REFUSED: yes\n"
    "- 'The context discusses tokenizers, but I don't have specific "
    "information about that topic.' → REFUSED: yes\n"
    "- 'AutoModel.from_pretrained takes a model name string.' → "
    "REFUSED: no\n"
    "- 'I'm not sure, but typically this is around 1 GB.' → "
    "REFUSED: no\n"
    "- 'The answer would require knowledge outside what's provided.' "
    "→ REFUSED: yes\n\n"
    "Respond in EXACTLY this format and nothing else:\n"
    "REFUSED: <yes or no>\n"
    "RATIONALE: <one short sentence explaining your decision>"
)

_REFUSAL_USER_TEMPLATE = "QUESTION:\n{question}\n\nANSWER:\n{answer}"


def _post_process_refusal(response: _RefusalResponse | None) -> RefusalJudgment:
    """Convert a parsed _RefusalResponse into a RefusalJudgment.

    Falls back to ``refused=False`` on parse failure (after retry).
    Same conservative default as the prior regex parser: a parse
    failure shouldn't accidentally inflate the refusal rate by
    classifying a confabulated answer as a refusal. The PARSE_FAILED
    marker on the rationale flags the case for review.
    """
    if response is None:
        return RefusalJudgment(
            refused=False,
            rationale="PARSE_FAILED: judge response did not produce valid JSON",
        )
    return RefusalJudgment(refused=response.refused, rationale=response.rationale)


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


def _post_process_relevance(
    response: _RelevanceResponse | None, metric: JudgeMetric = "relevance"
) -> JudgeScore:
    """Convert a parsed _RelevanceResponse into a JudgeScore.

    Snaps the raw score to the nearest five-anchor value
    (0.0/0.25/0.5/0.75/1.0). Even with JSON output the LLM may emit
    an off-anchor value like 0.7 — that's a "model meant 0.75" case,
    not a parse failure, so we snap silently.

    Falls back to ``score=0.0`` with a ``PARSE_FAILED`` marker if
    ``response`` is ``None`` (parse failure after retry). Same
    end-state as the prior regex parser.
    """
    if response is None:
        return JudgeScore(
            metric=metric,
            score=0.0,
            rationale="PARSE_FAILED: judge response did not produce valid JSON",
        )
    snapped = _snap_to_anchor(response.score)
    return JudgeScore(metric=metric, score=snapped, rationale=response.rationale)


def _post_process_claims(response: _ClaimsResponse | None) -> list[str]:
    """Convert a parsed _ClaimsResponse into a list of claim strings.

    Empty list signals "no factual claims" (e.g., refusal). ``None``
    signals parse failure — propagated as empty list since the caller
    (judge_faithfulness) treats both the same way (skips attribution).
    The score-level rationale on the caller's JudgeScore distinguishes
    "no claims to extract" (NO_CLAIMS_EXTRACTED) from "parse failure"
    (PARSE_FAILED).
    """
    if response is None:
        return []
    return [c.strip() for c in response.claims if c.strip()]


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

    _RETRY_REMINDER = (
        "Your previous response was not valid JSON matching the "
        "requested schema. Please respond with ONLY the JSON object. "
        "Do not include any other text, explanations, or markdown "
        "code fences."
    )

    def _call_with_json_retry[T: BaseModel](
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        max_new_tokens: int | None = None,
    ) -> T | None:
        """Single LLM call + JSON parse + one retry on parse failure.

        Returns the validated pydantic instance, or ``None`` if both
        the initial call and the retry fail to produce parseable
        output. The caller's post-processing layer (`_post_process_*`)
        handles the None case by emitting a `PARSE_FAILED` marker.

        The retry appends the assistant's first response and a
        clarifying user message to the conversation, asking for the
        JSON-only output again. This catches the most common failure
        mode (LLM added a prose preamble or wrapped in fences in a
        way `_extract_json_block` couldn't handle) without spending
        more than one extra LLM call.
        """
        text = self._run_inference(messages, max_new_tokens=max_new_tokens)
        parsed = _parse_json_response(text, schema)
        if parsed is not None:
            return parsed

        retry_messages = [
            *messages,
            {"role": "assistant", "content": text},
            {"role": "user", "content": self._RETRY_REMINDER},
        ]
        retry_text = self._run_inference(retry_messages, max_new_tokens=max_new_tokens)
        return _parse_json_response(retry_text, schema)

    def extract_claims(self, question: str, answer: str) -> list[str]:
        """LLM call 1 of faithfulness: decompose the answer into claims.

        Single LLM call (with one retry on JSON parse failure).
        Returns the claims as a list of strings. An empty list
        signals either the model emitted ``{"claims": []}`` (refusal
        or pure-code answer with no factual content) OR a parse
        failure after retry. ``judge_faithfulness`` distinguishes
        these via the score-level rationale on the resulting
        JudgeScore.
        """
        messages = [
            {"role": "system", "content": _EXTRACT_CLAIMS_SYSTEM},
            {
                "role": "user",
                "content": _EXTRACT_CLAIMS_USER_TEMPLATE.format(question=question, answer=answer),
            },
        ]
        response = self._call_with_json_retry(
            messages, _ClaimsResponse, max_new_tokens=self._EXTRACTION_MAX_TOKENS
        )
        return _post_process_claims(response)

    def attribute_claims_to_chunks(
        self, claims: list[str], chunks: list[ChunkRef]
    ) -> list[ClaimAttribution]:
        """LLM call 2 of faithfulness: attribute each claim to a chunk.

        Single batched LLM call (with one retry on JSON parse
        failure) covering all claims at once — N×K verification
        scaling collapses to a single inference. Returns one
        ClaimAttribution per claim, in input order. Out-of-range
        chunk indices and missing attribution entries are surfaced as
        ``supporting_chunk_idx=None`` with a marker in the rationale
        (``PARSE_FAILED`` for missing entries, ``OUT_OF_RANGE`` for
        chunk indices outside ``[1, n_chunks]``).

        Uses a much larger ``max_new_tokens`` than the default config —
        attribution output scales linearly with claim count
        (~75 tokens per JSON entry), and the default 256-token cap
        was silently truncating output on long answers. See the
        class-level ``_ATTRIBUTION_MAX_TOKENS`` constant for sizing.

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
        response = self._call_with_json_retry(
            messages, _AttributionsResponse, max_new_tokens=self._ATTRIBUTION_MAX_TOKENS
        )
        return _post_process_attributions(response, claims=claims, n_chunks=len(chunks))

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
        """Score relevance on the 5-anchor scale via JSON output."""
        messages = [
            {"role": "system", "content": _RELEVANCE_SYSTEM},
            {
                "role": "user",
                "content": _RELEVANCE_USER_TEMPLATE.format(question=question, answer=answer),
            },
        ]
        response = self._call_with_json_retry(messages, _RelevanceResponse)
        return _post_process_relevance(response, "relevance")

    def judge_refusal(self, question: str, answer: str) -> RefusalJudgment:
        """Decide whether ``answer`` indicates context-unavailability refusal.

        Single LLM call (with one retry on JSON parse failure). Output
        format is a JSON object ``{"refused": bool, "rationale": str}``
        — small response, default JudgeConfig.max_new_tokens (256)
        is plenty.

        Replaces the regex-based ``check_no_answer_behavior`` heuristic
        as the primary measurement. The rule-based check is still run
        alongside in the eval driver as a guardrail and cross-validation
        signal — see ``run_judging_phase`` in
        ``scripts/run_generation_eval.py``.
        """
        messages = [
            {"role": "system", "content": _REFUSAL_SYSTEM},
            {
                "role": "user",
                "content": _REFUSAL_USER_TEMPLATE.format(question=question, answer=answer),
            },
        ]
        response = self._call_with_json_retry(messages, _RefusalResponse)
        return _post_process_refusal(response)
