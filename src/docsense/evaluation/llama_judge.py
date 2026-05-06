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
- ``judge_faithfulness`` / ``judge_relevance`` build the prompt,
  invoke ``_run_inference``, and parse the response into a
  ``JudgeScore``.

The five-anchor scale (0.0 / 0.25 / 0.5 / 0.75 / 1.0) is repeated
verbatim in both prompts. The parser snaps whatever number the model
emits to the nearest anchor — so even if the LLM produces 0.7 the
score becomes 0.75 and stays in the [0.0, 1.0] anchor grid.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

from docsense.evaluation.judge import JudgeMetric, JudgeScore, LLMJudge

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from docsense.config import JudgeConfig


_ANCHORS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

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


_FAITHFULNESS_SYSTEM = (
    "You are an evaluator of LLM-generated answers. Score how FAITHFUL "
    "an answer is to its source context. An answer is faithful when "
    "every factual claim in it is directly supported by the provided "
    "context. Hallucinated facts, plausible-sounding inventions, and "
    "claims that go beyond the context all reduce faithfulness.\n\n"
    "Score on this five-point scale:\n"
    "- 1.0: Every claim is directly supported by the context.\n"
    "- 0.75: Most claims supported; minor reasonable extrapolations.\n"
    "- 0.5: About half the claims supported; some unsupported.\n"
    "- 0.25: Few claims supported; significant hallucination.\n"
    "- 0.0: Answer is largely fabricated relative to the context.\n\n"
    "Respond in EXACTLY this format and nothing else:\n"
    "SCORE: <one of 0.0, 0.25, 0.5, 0.75, 1.0>\n"
    "RATIONALE: <one or two sentences explaining the score>"
)

_FAITHFULNESS_USER_TEMPLATE = "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"

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
    """Snap an arbitrary float in [0, 1] to the nearest anchor value.

    The score parser receives whatever the LLM produced — sometimes
    exactly an anchor, sometimes off by a hair (0.7 instead of 0.75),
    occasionally a wildly out-of-band number. We clamp to the [0, 1]
    range first so the JudgeScore validator never trips on the snap
    output, then pick the closest anchor by absolute distance.
    """
    clamped = max(0.0, min(1.0, raw))
    return min(_ANCHORS, key=lambda a: abs(a - clamped))


def parse_judge_response(text: str, metric: JudgeMetric) -> JudgeScore:
    """Extract SCORE and RATIONALE from a judge response into a JudgeScore.

    On a clean response, returns the snapped anchor + rationale. On a
    response missing SCORE entirely, returns ``score=0.0`` with the
    rationale field flagged as a parse failure (with a truncated dump
    of the raw text). Returning a typed JudgeScore on every input —
    rather than raising — keeps the eval loop robust to occasional
    judge misbehavior; the report includes the parse-failure marker so
    a reviewer can spot which queries to recheck.
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


class LlamaJudge(LLMJudge):
    """LLM-judge backed by Llama 3.1 8B Instruct (NF4 4-bit by default).

    See module docstring for the structural mirror to ``Generator``.
    Tests should override ``_run_inference`` directly to avoid model
    loading; production use loads the model on first ``judge_*`` call.
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

    def _run_inference(self, messages: list[dict[str, str]]) -> str:
        """Run the judge model and return the generated text.

        Mirrors ``Generator._run_inference`` but returns text only —
        the judge doesn't need latency/token metadata; that lives on
        the eval report alongside the JudgeScore, not inside it.
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
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.temperature > 0,
        )
        completion_ids = output_ids[0][input_token_count:]
        return cast("str", self.tokenizer.decode(completion_ids, skip_special_tokens=True))

    def judge_faithfulness(self, question: str, context: str, answer: str) -> JudgeScore:
        messages = [
            {"role": "system", "content": _FAITHFULNESS_SYSTEM},
            {
                "role": "user",
                "content": _FAITHFULNESS_USER_TEMPLATE.format(
                    question=question, context=context, answer=answer
                ),
            },
        ]
        text = self._run_inference(messages)
        return parse_judge_response(text, "faithfulness")

    def judge_relevance(self, question: str, answer: str) -> JudgeScore:
        messages = [
            {"role": "system", "content": _RELEVANCE_SYSTEM},
            {
                "role": "user",
                "content": _RELEVANCE_USER_TEMPLATE.format(question=question, answer=answer),
            },
        ]
        text = self._run_inference(messages)
        return parse_judge_response(text, "relevance")
