"""LLM-judge abstraction for generation-quality evals.

Two judge methods, deliberately minimal:

- ``judge_faithfulness(question, context, answer)`` — does the answer
  follow from the supplied context, without inventing facts?
- ``judge_relevance(question, answer)`` — does the answer actually
  address the question that was asked?

Faithfulness needs context (the retrieved chunks the LLM saw).
Relevance does not — an off-topic answer is off-topic regardless of
what was retrieved. This shape mirrors the metrics in eval-methodology.md.

**Faithfulness scoring approach (as of 2026-05-06):** RAGAS-style
claim-level decomposition. The judge first extracts atomic factual
claims from the answer, then attributes each claim to a specific
retrieved chunk (or marks it unsupported). The score is
``n_supported_claims / n_total_claims``, computed continuously in
``[0.0, 1.0]``. This replaced an earlier 5-anchor absolute-scale
approach that exhibited LLM-as-judge anchor saturation at 0.75
(49 of 50 in-corpus answers scored exactly 0.75 — see the
2026-05-06 baseline analysis). Per-chunk attribution makes the
score grounded in concrete evidence and produces actionable
debugging data.

Relevance still uses the absolute-scale anchor approach
(0.0 / 0.25 / 0.5 / 0.75 / 1.0). It showed less saturation in the
baseline and a per-claim decomposition for "does the answer address
the question?" is awkward (the answer is one thing). We can revisit
if Phase 3 measurements suggest relevance also needs decomposition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field, model_validator

JudgeMetric = Literal["faithfulness", "relevance"]


class ClaimAttribution(BaseModel):
    """One atomic claim from an answer, attributed to a chunk (or unsupported).

    Produced by ``LlamaJudge``'s claim-level faithfulness flow:
    ``extract_claims`` decomposes the answer into atomic claims,
    ``attribute_claims_to_chunks`` decides which retrieved chunk
    (1-indexed) supports each claim, or returns ``None`` if no chunk
    does. The per-claim breakdown is preserved on the JudgeScore so
    the eval report carries grounded evidence for the score —
    "answer was scored 0.6 because 3 of 5 claims were attributed to
    chunks 1, 2, and 4; the other 2 had no supporting chunk."

    Stored on the eval report alongside the aggregate score so a
    Phase 3 fine-tune analyst can inspect *which specific claims*
    the model fails to ground, not just an aggregate number.
    """

    claim_idx: int = Field(ge=1, description="1-indexed claim number within this answer.")
    claim_text: str
    supporting_chunk_idx: int | None = Field(
        default=None,
        description=(
            "1-indexed chunk number that supports this claim, or None "
            "if the claim is not supported by any retrieved chunk."
        ),
    )
    rationale: str | None = Field(
        default=None,
        description=(
            "Brief judge-produced explanation. Optional — judges may not "
            "always emit one, and a missing rationale shouldn't crash the "
            "eval pipeline."
        ),
    )


class JudgeScore(BaseModel):
    """One judge-produced score for one metric on one Answer.

    ``score`` is constrained to ``[0.0, 1.0]``.

    For **faithfulness**, score is computed as
    ``n_supported_claims / n_total_claims`` from the
    ``claim_attributions`` field — a continuous value in ``[0, 1]``,
    not snapped to anchors. ``rationale`` carries a brief summary
    ("4 of 5 claims supported; claim 3 about cache location is
    ungrounded"); the per-claim detail lives in ``claim_attributions``.

    For **relevance**, score uses the 5-anchor scale
    (0.0 / 0.25 / 0.5 / 0.75 / 1.0); ``claim_attributions`` is empty.
    The concrete judge is expected to snap to anchors itself.

    The optional ``claim_attributions`` field is the cleanest way to
    extend JudgeScore for claim-level metrics without forking the
    type — relevance and faithfulness still flow through the same
    interface, downstream code (eval report, aggregation) just
    inspects the field when present.
    """

    metric: JudgeMetric
    score: float = Field(ge=0.0, le=1.0)
    rationale: str
    claim_attributions: list[ClaimAttribution] = Field(
        default_factory=list,
        description=(
            "Per-claim attribution from the faithfulness flow; empty for "
            "relevance and for any judge that doesn't decompose claims."
        ),
    )

    @model_validator(mode="after")
    def _validate_claim_indices_unique(self) -> JudgeScore:
        """If claim_attributions are present, the claim_idx values must
        be unique (1, 2, 3, ... — no duplicates, no gaps allowed since
        the parser produces dense 1-indexed sequences). Catches a
        parser bug that would silently double-count a claim in the
        score aggregation."""
        if not self.claim_attributions:
            return self
        idxs = [c.claim_idx for c in self.claim_attributions]
        if len(idxs) != len(set(idxs)):
            msg = (
                f"Duplicate claim_idx values in claim_attributions: {idxs}. "
                "Each claim must have a unique 1-indexed position."
            )
            raise ValueError(msg)
        return self


class LLMJudge(ABC):
    """Abstract LLM-judge interface.

    Concrete implementations (``LlamaJudge``, optionally ``AnthropicJudge``)
    own model loading and prompt construction. Tests use a ``MockJudge``
    that returns canned scores without any model.
    """

    @abstractmethod
    def judge_faithfulness(self, question: str, context: str, answer: str) -> JudgeScore:
        """Score how faithfully ``answer`` follows from ``context``.

        Concrete implementations should use claim-level decomposition
        (extract atomic claims, attribute each to a specific chunk or
        mark unsupported) and return a JudgeScore with
        ``claim_attributions`` populated. The aggregate ``score`` is
        ``n_supported / n_total`` in ``[0, 1]``.

        ``context`` should be the formatted retrieved-chunk text the
        Generator was prompted with. ``LlamaJudge`` parses the chunk
        boundaries from this string to build the per-claim attribution
        prompt — passing different context here measures something
        other than faithfulness-as-served.
        """

    @abstractmethod
    def judge_relevance(self, question: str, answer: str) -> JudgeScore:
        """Score whether ``answer`` addresses the question.

        An off-topic, evasive, or tangential answer is not relevant,
        even if its content is true. Returns a JudgeScore on the
        5-anchor scale (claim_attributions empty for relevance —
        the question/answer pair is a single judgment, not a
        decomposition). No ``context`` is needed.
        """
