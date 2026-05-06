"""LLM-judge abstraction for generation-quality evals.

Two judge methods, deliberately minimal:

- ``judge_faithfulness(question, context, answer)`` — does the answer
  follow from the supplied context, without inventing facts?
- ``judge_relevance(question, answer)`` — does the answer actually
  address the question that was asked?

Faithfulness needs context (the retrieved chunks the LLM saw).
Relevance does not — an off-topic answer is off-topic regardless of
what was retrieved. This shape mirrors the metrics in eval-methodology.md.

Score is in ``[0.0, 1.0]`` with anchor values at ``0.0, 0.25, 0.5,
0.75, 1.0``. The anchors are repeated in the judge prompt and the
parser snaps the LLM's response to the nearest one — getting Likert's
prompt-stability without losing arithmetic friendliness for averaging
across queries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

JudgeMetric = Literal["faithfulness", "relevance"]


class JudgeScore(BaseModel):
    """One judge-produced score for one metric on one Answer.

    ``score`` is constrained to ``[0.0, 1.0]``. Concrete judges are
    expected to snap their raw output to the five anchor values
    (0.0 / 0.25 / 0.5 / 0.75 / 1.0), but that snapping is the
    judge's job; this model just enforces the range.

    ``rationale`` carries the LLM's brief explanation. Stored on the
    eval report so a human reviewer can spot-check whether the score
    matches the reasoning.
    """

    metric: JudgeMetric
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class LLMJudge(ABC):
    """Abstract LLM-judge interface.

    Concrete implementations (``LlamaJudge``, optionally ``AnthropicJudge``)
    own model loading and prompt construction. Tests use a ``MockJudge``
    that returns canned scores without any model.
    """

    @abstractmethod
    def judge_faithfulness(self, question: str, context: str, answer: str) -> JudgeScore:
        """Score how faithfully ``answer`` follows from ``context``.

        Hallucinated claims, plausible-sounding inventions, and
        assertions that go beyond the context all reduce the score.
        ``context`` should be the same retrieved-chunk text the
        Generator was prompted with — passing different context here
        measures something other than faithfulness-as-served.
        """

    @abstractmethod
    def judge_relevance(self, question: str, answer: str) -> JudgeScore:
        """Score whether ``answer`` addresses the question.

        An off-topic, evasive, or tangential answer is not relevant,
        even if its content is true. No ``context`` is needed —
        relevance is between the question and the answer alone.
        """
