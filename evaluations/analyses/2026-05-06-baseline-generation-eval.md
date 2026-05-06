# Pre-Phase-3 baseline generation eval

**Date:** 2026-05-06
**Reports analyzed:**
- `evaluations/reports/generation-20260506T060710Z-curated.json` (n=20)
- `evaluations/reports/generation-20260506T060710Z-structural.json` (n=30)
- `evaluations/reports/generation-20260506T064240Z-no-answer.json` (n=8)
- baseline: `evaluations/baselines/pre_phase3_generation_base.json`

## TL;DR

The first end-to-end LLM-judged eval of docsense's generation pipeline
produces a coherent picture of what Qwen 2.5 7B Instruct does on the
production retrieval stack, **and surfaces three findings that any
serious Phase 3 plan needs to incorporate.**

1. **The Llama 3.1 8B judge clusters faithfulness at 0.75** — 49 of
   50 in-corpus answers scored *exactly* 0.75. This is a real
   LLM-as-judge anchor bias, not a small-sample artifact (curated
   and structural agree to within 0.008). Phase 3 needs a
   calibration check before treating absolute faithfulness numbers
   as actionable signal; Δ-faithfulness vs this baseline is still
   meaningful, absolute thresholds are not.
2. **Qwen produces citation markers on ~54% of in-corpus answers**,
   not 0% (the smoke run) and not 100% (what the prompt directive
   asks for). Strong cross-set agreement (55% curated, 53%
   structural) makes this a stable behavioral measurement. Citation
   rate is a clear Phase 3 fine-tuning target.
3. **Refusal on off-corpus is 100%** — but the rule-based heuristic
   originally read 38% because Qwen's preferred refusal phrasing
   ("I don't have enough context to answer that") didn't match any
   pattern. The story here is the *iteration*: eval data exposed a
   coverage gap, the patch was one new regex, and re-running closed
   it out. Rule-based evals need to be written against real model
   outputs, not hypothesized ones.

Plus two methodological observations:

- **Cross-eval-set agreement is the most reliable signal** in the
  data. Where curated and structural agree (citation rate 54-55%,
  faithfulness ~0.75), we have a stable property. Where they
  disagree (relevance: curated 0.762 / structural 0.733), the
  difference is itself a finding.
- **Sequential LLM loading on a 12 GB shared GPU works cleanly** —
  9 MB residual VRAM after both generator and judge unloads. The
  Phase B → Phase C handoff in `run_generation_eval.py` is the
  right architectural choice.

## Methodology

### Configuration

Production retrieval stack: hybrid (BM25 + dense FAISS) + RRF
fusion + cross-encoder rerank, recursive chunking, top_k=5,
max_context_tokens=3500. This is the stack `evaluations/performance.md`
declares as the production default — running the eval against any
other configuration would measure something we don't actually serve.

| Component | Setting |
|---|---|
| Generator model | `Qwen/Qwen2.5-7B-Instruct` |
| Generator quantization | NF4 (bnb_4bit_quant_type=nf4, double_quant=true) |
| Generator temperature | 0.1 |
| Judge model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Judge quantization | NF4 |
| Judge temperature | 0.0 (deterministic — eval reproducibility > variation) |
| Chunking strategy | recursive |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, batch_size=16 |

Judge prompts use a five-anchor scale (0.0 / 0.25 / 0.5 / 0.75 / 1.0)
with explicit anchor descriptions. Parser snaps raw LLM output to the
nearest anchor — gets Likert prompt-stability without losing
arithmetic friendliness for averaging. Parse failures are tracked
separately (zero across all 100 judge calls in this run).

### What got measured per query

| Metric | Type | Source |
|---|---|---|
| `faithfulness.score` | LLM judge (Llama) | `judge_faithfulness(question, context, answer)` — only for in-corpus |
| `relevance.score` | LLM judge (Llama) | `judge_relevance(question, answer)` — only for in-corpus |
| `citation_check.n_markers_in_text` | rule-based | regex over `answer.text` |
| `citation_check.all_markers_in_range` | rule-based | every `[N]` resolves to a chunk |
| `no_answer_check.refused` | rule-based | refusal-phrase regex set — only for off-corpus |
| `timing.{retrieve,assemble,generate}_ms` | wall-clock | per-stage timer in eval driver |

Faithfulness and relevance are skipped on no-answer queries —
"the answer is 'I don't know'" doesn't have a meaningful
faithfulness score against context that's deliberately irrelevant,
and the relevance prompt assumes a well-formed in-corpus question.
Citation grounding *is* run on no-answer answers (we'd want to know
if the model was hallucinating citations even while refusing —
spoiler: it isn't).

## Findings

### 1. Llama judge anchor saturation at 0.75

The most striking pattern in the data:

| Eval set | n | mean | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| curated, faithfulness | 20 | 0.750 | 0 | 0 | 0 | **20** | 0 |
| structural, faithfulness | 30 | 0.758 | 0 | 0 | 0 | **29** | 1 |
| **total in-corpus, faithfulness** | **50** | **0.754** | 0 | 0 | 0 | **49** | 1 |
| curated, relevance | 20 | 0.762 | 0 | 0 | 0 | 19 | 1 |
| structural, relevance | 30 | 0.733 | 0 | 3 | 0 | 23 | 4 |

For faithfulness, 49 of 50 answers got exactly 0.75. For relevance,
42 of 50 got exactly 0.75. The Llama 3.1 8B judge is anchoring
hard at the "mostly-supported, minor extrapolation" level —
regardless of whether the answer is actually a near-perfect
quote of the context or a partial extrapolation.

This is a known LLM-as-judge failure mode. The judge:
- Doesn't want to give 1.0 (looks ungrounded — there's *always*
  something that could be argued as extrapolation)
- Doesn't want to give 0.5 or below (would imply substantial
  hallucination, which usually isn't there for in-corpus questions
  with topical retrieval)
- 0.75 is the safe-but-positive default

**Implication for Phase 3:** the absolute faithfulness number is
**not interpretable** as "Qwen is 75% faithful." Δ-faithfulness
vs this baseline (e.g., a fine-tune that moves mean from 0.750 to
0.825) IS still a meaningful signal because the judge bias should
be approximately constant — but moving the mean off 0.75 at all
will require either (a) the model genuinely improving enough that
the judge feels comfortable giving 1.0, or (b) introducing a
calibration check.

Calibration options if this becomes load-bearing in Phase 3:
- **Anthropic-judge calibration mode** (already scoped as deferred
  follow-up in `docs/phase-2-4-scope.md`). Score the same answers
  with both Llama and Claude, compute correlation, validate the
  local-judge signal is shifted-but-not-uncorrelated.
- **Pairwise comparison instead of absolute scoring** — ask the
  judge "is answer A more faithful than answer B?" rather than
  "score answer A from 0 to 1". Less prone to anchor bias.
- **Different judge prompt anchors** — e.g., 5-point Likert with
  worded anchors at every level. May spread the distribution.

### 2. Relevance shows healthier spread, especially on structural

Where faithfulness collapsed to 0.75, relevance has more variation
on the structural set: 3 queries at 0.25, 23 at 0.75, 4 at 1.0.
Looking at the per-query data, the 3 low-relevance cases are all
in the same family:

```
[0.25] structural_004: "Usage example"
       → answer acknowledged the gap but produced unrelated content
[0.25] structural_005: "Usage examples"
       → similar — listed examples but not of the right kind
[0.25] structural_030: "Usage example"
       → same template
```

These are genuinely poor query-answer pairs: short, ambiguous
queries derived from generic doc headings ("Usage example") match
many sections, retrieval pulled chunks that *had* example-shaped
content but not the example the implicit user wanted, and Qwen
produced something that addressed the literal phrase without
addressing the question's intent.

This is a useful finding by itself — the structural eval set has
known noise from generic h2 headings (documented in the bakeoff
investigation), and these three queries demonstrate it
quantitatively. The four 1.0-relevance cases are all specific
queries where retrieval had a clear winner:

```
[1.00] structural_006: "Albert specific outputs"
[1.00] structural_008: "Load IMDb dataset"
[1.00] structural_020: "Initialising VisionEncoderDecoderModel..."
[1.00] structural_024: "Register a pipeline"
```

**Implication for Phase 3:** the 0.25-relevance triad is a real
pre-fine-tune target. Whether fine-tuning helps depends on whether
the failure is generation-side (the model could have refused or
clarified but didn't) or retrieval-side (the right chunks weren't
retrieved). Per-query inspection in Phase 3 will need to disentangle.

### 3. Citation rate: 54% with strong cross-set agreement

| Eval set | frac_with_any_marker | frac_all_in_range | mean_n_markers/answer |
|---|---:|---:|---:|
| curated (n=20) | 0.55 | 0.55 | 1.85 |
| structural (n=30) | 0.533 | 0.533 | 1.90 |

Two things stand out:

1. The two eval sets agree closely. **~54% citation rate is the
   stable behavioral measurement**, not noise. The smoke-run's 0%
   was on a single specific query.
2. `frac_with_any_marker == frac_all_in_range` on both sets —
   meaning when Qwen does cite, every marker resolves to a real
   chunk. There are no out-of-range hallucinated citations like
   `[7]` when only 5 chunks were retrieved. This is good news;
   the citation behavior is binary "cite correctly or don't cite."

54% is the gap. The system prompt says "Cite sources using [N]
notation that matches the numbered context items below" and Qwen
honors it on roughly half of queries. Looking at the per-query
data, citation behavior doesn't obviously correlate with question
type — some "How do I..." queries cite, some don't, similarly
for terminology queries.

**Implication for Phase 3:** citation rate is the most actionable
target the data surfaces. A fine-tune dataset of ~1000-2000 pairs
where Qwen *should* have cited but didn't, with target answers
that include `[N]` markers, would directly address it. This is
exactly the kind of behavioral gap fine-tuning is good at closing.

### 4. The refusal-pattern coverage gap (and how the eval caught it)

Worth narrating because it's the cleanest illustration of how
rule-based evals iterate.

**First measurement (from the initial full run):**
```
no-answer:    frac_correct=0.38  frac_refused=0.38
```

This looked alarming — only 3/8 off-corpus queries got a refusal?
Per-query inspection revealed all 8 actually refused:

```
[CONFABULATED] no_answer_001: "I don't have enough context to answer that...."
[CONFABULATED] no_answer_002: "I don't have enough context to answer that. The provided context focuses on..."
[CONFABULATED] no_answer_003: "I don't have enough context to answer that...."
[REFUSED     ] no_answer_004: "I don't have enough context to answer that. The provided context does not contain..."
[REFUSED     ] no_answer_005: "I don't have enough context to answer that. The provided context does not contain..."
[CONFABULATED] no_answer_006: "I don't have enough context to answer that...."
[CONFABULATED] no_answer_007: "I don't have enough context to answer that. The provided context discusses..."
[REFUSED     ] no_answer_008: "I don't have enough context to answer that. The provided context does not contain..."
```

Qwen consistently produced "I don't have enough context to answer
that" as its refusal phrase. The 3 marked "REFUSED" only differed
from the 5 "CONFABULATED" cases by having a *follow-up sentence*
mentioning "The provided context does not contain..." — which the
existing `context_doesnt` regex caught.

**Root cause:** the `cannot_action` pattern's verb list was
(`know|determine|tell|say|answer|provide|find|help|locate`). It
didn't include `have` because "I don't have time" / "I don't have
a pen" aren't refusals on their own. But "I don't have *enough
context*" unambiguously is.

**Fix (commit `87da219`):** new pattern `dont_have_context`,
specifically requiring a context/information noun (with optional
quantifier) after `don't have`. Three new test cases parametrized
in `test_rule_based.py`.

**Re-run:**
```
no-answer:    frac_correct=1.00  frac_refused=1.00
all 8 caught by:  pattern=dont_have_context
```

Qwen's actual refusal rate is 100% on this n=8 set. The model is
well-behaved on off-corpus questions. The lesson is meta:
rule-based evals are only as good as the patterns you write, and
**you can't write patterns for phrases you haven't seen**. The
ideal workflow is exactly what happened here — run an eval, look
at the per-query outputs, find phrases the heuristic missed, add
patterns, re-run. Anyone reading this analysis later who's
extending the no-answer eval should expect to do another iteration
when they switch to a different generator model — Llama 3.1 might
refuse with a different phrasing.

### 5. Latency

End-to-end per-query, measured across the eval (n=58):

| Eval set | retrieve p50 | generate p50 | generate p90 | generate max |
|---|---:|---:|---:|---:|
| curated | 315 ms | 15,125 ms | 24,311 ms | 26,566 ms |
| structural | 322 ms | 9,733 ms | 18,822 ms | 33,779 ms |
| no-answer | 393 ms | 2,438 ms | 3,994 ms | 4,286 ms |

Three observations:

- **Generation dominates** wall time. Retrieve+rerank is sub-second
  steady-state; the slow part is the LLM forward pass at ~14 tok/s.
- **Curated p50 > structural p50** because curated questions
  ("How do I install transformers?") elicit longer expository
  answers; structural queries ("Albert specific outputs") elicit
  shorter ones. Some curated answers hit max_new_tokens=512.
- **Refusals are short.** ~2.4 s p50 on no-answer means most
  refusal answers are < 50 tokens. Confirms the model isn't
  burning compute generating long fake answers when it can't
  answer.

The 10.6-second `retrieve_max` on the curated set's first query
is a one-time cross-encoder warm-up — the cross-encoder loads
lazily on first `predict()`. After that, retrieval is consistently
sub-second. Worth noting; not worth fixing for an eval driver.

## Implications for Phase 3

In rough priority order:

1. **Citation rate is the highest-leverage fine-tuning target.** The
   gap is large (54% vs 100% directive-stated), the failure mode
   is binary (cite-correctly-or-not, no out-of-range hallucinations
   to clean up), and the training data is easy to construct
   (existing curated answers + `[N]` markers). Δ-citation-rate
   is a direct, interpretable Phase 3 evaluation metric.

2. **Faithfulness measurement needs calibration before Phase 3
   uses it as a primary signal.** Either AnthropicJudge calibration
   (already scoped as a follow-up) or pairwise comparison. Without
   that, "fine-tuned model has higher faithfulness" claims will be
   suspect because the judge's anchor bias dominates.

3. **Refusal behavior is already strong** (100% on the n=8 set).
   Phase 3 should preserve this rather than improve on it — the
   training data needs to include refusal exemplars or the
   fine-tune may regress on what's currently working.

4. **Relevance has clearer signal than faithfulness** because the
   spread is healthier. It's a useful secondary signal for Phase 3,
   especially on the structural eval set where the failure modes
   (the "Usage example" triad) are concrete.

5. **Latency is fine for an offline eval** but matters for serving.
   p90 of 24 s on curated would be a poor user experience.
   Phase 4's tracing work will need to surface this.

## Methodology caveats

- **n=58 in-corpus is modest.** The cross-set agreement on citation
  rate and faithfulness is encouraging — both eval sets see the
  same patterns — but absolute numbers should be cited with
  ±0.05 plausible noise. The deferred 5c LLM-generated eval set
  would give a third independent distribution.
- **n=8 off-corpus is too small for a stable refusal-rate estimate.**
  100% on 8 queries means somewhere between ~70% and 100% true
  refusal rate at typical confidence. Worth growing this set when
  Phase 3 lands — the refusal exemplars are also training data
  for the fine-tune.
- **Single judge.** All faithfulness and relevance scores come
  from one Llama 3.1 8B Instruct instance at temperature=0.
  Determinism is good for reproducibility but masks variance from
  judge instability. A multi-judge ensemble or a higher-temperature
  variance check would be reasonable additions.
- **No statistical tests on Δ vs baseline.** When Phase 3 reports
  faithfulness 0.83 (or whatever) vs 0.75 baseline, we'll need to
  decide whether to bootstrap-test that delta. Hand-wavy "moved
  by 0.08" is fine for Block 1B but won't be fine for a fine-tune
  claim.
- **No cost / token-billing tracking.** Each eval run takes ~30-45
  minutes on the user's RTX 4070. As query counts grow, this
  budget should be tracked.

## What this changes about the project's narrative

Pre-Phase-3 closeout: docsense's generation pipeline is
**measurably end-to-end functional**. Real model, real corpus,
real judge, real query distributions, zero parse failures across
100 judge calls. The findings are concrete enough to design
Phase 3 around (citation-rate fine-tune target, judge calibration
follow-up, refusal-preservation constraint) rather than
hypothetical.

The honest framing for what Phase 3 will measure:

> Phase 1 measured retrieval (recursive chunking wins MRR on
> curated, then exposed eval-set bias). Phase 2 wired in the
> reranker and adopted hybrid+rerank as production default.
> Pre-Phase-3 measured generation, found that Qwen 2.5 7B Instruct
> answers in-corpus questions correctly but cites only 54% of the
> time, refuses off-corpus questions reliably, and that the local
> Llama judge anchors at 0.75 in a way that limits absolute-faithfulness
> claims. Phase 3 fine-tunes the citation gap and validates the
> faithfulness signal against a cross-judge calibration.
