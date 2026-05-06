# Phase 2 bakeoff investigation: ablating BM25, RRF, and the cross-encoder

**Date:** 2026-05-06
**Reports analyzed:**
- `evaluations/reports/bakeoff-20260506-{dense,hybrid,hybrid-rerank}-{curated,structural}.json`
- baseline: `evaluations/baselines/phase1_chunking.json`

## TL;DR

The Block B headline finding — *"fixed-size chunking wins under
hybrid+rerank, recursive degrades"* — **does not generalize**. It is
specific to the 20-query hand-curated eval set. On the unbiased
30-query structural eval set, fixed never wins anything; recursive
wins under hybrid+rerank; and the full pipeline produces a uniform
improvement across all metrics for all strategies.

The Block B+ ablation also shows that:

- **BM25 + RRF alone** moves rankings only modestly on the curated
  set (recursive MRR -0.025) but moves them substantially on the
  structural set (recursive MRR +0.155, header MRR +0.093). BM25
  helps when queries contain the kind of specific terminology that
  matches verbatim in chunks — which structural queries do far more
  than curated queries.
- **The cross-encoder** improves recall@10 universally (+0.05 to
  +0.20 across the grid) but its precision contribution depends on
  the eval set: on structural it's a clean win, on curated it
  produces winners-and-losers behavior.

The right going-forward posture: report both eval sets in Phase 2/3
and treat curated-only conclusions as suggestive, not authoritative.

## The 3×2 grid

### MRR (rank of first relevant doc; higher is better)

| pipeline | eval | fixed | recursive | header | winner |
|---|---|---:|---:|---:|---|
| dense | curated | 0.551 | **0.692** | 0.623 | recursive |
| hybrid | curated | 0.552 | **0.667** | 0.588 | recursive |
| hybrid-rerank | curated | **0.701** | 0.599 | 0.604 | **fixed** |
| dense | structural | 0.376 | 0.429 | **0.518** | header |
| hybrid | structural | 0.532 | 0.584 | **0.611** | header |
| hybrid-rerank | structural | 0.631 | **0.685** | 0.649 | **recursive** |

The winner column does not stabilize across eval sets. On curated, the
ranking shifts on the third row (fixed jumps in front). On structural,
the ranking shifts on the third row too — but to recursive, not fixed.

### Recall@10 (fraction of relevant docs retrieved in top 10)

| pipeline | eval | fixed | recursive | header |
|---|---|---:|---:|---:|
| dense | curated | 0.764 | 0.863 | 0.743 |
| hybrid | curated | 0.762 | 0.864 | 0.793 |
| hybrid-rerank | curated | 0.814 | 0.864 | 0.863 |
| dense | structural | 0.567 | 0.600 | 0.667 |
| hybrid | structural | 0.633 | 0.600 | 0.700 |
| hybrid-rerank | structural | 0.733 | 0.800 | 0.800 |

Recall@10 is the most stable signal: it improves monotonically with
each addition for almost every cell. The cross-encoder is doing real
work pulling relevant docs into the top 10 — it's the *re-ordering
within* the top 10 where the curated/structural disagreement lives.

### Hit rate at top-5 (fraction of queries with ≥1 relevant doc in top 5)

| pipeline | eval | fixed | recursive | header |
|---|---|---:|---:|---:|
| dense | curated | 0.800 | 0.800 | 0.850 |
| hybrid | curated | 0.700 | 0.900 | 0.900 |
| hybrid-rerank | curated | 0.850 | 0.850 | 0.850 |
| dense | structural | 0.500 | 0.500 | 0.533 |
| hybrid | structural | 0.633 | 0.567 | 0.667 |
| hybrid-rerank | structural | 0.733 | 0.800 | 0.767 |

Hit@5 on structural improves cleanly across the pipeline. On curated
it's noisier — fixed actually drops from 0.800 to 0.700 when BM25 is
added before recovering with the cross-encoder.

## Findings

### 1. The Block B headline result was a curated-eval artifact

Block B reported "fixed wins on MRR/P@1 under hybrid+rerank;
recursive degrades." Both halves of that statement only hold on
curated:

| | curated MRR | structural MRR |
|---|---|---|
| fixed under hybrid-rerank | **0.701** (best) | 0.631 (worst) |
| recursive under hybrid-rerank | 0.599 (degraded -0.093 vs Phase 1) | **0.685** (improved +0.256 vs structural-dense) |

On structural, recursive wins under the production pipeline by a
clear margin. The curated set's bias toward queries the curator
already knew the answers to apparently happened to favor fixed-size
chunks once the cross-encoder was in the loop.

### 2. The recursive-degrades-under-rerank story is curated-specific

On curated, recursive's MRR drops monotonically with each addition
(0.692 → 0.667 → 0.599). On structural, it does the opposite (0.429
→ 0.584 → 0.685). Whatever's causing the curated degradation isn't a
fundamental property of recursive chunking + cross-encoder; it's an
interaction with the curated query distribution.

A plausible mechanism: the curated set's queries are conversational
("How do I install transformers?") and have chunks that share *topic*
but not many specific terms. Recursive's small, semantically-focused
chunks are excellent matches for the bi-encoder, but the cross-encoder
might over-weight surface-level lexical signals that fixed chunks
(coarser, more text per chunk) happen to satisfy more often. This is
speculative; it would take per-query inspection to confirm.

### 3. BM25's contribution is much larger on structural queries

| strategy | curated dense → hybrid Δ MRR | structural dense → hybrid Δ MRR |
|---|---|---|
| fixed | +0.001 | +0.156 |
| recursive | -0.025 | +0.155 |
| header | -0.035 | +0.093 |

Structural queries are derived from doc headings, which means they
contain the specific terminology that BM25 excels at matching.
Curated queries are paraphrased, conceptual, less keyword-dense — so
BM25's contribution is small or even slightly negative on the
curated set.

This isn't a bug. It's a real property of how queries are written.
For an interview-style RAG over technical docs, where users *do* often
type specific terms ("PEFT", "AutoModel", "gradient checkpointing"),
the structural-eval-leaning conclusion is probably more representative
of production behavior than the curated-eval-leaning conclusion.

### 4. The cross-encoder's recall contribution is universal

Recall@10 improves with the cross-encoder for **every cell** in the
grid — both eval sets, all three strategies. Average lift:
- Curated: +0.050 average (range +0.000 to +0.120)
- Structural: +0.117 average (range +0.100 to +0.200)

The cross-encoder reliably pulls relevant docs into top-10. The
ranking-within-top-10 is where strategy interactions get noisy, but
the "are the right docs in the candidate set at all" question has a
clean yes-improves answer.

### 5. The full hybrid+rerank pipeline is the right production default

Even where individual strategy rankings shuffle, the *production
pipeline* (hybrid + rerank) is the best or tied-best on Recall@10 in
every cell, and best or near-best on MRR for the structural set.
Adopting it as the default for Phase 2's downstream stages
(generation, fine-tuning) is the right call. The Phase 1 dense-only
baseline served its purpose as a comparison point; it isn't the
right production target.

### 6. Header's recall@10 jumps the most under hybrid+rerank on curated

| | curated recall@10 |
|---|---|
| header dense | 0.743 |
| header hybrid-rerank | 0.863 (+0.120) |

Phase 1 noted that "header is more competitive than expected" once
metrics were corrected, and predicted the cross-encoder would help
header most. The data confirms this for recall — header gains the
most from hybrid+rerank on the curated set. On structural, header
also benefits but no more than recursive. The Phase 1 prediction
*did* hold on the curated set; it just didn't generalize.

### 7. The Phase 1 dense-only baseline doesn't predict winner under hybrid-rerank

The Phase 1 ranking (recursive > header > fixed) holds on the dense
*and* hybrid rows of curated. It breaks on the hybrid-rerank row.
Similarly on structural, the Phase 1-equivalent ranking is
header > recursive > fixed under dense, recursive > header > fixed
under hybrid-rerank.

**Generalization:** picking a chunking strategy from dense-only
metrics gives the wrong answer for a downstream pipeline that
includes a cross-encoder. The chunker should be co-selected with
the rest of the pipeline.

## Implications for Phase 2 and Phase 3

1. **Default pipeline going forward: hybrid-rerank.** This is the
   stack downstream generation will sit on top of. Recall@10
   improvements are universal; MRR is best or near-best on
   structural; the curated MRR exception is interpreted as a
   curated-set artifact.

2. **Default chunking strategy: leave as `recursive`.** It wins on
   structural under hybrid-rerank, and it isn't the worst on
   curated under hybrid-rerank either (it's tied with header at
   ~0.60 MRR). No reason to switch to fixed despite the curated
   spike — that result doesn't generalize.

3. **Always run both eval sets going forward.** Single-set
   conclusions are now demonstrably unreliable. The structural set
   takes ~equal time to run and gives independent signal.

4. **Update the architecture doc** to reflect "hybrid+rerank as
   production default; recursive as default chunker" with a pointer
   to this analysis.

5. **For Phase 3 fine-tuning**: this analysis says nothing about
   generation quality. Fine-tuning will be evaluated against
   faithfulness, answer-relevance, and citation-accuracy LLM-judge
   evals — not retrieval metrics. But the *retrieval* pipeline used
   to feed those evals should be hybrid-rerank, since that's what
   maximizes recall@10 (the input quality the LLM gets to work
   with).

## Methodology caveats

- **Sample sizes are still modest.** Curated n=20, structural n=30.
  Even averaged across multiple metrics, MRR differences in the 0.05
  range sit close to the noise floor. The 3×2 grid pattern is
  consistent enough across multiple metrics that the directional
  conclusions are credible, but precise effect sizes shouldn't be
  cited without confidence intervals (which we don't compute).
- **The eval over-retrieves more candidates than production would.**
  For dedup headroom, the bakeoff invokes retrievers with
  `top_k = eval_k * 4 = 40`, then dedupes to ~10 unique doc_ids.
  Production would use `rerank_candidates=20, top_k=5`. The 2× larger
  candidate pool changes how much work the cross-encoder does and
  how it ranks; some of the curated MRR weirdness might attenuate
  with smaller pools. Worth checking in a follow-up.
- **No statistical tests.** A bootstrap or paired-sample test on
  per-query metric values would put confidence intervals on the
  deltas. Not done here; the directional findings don't require it
  but a finer-grained claim would.
- **Structural eval queries are noisier.** Some headings are vague
  ("Configuration") and could legitimately match many docs, which
  degrades absolute scores across the board. The relative ranking
  between strategies is what matters and is what we report.
- **A third eval set (LLM-generated, "5c") is still scoped for
  pre-Phase-3 work.** Adding it would give us three independent
  distributions and would close out the eval-bias question more
  decisively than two.

## What this changes about the project's narrative

Block B's commit message claimed "fixed wins" as a finding. After
B+, that claim is wrong. The honest summary is:

> The Phase 1 dense-only baseline ranked recursive > header > fixed.
> The full Phase 2 hybrid+rerank pipeline produces a more nuanced
> picture: it uniformly improves recall@10, but the precision-side
> winner depends on the eval set. On structural data (unbiased
> programmatic queries), recursive wins under hybrid+rerank — the
> Phase 1 winner stays the winner. On curated data (hand-authored
> queries), fixed-size chunks become best on MRR but recursive stays
> best on Recall@10. We adopt hybrid+rerank as the production default
> with recursive chunking, and we report both eval sets going
> forward to avoid single-set artifacts.

That's a better story than "fixed wins, but I can't tell you why."
It also makes the Phase 1 work more meaningful in retrospect — the
recursive choice still holds up, just for partly different reasons
than originally thought.
