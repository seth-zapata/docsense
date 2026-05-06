# Block 3B.2 — training-dataset seeding strategy

> Builds on the [Phase 3 scope](phase-3-scope.md). The Block 3B.1 pilot
> ([`evaluations/datasets/training/pilot/comparison.md`](../evaluations/datasets/training/pilot/comparison.md))
> picked **Sonnet 4.5 (no thinking) on prompt v2** as the distillation
> model. This document covers the next layer up: where the ~800
> `(query, retrieved_chunks)` pairs the distillation script consumes
> actually come from.

## Goal

Generate ~800 training examples that satisfy four properties simultaneously:

1. **Realistic** — match the distribution of questions a developer actually asks the HF Transformers docs.
2. **Diverse** — broad corpus coverage, no concentration in popular docs (e.g., not 200 examples all about `Trainer`).
3. **Type-balanced** — proportional to a deliberate question-type distribution (see below).
4. **Eval-clean** — zero overlap with Phase 2 `curated_eval` and `structural_eval` sets, otherwise Phase 3's "did fine-tuning improve performance?" question is contaminated.

Hand-writing 800 quality queries is days of work. Reusing eval queries contaminates the eval. So: **synthetic generation seeded from the corpus**, with hand-curated quality controls and a hand-reviewable intermediate artifact.

## Question-type distribution

| Type | Share | Count | Definition |
|---|---:|---:|---|
| Procedural | 30% | 240 | "how do I X" / "how does X work" — answer derives from a chunk's procedural content (often a code block) |
| Comparison | 20% | 160 | "what's the difference between X and Y" / "when should I use X vs Y" — answer differentiates two related concepts |
| Best-practice | 20% | 160 | "what's the recommended way to X" — answer surfaces an advisory pattern ("we advise", "should") from a chunk |
| Pointer | 15% | 120 | "where do I find / start with X" — answer points to a script, file, model, or reference |
| Refusal | 15% | 120 | Answer should be the canonical refusal phrase |

Refusal split into two sub-types (deliberate, see "Per-type seeding" below):
- **Off-corpus** (~80, 66% of refusal): topic is genuinely outside HF Transformers docs (Linux, AWS, JAX internals, etc.)
- **In-corpus retrieval-failure** (~40, 33% of refusal): query is valid HF-flavored but paired with chunks from a different topic — chunks look relevant but don't actually answer

The 30/20/20/15/15 split is deliberate, not measured. Reasoning:
- Procedural is over-weighted because it's the most common shape in the docs (most chunks are "to do X, write Y") and the most common shape in user questions.
- Comparison and best-practice are co-equal at 20% each — both are skill-distinctive (Qwen needs different answer shapes) and both are common in real Q&A.
- Pointer is smaller because pointer answers are short and the skill is narrower (the model just needs to extract a reference, not synthesize).
- Refusal at 15% matches the pre-Phase-3 baseline's refusal rate (the eval set was 50% refusal, but production traffic will be much lower; 15% is a defensible mid-point).

We can revisit these percentages after the first training run if a particular type underperforms in the eval.

## Architecture: two-stage pipeline

```
Stage 1: Corpus → Query Pool
─────────────────────────────
  indexed chunks
       ↓
  classify chunks by type-affinity (procedural/comparison/best-practice/pointer)
       ↓
  for each in-corpus type: stratified sample → Haiku 4.5 generates ONE query per chunk
       ↓
  for refusals (off-corpus): topic seeds → Haiku 4.5 generates queries
  for refusals (retrieval-failure): pair valid queries with mismatched chunks
       ↓
  dedupe (embedding similarity > 0.85) + filter (eval contamination > 0.7, length floor)
       ↓
  run each query through OUR hybrid retriever to get realistic chunks
       ↓
  query_pool.jsonl  ← Stage 1 artifact, hand-reviewable, committed

Stage 2: Query Pool → Training Dataset
──────────────────────────────────────
  for each (query, retrieved_chunks):
    Sonnet 4.5 generates ideal_answer (existing build_training_dataset.py, prompt v2)
       ↓
  training_dataset.json  ← Stage 2 final artifact, fed to LoRAFineTuner
```

**Why two stages, not one:**

Stage 1 is one-shot — once the corpus is indexed, the query pool is deterministic given seed sampling. Stage 2 is iterative — if the first 50 spot-check reveals a Sonnet output issue, we re-run Stage 2 with a tweaked prompt without re-paying for query generation. Decoupling saves both money and time during the inevitable iteration. It also lets us hand-review the query pool (Stage 1 output) before paying for distillation (Stage 2), catching garbage queries early.

**Why query generation runs through OUR retriever, not just the seed chunk:**

When Sonnet sees a training example at distillation time, it sees the chunks our hybrid retriever returns for the query — not the chunk that seeded the query. Running through retrieval ensures the chunks Qwen sees during training match the distribution it'll see at inference time. It also surfaces realistic retrieval imperfections (the seed chunk may not be in the top-5, related chunks may be there instead) — Qwen learns to handle imperfect retrieval, not idealized retrieval.

## Per-type seeding

### Procedural (240 examples)

**Chunk-affinity heuristic**: chunks containing code blocks (`backtick-py` or similar), imperative verbs (`pass`, `set`, `call`, `add`), and "to X, do Y" phrasing.

**Generator prompt** (paraphrased): "Given this chunk, generate one 'how do I X' or 'how does X work' question. Be specific enough that the chunk's content is needed; phrase like a real developer (informal); 5-15 words."

**Sampling**: stratify across `model_doc/`, `tasks/`, `quantization/`, `trainer.md`, `peft.md`, etc. — with chunks-with-code weighted higher.

### Comparison (160 examples)

**Chunk-affinity heuristic**: chunks mentioning 2+ related concepts in close proximity (e.g., `AutoModel` and `AutoModelForCausalLM` both appear), or containing phrases like "vs", "instead of", "compared to", "two general types".

**Generator prompt**: "This chunk discusses two related concepts. Generate one 'what's the difference between X and Y' or 'when should I use X vs Y' question."

**Risk**: comparison questions can be too narrow ("difference between save_pretrained and save_pretrained_with_hub"). Mitigation: filter by checking that both X and Y are prominent corpus terms (appear in ≥ N chunks), not one-off mentions.

### Best-practice (160 examples)

**Chunk-affinity heuristic**: chunks containing "we advise", "recommended", "should", "best practice", "preferred", "make sure to".

**Generator prompt**: "This chunk contains a recommendation. Generate one 'what's the recommended way to X' question."

**Why this is its own type, separate from procedural**: procedural and best-practice both produce "how do I X"-shaped answers, but best-practice questions teach Qwen to recognize and surface the *recommended* pattern when chunks contain advisory language. Without this signal, Qwen learns to enumerate options rather than recommend.

### Pointer (120 examples)

**Chunk-affinity heuristic**: chunks pointing to scripts/files/models — phrases like "see the BERT conversion script", "refer to the original repository", "a good starting point is", "for X, look at Y".

**Generator prompt**: "This chunk directs readers to a specific resource. Generate one 'where do I find / start with X' question."

**Why it matters**: pointer questions have the shortest answers (the Q5 pilot was 120 chars). Without explicit pointer training, Qwen will over-elaborate when a one-line pointer is the right answer.

### Refusal — off-corpus (80 examples)

**Source**: `evaluations/datasets/training/refusal_topic_seeds.json` — hand-curated ~30 topic seeds, ~3 questions per seed.

Topic categories:
- **Adjacent ML but not transformers**: TensorFlow internals, JAX, scikit-learn, traditional ML, vector DBs (FAISS internals — careful, FAISS *usage* is in our docs)
- **General CS**: Linux, AWS, Docker, Kubernetes, networking, databases, Git internals
- **Completely unrelated** (calibration sanity-check): cooking, sports, weather

**Generator prompt** for each topic: "Generate one technical question on the topic '{topic}' that a developer might ask, but is NOT covered by HuggingFace Transformers docs."

**Pipeline**: even though chunks are returned by the retriever for these queries, the chunks won't actually answer. Sonnet should refuse with the canonical phrase.

### Refusal — in-corpus retrieval-failure (40 examples)

**Why this sub-type**: in production, retrieval will sometimes return topically-adjacent-but-non-answering chunks. The model must recognize this case rather than confabulating an answer from weakly-related context. Without this training, Qwen learns "if there are chunks, answer using them" — a pathology we observed in the pre-Phase-3 baseline analysis.

**Synthesis approach**:
1. Take an in-corpus question (e.g., "How do I save a fine-tuned model?")
2. Pair it with chunks from a *different* topic (e.g., chunks about quantization Marlin kernels)
3. Result: query is valid, chunks are tangentially related, no answer is derivable

This is the highest-leverage refusal training signal we can produce — it teaches the actual production failure mode.

## Quality controls

Catch garbage before it costs us:

1. **Embedding-based dedupe within the query pool.** sentence-transformers cosine similarity > 0.85 → drop the duplicate. Catches near-paraphrases like "How do I save a model?" and "How do I save my model to disk?".
2. **Eval-set contamination filter.** Embed all queries from `eval_queries.py` and `structural_eval`. Drop any new query with similarity > 0.7 to any eval query. Hard threshold — even a few contaminated examples destroy the eval comparison.
3. **Length floor.** Queries < 5 words are usually too vague (e.g., "What is BERT?"). Drop.
4. **Retrieval-quality canary.** After running through the hybrid retriever, if all top-5 chunks have BM25 score < threshold AND dense score < threshold, the query is probably malformed. Flag for review (don't auto-drop — refusals legitimately have low retrieval scores).
5. **Stage 1 hand-review.** Random sample 30 queries from the pool, eyeball pass/fail. Reject the batch if > 15% are unrealistic.
6. **Stage 2 spot-check at 50 examples.** Generate the first ~50 distilled answers with Sonnet, eyeball them as a batch. Abort + tweak prompt if Sonnet's answers regress vs the Block 3B.1 pilot.

## Validated decisions

| Decision | Choice | Validation |
|---|---|---|
| Total N | 800 | Picked over 600 (more diversity) and 1000 (over-budget) — fits Modal A10G ~30 min training and gives statistical power for eval comparisons. |
| Refusal split | 80 off-corpus + 40 retrieval-failure | Retrieval-failure is the production failure mode; without it Qwen learns to always cite. |
| Stage 1 review | 30-query sample, batch accept/reject | Finer review doesn't scale; mechanical filters catch most issues. |
| Topic seeds | Hand-curated (~30) | Taste matters more than scale at this size; LLM-generated topics risk overlapping with HF content. |
| **Distillation model (answer gen)** | **Sonnet 4.5, prompt v2** | Block 3B.1 4-way pilot — Sonnet won on citation density (26 markers), no tangents, prerequisite completeness. |
| **Query generation model** | **Haiku 4.5** | Mini-pilot validated (see [`pilot/query_gen_validation.md`](../evaluations/datasets/training/pilot/query_gen_validation.md)) — query quality indistinguishable from Sonnet across all 5 types; ~3.5× cheaper ($0.26 vs $0.92 for 800 queries). |

### Query-generation model mini-pilot

Validation pilot (5 queries, one per type, same prompt and chunks for both models) ran 2026-05-06. Findings:

- **Quality parity**: all 10 generated queries were realistic developer phrasings, type-appropriate, parsed cleanly, and within 5-15 words. No verbatim chunk quotes from either model. One slight stylistic edge to Sonnet on the best-practice case ("what's the recommended" vs "how should I"); one slight edge to Haiku on the refusal case (more generic phrasing, which is actually preferable for refusal training).
- **Cost**: Haiku $0.26 vs Sonnet $0.92 projected for 800 queries. Smaller absolute savings than projected ($0.66 vs ~$1.50) because Sonnet's outputs were more concise than expected. Cost ratio is ~3.5×, not 15×.
- **Decision**: Haiku 4.5 for query generation. Quality is a wash; smaller-model-first principle applies; saves enough to be worth doing.

## Cost & timeline

| Stage | What | Cost | Wall time |
|---|---|---:|---:|
| 1a | Query generation (~800 × Haiku) | ~$0.26 | ~10 min |
| 1b | Hybrid retrieval on each query (local) | $0 | ~2 min |
| 1c | Dedupe + filter | $0 | <1 min |
| 1d | Stage-1 hand-review (30 sample) | $0 | ~10 min |
| 2a | First 50 examples (Sonnet) | ~$0.40 | ~3 min |
| 2a' | Spot-check checkpoint | $0 | ~10 min review |
| 2b | Remaining ~750 (if checkpoint passes) | ~$5.50 | ~30 min |
| **Total** | | **~$6.16** | **~1 hr active** |

Comfortably within the $10 distillation budget allocated in the Phase 3 scope.

## Implementation surface

```
src/docsense/finetuning/
  query_generation.py       # NEW — TypeAwareQueryGenerator class (one prompt template per type)
  query_filters.py          # NEW — dedupe_by_embedding, filter_eval_contamination, length_floor
  chunk_classifier.py       # NEW — heuristic chunk-affinity classifier (procedural/comparison/etc.)

scripts/
  build_query_pool.py       # NEW — Stage 1 driver
  build_training_dataset.py # EXTEND — accept query_pool.jsonl as input format

evaluations/datasets/training/
  refusal_topic_seeds.json  # NEW — hand-curated ~30 off-corpus topic seeds
  query_pool.jsonl          # Stage 1 output, committed for reproducibility
  training_dataset.json     # Stage 2 output, fed to LoRAFineTuner

tests/unit/
  test_query_generation.py  # type-affinity classifiers, generator stubs
  test_query_filters.py     # dedupe, eval-contamination, length floor
  test_chunk_classifier.py  # heuristic affinity classifiers
```

The query-pool format extends `pilot_input.json`'s shape with two fields per entry:
- `type: "procedural" | "comparison" | "best_practice" | "pointer" | "refusal"`
- `seed_chunks: list[ChunkRef]` — the chunks that seeded the query (so we can audit "did Sonnet's answer cite the chunk we seeded the query from?")

## Deferred / open questions

- **Question-type distribution refinement.** The 30/20/20/15/15 split is a first guess. After the first training run, if the eval shows e.g. comparison answers are weak, bump comparison's share in the next dataset iteration.
- **Retrieval-failure refusal synthesis.** Approach 1 (manually pair queries with off-topic chunks) is what's planned. Approach 2 (use the retriever's top-K-from-N where K > 5 to surface deliberate near-misses) is more automated but may produce unrealistic chunks. Start with Approach 1; revisit if 40 manually-paired examples is too tedious.
- **Hand-review threshold for Stage 1.** "> 15% unrealistic = reject batch" is a guess. May need to be tighter (10%) or looser (25%) once we see what the actual pool looks like.
- **Retrieval-quality canary thresholds.** "BM25 < threshold AND dense < threshold" needs concrete numbers. Will calibrate against known-good queries from `eval_queries.py` once the pipeline is built.
