# Phase 2–4 scope: Git, CI, eval, and infrastructure

A durable planning doc for the docsense roadmap from Phase 2 onward.
Future sessions should read this *first* when picking up Phase 2 work
or beyond — it captures the agreed scope, the explicit non-goals, and
the block-by-block implementation pacing.

> **Companion docs:**
> - [`docs/architecture.md`](architecture.md) — what the system *is*; updated at phase boundaries.
> - [`CLAUDE.md`](../CLAUDE.md) — active worklist + dev instructions.
> - [`docs/journal/`](journal/) — gitignored personal journal; decisions and surprises.

---

## Adjustments locked in

This scope reflects the following constraints, agreed before any
implementation begins:

1. **Phase 1 metric comparability is preserved.** The bakeoff runner
   accepts an explicit `--eval-k` argument (default 10) so Phase 2
   reranked retrieval is fairly comparable to Phase 1 baselines
   measured at Recall@10 / nDCG@10. The pipeline's
   *delivered* `top_k` (e.g., 5 for the LLM) and the *eval k* (e.g.,
   10 for metric continuity) are decoupled.
2. **Naming semantics are fixed before reranker wiring.**
   - `RerankingConfig.batch_size` means **only** the cross-encoder
     inference batch size. It must not determine result count.
   - `RetrievalConfig.rerank_candidates` means how many fused
     candidates are sent to the reranker.
   - `RetrievalConfig.top_k` means the **final** number of results
     returned to the caller. (Aliased as `final_top_k` in the docs
     when disambiguation helps.)
3. **Mocked tests do not overclaim behavioral guarantees.** Cheap
   rule-based tests verify *contract*: prompt template shape,
   generator parses expected response forms, output types validate.
   Real no-answer behavior, faithfulness, answer relevance, and
   citation accuracy are LLM-judged evals run manually or nightly,
   not in PR CI.
4. **Implementation pauses at every block boundary.** This doc lays
   out blocks A–E. Future sessions must wait for explicit user
   approval before crossing each boundary. No "run the whole plan"
   unattended.
5. **Security work is staged, not front-loaded.** Don't duplicate
   GitHub-native secret scanning. Document the items that become
   important before Phase 4; implement when their context arrives.
6. **FastAPI / OpenAPI / Docker / deployment are Phase 4 only.**
   Documented here so the design constraints are visible upstream,
   but no scaffold lands until Phase 4 starts.

---

## A. Already covered — leave it alone

Don't rework these; they're solid and any changes would be churn.

- **Retrieval pipeline core** — ingestion, three chunkers, embedder,
  hybrid (BM25 + FAISS + RRF), retrieval metrics. 100% covered,
  mocked correctly.
- **Test infrastructure** — 115 tests, 98% coverage, 90% CI gate. The
  omit-list discipline (`fetcher.py`, `eval_queries.py`) is documented
  inline and only grows by deliberate commit.
- **CI pipeline** — lint + typecheck + test+coverage on Python 3.12.
  No matrix theater.
- **Tooling pins** — `ruff==0.15.8`, `mypy==1.19.1`, pre-commit hooks via
  `language: system`. Single source of truth in pyproject dev extras.
- **Docs structure** — README / architecture.md / CLAUDE.md / journal
  split with clear roles. Don't merge them.
- **Eval scaffolding** — curated 20-query set, structural query
  generator, deduplicate-preserving-order helper, retrieval metrics.
- **Engineering journal practice** — gitignored, narrative-first,
  append-only.

**Specifically: don't add a Python version matrix back, don't move
tool pins to lockfiles, don't gate coverage above 90%, don't add
nbstripout to pre-commit (notebook outputs are intentionally
committed for the bakeoff to render on GitHub).**

---

## B. Must-add now for Phase 2

Strict minimum to land Phase 2 cleanly. Everything else can wait.

### B1. Config naming hygiene (pre-reranker wiring)

Rename and split before any code calls into the reranker via the
hybrid path. The current `RerankingConfig.batch_size` is doing double
duty as cross-encoder batch size *and* default top_k for `rerank()` —
that's the misnomer to fix first.

Final shape:

```python
class RetrievalConfig(BaseSettings):
    top_k: int = 5                  # FINAL delivered count (chunks → LLM)
    rerank_candidates: int = 20     # NEW: fused candidates → reranker input
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    # Removed: rerank_top_k (was redundant with top_k)

class RerankingConfig(BaseSettings):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32            # cross-encoder inference batch only
    device: str = "cpu"
```

`HybridRetriever` over-retrieves `rerank_candidates` from the fused
results, the reranker scores them in batches of `batch_size`, the
final top_k is returned.

### B2. Reranker integration into HybridRetriever

The reranker class exists with 9 unit tests. Wiring is small:
`HybridRetriever.__init__` accepts an optional `reranker:
CrossEncoderReranker | None`. `search()` over-retrieves
`rerank_candidates` when a reranker is set, hands the fused list to
`reranker.rerank(query, results, top_k=top_k)`, returns the result.

Integration test in `tests/integration/test_hybrid_rerank.py` (the
directory is empty today): mocked cross-encoder, verify the right
candidate count flows through, cross-encoder scores replace fusion
scores, final length matches `top_k`.

### B3. Phase 1 baseline committed

`evaluations/baselines/phase1_chunking.json` — small JSON file with
the corrected bakeoff numbers per strategy at `eval_k=10`. Includes
metadata: corpus commit SHA, embedding model name, ranker
configuration. Without this committed, Phase 2 changes have nothing
to be measured against.

### B4. Bakeoff runner script with explicit eval_k

`scripts/run_bakeoff.py` — extracts the eval logic from the notebook
into a reproducible script. Critical CLI surface:

```
python scripts/run_bakeoff.py \
    --strategy {fixed,recursive,header} \
    --eval-k 10                # explicit; default 10 for Phase 1 parity
    [--rerank]                 # toggles cross-encoder reranking
    [--config configs/...yaml] # YAML preset
    --out evaluations/reports/<filename>.json
```

`--eval-k` is the metric horizon (Recall@k, nDCG@k). The pipeline's
delivered `top_k` is read from config / preset. **eval_k and top_k
are independent**: the bakeoff invokes retrieval with
`top_k=max(eval_k, config.top_k)` so there are always enough results
to compute the metric, regardless of what the production default
delivers.

The script also produces a markdown delta vs.
`evaluations/baselines/phase1_chunking.json` so PRs can include the
report directly.

The notebook stays as the GitHub-rendered artifact; the script is the
reproducible runner.

### B5. Generation response types (define before the LLM lands)

Pydantic models in `src/docsense/generation/types.py`:

```python
class ChunkRef(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    text: str

class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    quote: str | None = None    # exact span if extractable

class Answer(BaseModel):
    text: str
    citations: list[Citation]
    retrieved_chunks: list[ChunkRef]    # what was passed to the LLM
    generation_metadata: dict           # latency_ms, prompt_tokens, model_name
```

Defining these now — before the LLM lands — forces clarity on the
contract and lets context-assembly + prompt + generator be written
against a stable target. JSON Schema falls out for free when FastAPI
arrives in Phase 4.

### B6. Phase 2 generation tests — cheap, contract-shape only

These run on every PR. **They do not validate behavioral correctness
of the LLM**; they validate that *our* code delivers and consumes the
expected shapes. Behavioral evaluation is C1.

- **Context-assembly determinism** — given (query, chunks, config),
  `assemble_context()` returns identical output across two calls.
  Catches nondeterministic ordering or dict iteration drift.
- **Prompt snapshot test** — render the full prompt for fixed inputs,
  diff against `tests/snapshots/prompt_default.txt`. Snapshot updates
  are intentional commits and reviewed.
- **Citation-preservation test** — given an `Answer` with citations,
  every cited `doc_id` must reference a chunk that was in
  `retrieved_chunks`. Pure rule-based, no LLM needed.
- **Token-budget enforcement** — `assemble_context()` must not exceed
  the configured token budget. Use the model's tokenizer (or
  `tiktoken` analog).
- **Generator contract test, mocked LLM** — verifies `Generator.run`
  passes the assembled prompt to the underlying model, accepts the
  response, returns a valid `Answer`. **Does not assert anything
  about whether the model would refuse, hallucinate, or be
  faithful** — those are LLM-judged evals (C1).

### B7. Configs directory — ablation-friendly only

`configs/default.yaml`, `configs/no-rerank.yaml`,
`configs/header-strategy.yaml`. **Just experiment knobs**, not the
entire pydantic config tree. Consumed by `run_bakeoff.py`.

```yaml
# configs/no-rerank.yaml — example
chunking:
  strategy: recursive
retrieval:
  top_k: 10
  rerank_candidates: 0    # disables reranker path
```

Pydantic-settings supports YAML via a small loader. Don't convert the
full config system to YAML-first — that's the bureaucracy we're
avoiding. Project-structure config (data dir, log level) stays in
defaults / env vars.

---

## C. High-value before Phase 3

These prep for fine-tuning. Not Phase 2 blockers; gating point is
"about to start QLoRA work."

### C1. LLM-judge eval scaffolding

Faithfulness, answer relevance, citation accuracy, and *real* no-answer
behavior all require an LLM judge.

**Default judge: local Llama 3.1 8B Instruct** (NF4 4-bit), invoked via
HuggingFace transformers. Decision locked 2026-05-06 after weighing
local-judge vs Anthropic SDK. Reasoning summary:

- **Cost**: $0 vs ~$0.20 per full eval run with Claude Haiku. Real but
  small; not the deciding factor by itself.
- **Reproducibility**: pinned local weights vs API drift. Frozen
  baseline numbers can be reproduced years later.
- **Self-containment**: no external dependencies for normal operation.
  Important for the project's "I built this from scratch" narrative.
- **Acknowledged tradeoff**: smaller-model judges are noisier and less
  well-calibrated than frontier models. Egocentric bias would be an
  issue if judge=system; mitigated by using Llama (judge) +
  Qwen (system) — different model families.

An **optional Anthropic-judge mode** (`AnthropicJudge` class, gated
behind an env var like `DOCSENSE_USE_ANTHROPIC_JUDGE=1`) is acceptable
as a calibration check — "how do my Llama-judge scores correlate with
Claude-judge scores on the same outputs?" — but it is NOT the default
path. Adding it stays optional; full project must run without an
Anthropic API key.

Pluggable judge interface: an `LLMJudge` protocol (or ABC) with
implementations for local + (optional) Anthropic. Judge functions:

- `judge_faithfulness(question, context, answer)` — does the answer
  follow from the chunks? Returns `{score: float ∈ [0,1], rationale: str}`.
- `judge_relevance(question, answer)` — does the answer address the
  question?

Plus rule-based-but-uses-real-LLM-output evals (no judge call needed,
just pattern matching on real generated output):

- `check_no_answer_behavior(answer)` — when retrieval returns
  irrelevant chunks, does the model produce a refusal? Pattern match
  against the prompt template's refusal directive ("I don't have
  enough context").
- `check_citations_grounded(answer, retrieved_chunks)` — every
  citation references a real retrieved chunk. Already enforced as a
  pydantic invariant on `Answer` (Block D.1); the eval just confirms
  the invariant holds on the model's actual outputs at scale.

**These run manually or on a `nightly.yml` schedule, never on every
PR.** Latency + score noise make per-PR gating false-positive prone.

### C2. Baseline generation eval (before fine-tuning)

`evaluations/baselines/pre_phase3_generation_base.json` — the base
model's faithfulness / answer relevance / citation accuracy /
no-answer behavior numbers, on the curated + structural eval sets,
captured *before* any fine-tuning. Phase 3's fine-tuned model is
measured against this. Without it, the fine-tune has nothing to
claim improvement over.

**System (generator) model: `Qwen/Qwen2.5-7B-Instruct`.** Decision
locked 2026-05-06. Chosen over Mistral 7B Instruct v0.3 because
HumanEval ~85 vs Mistral ~40 directly reflects code-comprehension
quality, and the HF Transformers corpus is heavily Python documentation.
Apache 2.0 license matches our existing dep posture. Update
`GenerationConfig.model_name` default accordingly when the wiring
lands.

**Hardware reality**: vanilla RTX 4070 (12 GB VRAM). Both system and
judge models must load at NF4 4-bit quantization (~5-6 GB each) and
must be **swapped sequentially** during eval — load system, run all
queries through the pipeline, save intermediate outputs to disk, free
VRAM, load judge, score outputs. Code path must be device-agnostic
(`device="auto"`); also runnable on CPU for users without GPU
(slower, but functional).

### C3. Synthetic Q&A dataset generator

`src/docsense/training/dataset.py` (new module) — given the corpus,
produces `(question, answer, source_chunks)` triples via Claude. The
training data for QLoRA. Validate output (answer must be supported by
source chunks per the LLM judge; otherwise discard).

### C4. The deferred 5c eval set

Synthetic queries via Claude on a held-out doc subset. Three
independent eval distributions (curated, structural, LLM-generated)
× generation metrics × phase = a strong agreement story. Same
infrastructure as C3 but inputs are doc sections, outputs are
queries.

### C5. W&B integration

Already in deps, never used. Wire into `scripts/run_bakeoff.py` and
the eventual training script. Most valuable for fine-tuning where
many runs need to be compared.

---

## D. Phase 4 only — document, do not implement

Don't build any of this until Phase 4 starts. Building now means
rotting before being used.

- **FastAPI endpoint** with the `Answer` response model from B5
- **OpenAPI snapshot test** — render `/openapi.json` to a committed
  file, assert no diff. Catches accidental contract changes.
- **Tracing / structlog** with correlation IDs in every log line.
- **Debug-info gating** — `?debug=true` query exposes
  `retrieved_chunks` + prompt; default is text + citations only.
  Prevents prompt/chunk leakage to normal users.
- **Dockerfile** — multi-stage build; image scanning (Trivy or Grype)
- **CD pipeline** — build → push to GHCR → staging → smoke tests →
  manual gate → prod
- **Rate limiting / abuse protection** — only if hosted publicly
- **Document upload safety** — size limits, MIME validation,
  sanitization. Only matters if `/ingest`-style endpoint exists.
- **Adversarial / prompt-injection eval suite** — committed corpus
  of injection attempts (`Ignore previous instructions...`,
  `<|system|>...`), assert grounded-response behavior. Doable as
  judged eval against a small adversarial set.

---

## E. Specific files / workflows / tests

Concrete additions, ordered by phase. **None of these exist yet
unless noted.**

### Phase 2

```
configs/                                          (currently empty dir)
  default.yaml                                    NEW — knobs for default run
  no-rerank.yaml                                  NEW — ablation: skip reranker
  header-strategy.yaml                            NEW — ablation: chunking
evaluations/                                      NEW directory
  baselines/
    phase1_chunking.json                          NEW — corrected Phase 1 numbers
  reports/
    .gitkeep                                      NEW
src/docsense/
  generation/
    types.py                                      NEW — Answer, Citation, ChunkRef
    context.py                                    NEW — token-budget assembly
    prompt.py                                     NEW — template rendering
    generator.py                                  NEW — LLM wrapper, mockable
  retrieval/
    hybrid.py                                     CHANGE — accept optional reranker
  reranking/
    reranker.py                                   CHANGE — top_k from config.top_k
  config.py                                       CHANGE — split rerank_candidates
scripts/
  run_bakeoff.py                                  NEW — reproducible bakeoff runner
tests/
  unit/
    test_context_assembly.py                      NEW — determinism, dedup, budget
    test_prompt_construction.py                   NEW — snapshot
    test_generation_types.py                      NEW — schema validation
    test_citation_preservation.py                 NEW — citations valid
    test_token_budget.py                          NEW — budget enforcement
  integration/
    test_hybrid_rerank.py                         NEW — end-to-end mocked CE
    test_generation_pipeline.py                   NEW — full mocked pipeline
  snapshots/
    prompt_default.txt                            NEW — committed prompt
docs/
  eval-methodology.md                             NEW — what's measured, where, how
```

### Pre-Phase-3 (high-value)

```
src/docsense/
  training/
    __init__.py                                   NEW
    dataset.py                                    NEW — synthetic Q&A generation
    judge.py                                      NEW — Claude faithfulness/relevance
evaluations/
  baselines/
    phase2_generation_base.json                   NEW — base-model eval
  llm_judge_eval/                                 NEW directory
.github/workflows/
  nightly.yml                                     NEW — pip-audit + judged eval
```

### Phase 4 only — do not create yet

```
src/docsense/
  api/main.py, api/schemas.py                     Phase 4
  tracing/setup.py                                Phase 4
Dockerfile                                        Phase 4
.github/workflows/cd.yml, security.yml            Phase 4
tests/contract/                                   Phase 4
```

### GitHub Actions split

| Workflow | Trigger | Runs |
|---|---|---|
| `ci.yml` (existing) | Every PR + push to main | lint, typecheck, test+coverage. <3 min. |
| `nightly.yml` (Phase 3) | Schedule + manual | pip-audit, optional LLM-judge eval against committed baseline, full bakeoff with reranker, structural-eval comparison |
| `cd.yml` (Phase 4 only) | Push to main | Build container, push to GHCR, deploy staging, smoke tests, manual gate to prod |
| `security.yml` (Phase 4 only) | Schedule + push to main | Trivy on Docker image, supplementary scans if GitHub-native isn't enough |

**Never** run LLM-judge evals on every PR. Cost + latency + noise.

---

## F. Implementation plan — block-paced

The full Phase-2 plan is split into five blocks. **Implementation
pauses at every block boundary** for explicit approval before
proceeding. No autonomous run-through.

### Block A — Config cleanup + reranker wiring

1. `Split rerank_candidates from top_k in RetrievalConfig` — new field,
   remove redundant `rerank_top_k`, update tests.
2. `Untangle reranker top_k from cross-encoder batch_size` —
   `rerank()` takes explicit `top_k` or reads `RetrievalConfig.top_k`.
   `RerankingConfig.batch_size` is now exclusively the inference
   batch size. Update the 9 existing reranker tests.
3. `Wire optional cross-encoder reranker into HybridRetriever` — new
   `reranker` constructor arg; over-retrieves `rerank_candidates`
   when set.
4. `Add tests/integration/test_hybrid_rerank.py` — end-to-end with
   mocked cross-encoder.

⏸️ **Block A pause point.** This is a self-contained chunk that could
ship and stop here if priorities shift.

### Block B — Baselines + bakeoff runner

5. `Commit Phase 1 corrected baseline to evaluations/baselines/phase1_chunking.json`
   — at `eval_k=10`, with corpus + model metadata.
6. `Add scripts/run_bakeoff.py with explicit --eval-k flag` — Phase 1
   parity first; no `--rerank` yet. Verifies the script reproduces
   the baseline numbers within tolerance.
7. `Add --rerank flag to run_bakeoff.py + commit Phase 2 rerank report`
   — the experiment that resolves the parked header-vs-recursive
   question.
8. `Add configs/default.yaml + configs/no-rerank.yaml + YAML loader`
   — pydantic-settings YAML support; `run_bakeoff.py --config <path>`.
9. `Add configs/header-strategy.yaml ablation` — proves the config
   plumbing works for at least one non-default.

⏸️ **Block B pause point.** Phase 2's headline experimental result
exists by the end of this block. Worth a journal coda.

### Block B+ — Bakeoff investigation (added 2026-05-06)

Block B's headline result was unexpected: the full hybrid+rerank
pipeline flipped the strategy ranking — fixed-size chunking overtook
recursive on rank-sensitive metrics, and recursive (the Phase 1
winner) actually got *worse*. But ``--rerank`` enables BM25 + RRF +
cross-encoder simultaneously, so the comparison conflates three
additions. We can't tell whether the surprise is "rerank is bad for
recursive," "BM25 fusion is bad for recursive," or "the eval
over-retrieves more candidates than production would, and the
cross-encoder gets noisy at scale." Each has different downstream
implications.

Block B+ ablates each addition and adds an independent eval set
(structural queries) so we close out Phase 2's retrieval work with
properly interpretable data.

5+. `Update docs/phase-2-4-scope.md with Block B+ section` — formalize
   the scope before implementing it.
6+. `Replace --rerank with --pipeline {dense,hybrid,hybrid-rerank}` in
   ``run_bakeoff.py``. Hybrid mode uses ``HybridRetriever`` without
   the cross-encoder, isolating BM25 + RRF effect from the reranker
   effect.
7+. `Generate and commit structural eval set + add --eval-set flag` —
   the structural query generator was committed in Phase 1 cleanup
   but never run. Generate ~30 queries from doc headings, commit to
   ``evaluations/eval_sets/structural.json``. Add
   ``--eval-set {curated,structural}`` to ``run_bakeoff.py``.
8+. `Run bakeoff in 6 configurations + commit reports` — 3 pipelines
   × 2 eval sets. Reports go to ``evaluations/reports/`` with naming
   ``bakeoff-<date>-<pipeline>-<eval-set>.json``. Also add
   ``evaluations/README.md`` explaining the directory structure.
9+. `Add bakeoff investigation analysis` —
   ``evaluations/analyses/2026-05-06-bakeoff-investigation.md``
   interpreting the 3×2 grid. Two possible outcomes:
   - Result holds across all configurations → strong signal that
     fixed-chunking + production pipeline really is best for this
     corpus.
   - Result reverses or attenuates somewhere → we know exactly which
     addition (BM25 / fusion / rerank) flips the ranking, and on
     which query distribution.
   Either is a stronger story than the current "fixed wins, but I
   can't tell you why."

⏸️ **Block B+ pause point.** Phase 2 retrieval is properly understood
by the end of this block. Block C (generation scaffolding) starts on
trustworthy ground.

### Block C — Generation scaffolding

10. `Add src/docsense/generation/types.py` — Answer, Citation,
    ChunkRef + tests.
11. `Add src/docsense/generation/context.py` — token-budget assembly
    with determinism test.
12. `Add src/docsense/generation/prompt.py` + `tests/snapshots/prompt_default.txt`.
13. `Add src/docsense/generation/generator.py` — mockable LLM wrapper.
14. `Add tests/integration/test_generation_pipeline.py` — full
    mocked pipeline (query → retrieve → rerank → assemble → prompt
    → generator → typed Answer).

⏸️ **Block C pause point.** Generation surface exists, fully tested
against contract shape. No real LLM calls yet.

### Block D — Cheap rule-based generation tests

15. `Add citation-preservation test`
16. `Add token-budget enforcement test`
17. `Add generator contract test (mocked LLM)`

(Behavioral evals — faithfulness, answer relevance, *real* no-answer
behavior — are deliberately deferred to C1 / pre-Phase-3 with LLM
judges.)

⏸️ **Block D pause point.**

### Block E — Documentation update at phase boundary

18. `Update CLAUDE.md and docs/architecture.md to reflect Phase 2 closure`
    — single commit at the boundary, per the rule we set in CLAUDE.md.
19. `Add docs/eval-methodology.md` — short reference (~100 lines)
    explaining what's evaluated where, what's PR-safe vs nightly vs
    manual, where baselines live.

⏸️ **Phase 2 closes.** ~19 commits, paced through five blocks, each
with an explicit pause for approval.

### Pre-Phase-3 Block 1 — Generation infrastructure + LLM-judge scaffolding (added 2026-05-06)

Phase 2 closed with the generation surface scaffolded but never run
against a real LLM (only mocked). Pre-Phase-3 Block 1 closes that gap:
loads a real model end-to-end, wires NF4 quantization for the 12 GB
hardware constraint, refactors prompt construction to use the
tokenizer's chat template, and stands up the local-judge framework so
generation quality can be measured and tracked.

Split into **two sub-blocks** so the infrastructure (1A) can land and
be smoke-tested before committing to the larger judge work (1B). Each
has its own pause point.

#### Block 1A — Generation infrastructure

20. `Update scope doc with Pre-Phase-3 Block 1 spec + journal entry`
    — formalize before implementing (this commit).
21. `Switch GenerationConfig defaults to Qwen 2.5 7B Instruct` —
    `model_name = "Qwen/Qwen2.5-7B-Instruct"` and any related
    config (e.g., a new `use_4bit_quantization: bool = True`).
    Update tests that assert defaults.
22. `Refactor Generator._run_inference to use tokenizer chat template`
    — replace the raw "Question:/Answer:" completion-style invocation
    with `tokenizer.apply_chat_template([{role: ..., content: ...}, ...],
    add_generation_prompt=True, tokenize=True, return_tensors="pt")`.
    `PromptBuilder` returns a *list of messages* rather than a flat
    string; the chat-template formatting is the tokenizer's job. This
    makes Generator model-agnostic — Qwen, Llama, Mistral all work
    via the same code path. Update prompt snapshot test (snapshot is
    now the rendered chat-template output for the configured model).
23. `Wire NF4 4-bit quantization via bitsandbytes into Generator` —
    `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16)` passed to
    `AutoModelForCausalLM.from_pretrained` when
    `config.use_4bit_quantization` is True (default). Bitsandbytes is
    already in `pyproject.toml`'s `gpu` extras. Tests use the existing
    `_run_inference` override path; quantization is only exercised when
    a real model loads.
24. `Add scripts/smoke_generate.py and run it manually` — small driver
    that loads the configured model, runs one query through the full
    pipeline (retrieve → rerank → assemble → prompt → generate → typed
    Answer), prints the answer + citations + metadata. **Manual
    verification step**, not part of CI. User runs once on their
    hardware to confirm the model loads, generates plausible output,
    and the chat template is formatted correctly. Capture stdout in
    `evaluations/manual-runs/2026-05-XX-smoke.txt` and commit.

⏸️ **Block 1A pause point.** First real end-to-end generation works
on the user's hardware. We know the model loads, fits in 12 GB, and
produces plausible output. No automated quality measurement yet.

#### Block 1B — LLM-judge framework + baseline eval

25. `Add LLMJudge protocol/ABC in src/docsense/evaluation/judge.py`
    — abstract base with methods `judge_faithfulness(question, context,
    answer) -> JudgeScore` and `judge_relevance(question, answer) ->
    JudgeScore`. Plus the `JudgeScore` pydantic model
    (`{score: float, rationale: str}`).
26. `Add LlamaJudge implementation (Llama 3.1 8B Instruct, NF4 4-bit)`
    — concrete `LLMJudge` subclass. Same lazy-load pattern as
    `Generator`. Test with mocked `_run_inference` override.
27. `Add rule-based evals: check_no_answer_behavior + check_citations_grounded`
    — these don't need a judge call, only the model's actual output.
    Pure functions taking `Answer`. Tested with synthetic Answer
    instances.
28. `Add scripts/run_generation_eval.py` — end-to-end eval driver:
    load system → run all queries through pipeline (saving raw Answer
    objects to disk) → free system VRAM → load judge → score each
    Answer → save reports. Sequential model loading is the headline
    constraint; the script must explicitly free CUDA memory between
    stages.
29. `Run the eval, commit baseline reports` — first real LLM-judged
    eval run. Reports go to `evaluations/reports/generation-<date>-{curated,structural}.json`
    and `evaluations/baselines/pre_phase3_generation_base.json`.
30. `Add evaluations/analyses/2026-05-XX-baseline-generation-eval.md`
    — interpret the scores. What did Qwen 2.5 7B answer correctly?
    Where did it confabulate? Did the refusal directive activate? Did
    citations come out properly? This is Phase 3's starting line.

⏸️ **Block 1B pause point.** Phase 2 retrieval + Phase 2 generation
are now empirically measured end-to-end. Phase 3 fine-tuning has a
real baseline to improve on.

### Pre-Phase-3 Block 2 — eval methodology hardening (added 2026-05-06)

After Block 1B's first end-to-end eval landed, two methodology
flaws surfaced that warranted fixing before Phase 3 (rather than
"defer until they bite"). Phase 3 will fine-tune Qwen, which will
shift behavior in ways that compound silent measurement errors.
Cheaper to fix the methodology while we have a working baseline.

The two fixes break along the same conceptual line: **regex on
free-form generator output and regex on tool-controlled judge
output are different mistakes with different right answers**.

#### Block 2A — Latency reporting + observability conventions (PR A, closed 2026-05-06)

Switched latency schema from p50/p90/max to AWS-standard
p50/p95/p99 + mean + n. Re-aggregated existing eval reports from
preserved per_query timing data; no eval re-run needed. Added a
cross-cutting `Latency & observability conventions` section to
this scope doc (below) capturing the convention plus broader
observability standards (cold-start vs steady-state, per-stage
breakdowns, throughput, error-rate categories, anti-patterns).

#### Block 2B — Claim-level faithfulness with per-chunk attribution (PR B, closed 2026-05-06)

Replaced absolute-scale anchor-based faithfulness scoring (which
saturated at 0.75 in 49 of 50 in-corpus answers) with RAGAS-style
claim-level decomposition + per-chunk attribution. Two LLM calls
per query: extract atomic claims, then attribute each claim to a
specific chunk or "none". Score = `n_supported / n_total`,
continuous in [0, 1], no anchor snapping. Per-claim attributions
preserved on the JudgeScore. Surfaced one new finding: chunk usage
diverges sharply across eval sets (chunk 1 cited least on curated,
most on structural), suggesting query-style-dependent retrieval
behavior worth a Phase 3 audit. Cross-set faithfulness mean is
exactly 0.853 on both — methodology-validation signal.

#### Block 2C — Refusal detection via LLM-judge with cross-validation (PR C.1, closed 2026-05-06)

Replaced regex-only refusal detection (which had a phrasing-coverage
treadmill) with a `judge_refusal` LLM primitive. Critically, the
rule-based regex still runs alongside as a guardrail — both verdicts
are surfaced in the report, plus an `agreement_rate` metric that
becomes the early-warning signal for Phase 3 phrasing drift. Also
expanded the no-answer eval set from n=8 to n=25 (with the regex
maintenance burden removed, the size constraint went away). First
run of the new methodology: judge=1.000, rule=1.000,
agreement=1.000 across 25 queries — perfect cross-validation.

`RefusalJudgment(refused: bool, rationale: str)` is a new typed
result alongside `JudgeScore` — refusal is naturally categorical,
not a [0, 1] score.

#### Block 2D — Judge output → JSON, eliminate regex parsers (PR C.2, queued)

Final piece of the methodology cleanup. Refactor judge output
parsing for `extract_claims`, `attribute_claims_to_chunks`, and
`parse_relevance_response` from regex to JSON + pydantic
validation + retry-on-parse-failure. Eliminates the `curated_001`
8-of-8-PARSE_FAILED edge case documented in the baseline analysis
and aligns with industry-standard LLM-as-judge practice.

After PR C.2, the only regex left in the eval pipeline will be
`_CITATION_MARKER_RE` — extracting `[N]` markers from the
generator's output, which is the right level of mechanical for
the input shape we ask the model to produce.

### Optional follow-up — AnthropicJudge calibration mode

Not in Block 1 by default. Add later as a separate small PR if/when
the value of judge calibration justifies the API key + spend:

- `Add AnthropicJudge (env-var gated)` — same `LLMJudge` interface,
  uses Anthropic SDK. Only constructable if `DOCSENSE_USE_ANTHROPIC_JUDGE=1`
  and `ANTHROPIC_API_KEY` are set; otherwise raises a clear error.
- `Run calibration check + commit comparison report` — score the same
  Answer outputs with both LlamaJudge and AnthropicJudge, compute
  per-metric correlation, write `evaluations/analyses/judge-calibration.md`.
  This becomes a portfolio artifact: "I demonstrated that local-judge
  scores correlate with frontier-judge scores at ρ=X on my eval set,
  validating the local-only default."

---

## Latency & observability conventions

Cross-cutting standards for how docsense reports latency and other
operational metrics. Pinned here so every eval run, smoke artifact,
and (eventually) Phase 4 serving telemetry uses the same shapes —
inconsistent reporting is the enemy of system-state legibility.

### Latency: AWS-standard percentiles

**p50, p95, p99 + mean + n.** Adopted 2026-05-06 (`scripts/run_generation_eval.py`,
`evaluations/performance.md`, `evaluations/baselines/pre_phase3_generation_base.json`).

| Metric | What it captures |
|---|---|
| **p50 (median)** | Typical user experience. The "what does it feel like for most queries" number. |
| **p95** | Slow-tail signal that survives single outliers. The "1 in 20 users" boundary; the canonical SLO target in most prod systems. |
| **p99** | True tail signal. The "1 in 100 users" boundary; reveals heavy-tailed distributions. At small N (n < 100) p99 is dominated by single events — read it as "tail signal" not "production worst-case." |
| **mean** | Sanity check alongside the percentiles. mean ≫ p50 ⇒ right-skewed (heavy-tail); mean ≈ p50 ⇒ symmetric. |
| **n** | Sample size. Always reported; absolute percentile interpretations require it. |

Why not p90 + max:
- p90 is a lower-bar version of p95. Pick one — AWS convention is p95.
- max is a single-observation tail metric that's easy to pollute
  with one-off slow events (cold-start cross-encoder warmup, GC
  pause). p99 is the right tail metric in a controlled eval; max
  is recoverable from per_query data when needed for forensics.

### Cold-start vs steady-state

Distinguish in every report. Currently informal — a one-time model
load (~163 s for Qwen 7B at NF4) inflates wall time on the first
query but doesn't represent steady-state. Eval timings already
exclude model load (Phase B starts post-load). For Phase 4 serving,
warm vs cold starts must be separately tracked: a request that
hit a freshly-spawned worker shouldn't be aggregated with requests
to a warm worker — the distributions are bimodal.

### Per-stage breakdown — pending

Currently `retrieve_ms` is a single bucket covering dense + sparse
+ RRF + cross-encoder rerank. The cross-encoder dominates (~1.3 s
at `rerank_candidates=20`); dense + sparse + RRF are sub-second.
Splitting these out is a small follow-up — useful when Phase 3
lands and we want to see whether changes in retrieval (e.g., a
reranker swap) shift the latency distribution.

### Resource utilization — Phase 3 onwards

Track peak VRAM per phase: relevant when comparing fine-tuned vs
base models (a fine-tune that doesn't change quality but bloats
VRAM is a real regression). Peak RSS and CUDA fragmentation are
secondary signals; matter at Phase 4 serving but not in offline eval.

### Throughput

`tokens/sec` (generation) is the most useful throughput metric for
LLM eval — captures inference speed independent of answer length.
Baseline: ~14.6 tok/s for Qwen 2.5 7B Instruct at NF4 on RTX 4070.
Track Δtok/s after Phase 3 fine-tuning — should be flat (LoRA
adapters add ~1-5% overhead, not more).

`queries/sec` matters at Phase 4 serving, not earlier.

### Error rate

Currently zero across Block 1B's 100+ judge calls (parse failures
tracked separately). For Phase 4 serving, error rate joins p99 as
a primary SLO. Categories worth distinguishing:

- Generator parse / format errors
- Judge parse failures (already tracked via `n_parse_failures`)
- Retrieval errors (FAISS search failure, BM25 index miss)
- Network / dependency errors (Phase 4 only)
- Timeouts (Phase 4 only)

### Anti-patterns to avoid

- **Reporting only mean.** Mean alone hides distribution shape; a
  bimodal distribution (warm vs cold start) reads identically to a
  unimodal one if you only see mean.
- **Reporting only max.** As above for the tail end. One slow event
  dominates.
- **Mixing wall time and CPU time.** Wall clock is what users
  experience; report it explicitly.
- **Aggregating across hardware classes.** A laptop CPU run and an
  RTX 4070 NF4 run are different distributions. Don't combine.

---

## Security and dependency hygiene — staged

The agreement is to document these now and implement them as their
context arrives, **not all at once and not as PR theater**.

| Item | Becomes important at | Notes |
|---|---|---|
| Secret scanning | Now (already covered) | GitHub-native scanning is on for public repos. **Don't add gitleaks if it duplicates this** — only add if a specific gap appears. |
| `pip-audit` | Phase 3 onwards | Add to `nightly.yml`, not every PR. Once dep set stabilizes after fine-tuning settles. |
| Upload / file safety | Phase 4 (when `/ingest`-style endpoint exists) | Size limits, MIME validation, sanitization. Pure boundary work. |
| Debug-data leakage gating | Phase 4 (when API exposes retrieved chunks / prompts) | `?debug=true` flag with default off. Prevents internal state leaking to normal users. |
| Docker image scanning | Phase 4 | Trivy or Grype, `security.yml`. Only meaningful once the Dockerfile exists. |
| API security (rate limit, auth) | Phase 4 if hosted publicly | Otherwise N/A. |
| Adversarial / prompt-injection corpus | Late Phase 3 or Phase 4 | A judged eval over a small committed adversarial set. Don't gate PRs on it. |

---

## What we explicitly *aren't* doing

The conservative bias, made visible:

- **Don't add `pip-audit` to PR CI** — nightly is fine, per-PR is noise.
- **Don't add `nbmake` for notebook smoke testing in Phase 2** — the
  bakeoff notebook is rendered manually after `run_bakeoff.py`
  produces the data. Automating it requires committing data files
  (gitignored) or reproducing them in CI.
- **Don't add gitleaks to CI** — GitHub-native secret scanning covers
  public repos.
- **Don't move retrieval / embedder / chunking config to YAML** —
  those aren't experiment knobs, they're project structure. YAML
  presets are for *experiment* knobs only.
- **Don't add property-based testing (hypothesis) in Phase 2** — bug
  surface is integration shape, not edge-case math.
- **Don't write end-to-end FastAPI tests now** — no FastAPI exists.
  Mock-only tests of an unwritten API are pure cargo.
- **Don't implement adversarial / prompt-injection eval before
  there's a real surface** — Phase 4 territory at the earliest.
- **Don't claim no-answer / faithfulness / answer-relevance / citation
  accuracy from mocked tests** — those are LLM-judged behavioral
  evals; mocked tests verify only contract shape.

---

## Resuming from this doc

When picking up Phase 2 work in a future session:

1. Read this file first.
2. Check `git log --oneline` for which Block A–E commits already
   landed; the last completed pause point is your starting line.
3. Look at the unstaged tree — there should be no work in progress
   from a prior session unless an in-flight commit was abandoned.
4. Confirm with the user before executing the next block. The
   expectation is **explicit approval per block**, not autonomy
   across the full plan.
5. Update this scope doc only if the user changes the agreed scope.
   Routine implementation does not modify it.

When closing Phase 2: update `CLAUDE.md` and `docs/architecture.md`
in a single commit (per the rule in `CLAUDE.md`'s Roadmap section)
and add a journal entry. This scope doc remains as-is until Phase 3
planning starts, at which point it gets a Phase 3 section appended
or is superseded by `docs/phase-3-scope.md`.
