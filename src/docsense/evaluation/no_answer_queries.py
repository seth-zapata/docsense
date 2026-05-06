"""Queries deliberately outside the HF Transformers docs domain.

Used by the LLM-judge eval driver to measure whether the system
refuses to answer when the corpus has no relevant content. A
well-behaved RAG pipeline should retrieve some semantically-adjacent
chunks (the embedder will always return its top-k neighbors, no
matter how off-topic the query) but the *generator* should recognize
that those chunks don't actually answer the question and respond
with a refusal phrase.

**Set size: 25** (expanded from 8 on 2026-05-06 alongside the
``judge_refusal`` LLM-judge introduction). The smaller n=8 set was
partly chosen because the regex-based ``check_no_answer_behavior``
heuristic required maintaining a refusal-phrase pattern set, so
each new query was a potential coverage-gap risk. The LLM-judge
removed that constraint, and a tighter confidence interval matters
more for Phase 3 Δ-comparisons (n=8 100% true rate could be
70-100%; n=25 100% pins it tighter, ~88-100%).

Domains covered (curated for breadth, not depth — refusal behavior
should generalize across topics, not be a per-domain signal):

- Cloud / pricing: AWS Lambda
- Tax / legal / regulatory: CA federal taxes, GDPR
- Consumer products: iPhone, MacBook
- Lifestyle: sourdough, mechanical keyboard cleaning, chocolate
  chip cookies, Tokyo travel timing, SF→LA route
- Real-time data: Lakers score, World Series winner, Bitcoin price,
  unemployment rate, stock prediction (hedging-prone)
- Different tech stacks: Postgres replication, nginx reverse proxy,
  MySQL JOIN syntax, React/Vercel deploy, JavaScript var/let
- Math / chemistry / facts: caffeine formula, compound interest
  formula, Reykjavik population
- Entertainment / culture: Inception director, 2025 box office

Some of these (caffeine formula, Inception director, MySQL syntax)
are things the generator may have in its parametric memory. They're
intentionally included — a well-behaved RAG should still refuse
when the *retrieved context* doesn't contain the answer, regardless
of what the model "knows" from training. The generator answering
"based on what I know" rather than "based on what was retrieved" is
exactly the failure mode we want refusal eval to catch.
"""

from __future__ import annotations

NO_ANSWER_QUERIES: list[str] = [
    # Cloud / pricing
    "What's the AWS Lambda free-tier limit per month?",
    # Tax / legal / regulatory
    "How do I file federal income taxes in California?",
    "What are the GDPR data-retention requirements for EU users?",
    # Consumer products
    "What's the latest iPhone model and its release date?",
    "Should I buy a MacBook Pro M4 or M3?",
    # Lifestyle / cooking / household
    "How do I make sourdough bread starter from scratch?",
    "How do I clean a mechanical keyboard?",
    "What's a good recipe for chocolate chip cookies?",
    "What's the best time of year to visit Tokyo?",
    "What's the best route to drive from San Francisco to Los Angeles?",
    # Real-time / current data
    "What's the current score of the Lakers game?",
    "Who won the 2026 World Series?",
    "What's the current price of Bitcoin?",
    "What's the latest US unemployment rate?",
    "Will the stock market go up next year?",
    # Different tech stacks (intentionally adjacent-but-out-of-corpus)
    "How do I configure Postgres streaming replication?",
    "How do I configure nginx as a reverse proxy?",
    "What's the syntax for a SELECT JOIN in MySQL?",
    "How do I deploy a React app to Vercel?",
    "What's the difference between let and var in JavaScript?",
    # Math / chemistry / facts (parametric-memory tempters)
    "What's the chemical formula of caffeine?",
    "What's the formula for compound interest?",
    "What's the population of Reykjavik?",
    # Entertainment / culture
    "Who directed the movie Inception?",
    "What was the highest-grossing movie of 2025?",
]
