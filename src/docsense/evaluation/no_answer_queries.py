"""Queries deliberately outside the HF Transformers docs domain.

Used by the LLM-judge eval driver to measure whether the system
refuses to answer when the corpus has no relevant content. A
well-behaved RAG pipeline should retrieve some semantically-adjacent
chunks (the embedder will always return its top-k neighbors, no
matter how off-topic the query) but the *generator* should recognize
that those chunks don't actually answer the question and respond
with a refusal phrase.

Curated by hand to span domains that are clearly outside the HF
Transformers documentation:

- Pricing / billing for cloud services
- Legal / regulatory questions
- Consumer / lifestyle topics
- Database / infrastructure outside ML
- Real-time data (model can't have it)
- Personal / subjective questions

Keep this list small. The eval cost grows linearly with set size and
the signal saturates after ~10 queries — we just need to be confident
the refusal behavior triggers, not measure it to three decimal places.
"""

from __future__ import annotations

NO_ANSWER_QUERIES: list[str] = [
    "What's the AWS Lambda free-tier limit per month?",
    "How do I file federal income taxes in California?",
    "What's the latest iPhone model and its release date?",
    "How do I make sourdough bread starter from scratch?",
    "What's the current score of the Lakers game?",
    "How do I configure Postgres streaming replication?",
    "What are the GDPR data-retention requirements for EU users?",
    "What's the best route to drive from San Francisco to Los Angeles?",
]
