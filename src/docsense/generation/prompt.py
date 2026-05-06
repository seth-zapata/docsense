"""Prompt construction for the generation pipeline.

`PromptBuilder.build()` composes the full LLM input from three pieces:

1. A **system prompt** that sets the assistant's role and tells it how
   to answer (cite by `[N]`, refuse when context is insufficient).
2. The **assembled context** produced by `ContextAssembler` — already
   formatted as numbered chunks with source attribution.
3. The user's **query**.

The default template targets instruction-tuned chat models (Mistral
Instruct, Llama 3 Instruct) where the user-visible content is rendered
into the model's chat-format under the hood. Different LLMs may want
different templates; callers can pass overrides via the constructor.

Snapshot testing: the rendered prompt for a fixed input is checked
against a committed file (``tests/snapshots/prompt_default.txt``).
Accidental changes to the prompt — e.g., a stray refactor that drops
the citation directive — break the snapshot loudly. Intentional
changes update the snapshot in the same commit and become reviewable.
"""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "You are a technical documentation assistant. Answer the user's "
    "question using only the provided context. Cite sources using [N] "
    "notation that matches the numbered context items below.\n"
    "\n"
    "If the context does not contain enough information to answer the "
    'question, reply with: "I don\'t have enough context to answer that." '
    "Do not invent information that is not in the context."
)

DEFAULT_TEMPLATE = "{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"


class PromptBuilder:
    """Compose the full LLM input from system prompt, context, and query.

    The default template/system_prompt is calibrated for instruction-tuned
    chat models. Callers needing a different surface (e.g., raw completion,
    a different chat schema) can pass overrides at construction.
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        template: str = DEFAULT_TEMPLATE,
    ) -> None:
        self.system_prompt = system_prompt
        self.template = template

    def build(self, query: str, context: str) -> str:
        """Render the prompt with the given query and assembled context.

        ``context`` is expected to be the output of
        ``ContextAssembler.assemble()`` — already numbered with source
        attribution. ``query`` is the user's question, unmodified.
        """
        return self.template.format(
            system_prompt=self.system_prompt,
            context=context,
            query=query,
        )
