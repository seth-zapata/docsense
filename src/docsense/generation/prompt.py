"""Prompt construction for the generation pipeline.

`PromptBuilder.build()` composes the LLM input as a structured list of
**chat messages** — `[{"role": "system", ...}, {"role": "user", ...}]`
— rather than a flat completion-style string. The downstream
``Generator`` invokes ``tokenizer.apply_chat_template(messages, ...)``,
which produces the model-specific chat formatting (Qwen's
``<|im_start|>``, Llama's ``<|begin_of_text|>``, Mistral's ``[INST]``)
under the hood. This makes the Generator model-agnostic — same code
path for every Instruct model.

The system message sets the assistant's role and tells it how to
answer (cite by `[N]`, refuse when context is insufficient). The user
message wraps the assembled context and the user's question.

Snapshot testing: the messages structure for a fixed input is checked
against ``tests/snapshots/prompt_default.json``. Accidental changes to
the system prompt or user template — e.g., a stray refactor that drops
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

# Template for the *user* message body only. The "Answer:" suffix from
# the prior completion-style template is not needed — the tokenizer's
# ``apply_chat_template(..., add_generation_prompt=True)`` adds the
# model-specific assistant-turn marker automatically.
DEFAULT_USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {query}"


class PromptBuilder:
    """Compose chat messages for the generation pipeline.

    The default ``user_template`` and ``system_prompt`` are calibrated
    for technical-documentation Q&A with citation grounding. Callers
    needing a different surface (e.g., a different role split, a
    different citation directive) can pass overrides at construction.

    The returned messages list is independent of which LLM will consume
    it; model-specific chat formatting is applied later by
    ``tokenizer.apply_chat_template`` inside the Generator.
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_template: str = DEFAULT_USER_TEMPLATE,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_template = user_template

    def build(self, query: str, context: str) -> list[dict[str, str]]:
        """Build the chat-message list for one query.

        ``context`` is expected to be the output of
        ``ContextAssembler.assemble()`` — already numbered with source
        attribution. ``query`` is the user's question, unmodified.

        Returns ``[{"role": "system", ...}, {"role": "user", ...}]``,
        ready to hand to ``Generator.generate(messages, ...)``.
        """
        user_content = self.user_template.format(context=context, query=query)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
