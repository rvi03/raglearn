"""History-aware query rewrite.

Turns a context-dependent follow-up ("what about its net income?") into a
standalone question ("what was Apple's net income?") by resolving references
against the conversation so far. A standalone query routes and retrieves far
better than one that leans on prior turns.

No history → no rewrite (and no LLM call): the first turn of a conversation, and
every single-shot ``/query``, passes through untouched. The rewrite only earns
its LLM call once a chat actually accumulates history.
"""

from __future__ import annotations

from finrag.core.interfaces.generation import LLMBackend
from finrag.core.registry import registry
from finrag.core.types import Query


def _rewrite_prompt(query: Query) -> str:
    """Build the prompt that rewrites a follow-up into a standalone question."""
    conversation = "\n".join(query.history)
    return (
        "Rewrite the FOLLOW-UP as a standalone question that needs no prior "
        "context: resolve pronouns and references to entities mentioned earlier. "
        "Output only the rewritten question, nothing else.\n\n"
        f"CONVERSATION:\n{conversation}\n\n"
        f"FOLLOW-UP: {query.text}\n\n"
        "STANDALONE QUESTION:"
    )


@registry.register("query_transform", "rewrite")
class RewriteTransform:
    """A :class:`~finrag.core.interfaces.QueryTransform` doing coref rewrite."""

    def __init__(self, llm: LLMBackend) -> None:
        """Bind the rewriter to the LLM it uses to resolve references."""
        self._llm = llm

    def transform(self, query: Query) -> Query:
        """Return the query rewritten to stand alone, or unchanged if no history.

        Args:
          query: The user question and its conversation history.

        Returns:
          A query whose ``text`` is the standalone rewrite; the original query
          when there is no history or the rewrite comes back empty.
        """
        if not query.history:
            return query
        rewritten = self._llm.generate(_rewrite_prompt(query)).text.strip()
        if not rewritten:
            return query
        return query.model_copy(update={"text": rewritten})
