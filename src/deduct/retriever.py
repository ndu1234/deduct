"""Build a compact memory context for a query, within a token budget."""

from __future__ import annotations

from dataclasses import dataclass, field

from .store import MemoryStore, Triple


def estimate_tokens(text: str) -> int:
    """Cheap ~4 chars/token estimate; good enough for budgeting, not billing."""
    return max(1, len(text) // 4)


@dataclass
class MemoryContext:
    session_summary: str | None = None
    related_summaries: list[str] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)

    def render(self) -> str:
        parts = []
        if self.triples:
            parts.append("Known facts (subject | predicate | object):\n" + "\n".join(t.render() for t in self.triples))
        if self.session_summary:
            parts.append(f"This conversation so far:\n{self.session_summary}")
        if self.related_summaries:
            parts.append("Related past conversations:\n" + "\n".join(f"- {s}" for s in self.related_summaries))
        if not parts:
            return ""
        return "<memory>\n" + "\n\n".join(parts) + "\n</memory>"

    @property
    def tokens(self) -> int:
        text = self.render()
        return estimate_tokens(text) if text else 0

    def to_dict(self) -> dict:
        return {
            "session_summary": self.session_summary,
            "related_summaries": self.related_summaries,
            "triples": [t.as_list() for t in self.triples],
            "tokens": self.tokens,
        }


class Retriever:
    def __init__(self, store: MemoryStore, budget_tokens: int = 1500, max_triples: int = 30, max_summaries: int = 2):
        self.store = store
        self.budget = budget_tokens
        self.max_triples = max_triples
        self.max_summaries = max_summaries

    def build(self, query: str, session_id: str) -> MemoryContext:
        """Fill the budget in priority order: current summary, matching triples, related summaries."""
        # Measure the rendered block (headers and wrapper included) after each addition;
        # candidate lists are capped, so re-rendering is cheap.
        ctx = MemoryContext(session_summary=self.store.get_summary(session_id))
        if ctx.tokens > self.budget:
            ctx.session_summary = None

        for t in self.store.search_triples(query, limit=self.max_triples):
            ctx.triples.append(t)
            if ctx.tokens > self.budget:
                ctx.triples.pop()
                break

        for s in self.store.search_summaries(query, exclude_session=session_id, limit=self.max_summaries):
            ctx.related_summaries.append(s)
            if ctx.tokens > self.budget:
                ctx.related_summaries.pop()
                break

        return ctx
