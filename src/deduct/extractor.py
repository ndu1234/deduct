"""Advanced Augmentation: turn one dialogue exchange into triples + a rolling summary.

A single structured-output call per exchange produces:
  * new facts as (subject, predicate, object) triples
  * facts that are now false (so they can be retracted)
  * the session summary, updated to include this exchange
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel, Field

from .store import Triple

FALLBACK_BETA = "server-side-fallback-2026-07-01"

EXTRACTION_SYSTEM = """\
You maintain long-term memory for an AI assistant. Given the current session \
summary and the newest exchange, return:

1. `add`: durable facts as semantic triples (subject, predicate, object).
   - Use short canonical forms: subject "user" for the person talking; lowercase \
snake_case predicates such as works_at, prefers, lives_in, has_deadline, decided, uses.
   - One fact per triple; split compound statements.
   - Record preferences, identity, relationships, plans, decisions, dates and \
project details. Skip greetings, filler and anything the assistant merely speculated.
   - Resolve pronouns and relative dates using the summary when possible.
2. `remove`: previously stated facts the new exchange makes false (e.g. the user \
moved cities), written exactly as they were likely stored. Empty if none.
3. `summary`: the session summary rewritten to cover the whole session so far, in \
at most 120 words. Keep goals, decisions, open questions and the current topic; \
drop small talk."""


class TripleOut(BaseModel):
    subject: str
    predicate: str
    object: str


class Extraction(BaseModel):
    add: list[TripleOut] = Field(default_factory=list)
    remove: list[TripleOut] = Field(default_factory=list)
    summary: str


class Extractor:
    def __init__(self, client: anthropic.Anthropic, model: str = "claude-opus-5-5", effort: str = "low"):
        self.client = client
        self.model = model
        self.effort = effort

    def extract(self, prior_summary: str | None, user_msg: str, assistant_msg: str) -> tuple[Extraction, int]:
        """Returns the extraction and the tokens spent on it (input + output)."""
        prompt = (
            f"<session_summary>\n{prior_summary or '(new session)'}\n</session_summary>\n\n"
            f"<exchange>\nUSER: {user_msg}\n\nASSISTANT: {assistant_msg}\n</exchange>"
        )
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=4000,
            system=EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=Extraction,
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        spent = response.usage.input_tokens + response.usage.output_tokens
        if response.stop_reason == "refusal" or response.parsed_output is None:
            # Keep the old summary rather than losing memory on a declined/failed extraction.
            return Extraction(summary=prior_summary or ""), spent
        return response.parsed_output, spent


def to_triples(items: list[TripleOut]) -> list[Triple]:
    return [Triple(t.subject, t.predicate, t.object) for t in items]
