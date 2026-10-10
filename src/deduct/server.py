"""Web UI + JSON API for deduct.

    deduct serve [--port 8000]

POST /api/sessions/{id}/chat streams Server-Sent Events:
    memory  -> recalled memory block for this turn (sent before the model call)
    delta   -> answer text as it streams
    answer  -> final answer + token stats
    saved   -> facts added/removed and extraction cost
    error   -> {message}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import MemoryAgent
from .store import MemoryStore, Triple

WEB_DIR = Path(__file__).parent / "web"
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ChatIn(BaseModel):
    message: str


class FactIn(BaseModel):
    subject: str
    predicate: str
    object: str
    session_id: str = "manual"


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def error_message(exc: Exception) -> str:
    if isinstance(exc, anthropic.AuthenticationError):
        return "Anthropic rejected the credentials. Check ANTHROPIC_API_KEY and restart the server."
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited by the Anthropic API. Wait a moment and try again."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Couldn't reach the Anthropic API. Check your network connection."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Anthropic API error {exc.status_code}: {exc.message}"
    return f"{type(exc).__name__}: {exc}"


def create_app(
    db_path: str = "memory.db",
    model: str = "claude-opus-5-5",
    effort: str = "medium",
    memory_budget: int = 1500,
    client: anthropic.Anthropic | None = None,
) -> FastAPI:
    store = MemoryStore(db_path)
    setup_error = None
    if client is None:
        try:
            client = anthropic.Anthropic()
        except Exception as exc:
            setup_error = str(exc)
        else:
            # The client constructs even with nothing configured; it only fails on the first request.
            has_profile = (Path.home() / ".config" / "anthropic").exists()
            if not (client.api_key or getattr(client, "auth_token", None) or has_profile):
                setup_error = "set ANTHROPIC_API_KEY (or run `ant auth login`) and restart"

    app = FastAPI(title="deduct")

    def check_session(session_id: str) -> None:
        if not SESSION_ID.match(session_id):
            raise HTTPException(400, "invalid session id")

    @app.get("/api/health")
    def health():
        return {"ok": setup_error is None, "error": setup_error, "model": model, "budget": memory_budget}

    @app.get("/api/sessions")
    def sessions():
        return store.sessions()

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str):
        check_session(session_id)
        return {"messages": store.session_messages(session_id), "summary": store.get_summary(session_id)}

    @app.get("/api/facts")
    def facts():
        return store.facts()

    @app.post("/api/facts")
    def add_fact(fact: FactIn):
        added = store.add_triples([Triple(fact.subject, fact.predicate, fact.object)], fact.session_id)
        return {"added": [t.as_list() for t in added]}

    @app.delete("/api/facts")
    def delete_fact(fact: FactIn):
        removed = store.remove_triples([Triple(fact.subject, fact.predicate, fact.object)])
        if not removed:
            raise HTTPException(404, "fact not found")
        return {"removed": [t.as_list() for t in removed]}

    @app.get("/api/stats")
    def stats():
        return store.stats()

    @app.post("/api/sessions/{session_id}/chat")
    def chat(session_id: str, body: ChatIn):
        check_session(session_id)
        message = body.message.strip()
        if not message:
            raise HTTPException(400, "empty message")
        if setup_error:
            raise HTTPException(503, f"No Anthropic credentials: {setup_error}")

        agent = MemoryAgent(
            store, client=client, session_id=session_id, model=model, effort=effort, memory_budget=memory_budget
        )

        def events():
            try:
                turn = agent.prepare(message)
                yield sse("memory", turn.memory.to_dict())

                with client.beta.messages.stream(**agent.request(turn)) as stream:
                    for text in stream.text_stream:
                        yield sse("delta", {"text": text})
                    final = stream.get_final_message()

                report = agent.record(turn, final)
                yield sse("answer", {"message_id": report.message_id, "content": report.answer, **report.to_dict()})
            except Exception as exc:
                yield sse("error", {"message": error_message(exc)})
                return

            try:
                agent.augment(turn, report)
                yield sse("saved", {"message_id": report.message_id, **report.to_dict()})
            except Exception as exc:
                yield sse("error", {"message": f"Answer saved, but memory extraction failed. {error_message(exc)}"})

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
