"""SQLite-backed memory store: semantic triples + conversation summaries.

Triples and summaries are indexed with FTS5 so retrieval is a cheap BM25
lookup instead of re-reading raw conversation history.
"""

from __future__ import annotations

import functools
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS triples (
    id          INTEGER PRIMARY KEY,
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    mentions    INTEGER NOT NULL DEFAULT 1,
    UNIQUE (subject, predicate, object)
);
-- Porter stemming so "work" matches works_at and "moving" matches moving_to.
CREATE VIRTUAL TABLE IF NOT EXISTS triples_fts USING fts5(
    subject, predicate, object, content='triples', content_rowid='id',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS triples_ai AFTER INSERT ON triples BEGIN
    INSERT INTO triples_fts(rowid, subject, predicate, object)
    VALUES (new.id, new.subject, new.predicate, new.object);
END;
CREATE TRIGGER IF NOT EXISTS triples_ad AFTER DELETE ON triples BEGIN
    INSERT INTO triples_fts(triples_fts, rowid, subject, predicate, object)
    VALUES ('delete', old.id, old.subject, old.predicate, old.object);
END;

CREATE TABLE IF NOT EXISTS summaries (
    session_id  TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    turns       INTEGER NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
    session_id UNINDEXED, summary, tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);

-- One row per answered turn, keyed by the assistant message, so token stats
-- and recalled memory can be shown again after a reload.
CREATE TABLE IF NOT EXISTS turns (
    message_id          INTEGER PRIMARY KEY,
    session_id          TEXT NOT NULL,
    memory              TEXT NOT NULL,
    answer_input_tokens INTEGER NOT NULL,
    full_history_tokens INTEGER NOT NULL,
    extraction_tokens   INTEGER NOT NULL DEFAULT 0,
    added               TEXT NOT NULL DEFAULT '[]',
    removed             TEXT NOT NULL DEFAULT '[]',
    created_at          REAL NOT NULL
);
"""

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has
    have he her him his how i if in into is it its me my no not of on or our
    she so than that the their them then there these they this to us was we
    were what when where which who why will with would you your about tell
    know remember please""".split()
)


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str

    def render(self) -> str:
        return f"({self.subject} | {self.predicate} | {self.object})"

    def as_list(self) -> list[str]:
        return [self.subject, self.predicate, self.object]


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def fts_query(text: str) -> str | None:
    """Turn free text into an OR-joined FTS5 query of quoted keywords."""
    words = {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1}
    if not words:
        return None
    return " OR ".join(f'"{w}"' for w in sorted(words))


def _locked(method):
    """Serialize access: the web server calls the store from worker threads."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class MemoryStore:
    def __init__(self, path: str | Path = "memory.db"):
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # --- writes ---------------------------------------------------------

    @_locked
    def add_triples(self, triples: list[Triple], session_id: str) -> list[Triple]:
        """Upsert triples; duplicates bump `mentions` instead of adding rows. Returns the new ones."""
        now = time.time()
        added = []
        with self.db:
            for t in triples:
                s, p, o = _norm(t.subject), _norm(t.predicate), _norm(t.object)
                if not (s and p and o):
                    continue
                cur = self.db.execute(
                    "UPDATE triples SET mentions = mentions + 1, updated_at = ? "
                    "WHERE subject = ? AND predicate = ? AND object = ?",
                    (now, s, p, o),
                )
                if cur.rowcount == 0:
                    self.db.execute(
                        "INSERT INTO triples (subject, predicate, object, session_id, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (s, p, o, session_id, now, now),
                    )
                    added.append(Triple(s, p, o))
        return added

    @_locked
    def remove_triples(self, triples: list[Triple]) -> list[Triple]:
        """Delete facts that are no longer true. Returns the ones that existed."""
        removed = []
        with self.db:
            for t in triples:
                s, p, o = _norm(t.subject), _norm(t.predicate), _norm(t.object)
                cur = self.db.execute(
                    "DELETE FROM triples WHERE subject = ? AND predicate = ? AND object = ?", (s, p, o)
                )
                if cur.rowcount:
                    removed.append(Triple(s, p, o))
        return removed

    @_locked
    def set_summary(self, session_id: str, summary: str) -> None:
        now = time.time()
        with self.db:
            self.db.execute(
                "INSERT INTO summaries (session_id, summary, turns, updated_at) VALUES (?, ?, 1, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET summary = excluded.summary, "
                "turns = turns + 1, updated_at = excluded.updated_at",
                (session_id, summary, now),
            )
            self.db.execute("DELETE FROM summaries_fts WHERE session_id = ?", (session_id,))
            self.db.execute(
                "INSERT INTO summaries_fts (session_id, summary) VALUES (?, ?)", (session_id, summary)
            )

    @_locked
    def add_message(self, session_id: str, role: str, content: str) -> int:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time()),
            )
        return cur.lastrowid

    @_locked
    def save_turn(self, message_id: int, session_id: str, turn: dict) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO turns (message_id, session_id, memory, answer_input_tokens, "
                "full_history_tokens, extraction_tokens, added, removed, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    session_id,
                    json.dumps(turn["memory"]),
                    turn["answer_input_tokens"],
                    turn["full_history_tokens"],
                    turn["extraction_tokens"],
                    json.dumps(turn["added"]),
                    json.dumps(turn["removed"]),
                    time.time(),
                ),
            )

    # --- reads ----------------------------------------------------------

    @_locked
    def search_triples(self, query: str, limit: int = 20) -> list[Triple]:
        q = fts_query(query)
        if q is None:
            return []
        rows = self.db.execute(
            "SELECT t.subject, t.predicate, t.object FROM triples_fts f "
            "JOIN triples t ON t.id = f.rowid WHERE triples_fts MATCH ? "
            "ORDER BY bm25(triples_fts), t.mentions DESC, t.updated_at DESC LIMIT ?",
            (q, limit),
        ).fetchall()
        return [Triple(*r) for r in rows]

    @_locked
    def search_summaries(self, query: str, exclude_session: str | None = None, limit: int = 3) -> list[str]:
        q = fts_query(query)
        if q is None:
            return []
        rows = self.db.execute(
            "SELECT session_id, summary FROM summaries_fts WHERE summaries_fts MATCH ? "
            "ORDER BY bm25(summaries_fts) LIMIT ?",
            (q, limit + 1),
        ).fetchall()
        return [r["summary"] for r in rows if r["session_id"] != exclude_session][:limit]

    @_locked
    def get_summary(self, session_id: str) -> str | None:
        row = self.db.execute("SELECT summary FROM summaries WHERE session_id = ?", (session_id,)).fetchone()
        return row["summary"] if row else None

    @_locked
    def recent_messages(self, session_id: str, n: int) -> list[dict]:
        if n <= 0:
            return []
        rows = self.db.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, n),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    @_locked
    def all_messages(self, session_id: str | None = None) -> list[dict]:
        sql = "SELECT role, content FROM messages"
        args: tuple = ()
        if session_id:
            sql += " WHERE session_id = ?"
            args = (session_id,)
        rows = self.db.execute(sql + " ORDER BY id", args).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    @_locked
    def session_messages(self, session_id: str) -> list[dict]:
        """Messages for display, with each assistant message's turn stats attached."""
        rows = self.db.execute(
            "SELECT m.id, m.role, m.content, m.created_at, t.memory, t.answer_input_tokens, "
            "t.full_history_tokens, t.extraction_tokens, t.added, t.removed "
            "FROM messages m LEFT JOIN turns t ON t.message_id = m.id "
            "WHERE m.session_id = ? ORDER BY m.id",
            (session_id,),
        ).fetchall()
        out = []
        for r in rows:
            msg = {"id": r["id"], "role": r["role"], "content": r["content"], "created_at": r["created_at"]}
            if r["memory"] is not None:
                msg["turn"] = {
                    "memory": json.loads(r["memory"]),
                    "answer_input_tokens": r["answer_input_tokens"],
                    "full_history_tokens": r["full_history_tokens"],
                    "extraction_tokens": r["extraction_tokens"],
                    "added": json.loads(r["added"]),
                    "removed": json.loads(r["removed"]),
                }
            out.append(msg)
        return out

    @_locked
    def sessions(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT m.session_id, MAX(m.created_at) AS updated_at, "
            "SUM(m.role = 'user') AS turns, s.summary, "
            "(SELECT content FROM messages f WHERE f.session_id = m.session_id AND f.role = 'user' "
            " ORDER BY f.id LIMIT 1) AS first_message "
            "FROM messages m LEFT JOIN summaries s ON s.session_id = m.session_id "
            "GROUP BY m.session_id ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    @_locked
    def all_triples(self) -> list[Triple]:
        rows = self.db.execute("SELECT subject, predicate, object FROM triples ORDER BY subject, predicate").fetchall()
        return [Triple(*r) for r in rows]

    @_locked
    def facts(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT subject, predicate, object, session_id, mentions, updated_at FROM triples "
            "ORDER BY subject, predicate, object"
        ).fetchall()
        return [dict(r) for r in rows]

    @_locked
    def stats(self) -> dict:
        one = lambda sql: self.db.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "triples": one("SELECT COUNT(*) FROM triples"),
            "sessions": one("SELECT COUNT(DISTINCT session_id) FROM messages"),
            "messages": one("SELECT COUNT(*) FROM messages"),
            "answer_input_tokens": one("SELECT COALESCE(SUM(answer_input_tokens), 0) FROM turns"),
            "full_history_tokens": one("SELECT COALESCE(SUM(full_history_tokens), 0) FROM turns"),
            "extraction_tokens": one("SELECT COALESCE(SUM(extraction_tokens), 0) FROM turns"),
        }
