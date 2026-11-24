"""Durable memory store (SQLite).

This is the assistant-level memory: facts, preferences, decisions
and commitments about one user, each with provenance (where it came
from, when it was learned, confidence, salience) and a full
correction history.

Correction handling is the core rule of this module: a newer
correction SUPERSEDES an older claim. The old claim is never
deleted and never left active as a contradiction — it is marked
``replaced``, linked to the claim that replaced it
(``superseded_by``), and excluded from every retrieval. History is
preserved; only the newest claim on a subject is active.

Tables
------
claims
    One row per claim ever learned. ``subject`` is the normalized
    topic the claim is about (e.g. ``"home city"``); two active
    claims on the same subject cannot coexist.
sessions / turns
    Every session and every turn, so sessions can be reopened and
    compaction can be verified against the original turns.
compactions
    One row per compaction: token counts before/after, the summary,
    and which turns it covered, so a summary can be checked
    against the turns it claims to cover.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .tokens import estimate_tokens

KINDS = ("fact", "preference", "decision", "commitment")
STATUSES = ("active", "replaced", "forgotten")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_subject(subject: str) -> str:
    return " ".join(subject.strip().lower().split())


class MemoryStore:
    """SQLite-backed durable memory + sessions + turns, keyed by user."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS claims (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id          TEXT NOT NULL,
                subject          TEXT NOT NULL,
                kind             TEXT NOT NULL,
                text             TEXT NOT NULL,
                source_session_id TEXT,
                source_quote     TEXT,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL,
                confidence       REAL NOT NULL DEFAULT 1.0,
                salience         REAL NOT NULL DEFAULT 0.5,
                status           TEXT NOT NULL DEFAULT 'active',
                superseded_by    INTEGER,
                supersedes       INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_claims_user_subject
                ON claims (user_id, subject, status);
            CREATE TABLE IF NOT EXISTS sessions (
                id                TEXT PRIMARY KEY,
                user_id           TEXT NOT NULL,
                title             TEXT,
                created_at        TEXT NOT NULL,
                parent_session_id TEXT,
                handoff_summary   TEXT
            );
            CREATE TABLE IF NOT EXISTS turns (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_turns_session
                ON turns (session_id, id);
            CREATE TABLE IF NOT EXISTS compactions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                tokens_before   INTEGER NOT NULL,
                tokens_after    INTEGER NOT NULL,
                summary         TEXT NOT NULL,
                turn_ids        TEXT NOT NULL,
                facts_preserved INTEGER NOT NULL,
                facts_total     INTEGER NOT NULL
            );
            """
        )
        self._conn.commit()

    # -- claims ---------------------------------------------------------

    def remember(self, user_id: str, subject: str, text: str,
                 kind: str = "fact", source_session_id: str | None = None,
                 source_quote: str | None = None,
                 confidence: float = 1.0, salience: float = 0.5) -> dict:
        """Store a claim. If an active claim on the same subject
        already exists with different text, it is superseded
        (marked replaced, linked to the new claim). An identical
        re-statement returns the existing claim unchanged."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        subject = _norm_subject(subject)
        if not subject or not text.strip():
            raise ValueError("subject and text must be non-empty")
        if not 0.0 <= confidence <= 1.0 or not 0.0 <= salience <= 1.0:
            raise ValueError("confidence and salience must be in [0, 1]")
        text = text.strip()
        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM claims WHERE user_id = ? AND subject = ? "
                "AND status = 'active' ORDER BY id DESC LIMIT 1",
                (user_id, subject)).fetchone()
            if existing and existing["text"] == text:
                return dict(existing)
            now = _now()
            cur = self._conn.execute(
                "INSERT INTO claims (user_id, subject, kind, text, "
                "source_session_id, source_quote, created_at, updated_at, "
                "confidence, salience, status, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)",
                (user_id, subject, kind, text, source_session_id,
                 source_quote, now, now, confidence, salience,
                 existing["id"] if existing else None))
            new_id = cur.lastrowid
            if existing:
                self._conn.execute(
                    "UPDATE claims SET status = 'replaced', "
                    "superseded_by = ?, updated_at = ? WHERE id = ?",
                    (new_id, now, existing["id"]))
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM claims WHERE id = ?", (new_id,)).fetchone()
            return dict(row)

    def correct(self, user_id: str, subject: str, text: str, **kwargs) -> dict:
        """Explicit correction: same as remember(), but the intent is
        recorded in the call site. A correction always supersedes."""
        return self.remember(user_id, subject, text, **kwargs)

    def active_claims(self, user_id: str, kind: str | None = None) -> list[dict]:
        with self._lock:
            if kind is None:
                rows = self._conn.execute(
                    "SELECT * FROM claims WHERE user_id = ? AND status = 'active' "
                    "ORDER BY salience DESC, created_at DESC", (user_id,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM claims WHERE user_id = ? AND status = 'active' "
                    "AND kind = ? ORDER BY salience DESC, created_at DESC",
                    (user_id, kind)).fetchall()
        return [dict(r) for r in rows]

    def claim_history(self, user_id: str, subject: str) -> list[dict]:
        """Full history for a subject: active + replaced, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM claims WHERE user_id = ? AND subject = ? "
                "ORDER BY id", (user_id, _norm_subject(subject))).fetchall()
        return [dict(r) for r in rows]

    def get_active(self, user_id: str, subject: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM claims WHERE user_id = ? AND subject = ? "
                "AND status = 'active' ORDER BY id DESC LIMIT 1",
                (user_id, _norm_subject(subject))).fetchone()
        return dict(row) if row else None

    def forget(self, user_id: str, subject: str) -> bool:
        """Mark the active claim on a subject forgotten (history kept)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM claims WHERE user_id = ? AND subject = ? "
                "AND status = 'active' ORDER BY id DESC LIMIT 1",
                (user_id, _norm_subject(subject))).fetchone()
            if not row:
                return False
            self._conn.execute(
                "UPDATE claims SET status = 'forgotten', updated_at = ? "
                "WHERE id = ?", (_now(), row["id"]))
            self._conn.commit()
            return True

    def find_retired_claim_by_text(self, user_id: str, text: str) -> dict | None:
        """Most recent non-active claim with this exact text
        (case-insensitive), or None. Lets compaction write-back check
        whether a fact was deliberately retired (replaced or
        forgotten) so it is never resurrected as a live claim."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM claims WHERE user_id = ? "
                "AND text = ? COLLATE NOCASE AND status != 'active' "
                "ORDER BY id DESC LIMIT 1",
                (user_id, text.strip())).fetchone()
        return dict(row) if row else None

    # -- sessions / turns ------------------------------------------------

    def create_session(self, session_id: str, user_id: str, title: str | None = None,
                       parent_session_id: str | None = None,
                       handoff_summary: str | None = None) -> dict:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions "
                "(id, user_id, title, created_at, parent_session_id, handoff_summary) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, user_id, title, _now(), parent_session_id, handoff_summary))
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row)

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self, user_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at",
                (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def add_turn(self, session_id: str, user_id: str, role: str, content: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO turns (session_id, user_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, role, content, _now()))
            self._conn.commit()
            return cur.lastrowid

    def get_turns(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, role, content FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def session_tokens(self, session_id: str) -> int:
        return sum(estimate_tokens(t["content"]) for t in self.get_turns(session_id))

    # -- compactions ------------------------------------------------------

    def record_compaction(self, session_id: str, user_id: str, tokens_before: int,
                          tokens_after: int, summary: str, turn_ids: list[int],
                          facts_preserved: int, facts_total: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO compactions (session_id, user_id, created_at, "
                "tokens_before, tokens_after, summary, turn_ids, "
                "facts_preserved, facts_total) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, user_id, _now(), tokens_before, tokens_after,
                 summary, ",".join(str(i) for i in turn_ids),
                 facts_preserved, facts_total))
            self._conn.commit()
            return cur.lastrowid

    def list_compactions(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM compactions WHERE session_id = ? ORDER BY id",
                (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
