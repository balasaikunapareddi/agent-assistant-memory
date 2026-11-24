"""Session management + working-context window.

A session is one conversation: its own transcript (turns in the
store), its own working context, and a link to the session it was
handed off from. Memory is *not* per-session — it lives in the
durable store — so a fact learned in session A is available in
session B. That split (ephemeral transcript, durable memory) is
the whole point of the design.

The working context is the window the LLM actually sees: an
optional compacted summary, plus recent turns, inside a token
budget. When the window exceeds the budget, the oldest turns are
handed to the compactor instead of being silently dropped.
"""

from __future__ import annotations

import uuid

from .tokens import count_messages, estimate_tokens


class ContextWindow:
    """Ordered working context with a hard token budget."""

    def __init__(self, budget_tokens: int = 2000):
        if budget_tokens <= 0:
            raise ValueError("budget_tokens must be positive")
        self.budget_tokens = budget_tokens
        self.summary: str | None = None
        self.messages: list[dict] = []

    def tokens(self) -> int:
        total = count_messages(self.messages)
        if self.summary:
            total += estimate_tokens(self.summary)
        return total

    def over_budget(self) -> bool:
        return self.tokens() > self.budget_tokens

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def set_summary(self, summary: str) -> None:
        self.summary = summary

    def compactable_messages(self) -> list[dict]:
        """Oldest half of the messages — the compaction candidates.
        At least one message is returned when there are 2+; the
        most recent turn is never compacted away in the same pass."""
        if len(self.messages) < 2:
            return []
        return self.messages[: max(1, len(self.messages) // 2)]

    def drop_messages(self, messages: list[dict]) -> None:
        drop = {id(m) for m in messages}
        self.messages = [m for m in self.messages if id(m) not in drop]

    def as_messages(self) -> list[dict]:
        out = []
        if self.summary:
            out.append({"role": "system",
                        "content": f"Summary of earlier conversation in this session:\n{self.summary}"})
        return out + list(self.messages)


class SessionManager:
    """Create / reopen / hand off sessions; keep a live ContextWindow
    per open session."""

    def __init__(self, store, budget_tokens: int = 2000):
        self.store = store
        self.budget_tokens = budget_tokens
        self._windows: dict[str, ContextWindow] = {}

    def start_session(self, user_id: str, title: str | None = None) -> str:
        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        self.store.create_session(session_id, user_id, title=title)
        self._windows[session_id] = ContextWindow(self.budget_tokens)
        return session_id

    def open_session(self, session_id: str) -> ContextWindow:
        """Reopen an existing session, reloading its turns into a
        fresh working context."""
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(f"unknown session: {session_id}")
        window = ContextWindow(self.budget_tokens)
        for turn in self.store.get_turns(session_id):
            window.add(turn["role"], turn["content"])
        self._windows[session_id] = window
        return window

    def window(self, session_id: str) -> ContextWindow:
        if session_id not in self._windows:
            return self.open_session(session_id)
        return self._windows[session_id]

    def add_turn(self, session_id: str, user_id: str, role: str, content: str) -> int:
        turn_id = self.store.add_turn(session_id, user_id, role, content)
        self.window(session_id).add(role, content)
        return turn_id

    def handoff(self, user_id: str, from_session_id: str,
                summary: str, title: str | None = None) -> str:
        """Start a new session that continues the old one. The new
        session carries the handoff summary as its working-context
        summary, so the next session starts informed, not blank.
        The old session's transcript stays intact under its own id."""
        new_id = self.start_session(
            user_id, title=title or f"continued from {from_session_id}")
        self.store.create_session(new_id, user_id,
                                  title=title or f"continued from {from_session_id}",
                                  parent_session_id=from_session_id,
                                  handoff_summary=summary)
        # create_session used INSERT OR IGNORE, so update the handoff fields.
        with self.store._lock:
            self.store._conn.execute(
                "UPDATE sessions SET parent_session_id = ?, handoff_summary = ? "
                "WHERE id = ?", (from_session_id, summary, new_id))
            self.store._conn.commit()
        self._windows[new_id].set_summary(summary)
        return new_id

    def list_sessions(self, user_id: str) -> list[dict]:
        return self.store.list_sessions(user_id)
