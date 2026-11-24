"""AssistantAgent — the visible agent class.

Owns the durable memory store, the session manager, the retriever
and the compactor, and wires them into one loop:

1. Retrieve the top relevant active memories for the user's message.
2. Inject them into the session's working context.
3. Ask the LLM (live Gemini via LiteLLM, or the deterministic stub).
4. Store both turns, and compact the window when it exceeds budget
   — writing any durable facts from the compaction back to memory.

A fact learned in one session is therefore available in the next:
retrieval reads the store, not the session transcript.
"""

from __future__ import annotations

import re

from .compaction import Compactor
from .llm import make_llm
from .prompts import MEMORY_PREAMBLE_TEMPLATE, SYSTEM_PROMPT
from .retrieval import format_memories, search_memory
from .sessions import SessionManager
from .store import MemoryStore
from .tokens import estimate_tokens


def _subject_from_fact(fact: str) -> str:
    """Derive a subject for a compaction-extracted fact, e.g.
    'Alex lives in Dallas, Texas' -> 'home city' is unknowable in
    general, so use the text before ' is '/' lives in ' / ' prefers '
    as the subject, else the first three words."""
    low = fact.lower()
    for sep in (" lives in ", " is ", " prefers ", " likes ", " works as "):
        if sep in low:
            head = fact[: low.index(sep)].strip()
            if head:
                return head.lower()
    words = fact.split()
    return " ".join(words[:3]).lower() if words else "general"


class AssistantAgent:
    """An AI assistant with durable memory and session management."""

    def __init__(self, store: MemoryStore, llm=None, budget_tokens: int = 2000,
                 user_id: str = "default-user"):
        self.store = store
        self.user_id = user_id
        self.llm = llm or make_llm("stub")
        self.llm_name = getattr(self.llm, "name", type(self.llm).__name__)
        self.sessions = SessionManager(store, budget_tokens=budget_tokens)
        self.compactor = Compactor(self.llm)
        self.budget_tokens = budget_tokens
        self.last_chat: dict = {}

    # -- durable memory (explicit, user-driven) -------------------------

    def learn(self, subject: str, text: str, kind: str = "fact",
              session_id: str | None = None, **kwargs) -> dict:
        return self.store.remember(self.user_id, subject, text, kind=kind,
                                   source_session_id=session_id, **kwargs)

    def correct(self, subject: str, text: str, session_id: str | None = None,
                **kwargs) -> dict:
        """A correction supersedes the old claim on that subject."""
        return self.store.correct(self.user_id, subject, text,
                                  source_session_id=session_id, **kwargs)

    def forget(self, subject: str) -> bool:
        return self.store.forget(self.user_id, subject)

    def recall(self, query: str, top_k: int = 5, kind: str | None = None) -> list[dict]:
        return search_memory(self.store, self.user_id, query, top_k=top_k, kind=kind)

    # -- sessions --------------------------------------------------------

    def start_session(self, title: str | None = None) -> str:
        return self.sessions.start_session(self.user_id, title=title)

    def handoff(self, from_session_id: str, summary: str, title: str | None = None) -> str:
        return self.sessions.handoff(self.user_id, from_session_id, summary, title=title)

    # -- chat -------------------------------------------------------------

    def chat(self, session_id: str, user_text: str, top_k: int = 3) -> str:
        """One turn: retrieve memories, answer, store turns, compact
        if the working context is over budget. Details of the turn
        (memories used, tokens, whether compaction ran) are kept in
        ``self.last_chat`` for the demo / live test to report."""
        window = self.sessions.window(session_id)
        hits = self.recall(user_text, top_k=top_k)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if hits:
            messages.append({"role": "system",
                             "content": MEMORY_PREAMBLE_TEMPLATE.format(
                                 memories=format_memories(hits))})
        messages += window.as_messages()
        messages.append({"role": "user", "content": user_text})

        reply = self.llm.reply(messages)
        tokens_before = window.tokens()
        self.sessions.add_turn(session_id, self.user_id, "user", user_text)
        self.sessions.add_turn(session_id, self.user_id, "assistant", reply)

        compacted = None
        if window.over_budget():
            compacted = self._compact_window(session_id, window)

        self.last_chat = {
            "session_id": session_id,
            "memories_used": [h["subject"] for h in hits],
            "memory_hits": hits,
            "reply": reply,
            "tokens_before_compaction": tokens_before,
            "tokens_after": window.tokens(),
            "compacted": compacted is not None,
            "compaction": compacted,
        }
        return reply

    def _compact_window(self, session_id: str, window) -> dict:
        """Compact the oldest half of the window; persist the facts
        the compaction surfaced and record the compaction."""
        candidates = window.compactable_messages()
        if not candidates:
            return {}
        # Map window messages back to stored turn ids: the window
        # holds turns in store order, minus turns already compacted
        # away in earlier passes (recorded in the store).
        stored = self.store.get_turns(session_id)
        already = set()
        for rec in self.store.list_compactions(session_id):
            already.update(int(x) for x in rec["turn_ids"].split(",") if x.strip())
        remaining = [t for t in stored if t["id"] not in already]
        n = len(candidates)
        compacted_turns = remaining[:n] if len(remaining) >= n else remaining
        turn_ids = [t["id"] for t in compacted_turns]
        result = self.compactor.compact(candidates, existing_summary=window.summary)
        window.drop_messages(candidates)
        window.set_summary(result.summary)
        # Persist durable facts from the compaction (dedupe by subject+text
        # is handled in the store: identical re-statements are no-ops).
        stored_facts = []
        for fact in result.durable_facts:
            subject = _subject_from_fact(fact)
            # Skip facts that merely restate an already-active claim.
            existing = self.store.get_active(self.user_id, subject)
            if existing and existing["text"].strip().lower() == fact.strip().lower():
                continue
            claim = self.store.remember(
                self.user_id, subject, fact, kind="fact",
                source_session_id=session_id, salience=0.6)
            stored_facts.append(claim)
        self.store.record_compaction(
            session_id, self.user_id, result.tokens_before, result.tokens_after,
            result.summary, turn_ids, result.facts_preserved, result.facts_total)
        return {"summary": result.summary, "facts": result.durable_facts,
                "stored_facts": len(stored_facts),
                "tokens_before": result.tokens_before,
                "tokens_after": result.tokens_after,
                "coverage": result.coverage,
                "facts_preserved": result.facts_preserved,
                "facts_total": result.facts_total,
                "turn_ids": turn_ids}
