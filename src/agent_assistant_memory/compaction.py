"""Compaction: summarize old turns into a short summary + durable facts.

When a session's working context exceeds its token budget, the
oldest turns are summarized (LLM) instead of dropped. The summary
replaces those turns in the window, and any durable facts in it
are written back into the memory store.

Verifiability: a compaction is only trusted if its summary can be
checked against the turns it covers. ``verify_summary`` measures
that check deterministically — for each durable fact, what
fraction of its content words appear in the summary — and the
result (facts preserved / total, coverage) is recorded in the
store alongside the token counts, so a later reader can audit
whether the compaction actually kept what mattered.
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompts import COMPACTION_PROMPT
from .retrieval import tokenize
from .tokens import count_messages, estimate_tokens


@dataclass
class CompactionResult:
    summary: str
    durable_facts: list[str]
    tokens_before: int
    tokens_after: int
    coverage: float
    facts_preserved: int
    facts_total: int


def parse_compaction_reply(reply: str) -> tuple[str, list[str]]:
    """Parse the LLM's fixed-format compaction reply into
    (summary, durable facts). Tolerates missing sections."""
    summary = ""
    facts: list[str] = []
    section = None
    for line in reply.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("summary:"):
            section = "summary"
            summary = stripped[len("Summary:"):].strip()
        elif stripped.lower().startswith("durable facts:"):
            section = "facts"
        elif stripped.lower().startswith("turns:"):
            section = "turns"
        elif section == "summary" and stripped:
            summary += " " + stripped
        elif section == "facts" and stripped.startswith("-"):
            fact = stripped[1:].strip()
            if fact and fact.lower() != "(none)":
                facts.append(fact)
    return summary.strip(), facts


def verify_summary(summary: str, facts: list[str]) -> tuple[int, int, float]:
    """Check each durable fact against the summary by content-word
    overlap. A fact counts as preserved when >= 50% of its content
    words appear in the summary. Returns (preserved, total, coverage).
    With no facts to check, coverage is 0.0 (unverified) — not a
    vacuous 1.0 — so a reply that extracted no facts can never
    masquerade as a perfect compaction."""
    summary_words = tokenize(summary)
    preserved = 0
    for fact in facts:
        words = tokenize(fact)
        if not words:
            continue
        overlap = len(words & summary_words) / len(words)
        if overlap >= 0.5:
            preserved += 1
    total = len([f for f in facts if tokenize(f)])
    coverage = (preserved / total) if total else 0.0
    return preserved, total, coverage


class Compactor:
    """LLM-backed compactor. ``llm`` is any object with
    ``reply(messages) -> str`` (live LiteLLM or the deterministic stub)."""

    def __init__(self, llm):
        self.llm = llm

    def compact(self, turns: list[dict], existing_summary: str | None = None) -> CompactionResult:
        """Summarize ``turns`` (each ``{"role", "content"}``).
        ``turns`` are the *original* turns, so the result is
        verifiable against them."""
        numbered = []
        for i, t in enumerate(turns, 1):
            numbered.append(f"{i}. [{t['role']}]: {t['content']}")
        prompt = COMPACTION_PROMPT.format(turns="\n".join(numbered))
        if existing_summary:
            prompt = (f"Previous summary of even earlier turns:\n{existing_summary}\n\n"
                      + prompt)
        reply = self.llm.reply([
            {"role": "system", "content": "You compact conversations. Follow the requested format exactly."},
            {"role": "user", "content": prompt},
        ])
        if not reply or not reply.strip():
            # Fail loudly: an empty reply must never silently drop the
            # turns it was meant to summarize. The caller keeps the
            # turns (still in the store and in the window) and can
            # retry; no compaction is recorded.
            raise ValueError(
                "compaction LLM returned an empty reply; turns left uncompacted")
        summary, facts = parse_compaction_reply(reply)
        if not summary:
            # LLM ignored the format: keep the raw reply as the summary
            # rather than losing the turns entirely.
            summary = reply.strip()
        tokens_before = count_messages(turns)
        tokens_after = estimate_tokens(summary)
        preserved, total, coverage = verify_summary(summary, facts)
        return CompactionResult(summary=summary, durable_facts=facts,
                                tokens_before=tokens_before,
                                tokens_after=tokens_after,
                                coverage=coverage,
                                facts_preserved=preserved, facts_total=total)
