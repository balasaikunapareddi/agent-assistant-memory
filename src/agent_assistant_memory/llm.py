"""LLM backends: live LiteLLM (Gemini by default) and a deterministic stub.

``gemini`` (default, live)
    LiteLLM chat completions; model from ``LLM_MODEL`` (default
    ``gemini/gemini-3.5-flash-lite``). Needs ``GEMINI_API_KEY`` (or the
    provider key for the chosen model) — the live test supplies it
    through the run_live.py harness, never committed.
``stub`` (offline tests and the one-command demo)
    Deterministic, no network. For compaction it echoes the turns
    back as a summary + durable facts, so compaction is verifiable
    offline. For answering it quotes the relevant memory injected
    in the context, so memory-grounding is testable without an LLM.
    Clearly labeled; never used for a live claim.
"""

from __future__ import annotations

import os
import re

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gemini/gemini-3.5-flash-lite")


class StubLLM:
    """Deterministic stand-in for offline tests and the demo."""

    name = "stub"

    def __init__(self, model: str = "stub"):
        self.model = model

    def reply(self, messages: list[dict], **kwargs) -> str:
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break
        low = user_text.lower()

        # --- compaction request: echo turns as summary + facts ---
        if "compacting an old conversation" in low or "durable facts:" in low and "turns:" in low:
            return self._stub_compaction(user_text)

        # --- memory-grounded answer: quote the injected memory ---
        memories = []
        for m in messages:
            content = m.get("content", "")
            if content.startswith("Relevant memories"):
                for line in content.splitlines():
                    if line.startswith("- ["):
                        memories.append(line[2:].strip())
        if memories:
            q_words = set(re.findall(r"[a-z0-9]+", low))
            best = None
            best_overlap = -1
            for mem in memories:
                overlap = len(q_words & set(re.findall(r"[a-z0-9]+", mem.lower())))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = mem
            if best is not None and best_overlap > 0:
                return f"Based on what I remember: {best}"
            return f"Based on what I remember: {memories[0]}"
        if not user_text.strip():
            return "I didn't catch that — could you say it again?"
        return f"You said: {user_text}. I don't have a stored memory about that."

    def _stub_compaction(self, prompt: str) -> str:
        # Turns are numbered "N. [role]: content" after the "Turns:" line.
        # Only USER turns carry durable facts; assistant turns here are
        # echoes of already-stored memories, not new facts.
        from .retrieval import STOPWORDS

        in_turns = False
        user_turns: list[str] = []
        for line in prompt.splitlines():
            if line.strip().lower().startswith("turns:"):
                in_turns = True
                continue
            if in_turns:
                m = re.match(r"\s*\d+\.\s*\[(user|assistant|system)\]:\s*(.*)", line)
                if m and m.group(1) == "user":
                    user_turns.append(m.group(2).strip())
        # Condensed summary: each user turn reduced to its content
        # words (stopwords dropped). Shorter than the original turns,
        # but every content word survives, so the summary is
        # verifiable against the facts it claims to cover.
        bits = []
        for t in user_turns:
            words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", t)
                     if w.lower() not in STOPWORDS]
            bits.append(" ".join(words))
        summary = "Summary: " + "; ".join(bits) + "." if bits \
            else "Summary: No substantive turns."
        out = [summary, "", "Durable facts:"]
        if user_turns:
            out += [f"- {t}" for t in user_turns]
        else:
            out.append("- (none)")
        return "\n".join(out)


class LiteLLMChat:
    """Live LLM via LiteLLM. Lazy import, provider-agnostic."""

    name = "gemini"

    def __init__(self, model: str | None = None, **kwargs):
        self.model = model or DEFAULT_MODEL
        kwargs.setdefault("timeout", 120)
        self.kwargs = kwargs

    def reply(self, messages: list[dict], **kwargs) -> str:
        import litellm
        resp = litellm.completion(
            model=self.model, messages=messages, **self.kwargs)
        try:
            return resp.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return str(resp)


def make_llm(name: str, **kwargs):
    """Factory: ``gemini`` (default, live via LiteLLM), ``stub``."""
    name = (name or "gemini").lower()
    if name in ("gemini", "litellm", "live"):
        return LiteLLMChat(**kwargs)
    if name == "stub":
        return StubLLM(**kwargs)
    raise ValueError(f"unknown LLM backend: {name!r} (gemini, stub)")
