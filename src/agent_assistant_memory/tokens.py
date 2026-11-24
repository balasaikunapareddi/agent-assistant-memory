"""Token counting.

A real tokenizer is not a dependency of this project. The estimate
``len(text) // 4`` (≈ 1 token per 4 characters, the usual English
rule of thumb) is used everywhere a budget is enforced, and every
report labels it as an estimate. The point of the budget is the
*mechanism* — compaction triggers, the window shrinks, counts are
reported before/after — not an exact tokenizer match.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Rough token count for a string (chars / 4, minimum 1)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def count_messages(messages: list[dict]) -> int:
    """Estimated tokens for a list of ``{"role": ..., "content": ...}``."""
    return sum(estimate_tokens(m.get("content", "")) for m in messages)
