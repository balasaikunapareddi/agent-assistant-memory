"""Retrieval: search durable memory by query, with scoring.

Score = keyword match + recency + salience. Only ``active`` claims
are ever returned — replaced and forgotten claims are history, not
answers, so a correction can never lose to the fact it replaced.

Keyword match: fraction of the query's content words that appear
in the claim text or subject (stopwords removed). Recency: newer
claims score higher, decaying over ~30 days. Salience: the
importance the claim was stored with. Weights are explicit and
reported per hit, so a ranking can be explained, not just trusted.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

STOPWORDS = frozenset("""
a an the and or of to in on for with is are was were be been it its this that
what where when who how do does did my your his her their our i you he she we they
me him us them am is are was were will would can could should may might
""".split())

W_KEYWORD = 0.6
W_RECENCY = 0.15
W_SALIENCE = 0.25


def _stem(word: str) -> str:
    """Light stemming so 'live' matches 'lives', 'prefers' matches
    'prefer', etc. Only suffix-stripping, no dictionary."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {_stem(w) for w in words if w not in STOPWORDS and len(w) > 1}


def _recency_score(created_at: str, now: datetime | None = None) -> float:
    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.5
    now = now or datetime.now(timezone.utc)
    age_days = max(0.0, (now - created).total_seconds() / 86400.0)
    # 1.0 today, ~0.5 after 30 days, ~0.25 after 90 days.
    return 1.0 / (1.0 + age_days / 30.0)


def score_claim(query: str, claim: dict, now: datetime | None = None) -> dict:
    """Score one active claim against a query. Returns components."""
    q_words = tokenize(query)
    claim_words = tokenize(claim["text"] + " " + claim["subject"])
    if q_words:
        keyword = len(q_words & claim_words) / len(q_words)
    else:
        keyword = 0.0
    recency = _recency_score(claim["created_at"], now)
    salience = float(claim.get("salience", 0.5))
    total = W_KEYWORD * keyword + W_RECENCY * recency + W_SALIENCE * salience
    return {"keyword": keyword, "recency": recency, "salience": salience,
            "total": total}


def search_memory(store, user_id: str, query: str, top_k: int = 5,
                  kind: str | None = None) -> list[dict]:
    """Ranked active claims for a query. Hits with no keyword overlap
    are dropped unless the query is empty (then salience/recency rank)."""
    claims = store.active_claims(user_id, kind=kind)
    scored = []
    for claim in claims:
        s = score_claim(query, claim)
        if query.strip() and s["keyword"] == 0.0:
            continue
        hit = dict(claim)
        hit["score"] = s["total"]
        hit["score_parts"] = s
        scored.append(hit)
    scored.sort(key=lambda h: (-h["score"], h["subject"]))
    return scored[:top_k]


def format_memories(hits: list[dict]) -> str:
    """One line per hit, for injection into a session's context."""
    lines = []
    for h in hits:
        lines.append(f"- [{h['kind']}] {h['subject']}: {h['text']} "
                     f"(confidence {h['confidence']:.2f}, "
                     f"learned {h['created_at'][:10]})")
    return "\n".join(lines)
