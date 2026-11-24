"""Retrieval tests: ranking by keyword + recency + salience."""

import pytest

from agent_assistant_memory.retrieval import format_memories, score_claim, search_memory, tokenize
from agent_assistant_memory.store import MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    s.remember("u1", "home city", "Alex lives in Dallas, Texas", salience=0.9)
    s.remember("u1", "reply style", "Alex prefers short, plain replies",
               kind="preference", salience=0.8)
    s.remember("u1", "favourite food", "Alex loves tacos with hot sauce", salience=0.5)
    s.remember("u1", "project deadline", "Alex will ship the garden-planner app by Friday",
               kind="commitment", salience=0.8)
    yield s
    s.close()


def test_tokenize_removes_stopwords_and_stems():
    assert "live" in tokenize("Alex lives in Dallas")
    assert "the" not in tokenize("the Dallas house")
    assert tokenize("what is the time") == {"time"}


def test_keyword_match_ranks_first(store):
    hits = search_memory(store, "u1", "Where do I live?")
    assert hits[0]["subject"] == "home city"
    assert "Dallas" in hits[0]["text"]


def test_unrelated_query_returns_nothing(store):
    assert search_memory(store, "u1", "quantum entanglement") == []


def test_replaced_claim_never_retrieved(store):
    store.remember("u1", "home city", "Alex lives in Austin, Texas")
    store.correct("u1", "home city", "Alex lives in Dallas, Texas")
    hits = search_memory(store, "u1", "where do I live Austin Dallas")
    assert all("Austin" not in h["text"] for h in hits)
    assert hits[0]["text"].startswith("Alex lives in Dallas")


def test_query_matching_only_superseded_claim_returns_nothing(store):
    """A query whose only keyword match is the retired wording must
    return nothing — the replaced claim is history, not an answer."""
    store.remember("u1", "home city", "Alex lives in Austin, Texas")
    store.correct("u1", "home city", "Alex lives in Dallas, Texas")
    assert search_memory(store, "u1", "Austin") == []


def test_forgotten_claim_never_retrieved(store):
    store.forget("u1", "favourite food")
    assert search_memory(store, "u1", "what food do I love tacos") == []


def test_salience_breaks_keyword_tie(store):
    # Both claims mention Dallas; the higher-salience one should win.
    store.remember("u1", "dallas weather", "Dallas weather is hot in summer", salience=0.3)
    hits = search_memory(store, "u1", "Dallas")
    assert hits[0]["subject"] == "home city"


def test_top_k_is_respected(store):
    hits = search_memory(store, "u1", "Alex", top_k=2)
    assert len(hits) == 2


def test_kind_filter(store):
    hits = search_memory(store, "u1", "Alex", kind="preference")
    assert all(h["kind"] == "preference" for h in hits)
    assert len(hits) == 1


def test_score_components_are_reported(store):
    hits = search_memory(store, "u1", "Where do I live?")
    hit = hits[0]
    assert 0.0 <= hit["score"] <= 1.0
    assert hit["score_parts"]["keyword"] > 0
    assert hit["score_parts"]["recency"] > 0.9  # just learned


def test_empty_query_ranks_by_salience_and_recency(store):
    hits = search_memory(store, "u1", "")
    assert hits[0]["subject"] == "home city"  # highest salience


def test_format_memories_lists_active_only(store):
    hits = search_memory(store, "u1", "Alex", top_k=10)
    text = format_memories(hits)
    assert "Dallas" in text and "home city" in text
    assert "confidence" in text
