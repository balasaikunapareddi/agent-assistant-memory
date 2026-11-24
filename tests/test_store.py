"""Store tests: remember / supersede / history / forget / provenance."""

import pytest

from agent_assistant_memory.store import MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_remember_and_get_active(store):
    c = store.remember("u1", "home city", "Alex lives in Dallas, Texas")
    assert c["status"] == "active"
    assert c["subject"] == "home city"
    assert store.get_active("u1", "home city")["text"] == "Alex lives in Dallas, Texas"


def test_remember_records_provenance(store):
    c = store.remember("u1", "name", "Alex Rivera", source_session_id="sess-1",
                       source_quote="I'm Alex", confidence=0.9, salience=0.8,
                       kind="fact")
    assert c["source_session_id"] == "sess-1"
    assert c["source_quote"] == "I'm Alex"
    assert c["confidence"] == 0.9
    assert c["salience"] == 0.8
    assert c["created_at"] and c["updated_at"]


def test_identical_restatement_is_noop(store):
    a = store.remember("u1", "name", "Alex Rivera")
    b = store.remember("u1", "name", "Alex Rivera")
    assert a["id"] == b["id"]
    assert len(store.claim_history("u1", "name")) == 1


def test_correction_supersedes_old_claim(store):
    old = store.remember("u1", "home city", "Alex lives in Austin, Texas")
    new = store.correct("u1", "home city", "Alex lives in Dallas, Texas")
    assert new["status"] == "active"
    assert new["supersedes"] == old["id"]
    history = store.claim_history("u1", "home city")
    assert len(history) == 2
    assert history[0]["status"] == "replaced"
    assert history[0]["superseded_by"] == new["id"]
    assert history[1]["status"] == "active"
    assert store.get_active("u1", "home city")["id"] == new["id"]


def test_only_one_active_claim_per_subject(store):
    store.remember("u1", "home city", "Austin")
    store.remember("u1", "home city", "Dallas")
    store.remember("u1", "home city", "Houston")
    active = store.active_claims("u1")
    assert [c for c in active if c["subject"] == "home city"][0]["text"] == "Houston"
    assert len(store.claim_history("u1", "home city")) == 3


def test_subject_is_normalized(store):
    store.remember("u1", "  Home   City ", "Dallas")
    assert store.get_active("u1", "home city") is not None
    assert store.get_active("u1", "HOME CITY") is not None


def test_forget_keeps_history_but_removes_from_active(store):
    store.remember("u1", "name", "Alex Rivera")
    assert store.forget("u1", "name") is True
    assert store.get_active("u1", "name") is None
    assert store.active_claims("u1") == []
    assert len(store.claim_history("u1", "name")) == 1
    assert store.claim_history("u1", "name")[0]["status"] == "forgotten"
    assert store.forget("u1", "name") is False


def test_kinds_are_stored(store):
    store.remember("u1", "reply style", "short replies", kind="preference")
    store.remember("u1", "deadline", "ship by Friday", kind="commitment")
    store.remember("u1", "choice", "chose Postgres", kind="decision")
    assert len(store.active_claims("u1", kind="preference")) == 1
    assert len(store.active_claims("u1", kind="commitment")) == 1
    assert len(store.active_claims("u1", kind="decision")) == 1


def test_invalid_kind_and_empty_text_rejected(store):
    with pytest.raises(ValueError):
        store.remember("u1", "x", "y", kind="rumour")
    with pytest.raises(ValueError):
        store.remember("u1", "x", "   ")
    with pytest.raises(ValueError):
        store.remember("u1", "x", "y", confidence=1.5)


def test_users_are_isolated(store):
    store.remember("u1", "name", "Alex")
    store.remember("u2", "name", "Priya")
    assert store.get_active("u1", "name")["text"] == "Alex"
    assert store.get_active("u2", "name")["text"] == "Priya"


def test_turns_and_sessions_persist(store):
    store.create_session("s1", "u1", title="first")
    store.add_turn("s1", "u1", "user", "hello")
    store.add_turn("s1", "u1", "assistant", "hi there")
    turns = store.get_turns("s1")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert store.get_session("s1")["title"] == "first"
    assert store.session_tokens("s1") > 0
