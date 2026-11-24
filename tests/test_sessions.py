"""Session tests: isolation, cross-session recall, handoff, budget."""

import pytest

from agent_assistant_memory.sessions import ContextWindow, SessionManager
from agent_assistant_memory.store import MemoryStore


@pytest.fixture
def mgr():
    s = MemoryStore(":memory:")
    m = SessionManager(s, budget_tokens=200)
    yield m
    s.close()


def test_sessions_have_separate_transcripts(mgr):
    a = mgr.start_session("u1", title="A")
    b = mgr.start_session("u1", title="B")
    mgr.add_turn(a, "u1", "user", "hello from A")
    mgr.add_turn(b, "u1", "user", "hello from B")
    assert [t["content"] for t in mgr.store.get_turns(a)] == ["hello from A"]
    assert [t["content"] for t in mgr.store.get_turns(b)] == ["hello from B"]


def test_reopening_a_session_reloads_its_turns(mgr):
    a = mgr.start_session("u1")
    mgr.add_turn(a, "u1", "user", "first")
    mgr.add_turn(a, "u1", "assistant", "second")
    window = mgr.open_session(a)
    assert [m["content"] for m in window.messages] == ["first", "second"]


def test_opening_unknown_session_raises(mgr):
    with pytest.raises(KeyError):
        mgr.open_session("sess-nope")


def test_handoff_carries_summary_and_links_parent(mgr):
    a = mgr.start_session("u1", title="old")
    mgr.add_turn(a, "u1", "user", "planning the app")
    new_id = mgr.handoff("u1", a, "User is planning the garden app; deadline Friday.")
    assert new_id != a
    session = mgr.store.get_session(new_id)
    assert session["parent_session_id"] == a
    assert session["handoff_summary"].startswith("User is planning")
    assert mgr.window(new_id).summary.startswith("User is planning")
    # Old transcript is untouched.
    assert len(mgr.store.get_turns(a)) == 1
    assert mgr.store.get_turns(new_id) == []


def test_handoff_summary_counts_toward_budget(mgr):
    a = mgr.start_session("u1")
    new_id = mgr.handoff("u1", a, "x" * 400)  # ~100 tokens
    assert mgr.window(new_id).tokens() >= 100


def test_context_window_budget_enforcement():
    w = ContextWindow(budget_tokens=20)
    assert not w.over_budget()
    w.add("user", "word " * 100)  # ~125 tokens
    assert w.over_budget()
    assert w.tokens() > 20


def test_compactable_messages_keep_the_newest():
    w = ContextWindow(budget_tokens=10)
    for i in range(6):
        w.add("user", f"turn {i}")
    candidates = w.compactable_messages()
    assert len(candidates) == 3
    assert candidates[-1]["content"] == "turn 2"
    assert w.messages[-1]["content"] == "turn 5"


def test_compactable_messages_empty_for_tiny_window():
    w = ContextWindow(budget_tokens=10)
    w.add("user", "only one")
    assert w.compactable_messages() == []


def test_drop_messages_removes_only_candidates():
    w = ContextWindow(budget_tokens=10)
    for i in range(4):
        w.add("user", f"turn {i}")
    w.drop_messages(w.compactable_messages())
    assert [m["content"] for m in w.messages] == ["turn 2", "turn 3"]


def test_as_messages_puts_summary_first():
    w = ContextWindow(budget_tokens=100)
    w.set_summary("Earlier: planning the app.")
    w.add("user", "continue")
    msgs = w.as_messages()
    assert msgs[0]["role"] == "system"
    assert "planning the app" in msgs[0]["content"]
    assert msgs[1]["content"] == "continue"


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        ContextWindow(budget_tokens=0)
