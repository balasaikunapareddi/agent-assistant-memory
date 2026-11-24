"""Compaction + LLM stub tests, and agent-level integration tests."""

import pytest

from agent_assistant_memory.agent import AssistantAgent
from agent_assistant_memory.compaction import (
    Compactor, parse_compaction_reply, verify_summary,
)
from agent_assistant_memory.llm import StubLLM, make_llm
from agent_assistant_memory.store import MemoryStore


@pytest.fixture
def agent():
    store = MemoryStore(":memory:")
    a = AssistantAgent(store, llm=StubLLM(), budget_tokens=200, user_id="alex")
    yield a
    store.close()


TURNS = [
    {"role": "user", "content": "Alex lives in Dallas, Texas"},
    {"role": "assistant", "content": "Noted, Dallas."},
    {"role": "user", "content": "Alex prefers short, plain replies"},
    {"role": "assistant", "content": "Understood."},
]


def test_parse_compaction_reply():
    reply = ("Summary: Alex lives in Dallas and likes short replies.\n\n"
             "Durable facts:\n- Alex lives in Dallas, Texas\n"
             "- Alex prefers short, plain replies\n")
    summary, facts = parse_compaction_reply(reply)
    assert "Dallas" in summary
    assert facts == ["Alex lives in Dallas, Texas",
                     "Alex prefers short, plain replies"]


def test_parse_compaction_reply_none_facts():
    summary, facts = parse_compaction_reply("Summary: Small talk only.\n\nDurable facts:\n- (none)\n")
    assert facts == []


def test_verify_summary_full_coverage():
    preserved, total, cov = verify_summary(
        "Alex lives in Dallas, Texas and prefers short plain replies",
        ["Alex lives in Dallas, Texas", "Alex prefers short, plain replies"])
    assert (preserved, total, cov) == (2, 2, 1.0)


def test_verify_summary_partial_coverage():
    preserved, total, cov = verify_summary(
        "Alex lives in Dallas", ["Alex lives in Dallas, Texas",
                                 "Alex loves tacos with hot sauce"])
    assert preserved == 1 and total == 2 and cov == 0.5


def test_verify_summary_no_facts_is_unverified():
    # No facts extracted -> nothing was verified preserved, so coverage
    # is 0.0, not a vacuous 1.0. A malformed reply that yields no
    # durable facts must not masquerade as a perfect compaction.
    assert verify_summary("anything", []) == (0, 0, 0.0)


def test_malformed_compaction_reply_keeps_raw_summary_with_zero_coverage():
    """A live model that ignores the fixed format: the raw reply is kept
    as the summary (nothing is silently dropped), but with no durable
    facts extracted the coverage is 0.0 — the record shows the
    compaction is unverified instead of claiming perfection."""
    class RamblingLLM:
        name = "rambling"
        def reply(self, messages, **kwargs):
            return "Some rambling prose with no format at all."
    result = Compactor(RamblingLLM()).compact(TURNS)
    assert result.summary == "Some rambling prose with no format at all."
    assert result.durable_facts == []
    assert (result.facts_preserved, result.facts_total, result.coverage) == (0, 0, 0.0)


def test_stub_compaction_preserves_facts():
    compactor = Compactor(StubLLM())
    result = compactor.compact(TURNS)
    assert result.tokens_after < result.tokens_before
    assert result.coverage == 1.0
    assert result.facts_preserved == result.facts_total
    assert any("Dallas" in f for f in result.durable_facts)


def test_stub_compaction_tokens_shrink_on_long_turns():
    long_turns = [{"role": "user", "content": "word " * 100},
                  {"role": "assistant", "content": "reply " * 50}]
    result = Compactor(StubLLM()).compact(long_turns)
    assert result.tokens_after < result.tokens_before


def test_make_llm_factory():
    assert make_llm("stub").name == "stub"
    with pytest.raises(ValueError):
        make_llm("nope")


def test_stub_answer_uses_injected_memory():
    llm = StubLLM()
    reply = llm.reply([
        {"role": "system", "content": "Relevant memories (from earlier sessions):\n"
         "- [fact] home city: Alex lives in Dallas, Texas (confidence 1.00, learned 2026-11-24)"},
        {"role": "user", "content": "Where do I live?"},
    ])
    assert "Dallas" in reply


def test_stub_answer_without_memory_says_so():
    reply = StubLLM().reply([{"role": "user", "content": "What is my favourite colour?"}])
    assert "don't have a stored memory" in reply


# --- agent-level ---------------------------------------------------------

def test_chat_uses_corrected_fact_in_fresh_session(agent):
    a = agent.start_session("day 1")
    agent.learn("home city", "Alex lives in Austin, Texas", session_id=a)
    agent.correct("home city", "Alex lives in Dallas, Texas")
    b = agent.start_session("day 2")
    answer = agent.chat(b, "Where do I live?")
    assert "Dallas" in answer
    assert "Austin" not in answer


def test_cross_session_recall_via_store_not_transcript(agent):
    a = agent.start_session("day 1")
    agent.learn("name", "Alex Rivera", session_id=a)
    b = agent.start_session("day 2")
    assert agent.store.get_turns(b) == []  # new session, no transcript
    answer = agent.chat(b, "What is my name?")
    assert "Alex Rivera" in answer


def test_chat_stores_both_turns(agent):
    s = agent.start_session()
    agent.chat(s, "hello")
    turns = agent.store.get_turns(s)
    assert len(turns) == 2
    assert turns[0]["role"] == "user" and turns[1]["role"] == "assistant"


def test_long_session_triggers_compaction_and_shrinks_window(agent):
    agent.budget_tokens = 150
    agent.sessions.budget_tokens = 150
    s = agent.start_session("long")
    compacted = False
    for i in range(12):
        agent.chat(s, f"Planning note {i}: the garden-planner app needs a Dallas "
                      f"planting calendar with frost warnings and watering reminders.")
        if agent.last_chat["compacted"]:
            compacted = True
            break
    assert compacted, "no compaction triggered in 12 long turns"
    assert agent.last_chat["compaction"]["tokens_after"] < \
        agent.last_chat["compaction"]["tokens_before"]
    # The loop runs until the window fits the budget (a single pass is
    # not guaranteed to fit). At least one recorded pass must have
    # actually preserved facts — asserting on the last pass alone is
    # wrong, since a later pass can legitimately cover turns with no
    # durable facts (0/0, coverage 0.0).
    assert agent.sessions.window(s).tokens() <= 150
    records = agent.store.list_compactions(s)
    assert records, "compaction not recorded in store"
    assert any(r["facts_total"] > 0 and r["facts_preserved"] == r["facts_total"]
               for r in records)


def test_compaction_facts_written_back_to_store(agent):
    agent.budget_tokens = 150
    agent.sessions.budget_tokens = 150
    s = agent.start_session("long")
    for i in range(12):
        agent.chat(s, f"Planning note {i}: Alex lives in Dallas, Texas and the "
                      f"garden-planner app ships Friday.")
        if agent.last_chat["compacted"]:
            break
    # The compacted fact itself must be a live claim — not merely "some
    # claim exists". (The old assertion passed whenever anything at all
    # was active.)
    texts = [c["text"] for c in agent.store.active_claims("alex")]
    assert any("ships Friday" in t for t in texts)
    assert any("Dallas" in t for t in texts)


def test_handoff_then_chat_keeps_context(agent):
    a = agent.start_session("old")
    agent.chat(a, "We are planning the garden app.")
    b = agent.handoff(a, "Planning the garden app; deadline Friday.")
    assert agent.sessions.window(b).summary is not None
    assert agent.store.get_session(b)["parent_session_id"] == a


def test_learn_correct_forget_round_trip(agent):
    agent.learn("home city", "Austin")
    assert agent.store.get_active("alex", "home city")["text"] == "Austin"
    agent.correct("home city", "Dallas")
    assert agent.store.get_active("alex", "home city")["text"] == "Dallas"
    assert agent.forget("home city") is True
    assert agent.recall("where do I live") == []


def test_demo_runs_with_stub_and_uses_corrected_fact():
    from agent_assistant_memory.demo import run_demo
    result = run_demo(db_path=":memory:", llm_name="stub", budget_tokens=400)
    assert result["corrected_used"] is True
    assert result["old_leaked"] is False
    assert result["compaction"]["tokens_after"] < result["compaction"]["tokens_before"]
    assert result["active_claims"] >= 4
    assert result["sessions"] == 4


# --- review fixes: compaction failures + write-back guards ------------

class FixedReplyLLM:
    """LLM that always returns the same reply (or raises)."""

    name = "fixed"

    def __init__(self, reply=None, exc=None):
        self._reply = reply
        self._exc = exc

    def reply(self, messages, **kwargs):
        if self._exc is not None:
            raise self._exc
        return self._reply


def _agent_with_fixed_compaction(reply, budget=1000):
    store = MemoryStore(":memory:")
    a = AssistantAgent(store, llm=StubLLM(), budget_tokens=budget, user_id="alex")
    a.compactor = Compactor(FixedReplyLLM(reply=reply))
    return a, store


def _fill_window_with_stored_turns(a, s, n=4):
    for i in range(n):
        a.sessions.add_turn(s, "alex", "user",
                            "planning filler %d " % i + "word " * 30)


def test_compaction_empty_llm_reply_raises_and_preserves_turns(agent):
    """An empty compaction reply fails loudly: no summary, no record,
    window and stored turns untouched — nothing silently dropped."""
    s = agent.start_session("long")
    agent.chat(s, "Planning note: Alex lives in Dallas, Texas.")
    agent.chat(s, "Planning note: the garden-planner app ships Friday.")
    window = agent.sessions.window(s)
    before = [m["content"] for m in window.messages]
    agent.compactor = Compactor(FixedReplyLLM(reply="   "))
    with pytest.raises(ValueError, match="empty reply"):
        agent._compact_window(s, window)
    assert [m["content"] for m in window.messages] == before
    assert window.summary is None
    assert agent.store.list_compactions(s) == []
    assert len(agent.store.get_turns(s)) == 4  # turns still in the store


def test_chat_stores_user_turn_when_answer_llm_fails():
    """A 503 on the answer must not lose the user's message: the user
    turn is stored before the LLM is called, and the error propagates."""
    store = MemoryStore(":memory:")
    failing = AssistantAgent(
        store, llm=FixedReplyLLM(exc=RuntimeError("503 model high demand")),
        budget_tokens=200, user_id="alex")
    s = failing.start_session("x")
    with pytest.raises(RuntimeError, match="503"):
        failing.chat(s, "hello, remember this")
    turns = store.get_turns(s)
    assert len(turns) == 1
    assert turns[0]["role"] == "user"
    assert turns[0]["content"] == "hello, remember this"
    store.close()


def test_compact_window_returns_none_when_nothing_to_compact(agent):
    s = agent.start_session("tiny")
    agent.chat(s, "hi")
    window = agent.sessions.window(s)
    window.messages = window.messages[:1]  # fewer than 2: nothing compactable
    assert agent._compact_window(s, window) is None
    assert agent.store.list_compactions(s) == []


def test_compaction_loops_until_window_fits_budget():
    """A single compaction pass is not guaranteed to fit the window in
    budget, so chat must keep compacting (bounded passes) until it
    fits, instead of leaving the window over budget after one pass."""
    store = MemoryStore(":memory:")
    llm = FixedReplyLLM(
        reply="Summary: Planning the garden app.\n\nDurable facts:\n- (none)\n")
    a = AssistantAgent(store, llm=llm, budget_tokens=60, user_id="alex")
    s = a.start_session("long")
    for i in range(6):
        a.sessions.add_turn(s, "alex", "user",
                            "Planning note %d " % i + "word " * 40)
    assert a.sessions.window(s).tokens() > 60
    a.chat(s, "continue")
    assert a.sessions.window(s).tokens() <= 60
    assert len(store.list_compactions(s)) >= 2
    store.close()


def test_compaction_writeback_never_resurrects_superseded_fact():
    """The core guarantee: a correction can never lose to the fact it
    replaced — including via compaction. Old turns still contain the
    superseded city, so a compaction that echoes them must not bring
    it back as a live claim (previously it did, under a shadow
    subject, and retrieval ranked it first)."""
    reply = ("Summary: Old planning turns.\n\nDurable facts:\n"
             "- Alex lives in Austin, Texas\n"
             "- The garden-planner app needs a Dallas planting calendar "
             "with frost warnings and watering reminders.\n"
             "- The garden-planner app ships by Friday.\n")
    a, store = _agent_with_fixed_compaction(reply)
    s = a.start_session("c")
    a.learn("home city", "Alex lives in Austin, Texas", session_id=s)
    a.correct("home city", "Alex lives in Dallas, Texas", session_id=s)
    _fill_window_with_stored_turns(a, s)
    result = a._compact_window(s, a.sessions.window(s))
    assert result is not None
    # Only the genuinely new fact is stored: the superseded-city fact
    # is skipped (it restates the active Dallas claim), and the
    # shipping fact is skipped as a near-duplicate of the calendar
    # fact stored just before it in the same pass.
    assert result["stored_facts"] == 1
    assert store.get_active("alex", "home city")["text"] == \
        "Alex lives in Dallas, Texas"
    assert store.get_active("alex", "alex") is None  # no shadow claim
    assert all("Austin" not in c["text"] for c in store.active_claims("alex"))
    hits = a.recall("Where do I live?")
    assert hits and hits[0]["text"] == "Alex lives in Dallas, Texas"
    assert all("Austin" not in h["text"] for h in hits)
    store.close()


def test_compaction_writeback_never_supersedes_an_existing_fact():
    """Two compaction facts that map to the same derived subject must
    not supersede each other (previously the second silently replaced
    the first, losing a durable fact the compaction had just found)."""
    reply = ("Summary: Planning.\n\nDurable facts:\n"
             "- The garden-planner app needs a Dallas planting calendar.\n"
             "- The garden-planner app ships by Friday.\n")
    a, store = _agent_with_fixed_compaction(reply)
    s = a.start_session("c")
    _fill_window_with_stored_turns(a, s)
    a._compact_window(s, a.sessions.window(s))
    history = store.claim_history("alex", "the garden-planner app")
    assert len(history) == 1
    assert history[0]["status"] == "active"
    assert "planting calendar" in history[0]["text"]
    store.close()


def test_compaction_writeback_skips_identical_fact_under_other_subject():
    """A compaction fact identical to an active claim must not create
    a duplicate claim under a different (derived) subject."""
    a, store = _agent_with_fixed_compaction(
        "Summary: Chat.\n\nDurable facts:\n- Alex lives in Dallas, Texas\n")
    s = a.start_session("c")
    a.learn("home city", "Alex lives in Dallas, Texas", session_id=s)
    before = len(store.active_claims("alex"))
    _fill_window_with_stored_turns(a, s)
    result = a._compact_window(s, a.sessions.window(s))
    assert result["stored_facts"] == 0
    assert len(store.active_claims("alex")) == before
    assert store.get_active("alex", "alex") is None
    store.close()


def test_compaction_writeback_never_resurrects_forgotten_fact():
    """A fact the user asked to forget must not come back because old
    turns still mention it."""
    a, store = _agent_with_fixed_compaction(
        "Summary: Old turns.\n\nDurable facts:\n- Alex loves tacos with hot sauce\n")
    s = a.start_session("c")
    a.learn("favourite food", "Alex loves tacos with hot sauce", session_id=s)
    assert a.forget("favourite food") is True
    _fill_window_with_stored_turns(a, s)
    result = a._compact_window(s, a.sessions.window(s))
    assert result["stored_facts"] == 0
    assert store.get_active("alex", "favourite food") is None
    assert store.active_claims("alex") == []
    store.close()
