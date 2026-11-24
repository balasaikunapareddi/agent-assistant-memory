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


def test_verify_summary_empty_facts_is_full_coverage():
    assert verify_summary("anything", []) == (0, 0, 1.0)


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
    assert agent.last_chat["compaction"]["coverage"] > 0
    assert agent.store.list_compactions(s), "compaction not recorded in store"


def test_compaction_facts_written_back_to_store(agent):
    agent.budget_tokens = 150
    agent.sessions.budget_tokens = 150
    s = agent.start_session("long")
    for i in range(12):
        agent.chat(s, f"Planning note {i}: Alex lives in Dallas, Texas and the "
                      f"garden-planner app ships Friday.")
        if agent.last_chat["compacted"]:
            break
    assert agent.store.get_active("alex", "alex") is not None or \
        len(agent.store.active_claims("alex")) > 0


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
