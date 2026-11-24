"""One-command demo: an assistant over several sessions / days.

Simulates a fictional user, Alex Rivera, so the demo never mixes
with any real person's facts:

- Day 1, session A: the assistant learns facts (name, home city,
  a preference, a commitment).
- Day 2, session B: Alex corrects the home city — the correction
  supersedes the old claim; history is preserved.
- Day 3, session C: a long working session exceeds the token budget
  and triggers LLM compaction; durable facts are written back.
- Day 4, session D (fresh session): Alex asks where they live.
  The answer must use the *corrected* city, retrieved from the
  durable store — proof of cross-session continuity.

Prints the memory store, the compaction summary, and token counts
before/after. Run offline (stub LLM, default) or live::

    python -m agent_assistant_memory.demo            # stub
    python -m agent_assistant_memory.demo --llm gemini  # live, needs GEMINI_API_KEY
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agent_assistant_memory.agent import AssistantAgent  # noqa: E402
from agent_assistant_memory.llm import make_llm  # noqa: E402
from agent_assistant_memory.store import MemoryStore  # noqa: E402


def run_demo(db_path: str = ":memory:", llm_name: str = "stub",
             budget_tokens: int = 400) -> dict:
    store = MemoryStore(db_path)
    agent = AssistantAgent(store, llm=make_llm(llm_name),
                           budget_tokens=budget_tokens, user_id="alex")
    print(f"=== agent-assistant-memory demo (llm={llm_name}, budget={budget_tokens} est. tokens) ===\n")

    # --- Day 1, session A: learn facts --------------------------------
    sess_a = agent.start_session("Day 1 — getting to know Alex")
    print(f"[Day 1] session A: {sess_a}")
    agent.learn("name", "Alex Rivera", session_id=sess_a, salience=0.9)
    agent.learn("home city", "Alex lives in Austin, Texas", session_id=sess_a,
                salience=0.9, source_quote="I live in Austin, Texas")
    agent.learn("reply style", "Alex prefers short, plain replies",
                kind="preference", session_id=sess_a, salience=0.8)
    agent.learn("project deadline", "Alex will ship the garden-planner app by Friday",
                kind="commitment", session_id=sess_a, salience=0.8)
    print(agent.chat(sess_a, "Hi, I'm Alex. Remember that I live in Austin, Texas."))
    print(agent.chat(sess_a, "I prefer short, plain replies."))

    # --- Day 2, session B: correction supersedes -----------------------
    sess_b = agent.start_session("Day 2 — correction")
    print(f"\n[Day 2] session B: {sess_b} (fresh session, same memory store)")
    old = agent.store.get_active("alex", "home city")
    print(f"  home city before correction: {old['text']!r} (status {old['status']})")
    agent.correct("home city", "Alex lives in Dallas, Texas", session_id=sess_b,
                  salience=0.9, source_quote="Actually, I moved — I live in Dallas now")
    print(agent.chat(sess_b, "Actually, I moved — I live in Dallas, Texas now, not Austin."))
    history = agent.store.claim_history("alex", "home city")
    print("  home-city history:")
    for h in history:
        print(f"    #{h['id']} {h['status']:9s} {h['text']!r} "
              f"(superseded_by={h['superseded_by']}, supersedes={h['supersedes']})")

    # --- Day 3, session C: long session triggers compaction ------------
    sess_c = agent.start_session("Day 3 — long planning session")
    print(f"\n[Day 3] session C: {sess_c} (long session, budget {budget_tokens})")
    topics = [
        "Let's plan the garden-planner app: it needs a planting calendar for Dallas weather.",
        "The planting calendar should warn about the last frost date in Dallas.",
        "Add a watering reminder every three days for tomatoes in Dallas heat.",
        "The app should store each garden bed and what is planted in it.",
        "Users in Dallas need a heat warning when it is over 100 degrees.",
        "Keep the interface simple: one screen per garden bed in Dallas.",
        "The Friday deadline still stands for the garden-planner app.",
        "Remind me that Alex lives in Dallas, Texas when planning the weather features.",
    ]
    compaction = None
    for i, topic in enumerate(topics, 1):
        reply = agent.chat(sess_c, topic)
        info = agent.last_chat
        print(f"  turn {i}: window tokens now {info['tokens_after']} "
              f"(compacted={info['compacted']})")
        if info["compacted"]:
            compaction = info["compaction"]
    if compaction is None:
        # Force one compaction pass so the demo always shows it.
        window = agent.sessions.window(sess_c)
        compaction = agent._compact_window(sess_c, window)
        print("  (forced one compaction pass at end of session C)")

    print("\n--- Compaction ---")
    print(f"tokens before: {compaction['tokens_before']}  after: {compaction['tokens_after']}")
    print(f"facts preserved: {compaction['facts_preserved']}/{compaction['facts_total']} "
          f"(coverage {compaction['coverage']:.2f})")
    print(f"summary: {compaction['summary']}")
    print("durable facts from compaction:")
    for f in compaction["facts"]:
        print(f"  - {f}")

    # --- Day 4, session D: fresh session, memory-grounded answer -------
    sess_d = agent.start_session("Day 4 — fresh session")
    print(f"\n[Day 4] session D: {sess_d} (brand-new session, no transcript)")
    answer = agent.chat(sess_d, "Where do I live?")
    print(f"  Alex: Where do I live?\n  Assistant: {answer}")
    corrected_used = "Dallas" in answer
    old_used = "Austin" in answer
    print(f"  corrected fact used (Dallas): {corrected_used}; "
          f"superseded fact leaked (Austin): {old_used}")

    # --- Final memory store ---------------------------------------------
    print("\n--- Memory store (active claims) ---")
    for c in store.active_claims("alex"):
        print(f"  [{c['kind']}] {c['subject']}: {c['text']} "
              f"(confidence {c['confidence']:.2f}, salience {c['salience']:.2f}, "
              f"source {c['source_session_id']}, learned {c['created_at'][:19]})")
    print(f"\nSessions: {[s['id'] for s in store.list_sessions('alex')]}")
    print(f"Compaction records: {len(store.list_compactions(sess_c))} "
          f"for session C")

    result = {
        "llm": llm_name,
        "corrected_used": corrected_used,
        "old_leaked": old_used,
        "compaction": compaction,
        "active_claims": len(store.active_claims("alex")),
        "sessions": len(store.list_sessions("alex")),
        "final_answer": answer,
    }
    print("\n=== demo done: corrected fact used = "
          f"{corrected_used}, old fact leaked = {old_used} ===")
    store.close()
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="agent-assistant-memory demo")
    p.add_argument("--llm", default="stub", choices=["stub", "gemini"])
    p.add_argument("--db", default=":memory:")
    p.add_argument("--budget", type=int, default=300)
    args = p.parse_args()
    result = run_demo(db_path=args.db, llm_name=args.llm, budget_tokens=args.budget)
    return 0 if result["corrected_used"] and not result["old_leaked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
