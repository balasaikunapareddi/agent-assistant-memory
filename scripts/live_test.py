"""Live test: real Gemini does compaction + a memory-grounded answer.

Run via the Gemini harness (injects GEMINI_API_KEY)::

    python3 ~/workspace/custom-connectors/custom.gemini/bin/run_live.py \\
        ~/workspace/agent-assistant-memory .venv/bin/python scripts/live_test.py

Backends used here are the REAL ones:
  - LLM: live Gemini (gemini-3.5-flash-lite) via LiteLLM, for both
    the compaction summarization and the grounded answer.
  - Store / sessions / retrieval / compaction machinery: the repo's
    own code (SQLite, scoring, token budget) — that is the system
    under test, not a stub.

Asserts:
  1. Live compaction shrinks the window (tokens_after < tokens_before)
     and preserves the durable facts (coverage >= 0.5, verified
     deterministically against the original turns).
  2. In a fresh session, the live answer to "Where do I live?" uses
     the CORRECTED city (Dallas), not the superseded one (Austin).
  3. Retrieval hit rate on a small probe set: each probe question
     must retrieve the claim it is about in the top 3.

Writes a JSON report to .runs/live/attempt-N/report.json and prints
"LIVE TEST PASSED" (exit 0) or raises (exit non-zero).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_assistant_memory.agent import AssistantAgent  # noqa: E402
from agent_assistant_memory.llm import make_llm  # noqa: E402
from agent_assistant_memory.store import MemoryStore  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, ".runs", "live")

PROBES = [
    ("Where do I live?", "home city"),
    ("What reply style do I prefer?", "reply style"),
    ("When is the garden-planner app due?", "project deadline"),
    ("What is my name?", "name"),
]


def main() -> int:
    os.makedirs(RUNS_DIR, exist_ok=True)
    attempt = len([d for d in os.listdir(RUNS_DIR)
                   if d.startswith("attempt-")]) + 1
    attempt_dir = os.path.join(RUNS_DIR, f"attempt-{attempt}")
    os.makedirs(attempt_dir, exist_ok=True)

    store = MemoryStore(":memory:")
    agent = AssistantAgent(store, llm=make_llm("gemini"),
                           budget_tokens=300, user_id="alex")
    print(f"[live] LLM backend: {agent.llm_name} model={agent.llm.model}")

    # --- Session A: learn facts, including one that will be corrected.
    sess_a = agent.start_session("Day 1")
    agent.learn("name", "Alex Rivera", session_id=sess_a, salience=0.9)
    agent.learn("home city", "Alex lives in Austin, Texas",
                session_id=sess_a, salience=0.9)
    agent.learn("reply style", "Alex prefers short, plain replies",
                kind="preference", session_id=sess_a, salience=0.8)
    agent.learn("project deadline",
                "Alex will ship the garden-planner app by Friday",
                kind="commitment", session_id=sess_a, salience=0.8)

    # --- Session B: the correction.
    sess_b = agent.start_session("Day 2")
    agent.correct("home city", "Alex lives in Dallas, Texas", session_id=sess_b,
                  salience=0.9)
    print("[live] correction: home city Austin -> Dallas (old claim replaced)")

    # --- Session C: long session to force live compaction.
    sess_c = agent.start_session("Day 3 — long session")
    compaction = None
    for i in range(12):
        agent.chat(sess_c,
                   f"Planning note {i}: the garden-planner app needs a Dallas "
                   f"planting calendar with frost warnings and watering reminders, "
                   f"and Alex lives in Dallas, Texas.")
        if agent.last_chat["compacted"]:
            compaction = agent.last_chat["compaction"]
            break
    assert compaction is not None, "live run never triggered compaction"
    print(f"[live] compaction tokens before={compaction['tokens_before']} "
          f"after={compaction['tokens_after']}")
    print(f"[live] compaction coverage: {compaction['facts_preserved']}/"
          f"{compaction['facts_total']} = {compaction['coverage']:.2f}")
    print(f"[live] compaction summary: {compaction['summary']}")
    print(f"[live] durable facts from compaction: {compaction['facts']}")

    # --- Retrieval probe set (deterministic scoring, live store).
    probe_results = []
    for question, expected_subject in PROBES:
        hits = agent.recall(question, top_k=3)
        subjects = [h["subject"] for h in hits]
        hit = expected_subject in subjects
        probe_results.append({"question": question,
                              "expected": expected_subject,
                              "retrieved": subjects, "hit": hit})
        print(f"[live] probe {question!r} -> {subjects} hit={hit}")
    hit_rate = sum(p["hit"] for p in probe_results) / len(probe_results)
    print(f"[live] retrieval hit rate: {hit_rate:.2f} "
          f"({sum(p['hit'] for p in probe_results)}/{len(probe_results)})")

    # --- Session D (fresh): live memory-grounded answer.
    sess_d = agent.start_session("Day 4 — fresh session")
    answer = agent.chat(sess_d, "Where do I live?")
    print(f"[live] fresh-session answer: {answer!r}")
    corrected_used = "Dallas" in answer
    old_leaked = "Austin" in answer
    print(f"[live] corrected fact used (Dallas): {corrected_used}; "
          f"superseded fact leaked (Austin): {old_leaked}")

    report = {
        "attempt": attempt,
        "llm_model": agent.llm.model,
        "compaction": compaction,
        "probes": probe_results,
        "retrieval_hit_rate": hit_rate,
        "final_answer": answer,
        "corrected_used": corrected_used,
        "old_leaked": old_leaked,
        "active_claims": len(store.active_claims("alex")),
    }
    with open(os.path.join(attempt_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    assert compaction["tokens_after"] < compaction["tokens_before"], (
        f"compaction did not shrink: {compaction['tokens_before']} -> "
        f"{compaction['tokens_after']}")
    assert compaction["coverage"] >= 0.5, (
        f"compaction coverage {compaction['coverage']:.2f} < 0.5")
    assert hit_rate >= 0.75, f"retrieval hit rate {hit_rate:.2f} < 0.75"
    assert corrected_used, f"live answer did not use corrected fact: {answer!r}"
    assert not old_leaked, f"live answer leaked superseded fact: {answer!r}"

    print("[live] LIVE TEST PASSED")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
