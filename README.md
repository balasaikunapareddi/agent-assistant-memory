# agent-assistant-memory

How an AI assistant maintains memory and sessions: a durable memory store with provenance and corrections, multi-session continuity, and LLM compaction that keeps long conversations inside a token budget.

## The problem this solves

A real assistant — the kind you talk to every day, across days and devices — has to remember you without ever contradicting itself. That is three hard problems at once, and most demos solve none of them:

1. **It learns something, then you correct it.** If the old fact stays active next to the new one, the assistant will happily tell you both. The correction has to *win*.
2. **Conversations end, memory must not.** A fact learned on Monday has to be there on Thursday, in a brand-new session with an empty transcript.
3. **Conversations get long.** A context window has a token budget. When it fills up, you cannot just drop the oldest turns — that is where the facts live. You have to *summarize* them, and then prove the summary kept what mattered.

This repo implements all three, in working code, modelled on how Muse itself works (this is the pattern implemented here, not a claim about Muse's internals): a durable SQLite memory store, sessions that share it, scored retrieval, and LLM compaction with a verifiable summary.

## One-command demo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m agent_assistant_memory demo
```

The demo simulates a fictional user, Alex Rivera, over four days: it learns facts in session A, gets corrected on the home city in session B (Austin → Dallas), runs a long session C that triggers compaction, then in a fresh session D asks "Where do I live?" — and answers **Dallas**, from the durable store. It prints the memory store, the compaction summary, and token counts before/after.

```
=== agent-assistant-memory demo (llm=stub, budget=400 est. tokens) ===
[Day 1] session A: ...
[Day 2] session B: ... (fresh session, same memory store)
  home-city history:
    #2 replaced  'Alex lives in Austin, Texas' ...
    #3 active    'Alex lives in Dallas, Texas' ...
--- Compaction ---
tokens before: ...  after: ...
facts preserved: .../... (coverage ...)
summary: ...
[Day 4] session D: ... (brand-new session, no transcript)
  Assistant: Based on what I remember: [fact] home city: Alex lives in Dallas, Texas ...
  corrected fact used (Dallas): True; superseded fact leaked (Austin): False
```

With a live model instead of the stub (needs `GEMINI_API_KEY`):

```bash
python -m agent_assistant_memory demo --llm gemini
```

## How it works

### Durable memory store (`store.py`)

Facts, preferences, decisions and commitments, each with provenance: source session, source quote, when learned, confidence, salience. The core rule: **a newer correction supersedes an older claim**. The old row is marked `replaced` and linked to its replacement (`superseded_by`); it is never deleted and never returned by retrieval again. History is preserved; only the newest claim on a subject is active. `forget()` works the same way (status `forgotten`, history kept).

```python
store = MemoryStore("memory.db")
store.remember("alex", "home city", "Alex lives in Austin, Texas")
store.correct("alex", "home city", "Alex lives in Dallas, Texas")
store.get_active("alex", "home city")["text"]  # "Alex lives in Dallas, Texas"
store.claim_history("alex", "home city")       # both rows: replaced + active
```

### Sessions (`sessions.py`)

A session is one conversation: its own transcript, its own working context, and a link to the session it was handed off from. Memory is *not* per-session — it lives in the durable store — so a fact learned in session A is available in session B. `handoff()` starts a new session carrying a summary of the old one, so the user never repeats themselves; the old transcript stays intact under its own id.

When the window is over budget, the oldest turns go to the compactor instead of being dropped. Compaction runs in passes (up to 5 per chat turn, stopping early when a pass shrinks nothing) until the window fits the budget — one pass is not guaranteed to fit, so the budget is enforced by a loop, not a single attempt. If a single turn (or the summary itself) is larger than the whole budget, the window can stay over budget: the mechanism never drops turns just to force the number down.

### Retrieval (`retrieval.py`)

Search scores active claims only: **keyword match (0.6) + recency (0.15) + salience (0.25)**, with per-hit score components reported so a ranking can be explained. Claims with no keyword overlap are dropped. The top hits are injected into the session's context as a "Relevant memories" preamble before the LLM answers.

### Compaction (`compaction.py`)

When the window is over budget, the oldest half of the turns is summarized by the LLM in a fixed format — a short summary plus a `Durable facts:` list — and the summary replaces those turns in the window. Facts from the summary are written back into the memory store, but only when they add something new: a compaction fact is skipped when the store already holds it (under any subject), holds a near-identical active claim, or deliberately retired an identical one (replaced or forgotten) — and compaction never supersedes anything, so two facts that map to the same derived subject cannot overwrite each other. A skipped fact is not lost: it stays in the summary, the compaction record, and the original turns. Every compaction is recorded (token counts before/after, summary, turn ids, facts preserved/total), and `verify_summary()` checks it deterministically: for each durable fact, what fraction of its content words appear in the summary. A compaction with low coverage is visible in the record, not silently trusted. A compaction that extracts no facts at all records coverage 0.0 (unverified), never a vacuous 1.0.

### The agent (`agent.py`)

`AssistantAgent` wires it together: retrieve → inject → ask the LLM → store both turns → compact if over budget. The system prompt lives in `prompts.py` — separate and inspectable, never inline.

## Why plain Python + LiteLLM + SQLite (not a framework)

This is a *system*, not a framework demo. The interesting parts are the store semantics (supersede, provenance), the session/window bookkeeping, and the compaction audit trail — all of which a framework would hide behind its own abstractions. Plain Python keeps every rule visible in a few hundred lines; LiteLLM keeps the live layer provider-agnostic (model from `LLM_MODEL`, default `gemini/gemini-3.5-flash-lite`); SQLite makes memory durable across restarts with no server and no network.

## How this differs from `agent-memory`

Bala's earlier `agent-memory` is CrewAI episodic memory *for an agent*: a support agent remembering a user's issue and preferences through CrewAI tools. This repo is the **assistant-level** memory + session system: the durable claim store with superseding corrections, multi-session continuity and handoffs, a token-budgeted working context, and verifiable LLM compaction — the machinery an assistant like Muse itself needs to maintain memory across sessions, independent of any agent framework.

## Live test

Run with the Gemini harness (injects the key; never committed):

```bash
python3 ~/workspace/custom-connectors/custom.gemini/bin/run_live.py \
  ~/workspace/agent-assistant-memory .venv/bin/python scripts/live_test.py
```

Result (run firsthand, 2026-11-24, attempt 1 — no retry needed):

```
[live] LLM backend: gemini model=gemini/gemini-3.5-flash-lite
[live] compaction tokens before=166 after=53
[live] compaction coverage: 2/3 = 0.67
[live] compaction summary: The user outlined requirements for the garden-planner app (Dallas planting calendar, frost warnings, and watering reminders) ahead of its Friday ship date, and the assistant offered to help develop these features.
[live] durable facts from compaction: ['Alex lives in Dallas, Texas.', 'The garden-planner app needs a Dallas planting calendar with frost warnings and watering reminders.', 'The garden-planner app ships by Friday.']
[live] probe 'Where do I live?' -> ['home city', 'alex'] hit=True
[live] probe 'What reply style do I prefer?' -> ['reply style'] hit=True
[live] probe 'When is the garden-planner app due?' -> ['project deadline', 'the garden-planner app'] hit=True
[live] probe 'What is my name?' -> ['name'] hit=True
[live] retrieval hit rate: 1.00 (4/4)
[live] fresh-session answer: 'You live in Dallas, Texas.'
[live] corrected fact used (Dallas): True; superseded fact leaked (Austin): False
[live] LIVE TEST PASSED
```

Notes on the numbers: coverage is 2/3, not 3/3 — the summary prose omits the residence fact ("Alex lives in Dallas") even though the model's `Durable facts:` list captured all three and all three were written back to the store. The verification caught a real imperfection instead of hiding it; the threshold in the live test is ≥ 0.5. Full report: `.runs/live/attempt-1/report.json`. (Review note: in that run, one of the three stored facts superseded another under the same derived subject — "the garden-planner app" — so only two of the three stayed active. The write-back guards added in the 2026-11-24 review now prevent that: a re-run would store only the genuinely new fact. The attempt-1 numbers above are the as-run record, unchanged.)

The live *demo* (`demo --llm gemini`) did not complete: Gemini returned 503 (model high demand) in session C, on the first run and again on the one retry after ~2.5 minutes. The stub demo passes end to end, and the live test above — which exercises the same live compaction + grounded answer — passed on its first attempt.

Offline suite: **66 tests pass** (`pytest tests/ -q`), deterministic, stub LLM only.

## Honest limits

- Token counts are estimates (chars / 4), not a real tokenizer's counts — the budget mechanism is exact, the unit is approximate, and every report says so.
- Retrieval is keyword + recency + salience scoring, not embeddings — it is explainable and dependency-free, and it ranks short factual memories well; it will miss paraphrases that share no words with the stored claim.
- Compaction facts are parsed from the LLM's fixed-format reply; if a live model ignores the format, the raw reply is kept as the summary and the coverage check will show it (0.0 when no facts were extracted). An empty compaction reply raises instead of silently dropping the turns, and a backend failure (503) during the answer or compaction propagates — the turns already stored stay stored and the window is left untouched, so the caller sees the failure and can retry.
- Reopening a session restores its latest compaction summary (or its handoff summary) and skips turns a compaction already removed, so compacted turns are never resurrected into the window.
- The demo user is fictional (Alex Rivera) on purpose, so the demo never mixes with any real person's facts.

## Layout

```
src/agent_assistant_memory/
  prompts.py     system + compaction + handoff prompts (separate file, on purpose)
  tokens.py      token estimation (chars / 4)
  store.py       SQLite durable memory: claims, sessions, turns, compactions
  retrieval.py   scored search (keyword + recency + salience), active claims only
  sessions.py    SessionManager + ContextWindow (token budget)
  compaction.py  LLM summarization + deterministic verification
  llm.py         LiteLLM (live Gemini) / deterministic stub
  agent.py       AssistantAgent — retrieve → inject → answer → compact
  demo.py        four-day, four-session demo (stub or live)
scripts/live_test.py   live Gemini compaction + grounded answer + probe set
tests/           66 offline tests (stub LLM only)
```

## License

MIT — see `LICENSE`.
