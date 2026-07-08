# Porting notes: 08_simple-agent-with-memory

## Mostly a straightforward port — no model-serving quirks this time

Unlike recent examples, `MemoryManager` (load/save/dedup/format) is plain
file I/O + JSON + string logic with no model-specific behavior to fight.
Ported directly to `memory_manager.py` with the same method breakdown
(`load_memories`, `save_memories`, `add_memory`, `get_memory_summary`).
Tool calling reuses the same manual call → execute → feed-back loop
established in `07_simple-agent` (same category of gap: node-llama-cpp's
`session.prompt(prompt, {functions})` loops internally, Ollama's
`chat(tools=[...])` doesn't).

## Simplified: no old-schema migration logic

`memory-manager.js`'s `loadMemories()` includes a migration branch for an
older `{facts: [...], preferences: {...}}` file shape into the current
`{memories: [...]}` shape. This Python port skips that entirely —
`memory_manager.py` only ever writes its own file
(`examples_python/08_simple-agent-with-memory/agent-memory.json`), so
there's no legacy-format file to migrate from. Intentional simplification,
not a missing feature.

## Finding: qwen3:1.7b's tool-calling judgment is noticeably inconsistent
## run-to-run — more so than expected from earlier examples

The JS example's `CODE.md` documents this expected first-run behavior:

```
User: "Hi! My name is Alex and I love pizza."
AI: "Nice to meet you, Alex! I've noted that you love pizza."
[Calls saveMemory twice - new information saved]
```

Running the Python port against the *identical* prompt repeatedly (memory
file deleted between each run) produced a different outcome almost every
time:

| Run | `save_memory` calls | Keys used | Reply style |
|-----|---------------------|-----------|--------------|
| 1 | 1 (`favorite_food`) | standard | flat: `"Memory updated: favorite_food = pizza"` |
| 2 | 1 (`favorite_food`) | standard | flat + emoji |
| 3 | 2 (`name`, `love`) | **non-standard**, despite the system prompt's exact example keys (`user_name`, `favorite_food`) | flat |
| 4 | **0** | — | hallucinated already knowing the info ("Yes, I remember your name is Alex...") despite an empty memory file — never actually called the tool |
| 5 | 1 (`favorite_food`) | standard | fuller/natural: `"Your favorite food is now recorded as pizza. Let me know if..."` |

So across 5 clean runs: sometimes 0, 1, or 2 tool calls; key naming
sometimes ignores the system prompt's explicit examples; reply tone swings
between literally echoing the tool result and a genuinely natural
sentence. This is a real, substantial finding, not a one-off — **never**
matched the JS example's documented "calls saveMemory twice, warm natural
reply" behavior in any of the 5 runs.

**Ruled out quantization as the cause.** Initially suspected Ollama's
`qwen3:1.7b` defaulting to Q4_K_M quantization (checked via
`ollama show qwen3:1.7b`) vs. the JS example's Q8_0 GGUF. Pulled the
`qwen3:1.7b-q8_0` tag specifically to match, re-tested with a clean memory
file — same category of inconsistent behavior, so quantization isn't the
explanation.

**Not fully diagnosed** — likely candidates: Ollama's default sampling
parameters for this model (`ollama show` reports `temperature 0.6`,
`top_k 20`, `top_p 0.95` — non-zero temperature alone explains a lot of
this variance), the small 1.7B/2B parameter scale being inherently less
reliable at consistent multi-fact extraction + tool-call judgment than
larger models, and/or genuine differences in how Ollama's tool-calling
path vs. node-llama-cpp's shapes the model's decision compared to the JS
example's original run. The underlying **mechanism** is verified correct
(dedup/update logic, system-prompt injection, persistence across real
process restarts — see below) — it's specifically the model's per-run
*judgment* about what to save and how to phrase the reply that's
unreliable at this scale. Worth setting `temperature: 0` (or close to it)
if reproducibility matters more than natural-sounding replies for this
particular example.

## Verified end-to-end (genuine cross-process persistence, not just
## within-conversation memory)

Note the JS example's own two prompts run in the *same* process/session
(see `simple-agent-with-memory.js` lines 79-87) — the comment "even after
restarting the script" describes the *intent*, but doesn't actually
exercise a real process restart in the demo as written. Recalling
"favorite food" right after telling it in the same conversation could just
be ordinary message-history recall, not evidence that persistence-to-disk
actually works.

To test the real claim properly, ran two genuinely separate `uv run`
processes with only the relevant single prompt in each, `agent-memory.json`
deleted beforehand:
- **Process 1** ("Hi! My name is Alex and I love pizza."): model called
  `save_memory` twice this run (`name: Alex`, `love: pizza` — non-standard
  key names, per the variance documented above), persisted correctly to
  `agent-memory.json` with timestamps.
- **Process 2** ("What's my favorite food?", fresh process, no
  conversation history at all — only the memory summary injected into a
  fresh system prompt): correctly answered `"Your favorite food is pizza."`
  with **no tool call**, proving the recall came from the persisted file
  via the system-prompt injection, not from any in-memory conversation
  state.

This confirms the actual persistence mechanism — load → inject into system
prompt → recall without re-querying — works correctly across real process
boundaries, independent of the model's per-run variance in what it chooses
to save documented above.
