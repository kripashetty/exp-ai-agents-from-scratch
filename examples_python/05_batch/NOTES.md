# Porting notes: 05_batch

## Why this example switches from llama-cpp-python to Ollama

`batch.js` runs two independent chat sessions concurrently, sharing one
loaded model's weights but each with its own context sequence
(`context.getSequence()`), dispatched via `Promise.all` so total wall-clock
time is roughly `max(t1, t2)` instead of `t1 + t2`. It's specifically about
GPU batch efficiency — one model instance, multiple sequences processed
together.

`llama-cpp-python`'s high-level `Llama` class has no equivalent to
`context.getSequence()` — there's no way to run two independent
conversations against one loaded model instance without either:
1. Loading the model twice (two separate `Llama()` objects) — works, but
   uses ~2x model memory since weights aren't shared, unlike the JS version.
2. Using the low-level ctypes bindings directly (manual `llama_batch_init`,
   `llama_decode`, per-sequence KV cache management) to replicate
   node-llama-cpp's approach faithfully — a big jump in complexity beyond
   anything in the previous three ports.

**Decision:** switch to [`ollama`](https://github.com/ollama/ollama-python)
for this example onward. Ollama runs a local server that keeps one model
loaded and natively queues/serves concurrent requests to it — much closer
in spirit to node-llama-cpp's shared-model design than option 1, without
the complexity of option 2. Model also changed from `DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf`
(not available as a plain local GGUF pull via Ollama's library) to
`qwen2.5-coder:7b`, pulled directly via `ollama pull qwen2.5-coder:7b`.

## Python has no top-level await

`batch.js` is a plain ES module using top-level `await`. Python has no
equivalent — `asyncio.gather` (the equivalent of `Promise.all`) must run
inside an `async def main()`, invoked via `asyncio.run(main())` at the
bottom of the file (see `if __name__ == "__main__":` in `batch.py`).

## Speedup was modest, not the dramatic ~2x the concept suggests

Ran a quick side-by-side comparison of sequential vs. concurrent calls to
the same two prompts used in `batch.py`:

```
sequential: 0.82s
concurrent: 0.70s
```

Only a ~15% improvement, not the ~2x the CONCEPT.md's timeline diagrams
illustrate. Likely explanation: these are both very short prompts (a
greeting, a one-line arithmetic question) — most of the wall-clock time is
fixed per-request overhead (prompt processing, connection handling) rather
than token generation time, so there's little generation work to actually
overlap. The concept itself (shared model, concurrent requests, GPU batch
efficiency) is real and correctly demonstrated architecturally
(`asyncio.gather` genuinely dispatches both requests concurrently to
Ollama's server rather than serializing them) — it just doesn't show a
dramatic speedup on prompts this trivial. A fairer test would use longer,
generation-heavy prompts.
