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

## `OLLAMA_NUM_PARALLEL` must be >1, or requests are silently serialized

First timing comparison (using the short prompts in `batch.py`) only showed
a ~15% speedup, not the ~2x the concept implies:

```
sequential: 0.82s
concurrent: 0.70s
```

Checking the Ollama server log (`~/.ollama` startup log) explained why —
the server was running with the default `OLLAMA_NUM_PARALLEL:1`, meaning
it only processes **one request at a time per model**, no matter how many
concurrent requests the client sends. `asyncio.gather` was dispatching
both requests concurrently on the client side, but the server queued and
ran them one after another anyway. The small ~15% gain was just overlapping
network/queue overhead, not real parallel inference.

**Fix:** restart the server with `OLLAMA_NUM_PARALLEL=2` (or higher):

```bash
OLLAMA_NUM_PARALLEL=2 ollama serve
```

Re-tested with longer, generation-heavy prompts and per-request start/end
timestamps to directly verify overlap — `batch.py` now uses these longer
prompts ("Write a 200 word explanation of how binary search/quicksort
works") and prints this instrumentation directly, rather than needing a
separate diagnostic script:

```
[q1] started at t=+0.00s, took 3.10s
[q2] started at t=+0.00s, took 3.17s
Total batch time: 3.17s
```

Both concurrent requests started at the **exact same timestamp** (`t=+0.00s`
for both), and the concurrent total (3.17s) matches `max(3.10s, 3.17s)`
rather than the sum (`6.27s`) — the unambiguous signature of genuine
parallel execution on the server, not just concurrent dispatch from the
client. This confirms the port's `asyncio.gather` pattern does achieve the
same effect as `Promise.all` in the JS version, **but only when the Ollama
server itself is configured to allow more than one parallel slot.**
`batch.py`'s printed timings make this directly observable on every run;
the README documents the `OLLAMA_NUM_PARALLEL` requirement so the demo
doesn't silently run sequentially for someone using the server's default.
