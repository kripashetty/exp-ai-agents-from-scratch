# Porting notes: 01_intro

## `n_ctx` default differs from `node-llama-cpp`

`node-llama-cpp`'s `loadModel`/`createContext` (used in [`../../examples/01_intro/intro.js`](../../examples/01_intro/intro.js))
defaults to the model's full trained context size from the GGUF metadata.

`llama_cpp.Llama`, the Python equivalent, defaults `n_ctx=512`
(`llama_cpp/llama.py:75`) — a small, fixed window unrelated to the model.

Qwen3-1.7B is a reasoning model: it emits a `<think>...</think>` block before
its actual answer. With the 512-token default, that reasoning alone filled
the entire context budget and the response was cut off mid-sentence with no
final answer ever produced — confirmed by the eval stats from that run
(`17` prompt tokens + `494` generated tokens = `511`, right at the wall).

**Fix:** pass `n_ctx=0` to `Llama(...)`, which tells `llama-cpp-python` to
read the context size from the model itself (`llama.py:393-396`,
`n_ctx = self._model.n_ctx_train()`) — the closest match to
`node-llama-cpp`'s default behavior.
