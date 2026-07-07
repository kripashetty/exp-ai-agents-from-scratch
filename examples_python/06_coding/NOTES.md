# Porting notes: 06_coding

## What is the Harmony format?

Harmony is the conversation format OpenAI trained `gpt-oss` models on. An
LLM has no innate concept of "system message" or "turn" — chat/instruct
fine-tuning teaches it a specific textual convention (special tokens
marking role/turn boundaries) that it learns to expect. A "chat wrapper" /
"chat template" is the client-side code that serializes structured
`messages=[{role, content}, ...]` into that exact expected text shape. Get
the shape wrong and the model doesn't recognize the conversation structure
(concretely demonstrated with Apertus in `03_translation/NOTES.md`, where
the wrong fallback template caused rambling, un-stoppable output).

Harmony is a more elaborate shape than a plain system/user/assistant
template:
- **Multiple channels within one assistant turn** — `analysis` (the
  model's internal reasoning) and `final` (the actual answer), structurally
  separated in the raw text via special tokens (`<|channel|>analysis...`,
  `<|channel|>final...`), not just informally wrapped in `<think>` tags
  the way Qwen3 does.
- **A `developer` role** distinct from `system`, for higher-priority
  instructions layered above a regular system prompt.
- **Structured tool-call support** via a `commentary` channel carrying
  call metadata, rather than tools being bolted on separately.

Because this is a different *shape* of conversation, not just different
token spellings of the same idea, generic auto-detected templates usually
can't handle it — you need code that specifically understands "there are
channels, here's how to route them." That's what `HarmonyChatWrapper`
(node-llama-cpp) and Ollama's `gpt-oss:20b` library entry both provide,
each on their own side.

## HarmonyChatWrapper isn't needed — Ollama's gpt-oss entry handles it

`coding.js` explicitly sets `chatWrapper: new HarmonyChatWrapper()` because
gpt-oss models are trained on OpenAI's "Harmony" conversation format, not a
plain system/user/assistant template — node-llama-cpp needs to be told
explicitly how to serialize messages into that shape.

Ollama's `gpt-oss:20b` library entry has Harmony-aware handling built into
the server itself, so `client.chat(messages=[...])` works without
specifying any wrapper — same category of simplification as `qwen2.5-coder`
and `Qwen3` "just working" via their embedded/library chat templates,
versus Apertus needing manual template handling (see `03_translation/NOTES.md`).

## gpt-oss separates "thinking" from "content" — and the first version of
## this port silently dropped it

Harmony's format has distinct channels: an internal `analysis` (reasoning)
channel and a `final` (answer) channel. Ollama exposes this directly as two
separate fields on every streamed chunk: `message.thinking` and
`message.content`. Checked with a quick script:

```python
r = await client.chat(model="gpt-oss:20b", messages=[...])
print(r.message.model_dump())
# {'content': '4', 'thinking': 'We need to answer briefly...', ...}
```

Streaming chunks confirmed both fields arrive incrementally — `thinking`
deltas first, then `content` deltas once reasoning finishes (a final chunk
has `thinking: None` once `content` starts).

The first version of `coding.py` only read `chunk["message"]["content"]`,
completely ignoring `thinking`. In practice this means the script would sit
silently for a while (while gpt-oss "thinks") before any output appeared —
not a faithful port of the JS version's `onTextChunk`, which is meant to
show generation progress immediately.

**Fix (applied):** print both channels as they stream, labeled
`[thinking]` then `[answer]`, so the live-progress behavior `onTextChunk`
is meant to demonstrate is actually visible the whole time, not just once
the model starts producing its final answer.

## Full response structure (`ollama`'s ChatResponse)

`message.thinking`/`message.content` are only part of what a `chat()` call
returns. The `ollama` client returns Pydantic models, not plain dicts, so
`r["message"]["content"]`-style indexing only shows you the field you ask
for — to see everything at once, use `model_dump()`:

```python
import json
r = await client.chat(model="gpt-oss:20b", messages=[...])
print(json.dumps(r.model_dump(), indent=2, default=str))
```

```json
{
  "model": "gpt-oss:20b",
  "created_at": "2026-07-07T09:09:49.188363Z",
  "done": true,
  "done_reason": "stop",
  "total_duration": 5735318750,
  "load_duration": 4950736500,
  "prompt_eval_count": 77,
  "prompt_eval_duration": 243318292,
  "eval_count": 31,
  "eval_duration": 423372666,
  "message": {
    "role": "assistant",
    "content": "4",
    "thinking": "The user asks \"What is 2+2? Answer briefly.\" The answer is 4.",
    "images": null,
    "tool_name": null,
    "tool_calls": null
  },
  "logprobs": null
}
```

Field notes:
- `done` / `done_reason` — `"stop"` means the model hit a natural stop
  token; `"length"` would mean `num_predict` cut it off instead — useful
  for detecting when the token cap actually kicked in (see below).
- `*_duration` fields are all in **nanoseconds**. `load_duration` dominates
  `total_duration` on a cold call (model has to load into memory first);
  `eval_duration` is the actual token-generation time.
- `prompt_eval_count` / `eval_count` — tokens in the prompt vs. tokens
  generated, Ollama's equivalent of OpenAI's `usage.total_tokens` from
  `02_openai-intro`.
- `tool_calls` / `tool_name` populate when the model invokes a tool —
  relevant once we port the agent/tool-use examples (07+).
- For `stream=True`, each yielded chunk has this same shape, but `done` is
  `False` until the final chunk, and the `*_duration`/`eval_count` fields
  are only populated on that last chunk.

## What is "runaway generation", and how do maxTokens / num_predict prevent it?

An LLM generates one token at a time and keeps going, feeding each token
back in as input for the next, until it either produces a special
"end of turn" token on its own or an external limit stops it. Nothing
inherently guarantees it'll choose to stop at a reasonable length —
"runaway generation" is when it just keeps producing text well past the
point of being useful: padding out an already-complete answer, repeating
itself, or (as seen very concretely with Apertus's chat-template bug in
`03_translation`/`04_think`) rambling indefinitely because it never
recognizes a clean stopping point in the first place.

`maxTokens` (node-llama-cpp) / `num_predict` (Ollama) is the blunt, reliable
fix: a hard ceiling on how many tokens a single generation call is allowed
to produce, enforced by the inference library itself regardless of whether
the model "wants" to keep going. It trades a small risk of cutting off a
genuinely-still-useful answer (as happened in our test run below) for a
hard guarantee against unbounded cost/latency — cheap insurance, which is
why `coding.js`'s CONCEPT.md lists "always set maxTokens" as a best
practice rather than an optional tuning knob.

## Verified end-to-end

Ran `coding.py` against the real prompt ("What is hoisting in JavaScript?
Explain with examples."). Output showed a lengthy `[thinking]` block
(the model planning its answer), followed by a well-structured `[answer]`
covering `var`/`let`/`const`, function vs. expression hoisting, TDZ, and
classes — cut off cleanly at the `num_predict: 2000` token cap (Ollama's
equivalent of `maxTokens`), matching the JS example's intent of preventing
runaway generation.
