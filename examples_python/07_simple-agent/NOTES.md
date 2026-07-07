# Porting notes: 07_simple-agent

## The core porting challenge: no automatic tool-call loop

`simple-agent.js` calls tools through `session.prompt(prompt, {functions})`.
That single call handles the *entire* decide → call → respond loop
internally: the model decides to call `getCurrentTime`, node-llama-cpp
intercepts that decision, runs the JS `handler()`, feeds the result back
to the model, and only returns once the model has produced its final
text answer.

Ollama's `client.chat(messages, tools=[...])` only does the first half of
that. It returns a response where `message.tool_calls` may be populated —
but it does **not** execute anything or loop back on its own. Getting a
final answer requires writing that loop by hand:

1. Send the initial `chat()` call with `tools=` attached.
2. If `response.message.tool_calls` is non-empty, run the matching local
   Python function(s) yourself.
3. Append **both** the assistant's message (containing the tool call
   request) *and* a new `{"role": "tool", "content": ..., "tool_name": ...}`
   message with the result back onto `messages`.
4. Call `chat()` again with the updated `messages` to get the model's
   final natural-language answer.

This matches what we predicted while summarizing the JS example before
porting it, and is the same category of gap as `05_batch`'s missing
multi-sequence API — node-llama-cpp's `LlamaChatSession` bundles more
agent-loop logic into one call than Ollama's lower-level `chat()` does.

## Tool schema shape differs

node-llama-cpp's `defineChatSessionFunction` ties `description` +
`params` (JSON Schema) + `handler` together in one object, keyed by name
in a plain `functions = {getCurrentTime}` map:

```js
const getCurrentTime = defineChatSessionFunction({
    description: "Get the current time",
    params: { type: "object", properties: {} },
    async handler() { return new Date().toLocaleTimeString(); }
});
```

Ollama expects OpenAI-style tool schemas — a list of
`{"type": "function", "function": {"name", "description", "parameters"}}`
objects, with the handler kept completely separate (just a local Python
dict lookup by name, `AVAILABLE_FUNCTIONS[call.function.name]`):

```python
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}]
```

Note the explicit `"name"` field is required in Ollama's schema (the
function name doubles as its Python dict key and its identifier in
`tool_calls`) — there's no equivalent in node-llama-cpp's version, where
the object key in `functions = {getCurrentTime}` already serves that role.

## `tool_calls` arrive pre-parsed

`response.message.tool_calls` is a list of `ToolCall` objects, each with
`.function.name` (string) and `.function.arguments` (already a plain
`dict`, not a JSON string needing `json.loads`) — confirmed empirically:

```python
ToolCall(function=Function(name='get_current_time', arguments={}))
```

So calling the matched handler is just `handler(**call.function.arguments)`.

## General concepts: how tool calling actually works

Beyond the JS→Python porting mechanics above, some underlying concepts
worth capturing since they apply to every future tool-using example, not
just this one.

### The typical flow

1. **Define tools upfront** — name, plain-English `description` (the
   model's *only* signal for when to call it — vague descriptions cause
   wrong or missed calls), and a `parameters` JSON Schema. The executable
   handler is kept separate from this schema.
2. **Send the conversation + tool schemas to the model.** It decides:
   answer directly, or request a tool call (tool + arguments).
3. **Check the response for a tool call.** No tool call → done, return the
   text answer.
4. **Execute the tool(s) yourself** — nothing runs your code for you
   unless you're using a high-level wrapper like `LlamaChatSession` that
   hides this loop (see above).
5. **Feed the result back** — append both the assistant's tool-call
   request *and* a new `{"role": "tool", ...}` message onto history.
6. **Send the updated conversation back.** The model either gives a final
   answer or requests another tool call.
7. **Loop until there's no more tool call.** `simple_agent.py` only needs
   one round (steps 3-6 execute once); `09_react-agent`'s whole point is
   formalizing this as an explicit repeated loop (Thought → Action →
   Observation).

### The model doesn't "know" tool calling innately — it's learned behavior

An LLM's only mechanism is next-token prediction. Tool calling is
something specific fine-tuning data teaches: examples pairing (a block of
text describing available tools) with (a specially-formatted response,
e.g. Qwen's `<tool_call>{"name":..., "arguments":...}</tool_call>`, or
Harmony's channel syntax for gpt-oss) instead of plain prose. A model never
fine-tuned on this pattern won't reliably do it no matter what schema you
send it.

The JSON Schema shape we send (`{"type": "function", "function": {...}}`)
isn't derived from first principles — it's the convention OpenAI's
function-calling API popularized, now mirrored by most other providers
(Ollama, Anthropic, llama.cpp) for compatibility. It's documented per
provider, not something to guess at.

Critically, **passing the right schema is only useful if the model's chat
template correctly translates it into the exact raw-text tool-definition
format that specific model was fine-tuned on** — the same translation
layer responsible for the Apertus chat-template bug in `03_translation`.
If a model's template doesn't know how to serialize `tools=[...]`, passing
it can silently do nothing or produce garbage. Symmetrically, the
inference library has to *parse* the model's raw special-token tool-call
output back into the clean `tool_calls` structure we get — also
template/handler-specific. This is why we verified tool calling
empirically with a standalone test script against `qwen3:1.7b` *before*
writing `simple_agent.py`, rather than assuming `tools=[...]` would work.

### Why the assistant's tool-call message must stay in history

The second `chat()` call needs the *full* history, including the
assistant's own tool-call request — not just the tool result in isolation:

```python
messages = [
    {"role": "system", ...},
    {"role": "user", "content": "What time is it right now?"},
    response.message.model_dump(),  # assistant: "I'm calling get_current_time()"
    {"role": "tool", "content": "11:47:53 AM", "tool_name": "get_current_time"},
]
```

Without the assistant message, the conversation would look like a `tool`
result appearing out of nowhere, with no way for the model to connect it
to a request it made — a conversation shape it wasn't trained on. The
`user → assistant(tool_calls) → tool(result)` sequence mirrors exactly how
the model learned to produce a coherent final answer from a tool result.

### Stop conditions for the tool-call loop

The *implicit* stop signal is simply the model no longer requesting a tool
call (`if not response.message.tool_calls: return ...`). But relying on
that alone is risky — a model can get stuck calling the same tool
repeatedly, alternate between tools indefinitely, or never converge. Real
implementations add explicit guards:

- **Max iteration count** — cap the loop (e.g. 5-10 rounds), raise/bail
  after that. The most common and simplest guard.
- **Duplicate-call detection** — track `(tool_name, arguments)` pairs
  already executed; an exact repeat signals it's stuck.
- **Token/context budget** — bail before the growing message history
  overflows the model's context window.
- **Wall-clock/time budget** — useful when tool calls themselves are slow
  (e.g. network requests).
- **An explicit "finish" tool** (e.g. `submit_final_answer(answer)`) —
  some agent designs prefer this over relying on "no tool call = done,"
  for an unambiguous, machine-checkable stop signal.

`simple_agent.py` doesn't need any of these since it's a fixed one-tool,
one-round scenario (a plain `if`, not a loop) — but `09_react-agent` is
exactly where a real loop-with-a-cap becomes the point of the example, and
should implement one of these guards as a first-class part of the port.

## Verified end-to-end

Ran `simple_agent.py` against the real prompt ("What time is it right
now?"). The model correctly called `get_current_time`, got back a raw
12-hour string (e.g. `"11:45:53 AM"`), and converted it per the system
prompt to `"11:47"` — 24-hour format, no seconds, matching the JS
example's expected output exactly.
