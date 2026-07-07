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

## Verified end-to-end

Ran `simple_agent.py` against the real prompt ("What time is it right
now?"). The model correctly called `get_current_time`, got back a raw
12-hour string (e.g. `"11:45:53 AM"`), and converted it per the system
prompt to `"11:47"` — 24-hour format, no seconds, matching the JS
example's expected output exactly.
