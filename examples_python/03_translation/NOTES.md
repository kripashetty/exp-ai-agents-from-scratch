# Porting notes: 03_translation

## Stray `[INST] <<SYS>>` template leakage in output (not yet fixed)

`translation.py` currently uses `llm.create_chat_completion(messages=[...])`,
which produced output with a stray `[INST] <<SYS>>` fragment at the top,
ahead of the actual translation.

**Root cause:** `create_chat_completion` needs to know how to format
`messages` into the raw prompt text the model was trained on. It looks for
a `tokenizer.chat_template` field in the GGUF's metadata to do this. This
particular Apertus-8B GGUF conversion (`hf_giladgd_Apertus-8B-Instruct-2509.Q6_K.gguf`)
has **no such metadata at all** — confirmed via `llm.metadata`, which has no
`chat_template` key (unlike the Qwen3 model used in `01_intro`, which does).

When no template is found and nothing can be guessed, `llama_cpp/llama.py:555-556`
silently falls back to a hardcoded `chat_format = "llama-2"`, wrapping
messages in `[INST] <<SYS>> ... <</SYS>> ... [/INST]` — the Llama-2
convention. Apertus was never trained on that syntax, so the model gets
confused by the malformed prompt and starts echoing/continuing
template-like text instead of just answering.

**What the model actually expects:** inspecting the GGUF's vocabulary for
special tokens shows Apertus has its own purpose-built role tags:
`<|system_start|>`, `<|system_end|>`, `<|user_start|>`, `<|user_end|>`,
`<|assistant_start|>`, `<|assistant_end|>` (plus `<|developer_start|>`,
`<|tools_prefix|>`, etc. for other roles) — a custom schema, not Llama-2's.

**Potential fix (verified working in a standalone test, not yet applied
to `translation.py`):** bypass `create_chat_completion` for this model.
Manually build the prompt using the model's real tags, and use the
lower-level `create_completion` (plain text-in/text-out, no chat-format
guessing) instead:

```python
formatted_prompt = (
    f"<|system_start|>{system_prompt}<|system_end|>"
    f"<|user_start|>{prompt}<|user_end|>"
    f"<|assistant_start|>"
)

response = llm.create_completion(
    formatted_prompt,
    stop=["<|assistant_end|>"],
)

print("AI: " + response["choices"][0]["text"].strip())
```

Tested against the model directly with a small translation
("The weather is nice today." → "Das Wetter ist heute schön.") and produced
clean output with no template artifacts.
