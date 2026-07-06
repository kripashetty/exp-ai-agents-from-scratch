# Porting notes: 04_think

## Same chat-template bug as 03_translation, much worse impact here

Swapping `think.py`'s model to `hf_giladgd_Apertus-8B-Instruct-2509.Q6_K.gguf`
(same model used in `03_translation`) reproduces the chat-template issue
documented in [`../03_translation/NOTES.md`](../03_translation/NOTES.md):
this GGUF has no embedded `tokenizer.chat_template` metadata, so
`create_chat_completion` silently falls back to an incorrect Llama-2-style
`[INST] <<SYS>>` template that the model was never trained on.

**Here the impact is much worse than in `03_translation`.** The response
came back as **274.6KB** of text. The model actually reaches a correct-ish
answer (`4`) fairly early in the output, but then never recognizes a clean
stop point — because the malformed prompt gives it no proper turn boundary
— and keeps rambling: literal `<</SYS>>` template fragments leak into the
text, followed by the model repeating "The final answer is 4. You need 4
whole bags of potatoes." over and over until generation is finally cut off.
This is why the "bigger model" run felt dramatically slower than expected:
it wasn't just Apertus being an 8B model vs. Qwen3's 1.7B, it was
generating hundreds of times more tokens than a clean answer needs.

**Fix (same as `03_translation`, not yet applied):** bypass
`create_chat_completion` for this model and use the model's real role tags
with the lower-level `create_completion` instead:

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

This was already verified working cleanly for `03_translation`; the same
fix should resolve both the stray template text and the runaway output
length here.
