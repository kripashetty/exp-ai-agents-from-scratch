import asyncio

from ollama import AsyncClient

MODEL = "gpt-oss:20b"


async def main() -> None:
    client = AsyncClient()

    prompt = "What is hoisting in JavaScript? Explain with examples."

    full_response = ""
    thinking_started = False

    stream = await client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        # Tip: let Ollama choose or cap reasonably; using the whole context size can be wasteful
        options={"num_predict": 2000},
    )

    async for chunk in stream:
        # gpt-oss (Harmony format) streams two separate channels: "thinking"
        # (internal reasoning) and "content" (the final answer). Show both
        # live, like onTextChunk would, rather than silently dropping
        # thinking and only showing content once it starts.
        thinking = chunk["message"].get("thinking")
        if thinking:
            if not thinking_started:
                print("[thinking] ", end="", flush=True)
                thinking_started = True
            print(thinking, end="", flush=True)

        text = chunk["message"]["content"]
        if text:
            if thinking_started:
                print("\n\n[answer] ", end="", flush=True)
                thinking_started = False
            full_response += text
            print(text, end="", flush=True)  # optional: live print

    print("\n\nFinal answer:\n", full_response)


if __name__ == "__main__":
    asyncio.run(main())
