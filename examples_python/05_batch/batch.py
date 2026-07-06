import asyncio
import time

from ollama import AsyncClient

MODEL = "qwen2.5-coder:7b"


async def ask(client: AsyncClient, label: str, prompt: str, batch_start: float) -> str:
    request_start = time.monotonic() - batch_start
    response = await client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    duration = time.monotonic() - batch_start - request_start
    print(f"[{label}] started at t=+{request_start:.2f}s, took {duration:.2f}s")
    return response["message"]["content"]


async def main() -> None:
    client = AsyncClient()

    q1 = "Write a 200 word explanation of how binary search works."
    q2 = "Write a 200 word explanation of how quicksort works."

    print("Batching started...")

    batch_start = time.monotonic()
    a1, a2 = await asyncio.gather(
        ask(client, "q1", q1, batch_start),
        ask(client, "q2", q2, batch_start),
    )
    print(f"Total batch time: {time.monotonic() - batch_start:.2f}s")

    print("User: " + q1)
    print("AI: " + a1)

    print("User: " + q2)
    print("AI: " + a2)


if __name__ == "__main__":
    asyncio.run(main())
