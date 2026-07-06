import asyncio

from ollama import AsyncClient

MODEL = "qwen2.5-coder:7b"


async def ask(client: AsyncClient, prompt: str) -> str:
    response = await client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


async def main() -> None:
    client = AsyncClient()

    q1 = "Hi there, how are you?"
    q2 = "How much is 6+6?"

    print("Batching started...")

    a1, a2 = await asyncio.gather(
        ask(client, q1),
        ask(client, q2),
    )

    print("User: " + q1)
    print("AI: " + a1)

    print("User: " + q2)
    print("AI: " + a2)


if __name__ == "__main__":
    asyncio.run(main())
