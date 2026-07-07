import asyncio
from datetime import datetime

from ollama import AsyncClient

MODEL = "qwen3:1.7b"

SYSTEM_PROMPT = """You are a professional chronologist who standardizes time representations across different systems.

Always convert times from 12-hour format (e.g., "1:46:36 PM") to 24-hour format (e.g., "13:46") without seconds
before returning them."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]


def get_current_time() -> str:
    return datetime.now().strftime("%I:%M:%S %p")


AVAILABLE_FUNCTIONS = {"get_current_time": get_current_time}


async def main() -> None:
    client = AsyncClient()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What time is it right now?"},
    ]

    response = await client.chat(model=MODEL, messages=messages, tools=TOOLS)

    if response.message.tool_calls:
        # Unlike node-llama-cpp's session.prompt(prompt, { functions }), which
        # runs the whole decide -> call -> respond loop internally, Ollama's
        # chat() only returns the tool call request - executing it and
        # feeding the result back is on us.
        print(response.message.model_dump())
        messages.append(response.message.model_dump())
        for call in response.message.tool_calls:
            handler = AVAILABLE_FUNCTIONS[call.function.name]
            result = handler(**call.function.arguments)
            messages.append({"role": "tool", "content": str(result), "tool_name": call.function.name})

        response = await client.chat(model=MODEL, messages=messages, tools=TOOLS)

    print("AI: " + response.message.content)


if __name__ == "__main__":
    asyncio.run(main())
