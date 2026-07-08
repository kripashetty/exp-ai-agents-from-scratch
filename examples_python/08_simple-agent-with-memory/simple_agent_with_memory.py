import asyncio

from ollama import AsyncClient

from memory_manager import MemoryManager

MODEL = "qwen3:1.7b-q8_0"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save important information to long-term memory (user preferences, facts, personal details)",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["fact", "preference"]},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["type", "key", "value"],
            },
        },
    }
]


async def run_prompt(client: AsyncClient, messages: list, memory_manager: MemoryManager) -> str:
    response = await client.chat(model=MODEL, messages=messages, tools=TOOLS)

    if response.message.tool_calls:
        messages.append(response.message.model_dump())
        for call in response.message.tool_calls:
            args = call.function.arguments
            memory_manager.add_memory(type=args["type"], key=args["key"], value=args["value"])
            result = f"Memory saved: {args['key']} = {args['value']}"
            messages.append({"role": "tool", "content": result, "tool_name": call.function.name})

        response = await client.chat(model=MODEL, messages=messages, tools=TOOLS)

    messages.append(response.message.model_dump())
    return response.message.content


async def main() -> None:
    client = AsyncClient()

    memory_manager = MemoryManager("./agent-memory.json")

    # Load existing memories and add to system prompt
    memory_summary = memory_manager.get_memory_summary()

    system_prompt = f"""
You are a helpful assistant with long-term memory.

Before calling any function, always follow this reasoning process:

1. **Compare** new user statements against existing memories below.
2. **If the same key and value already exist**, do NOT call save_memory again.
   - Instead, simply acknowledge the known information.
   - Example: if the user says "My name is Malua" and memory already says "user_name: Malua", reply "Yes, I remember your name is Malua."
3. **If the user provides an updated value** (e.g., "I actually prefer sushi now"),
   then call save_memory once to update the value.
4. **Only call save_memory for genuinely new information.**

When saving new data, call save_memory with structured fields:
- type: "fact" or "preference"
- key: short descriptive identifier (e.g., "user_name", "favorite_food")
- value: the specific information (e.g., "Malua", "chinua")

Examples:
save_memory({{ "type": "fact", "key": "user_name", "value": "Malua" }})
save_memory({{ "type": "preference", "key": "favorite_food", "value": "chinua" }})

{memory_summary}
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Example conversation
    prompt1 = "Hi! My name is Alex and I love pizza."
    messages.append({"role": "user", "content": prompt1})
    response1 = await run_prompt(client, messages, memory_manager)
    print("AI: " + response1)

    # Later conversation (even after restarting the script)
    prompt2 = "What's my favorite food?"
    messages.append({"role": "user", "content": prompt2})
    response2 = await run_prompt(client, messages, memory_manager)
    print("AI: " + response2)


if __name__ == "__main__":
    asyncio.run(main())
