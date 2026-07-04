from pathlib import Path

from llama_cpp import Llama

model_path = Path(__file__).resolve().parent.parent.parent / "models" / "Qwen3-1.7B-Q8_0.gguf"

llm = Llama(model_path=str(model_path), verbose=False, n_ctx=0, n_gpu_layers=-1)

system_prompt = """You are an expert logical and quantitative reasoner.
    Your goal is to analyze real-world word problems involving families, quantities, averages, and relationships
    between entities, and compute the exact numeric answer.

    Goal: Return the correct final number as a single value - no explanation, no reasoning steps, just the answer.
    """

prompt = """My family reunion is this week, and I was assigned the mashed potatoes to bring.
The attendees include my married mother and father, my twin brother and his family, my aunt and her family, my grandma
and her brother, her brother's daughter, and his daughter's family. All the adults but me have been married, and no one
is divorced or remarried, but my grandpa and my grandma's sister-in-law passed away last year. All living spouses are attending.
My brother has two children that are still kids, my aunt has one six-year-old, and my grandma's brother's daughter has
three kids under 12. I figure each adult will eat about 1.5 potatoes and each kid will eat about 1/2 a potato, except my
second cousins don't eat carbs. The average potato is about half a pound, and potatoes are sold in 5-pound bags.

How many whole bags of potatoes do I need?
"""

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ],
)

print(f"AI: {response['choices'][0]['message']['content']}")
