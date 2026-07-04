from pathlib import Path

from llama_cpp import Llama

model_path = Path(__file__).resolve().parent.parent.parent / "models" / "Qwen3-1.7B-Q8_0.gguf"

llm = Llama(model_path=str(model_path), verbose=False, n_ctx=0, n_gpu_layers=-1)

prompt = "do you know node-llama-cpp"

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt}],
)

print("AI: " + response["choices"][0]["message"]["content"])
