# Python examples

Python ports of the JS examples in [`../examples`](../examples), for practice.

- **01_intro, 03_translation, 04_think** use
  [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) as the closest
  Python equivalent to `node-llama-cpp` for local GGUF models.
- **05_batch onward** use [`ollama`](https://github.com/ollama/ollama-python) instead —
  Ollama runs a local server that keeps one model loaded and natively handles
  concurrent requests to it, which is a closer match to node-llama-cpp's
  shared-model/multiple-sequence design than anything `llama-cpp-python`'s
  high-level API exposes.

## Setup

```bash
cd examples_python
uv sync
```

Models for `llama-cpp-python` examples are read from the repo-root `models/`
directory (not duplicated here) — see [`../DOWNLOAD.md`](../DOWNLOAD.md) for
how to fetch them.

Examples using Ollama require the Ollama server running locally
(`ollama serve`) and the relevant model pulled, e.g.:

```bash
ollama pull qwen2.5-coder:7b
```

## Running an example

```bash
uv run 01_intro/intro.py
```
