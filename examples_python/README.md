# Python examples

Python ports of the JS examples in [`../examples`](../examples), for practice. Uses
[`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) as the closest
Python equivalent to `node-llama-cpp` for local GGUF models.

## Setup

```bash
cd examples_python
uv sync
```

Models are read from the repo-root `models/` directory (not duplicated here) —
see [`../DOWNLOAD.md`](../DOWNLOAD.md) for how to fetch them.

## Running an example

```bash
uv run 01_intro/intro.py
```
