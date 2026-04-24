# LightMem Benchmark — How to Run

## Overview

This benchmark evaluates the impact of LLMLingua-2 pre-computation (streaming compression) on LightMem’s write latency.

It compares two operational modes:

- **Baseline mode**: compression occurs inside `add_memory()`.
- **Streaming mode**: compression happens earlier via `ingest_message_stream()`.

## Requirements

1. Python 3.9 or higher

2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```


## Set Environment Variables

Set the required variables before running the script:

```bash
export DATA_PATH=/path/to/longmemeval_s.json
export OLLAMA_MODEL_NAME=phi
export OLLAMA_HOST=http://localhost:11434
```

Optional variables:

```bash
export LLMLINGUA_MODEL_PATH=microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
export EMBEDDING_MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
export QDRANT_DATA_DIR=/path/to/qdrant_data
```

## Run the Benchmark

From the script directory:

```bash
python benchmark.py
```

Or using the full path:

```bash
python /path/to/LightMem/examples/benchmark.py
```

## What the Script Does

- Loads the dataset from `DATA_PATH`
- Iterates through conversations (sessions and turns)
- Runs multiple experiment configurations
- For each configuration:
  - runs the baseline
  - runs the streaming pre-compute
  - measures latency and runtime

## Output

Results are saved to `LightMem/results/`.

Generated files include:

- `.csv` — for analysis
- `.json` — full structured output

> Note: results are also printed directly in the terminal.

## Modes Explained

| Mode      | Description                                                      | Latency Impact       |
|-----------|------------------------------------------------------------------|----------------------|
| Baseline  | Compression occurs inside `add_memory()`                        | Higher write latency |
| Streaming | Compression occurs earlier via `lightmem.ingest_message_stream()` | Very low write latency |

## Modifying Dataset Size

Inside `benchmark.py`, adjust `experiment_grid` to scale your test:

```python
experiment_grid = [
    {"max_items": 1, "max_sessions_per_item": 1},
]
```

Increase these values for larger experiments.
