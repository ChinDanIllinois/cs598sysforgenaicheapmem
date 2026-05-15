# CS 598 - Systems for GenAI: LightMem Profiling Setup

This guide provides a comprehensive walkthrough for setting up and running the LightMem multitenant profiler on NCSA (which is what we used for our profiling experiments). It covers environment configuration, dependency management, and execution steps. We used the VS Code interface with an A100 GPU on NCSA to run our experiments.

## 1. Prerequisites

### Install `uv`
The project uses `uv` for fast Python package management.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install `git-lfs` (Local)
If `git-lfs` is not available globally, install it in your home directory:
```bash
wget https://github.com/git-lfs/git-lfs/releases/download/v3.2.0/git-lfs-linux-amd64-v3.2.0.tar.gz
tar xvf git-lfs-linux-amd64-v3.2.0.tar.gz
cd git-lfs-3.2.0/
chmod +x install.sh
sed -i 's|^prefix="/usr/local"$|prefix="$HOME/.local"|' install.sh
mkdir -p ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"
./install.sh
git-lfs --version
cd ~
```

## 2. Environment Setup

The system requires two separate virtual environments to manage dependencies effectively.

### vLLM Environment (Python 3.12)
Used for serving models via vLLM. Make sure to get the latest build that support the /batch endpoint
```bash
uv venv vllm-venv --python 3.12 --seed --managed-python
source vllm-venv/bin/activate

uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129

deactivate
```

### LightMem Environment (Python 3.11)
Used for the core LightMem logic and profiling.
```bash
uv venv --python 3.11 lightmem-venv
source lightmem-venv/bin/activate
uv pip install --upgrade pip setuptools wheel cffi

# Clone and install the repository
git clone https://github.com/ChinDanIllinois/cs598sysforgenaicheapmem
cd cs598sysforgenaicheapmem/

unset ALL_PROXY
pip install -e .
```

## 3. Data and Model Preparation

Clone the necessary models and datasets for the profiling tasks:
```bash
cd ~
git clone https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
git clone https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
```

## 4. Setup .env

Create a ".env" file in the root directory (i.e. inside cs598sysforgenaicheapmem/)
```bash
LLMLINGUA_MODEL_PATH="~/llmlingua-2-bert-base-multilingual-cased-meetingbank"
EMBEDDING_MODEL_PATH="~/all-MiniLM-L6-v2"
DATA_PATH="~/longmemeval-cleaned/longmemeval_s_cleaned.json"
QDRANT_DATA_DIR='~/LightMem/qdrant_data'

VLLM_MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
VLLM_BASE_URL="http://localhost:8000/v1" # or wherever your vllm is running

DASH_PROXY_PREFIX="/user/your_username/vscode/proxy/" # necessary to see dashboard on localhost on NCSA
```

## 5. Running the Profiler
Follow these steps to start the servers and initiate the profiling process.

### Step A: Start vLLM Server
```bash
vllm serve "Qwen/Qwen3-30B-A3B-Instruct-2507"
```

### Step B: Start LLMLingua-2 Server
```bash
uvicorn llmlingua_server:app --host 0.0.0.0 --port 8090 --workers 1
```

### Step C: Execute the Profiler (baseline)
```bash
python lightmem_multitenant_profiler.py \
    --provider vllm \
    --target-duration 120 \
    --concurrency-limit 64 \
    --start-date 2023-05-20 \
    --end-date 2023-05-21
```
Open up http://localhost:8501/ in your browser to see the dashboard

## Step D: Execute LightMem with our changes

```bash
python lightmem_multitenant_profiler.py \
    --provider vllm \
    --target-duration 120 \
    --concurrency-limit 64 \
    --start-date 2023-05-20 \
    --end-date 2023-05-21 \
    --vllm-adaptive-shaping \
    --streaming-precompress \
```

# 6. Running Accuracy Tests
Assuming you have all the setup from the previous steps:
```bash
python LightMem/examples/run_lightmem_vllm.py
```

# 7. Important Files
- `lightmem_multitenant_profiler.py`: Main entry point for the multitenant profiling experiments.
- `LightMem/src/lightmem/memory/lightmem.py`: Core LightMem memory management and multi-tenancy logic.
- `LightMem/examples/run_lightmem_vllm.py`: Script for running accuracy evaluation tests with vLLM.
- `llmlingua_server.py`: Server implementation for LLMLingua-2 context compression.
- `visualize_baselines.py`: Script used to generate the following 2 files.
- `baseline_comparison.html`: Comprehensive report visualizing performance comparisons between baseline and optimized runs.
- `latency_breakdowns.html`: Detailed breakdown of latency components during profiling.```
