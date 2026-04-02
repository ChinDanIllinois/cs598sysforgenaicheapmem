#!/bin/bash

# 1. Set environment for CPU
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/users/cud2/vllm-playground/.venv/lib/libiomp5.so:$LD_PRELOAD
export VLLM_CPU_OMP_THREADS_BIND=1

MODEL="Qwen/Qwen2.5-3B-Instruct"
PORT=8000

echo "Starting vLLM CPU server for $MODEL on port $PORT..."

# Using explicit --device cpu for x86_64 clusters
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --max-model-len 8192 \
    --trust-remote-code
