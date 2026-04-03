#!/bin/bash


# 1. Set environment for CPU
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/users/cud2/vllm-playground/.venv/lib/libiomp5.so:$LD_PRELOAD
export VLLM_CPU_OMP_THREADS_BIND=1

# 2. Parallelism (Use SCATTER to hit all 32 cores)
CORES=32
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
# UNSET conflicting variables if they were in your shell session
unset VLLM_CPU_OMP_THREADS_BIND
unset KMP_AFFINITY
unset OMP_PROC_BIND
unset OMP_PLACES


VLLM_MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
PORT=8000

echo "Starting vLLM CPU server on $CORES cores..."
# Use numactl to ensure memory is shared across the whole node
# (If numactl is not installed, the server will still run)
numactl --interleave=all vllm serve \
    --model "$VLLM_MODEL_NAME" \
    --port "$PORT" \
    --max-model-len 32768 \
    --trust-remote-code