"""
LightMem Multi-Tenant Load Test Harness
Simulates a realistic multi-tenant workload using LongMemEval dataset.
Events (historical session archiving and queries) are played back according to their timestamps.
Synced with configuration options from lightmem_profiler.py.
"""

import argparse
import asyncio
import csv
import datetime
import json
import logging
import os
import sys
import textwrap
import time
import uuid
import threading
from typing import List, Dict, Any, Tuple
from collections import deque

import dotenv
import numpy as np
import pandas as pd
from lightmem.memory.lightmem import LightMemory

# Suppress Logs
import flask.cli
flask.cli.show_server_banner = lambda *args: None
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('flask').setLevel(logging.ERROR)

dotenv.load_dotenv()

# ============================================================
# UTILS — RATE LIMITING
# ============================================================

class AsyncRateLimiter:
    """A simple token-bucket-like rate limiter for asyncio."""
    def __init__(self, rpm: float):
        self.rpm = rpm
        self.interval = 60.0 / rpm if rpm > 0 else 0
        self.last_call = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        if self.rpm <= 0:
            return
        async with self.lock:
            now = time.perf_counter()
            wait_time = self.last_call + self.interval - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_call = time.perf_counter()

# ============================================================
# CLI ARGUMENT PARSING
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LightMem Multi-Tenant Multi-User Load Test simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Provider & Backend Config (Synced with lightmem_profiler.py)
    parser.add_argument(
        "--provider",
        required=True,
        choices=["ollama", "gemini", "openai", "vllm", "mock"],
        help="LLM backend provider to use for memory management.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=1,
        help="LLM batch size for compatible providers (e.g., vllm). Default: 1 (no batching).",
    )
    parser.add_argument(
        "--llm-batch-timeout",
        type=int,
        default=10,
        help="LLM batch timeout in seconds. Default: 10.",
    )
    parser.add_argument(
        "--vllm-adaptive-shaping",
        action="store_true",
        help="Enable adaptive request shaping for vLLM (monitors engine metrics).",
    )
    parser.add_argument(
        "--vllm-metrics-url",
        type=str,
        default="",
        help="Custom metrics URL for vLLM (defaults to VLLM_BASE_URL/metrics).",
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=0,
        metavar="RPM",
        help="Global rate limit in Requests Per Minute (default: 0 = unlimited).",
    )
    
    # Multi-Tenant Simulation Control
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Path to LongMemEval dataset JSON. (Defaults to DATA_PATH env var).",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=3600.0,
        help="1 unit of dataset time (seconds) = X real-time seconds. Ignored if --target-duration is set.",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=0,
        help="Target total run time in seconds. If set, --time-scale is automatically calculated.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=50,
        help="Maximum number of users (tenants) to simulate.",
    )
    parser.add_argument(
        "--max-sessions-per-user",
        type=int,
        default=10,
        help="Maximum history sessions to load per user.",
    )
    parser.add_argument(
        "--concurrency-limit",
        type=int,
        default=10,
        help="Max simultaneous backend tasks (semaphore limit).",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Only send queries (no background archiving).",
    )
    parser.add_argument(
        "--skip-queries",
        action="store_true",
        help="Only perform archiving (no retrieval queries).",
    )
    
    # Dataset Slicing
    parser.add_argument(
        "--start-date",
        type=str,
        default="",
        help="Start date for slice (YYYY-MM-DD). If empty, starts from beginning.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="",
        help="End date for slice (YYYY-MM-DD). If empty, no end bound.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Maximum total events to include in the simulation.",
    )
    
    # Metadata & Logging
    parser.add_argument(
        "--run-name",
        type=str,
        default="multitenant_run",
        help="Label for this run, included in the CSV filename.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8052,
        help="Dashboard port (Not yet fully implemented for multitenant).",
    )
    
    return parser.parse_args()

# ============================================================
# ENV VALIDATION (Synced with lightmem_profiler.py)
# ============================================================

_SHARED_ENV_VARS = {
    "LLMLINGUA_MODEL_PATH": "Path to LLMLingua-2 model weights.",
    "EMBEDDING_MODEL_PATH": "Path to HuggingFace embedding model.",
    "QDRANT_DATA_DIR":      "Directory for Qdrant persistence.",
}

_PROVIDER_ENV_VARS = {
    "ollama": [("OLLAMA_MODEL_NAME", "Model name"), ("OLLAMA_HOST", "Host URL")],
    "gemini": [("GEMINI_MODEL_NAME", "Model name"), ("GEMINI_API_KEY", "API Key")],
    "openai": [("OPENAI_MODEL_NAME", "Model name"), ("OPENAI_API_KEY", "API Key")],
    "vllm":   [("VLLM_MODEL_NAME", "Model name"), ("VLLM_BASE_URL", "Base URL")],
    "mock":   [],
}

def validate_env(provider: str) -> None:
    missing = []
    for var, desc in _SHARED_ENV_VARS.items():
        if not os.getenv(var): missing.append(var)
    for var, desc in _PROVIDER_ENV_VARS[provider]:
        if not os.getenv(var): missing.append(var)
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

# ============================================================
# BACKEND BUILDERS
# ============================================================

def _memory_manager_config_ollama(args) -> dict:
    return {
        "model_name": "ollama",
        "configs": {
            "model": os.getenv("OLLAMA_MODEL_NAME"),
            "host":  os.getenv("OLLAMA_HOST"),
            "max_tokens": 16384,
        },
    }

def _memory_manager_config_gemini(args) -> dict:
    return {
        "model_name": "gemini",
        "configs": {
            "model":      os.getenv("GEMINI_MODEL_NAME"),
            "api_key":    os.getenv("GEMINI_API_KEY"),
            "max_tokens": 16384,
        },
    }

def _memory_manager_config_vllm(args) -> dict:
    return {
        "model_name": "vllm",
        "configs": {
            "model":      os.getenv("VLLM_MODEL_NAME"),
            "api_key":    os.getenv("VLLM_API_KEY", "EMPTY"),
            "max_tokens": 16384,
            "vllm_base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            "llm_batch_size": args.llm_batch_size,
            "llm_batch_timeout": args.llm_batch_timeout,
            "vllm_adaptive_shaping": args.vllm_adaptive_shaping,
            "vllm_metrics_url": args.vllm_metrics_url if args.vllm_metrics_url else None,
        },
    }

def _memory_manager_config_openai(args) -> dict:
    return {
        "model_name": "openai",
        "configs": {
            "model":      os.getenv("OPENAI_MODEL_NAME"),
            "api_key":    os.getenv("OPENAI_API_KEY"),
            "max_tokens": 16384,
        },
    }

def _memory_manager_config_mock(args) -> dict:
    return {
        "model_name": "mock",
        "configs": {"model": "mock-model"},
    }

_BUILDERS = {
    "ollama": _memory_manager_config_ollama,
    "gemini": _memory_manager_config_gemini,
    "openai": _memory_manager_config_openai,
    "vllm":   _memory_manager_config_vllm,
    "mock":   _memory_manager_config_mock,
}

# ============================================================
# DATA PREPARATION
# ============================================================

def parse_date(date_str: str) -> float:
    try:
        parts = date_str.split(" ")
        clean_str = f"{parts[0]} {parts[2]}" 
        dt = datetime.datetime.strptime(clean_str, "%Y/%m/%d %H:%M")
        return dt.timestamp()
    except:
        return time.time()

def load_events(data_path: str, args):
    if not data_path or not os.path.exists(data_path):
        print(f"ERROR: Data path {data_path} not found.")
        return []

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Date Bound Parsing
    start_ts = 0.0
    end_ts = float("inf")
    if args.start_date:
        start_ts = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").timestamp()
    if args.end_date:
        end_ts = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").timestamp()

    all_events = []
    for item in data[:args.max_users]:
        user_id = item["question_id"]
        # History
        if not args.skip_history:
            h_sessions = item.get("haystack_sessions", [])
            h_dates = item.get("haystack_dates", [])
            for session, date_str in zip(h_sessions[:args.max_sessions_per_user], h_dates[:args.max_sessions_per_user]):
                ts = parse_date(date_str)
                if start_ts <= ts <= end_ts:
                    # Tag every message with the timestamp required by LightMem
                    tagged_session = []
                    for msg in session:
                        msg_copy = dict(msg)
                        msg_copy["time_stamp"] = date_str
                        tagged_session.append(msg_copy)
                        
                    all_events.append({
                        "ts": ts,
                        "type": "archive",
                        "user_id": user_id,
                        "content": tagged_session
                    })
        
        # Query
        if not args.skip_queries:
            q_date = item.get("question_date")
            if q_date:
                ts = parse_date(q_date)
                if start_ts <= ts <= end_ts:
                    all_events.append({
                        "ts": ts,
                        "type": "query",
                        "user_id": user_id,
                        "content": item["question"]
                    })
            
    all_events.sort(key=lambda x: x["ts"])
    
    # Cap total events if requested
    if args.max_events > 0:
        all_events = all_events[:args.max_events]
        
    return all_events

# ============================================================
# SIMULATION ENGINE
# ============================================================

class LoadTestMetrics:
    def __init__(self):
        self.results = []
        self.throughput_records = []
        self.total_completed = 0
        self.total_errors = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def record(self, event_type, user_id, latency, status="success", stage_timings=None):
        row = {
            "wall_time": time.time() - self.start_time,
            "type": event_type,
            "user_id": user_id,
            "latency": latency,
            "status": status
        }
        if stage_timings:
            for k, v in stage_timings.items():
                row[f"stage_{k}"] = v
        with self.lock:
            self.results.append(row)
            self.total_completed += 1
            if status == "error":
                self.total_errors += 1

    def save(self, filename_base):
        df = pd.DataFrame(self.results)
        df.to_csv(f"{filename_base}.csv", index=False)
        
        if self.throughput_records:
            df_tput = pd.DataFrame(self.throughput_records)
            df_tput.to_csv(f"{filename_base}_throughput.csv", index=False)
            print(f"Throughput stats saved to {filename_base}_throughput.csv")
            
        print(f"Raw event metrics saved to {filename_base}.csv")

async def monitor_throughput(metrics, stop_event):
    last_count = 0
    last_time = time.time()
    while not stop_event.is_set():
        await asyncio.sleep(5)
        now = time.time()
        with metrics.lock:
            current_count = metrics.total_completed
            errors = metrics.total_errors
        
        delta_count = current_count - last_count
        delta_time = now - last_time
        tput = delta_count / delta_time if delta_time > 0 else 0
        elapsed = now - metrics.start_time
        
        metrics.throughput_records.append({
            "elapsed_sec": elapsed,
            "throughput_eps": tput,
            "completed_so_far": current_count,
            "errors_so_far": errors
        })
        print(f"   >>> [{elapsed:6.1f}s] T-Put: {tput:6.2f} eps | Total: {current_count:5} | Errors: {errors}")
        
        last_count = current_count
        last_time = now

async def run_simulation(events, args, memory, rate_limiter):
    metrics = LoadTestMetrics()
    sem = asyncio.Semaphore(args.concurrency_limit)
    
    if not events: return

    first_ts = events[0]["ts"]
    real_start = time.time()
    tasks = []
    stop_monitor = asyncio.Event()

    # Start throughput monitor task
    monitor_task = asyncio.create_task(monitor_throughput(metrics, stop_monitor))

    async def execute_event(event):
        async with sem:
            await rate_limiter.wait()
            start_t = time.perf_counter()
            status = "success"
            stage_timings = None
            try:
                if event["type"] == "archive":
                    result = await asyncio.to_thread(
                        memory.add_memory,
                        messages=event["content"],
                        user_id=event["user_id"],
                        force_segment=True,
                        force_extract=True
                    )
                    if isinstance(result, dict) and "extraction_future" in result:
                        await asyncio.wrap_future(result["extraction_future"])
                elif event["type"] == "query":
                    # Simulating a retrieval query
                    await asyncio.to_thread(
                        memory.retrieve_memory,
                        query=event["content"],
                        user_id=event["user_id"]
                    )
                
                # Try to get stage timings if available from memory stats
                stats = memory.get_token_statistics()
                stage_timings = stats.get("stage_timings", {})
                memory.reset_token_statistics() # Reset for next event to avoid accumulation overlap if serial
                
            except Exception as e:
                print(f"Error in {event['type']} for {event['user_id']}: {e}")
                status = "error"
            
            latency = time.perf_counter() - start_t
            metrics.record(event["type"], event["user_id"], latency, status, stage_timings)

    for i, event in enumerate(events):
        dataset_offset = event["ts"] - first_ts
        target_real_time = real_start + (dataset_offset / args.time_scale)
        delay = target_real_time - time.time()
        if delay > 0: await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(execute_event(event)))
        if (i+1) % 50 == 0: print(f"Dispatched {i+1}/{len(events)} events...")

    await asyncio.gather(*tasks)
    
    # Wait a moment for final metrics and stop monitor
    await asyncio.sleep(2)
    stop_monitor.set()
    await monitor_task
    
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"profiling_runs/{args.run_name}_{args.provider}_{run_id}"
    metrics.save(filename_base)

# ============================================================
# SETUP LIGHTMEM
# ============================================================

def setup_lightmem(args):
    builder = _BUILDERS.get(args.provider)
    memory_manager_cfg = builder(args)
    
    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name":    os.getenv("LLMLINGUA_MODEL_PATH"),
                    "device_map":    "cpu",
                    "use_llmlingua2": True,
                },
                "compress_config": {"rate": 0.6},
            },
        },
        "topic_segment":        True,
        "precomp_topic_shared": True,
        "topic_segmenter":      {"model_name": "llmlingua-2"},
        "messages_use":         "user_only",
        "metadata_generate":    True,
        "text_summary":         True,
        "memory_manager":       memory_manager_cfg,
        "extract_threshold":    0.1,
        "index_strategy":       "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model":          os.getenv("EMBEDDING_MODEL_PATH"),
                "embedding_dims": 384,
                "model_kwargs":   {"device": "cpu"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": f"mt_{uuid.uuid4().hex[:4]}",
                "embedding_model_dims": 384,
                "path": f"{os.getenv('QDRANT_DATA_DIR')}/multitenant",
            },
        },
        "update": "offline" if args.provider == "vllm" else "sync",
        "logging": {"level": "INFO"}
    }
    return LightMemory.from_config(config)

async def main():
    args = parse_args()
    validate_env(args.provider)
    os.makedirs("profiling_runs", exist_ok=True)
    
    data_path = args.data_path or os.getenv("DATA_PATH")
    events = load_events(data_path, args)
    
    if not events:
        print("No events to simulate.")
        return

    # Calculate time scale if target duration is provided
    if args.target_duration > 0:
        span = events[-1]["ts"] - events[0]["ts"]
        if span > 0:
            args.time_scale = span / args.target_duration
            print(f"Calculated time-scale: {args.time_scale:.2f} (Dataset span: {span/3600:.1f} hours -> Target: {args.target_duration} seconds)")
        else:
            args.time_scale = 1.0 # Or some default
            print("Warning: Dataset span is 0, using time-scale 1.0")
    else:
        print(f"Using fixed time-scale: {args.time_scale:.2f}")
    
    memory = setup_lightmem(args)
    rate_limiter = AsyncRateLimiter(args.rpm)
    
    await run_simulation(events, args, memory, rate_limiter)

if __name__ == "__main__":
    asyncio.run(main())
