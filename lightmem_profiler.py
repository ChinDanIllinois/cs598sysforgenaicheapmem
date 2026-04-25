"""
LightMem Load Test Harness — Backend-Agnostic Profiler
Realistic workload from LongMemEval dataset.
Tracks throughput, E2E latency (mean/p50/p95/p99), error rate,
LLM/embedding call rates, avg call times, and pipeline stage breakdown.

Usage:
    python lightmem_profiler.py --provider ollama
    python lightmem_profiler.py --provider gemini
    python lightmem_profiler.py --provider openai
"""

import argparse
import asyncio
import csv
import io
import json
import time
import threading
import uuid
import datetime
import sys
import textwrap
import numpy as np
from collections import deque

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import logging
import flask.cli
import concurrent.futures

# Suppress Flask / Dash / Werkzeug banner and logs
flask.cli.show_server_banner = lambda *args: None
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('flask').setLevel(logging.ERROR)
logging.getLogger('dash').setLevel(logging.ERROR)

import os
import dotenv
dotenv.load_dotenv()

from lightmem.memory.lightmem import LightMemory


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
        description="LightMem Load Test Harness — model-backend-agnostic profiler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Supported providers:
              ollama   — Local Ollama server (requires OLLAMA_MODEL_NAME, OLLAMA_HOST)
              gemini   — Google Gemini API  (requires GEMINI_MODEL_NAME, GEMINI_API_KEY)
              openai   — OpenAI API         (requires OPENAI_MODEL_NAME, OPENAI_API_KEY)

            All providers also require the following shared env vars:
              DATA_PATH            — Path to LongMemEval dataset JSON
              LLMLINGUA_MODEL_PATH — Path to LLMLingua-2 model weights
              EMBEDDING_MODEL_PATH — Path to local HuggingFace embedding model
              QDRANT_DATA_DIR      — Directory for Qdrant vector store persistence
        """),
    )
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
        help="Rate limit in Requests Per Minute (default: 0 = unlimited).",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32],
        metavar="N",
        help="Space-separated concurrency levels to sweep (default: 1 2 4 8 16 32).",
    )
    parser.add_argument(
        "--test-seconds",
        type=int,
        default=20,
        metavar="SEC",
        help="Duration of each concurrency level test in seconds (default: 20).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8051,
        metavar="PORT",
        help="Dashboard port (default: 8051).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        metavar="NAME",
        help=(
            "Optional label for this run, included in the CSV filename "
            "(e.g. 'baseline', 'batch4', 'no-compress').  "
            "Spaces are replaced with underscores."
        ),
    )
    return parser.parse_args()


# ============================================================
# ENVIRONMENT VARIABLE VALIDATION
# ============================================================

# Shared env vars required by every provider.
_SHARED_ENV_VARS = {
    "LLMLINGUA_MODEL_PATH": (
        "Path to the LLMLingua-2 model weights used for prompt compression and "
        "topic segmentation.  Example: /models/llmlingua2"
    ),
    "EMBEDDING_MODEL_PATH": (
        "Path to the local HuggingFace sentence-embedding model.  "
        "Example: /models/all-MiniLM-L6-v2"
    ),
    "QDRANT_DATA_DIR": (
        "Directory where Qdrant will persist its vector store collections.  "
        "Example: /data/qdrant"
    ),
}

# Per-provider env vars.  Each entry is (env_var_name, human_description).
_PROVIDER_ENV_VARS = {
    "ollama": [
        (
            "OLLAMA_MODEL_NAME",
            "The name of the Ollama model to use (e.g. 'llama3', 'mistral').  "
            "Run `ollama list` to see available models.",
        ),
        (
            "OLLAMA_HOST",
            "Base URL of the Ollama server (e.g. 'http://localhost:11434').  "
            "The server must be running before starting the profiler.",
        ),
    ],
    "gemini": [
        (
            "GEMINI_MODEL_NAME",
            "The Gemini model identifier (e.g. 'gemini-1.5-pro', 'gemini-1.5-flash').  "
            "See https://ai.google.dev/models for the full list.",
        ),
        (
            "GEMINI_API_KEY",
            "Your Google AI Studio or Vertex AI API key.  "
            "Obtain one at https://aistudio.google.com/app/apikey",
        ),
    ],
    "openai": [
        (
            "OPENAI_MODEL_NAME",
            "The OpenAI model identifier (e.g. 'gpt-4o', 'gpt-4-turbo').  "
            "See https://platform.openai.com/docs/models for the full list.",
        ),
        (
            "OPENAI_API_KEY",
            "Your OpenAI API key.  "
            "Obtain one at https://platform.openai.com/api-keys",
        ),
    ],
    "vllm": [
        (
            "VLLM_MODEL_NAME",
            "The vLLM model identifier (e.g. 'Qwen/Qwen2.5-3B-Instruct').",
        ),
        (
            "VLLM_BASE_URL",
            "The URL for the vLLM server (e.g. 'http://localhost:8000/v1').",
        ),
    ],
    "mock": [],
}


def validate_env(provider: str) -> None:
    """
    Check that all required environment variables are set for the given provider.
    Prints a clear, actionable error for every missing variable and exits if any
    are absent.
    """
    missing: list[tuple[str, str]] = []

    # Check shared vars (DATA_PATH is optional — warn but don't error)
    for var, description in _SHARED_ENV_VARS.items():
        if not os.getenv(var):
            missing.append((var, description))

    # Check provider-specific vars
    for var, description in _PROVIDER_ENV_VARS[provider]:
        if not os.getenv(var):
            missing.append((var, description))

    if missing:
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════╗",
            f"║  ERROR: Missing required environment variables for provider: {provider:<6}║",
            "╠══════════════════════════════════════════════════════════════════╣",
        ]
        for var, description in missing:
            lines.append(f"║  ✗  {var}")
            # Word-wrap the description to 62 chars so it fits in the box
            wrapped = textwrap.wrap(description, width=62)
            for wline in wrapped:
                lines.append(f"║       {wline}")
            lines.append("║")
        lines += [
            "║  Set the variables above in your shell or in a .env file,",
            "║  then re-run the profiler.",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
        ]
        print("\n".join(lines), file=sys.stderr)
        sys.exit(1)

    # Soft warning for DATA_PATH (synthetic workload will be used if absent)
    if not os.getenv("DATA_PATH") or not os.path.exists(os.getenv("DATA_PATH", "")):
        print(
            "WARNING: DATA_PATH is not set or the file does not exist — "
            "the profiler will fall back to a synthetic workload.",
            file=sys.stderr,
        )


# ============================================================
# CONFIG (populated after arg parse)
# ============================================================

args = parse_args()
validate_env(args.provider)

# Resolve the model name from the appropriate env var for the chosen provider.
_MODEL_NAME_ENV = {
    "ollama": "OLLAMA_MODEL_NAME",
    "gemini": "GEMINI_MODEL_NAME",
    "openai": "OPENAI_MODEL_NAME",
    "vllm": "VLLM_MODEL_NAME",
    "mock": "MOCK_MODEL_NAME"
}
_resolved_model_name = os.getenv(_MODEL_NAME_ENV[args.provider], "unknown")

CONFIG = {
    "provider":           args.provider,
    "model_name":         _resolved_model_name,
    "run_name":           args.run_name.strip().replace(" ", "_"),
    "concurrency_levels": args.concurrency,
    "test_seconds":       args.test_seconds,
    "dashboard_port":     args.port,
    "rpm":                args.rpm,
    "llm_batch_size":     args.llm_batch_size,
    "llm_batch_timeout":  args.llm_batch_timeout,
    "vllm_adaptive_shaping": args.vllm_adaptive_shaping,
    "vllm_metrics_url":   args.vllm_metrics_url,
}

rate_limiter = AsyncRateLimiter(CONFIG["rpm"])

DATA_PATH            = os.getenv("DATA_PATH")
LLMLINGUA_MODEL_PATH = os.getenv("LLMLINGUA_MODEL_PATH")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
QDRANT_DATA_DIR      = os.getenv("QDRANT_DATA_DIR")


# ============================================================
# PROVIDER-SPECIFIC MEMORY MANAGER CONFIG BUILDERS
# ============================================================

def _memory_manager_config_ollama() -> dict:
    return {
        "model_name": "ollama",
        "configs": {
            "model": os.getenv("OLLAMA_MODEL_NAME"),
            "host":  os.getenv("OLLAMA_HOST"),
            "max_tokens": 16384,
        },
    }


def _memory_manager_config_gemini() -> dict:
    return {
        "model_name": "gemini",
        "configs": {
            "model":      os.getenv("GEMINI_MODEL_NAME"),
            "api_key":    os.getenv("GEMINI_API_KEY"),
            "max_tokens": 16384,
        },
    }


def _memory_manager_config_vllm() -> dict:
    return {
        "model_name": "vllm",
        "configs": {
            "model":      os.getenv("VLLM_MODEL_NAME"),
            "api_key":    os.getenv("VLLM_API_KEY", "EMPTY"),
            "max_tokens": 16384,
            "vllm_base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            "llm_batch_size": CONFIG["llm_batch_size"],
            "llm_batch_timeout": CONFIG["llm_batch_timeout"],
            "vllm_adaptive_shaping": CONFIG["vllm_adaptive_shaping"],
            "vllm_metrics_url": CONFIG["vllm_metrics_url"] if CONFIG["vllm_metrics_url"] else None,
        },
    }


def _memory_manager_config_openai() -> dict:
    return {
        "model_name": "openai",
        "configs": {
            "model":      os.getenv("OPENAI_MODEL_NAME"),
            "api_key":    os.getenv("OPENAI_API_KEY"),
            "max_tokens": 16384,
        },
    }


def _memory_manager_config_mock() -> dict:
    return {
        "model_name": "mock",
        "configs": {
            "model": "mock-model",
        },
    }


_MEMORY_MANAGER_BUILDERS = {
    "ollama": _memory_manager_config_ollama,
    "gemini": _memory_manager_config_gemini,
    "openai": _memory_manager_config_openai,
    "vllm":   _memory_manager_config_vllm,
    "mock":   _memory_manager_config_mock,
}


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(data_path: str):
    """
    Load all (messages, timestamp) pairs from LongMemEval dataset.
    Returns a flat list of turn_messages lists ready for add_memory.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        sessions   = item.get("haystack_sessions", [])
        timestamps = item.get("haystack_dates", [])
        for session, timestamp in zip(sessions, timestamps):
            # Align to user-start
            while session and session[0]["role"] != "user":
                session.pop(0)
            num_turns = len(session) // 2
            for turn_idx in range(num_turns):
                turn_messages = session[turn_idx * 2 : turn_idx * 2 + 2]
                if (
                    len(turn_messages) < 2
                    or turn_messages[0]["role"] != "user"
                    or turn_messages[1]["role"] != "assistant"
                ):
                    continue
                tagged = []
                for msg in turn_messages:
                    m = dict(msg)
                    m["time_stamp"] = timestamp
                    tagged.append(m)
                samples.append(tagged)
    return samples


print("Loading dataset...")
if DATA_PATH and os.path.exists(DATA_PATH):
    DATASET = load_dataset(DATA_PATH)
    print(f"Dataset loaded: {len(DATASET)} conversation turns available.")
else:
    print("WARNING: DATA_PATH not set or missing — using synthetic workload.")
    DATASET = None


def get_current_timestamp():
    now = datetime.datetime.now()
    return now.strftime(f"%Y/%m/%d ({now.strftime('%a')}) %H:%M:%S")


def get_sample(worker_id: int, concurrency: int, i: int):
    """Return a message list for this worker's i-th request."""
    if DATASET:
        idx = (worker_id * 70 * concurrency + i) % len(DATASET)
        return DATASET[idx]
    return [
        {
            "role": "user",
            "content": f"Worker {worker_id} turn {i}: I prefer apples over bananas.",
            "time_stamp": get_current_timestamp(),
        },
        {
            "role": "assistant",
            "content": "Noted, you prefer apples.",
            "time_stamp": get_current_timestamp(),
        },
    ]


# ============================================================
# LIGHTMEM SETUP
# ============================================================

def load_lightmem(collection_name: str) -> LightMemory:
    provider = CONFIG["provider"]
    memory_manager_cfg = _MEMORY_MANAGER_BUILDERS[provider]()

    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name":    LLMLINGUA_MODEL_PATH,
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
                "model":          EMBEDDING_MODEL_PATH,
                "embedding_dims": 384,
                "model_kwargs":   {"device": "cpu"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name":      collection_name,
                "embedding_model_dims": 384,
                "path": f"{QDRANT_DATA_DIR}/{collection_name}",
            },
        },
        "update": "offline",
        "logging": {
            "level":                "INFO",
            "file_enabled":         True,
            "log_dir":              "logs",
            "log_filename_prefix":  "run",
            "console_enabled":      True,
            "file_level":           "DEBUG",
        },
    }
    return LightMemory.from_config(config)


def reset_stats(mem):
    mem.reset_token_statistics()


print(f"Preparing LightMem with provider: {CONFIG['provider']} ...")
collection_id = "load_test_" + uuid.uuid4().hex[:8]
memory = None
memory_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="lightmem-profiler")


def _create_memory_instance() -> LightMemory:
    return load_lightmem(collection_id)


async def _memory_call(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(memory_executor, lambda: fn(*args, **kwargs))


async def _init_memory_for_sweep_thread() -> None:
    global memory
    memory = await _memory_call(_create_memory_instance)
    print("LightMem initialized in dedicated worker thread.")


# ============================================================
# GLOBAL STATE
# ============================================================

current_concurrency = None
current_running     = False
sweep_finished      = False

live_timestamps    = deque(maxlen=500)
live_throughput    = deque(maxlen=500)
live_latency       = deque(maxlen=500)
live_llm_rate      = deque(maxlen=500)
live_embedding_rate = deque(maxlen=500)
live_errors_per_sec = deque(maxlen=500)

latency_samples: list[float] = []
latency_samples_lock = threading.Lock()

sweep_results: list[dict] = []

metrics_lock   = threading.Lock()
total_writes   = 0
total_latency  = 0.0
total_errors   = 0
total_attempts = 0


# ============================================================
# WORKERS
# ============================================================

async def worker(worker_id, concurrency, stop_event):
    global total_writes, total_latency, total_errors, total_attempts
    i = 0
    while not stop_event.is_set():
        messages = get_sample(worker_id, concurrency, i)
        start_t  = time.perf_counter()
        with metrics_lock:
            total_attempts += 1
        try:
            await rate_limiter.wait()

            await _memory_call(
                memory.add_memory,
                messages=messages,
                force_segment=True,
                force_extract=True,
            )
            duration = time.perf_counter() - start_t
            with metrics_lock:
                total_writes  += 1
                total_latency += duration
            with latency_samples_lock:
                latency_samples.append(duration)
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            with metrics_lock:
                total_errors += 1
        i += 1


# ============================================================
# SINGLE CONCURRENCY RUN
# ============================================================

async def run_single_concurrency(concurrency):
    global current_running, current_concurrency
    global total_writes, total_latency, total_errors, total_attempts

    print(f"\nRunning concurrency={concurrency}")
    current_concurrency = concurrency
    current_running     = True

    await _memory_call(reset_stats, memory)
    total_writes   = 0
    total_latency  = 0.0
    total_errors   = 0
    total_attempts = 0

    with latency_samples_lock:
        latency_samples.clear()

    live_timestamps.clear()
    live_throughput.clear()
    live_latency.clear()
    live_llm_rate.clear()
    live_embedding_rate.clear()
    live_errors_per_sec.clear()

    stop_event = asyncio.Event()
    workers    = [asyncio.create_task(worker(i, concurrency, stop_event)) for i in range(concurrency)]

    start       = time.perf_counter()
    last_sample = start
    last_writes  = 0
    last_latency = 0.0
    last_errors  = 0

    while time.perf_counter() - start < CONFIG["test_seconds"]:
        await asyncio.sleep(1)

        now      = time.perf_counter()
        elapsed  = now - start
        interval = now - last_sample

        with metrics_lock:
            writes_now  = total_writes
            latency_now = total_latency
            errors_now  = total_errors

        delta_writes  = writes_now - last_writes
        delta_latency = latency_now - last_latency
        delta_errors  = errors_now - last_errors
        
        throughput      = delta_writes / interval
        avg_latency_sec = (delta_latency / delta_writes) if delta_writes > 0 else 0.0
        errors_per_sec  = delta_errors / interval

        stats = await _memory_call(memory.get_token_statistics)
        llm_calls = stats.get("summary", {}).get("total_llm_calls", 0)
        emb_calls = stats.get("summary", {}).get("total_embedding_calls", 0)
        llm_rate       = llm_calls / elapsed if elapsed > 0 else 0
        embedding_rate = emb_calls / elapsed if elapsed > 0 else 0

        live_timestamps.append(elapsed)
        live_throughput.append(throughput)
        live_latency.append(avg_latency_sec)
        live_llm_rate.append(llm_rate)
        live_embedding_rate.append(embedding_rate)
        live_errors_per_sec.append(errors_per_sec)

        last_sample  = now
        last_writes  = writes_now
        last_latency = latency_now
        last_errors  = errors_now

    stop_event.set()
    await asyncio.gather(*workers)

    total_time  = time.perf_counter() - start
    final_stats = await _memory_call(memory.get_token_statistics)

    final_llm   = final_stats.get("summary", {}).get("total_llm_calls", 0)
    final_llm_t = final_stats.get("summary", {}).get("total_llm_time", 0.0)
    final_emb   = final_stats.get("summary", {}).get("total_embedding_calls", 0)
    final_emb_t = final_stats.get("summary", {}).get("total_embedding_time", 0.0)
    stage_t     = final_stats.get("stage_timings", {})

    with latency_samples_lock:
        samples = list(latency_samples)

    p50 = float(np.percentile(samples, 50)) if samples else 0.0
    p95 = float(np.percentile(samples, 95)) if samples else 0.0
    p99 = float(np.percentile(samples, 99)) if samples else 0.0

    with metrics_lock:
        _attempts = total_attempts
        _errors   = total_errors
        _writes   = total_writes

    error_rate = _errors / _attempts if _attempts > 0 else 0.0
    n          = _writes if _writes > 0 else 1

    sweep_results.append({
        "concurrency":       concurrency,
        "throughput":        _writes / total_time,
        "avg_latency":       total_latency / _writes if _writes > 0 else 0.0,
        "p50":               p50,
        "p95":               p95,
        "p99":               p99,
        "error_rate":        error_rate,
        "total_writes":      _writes,
        "total_errors":      _errors,
        "llm_rate":          final_llm / total_time,
        "embedding_rate":    final_emb / total_time,
        "avg_llm_time":      final_llm_t / final_llm if final_llm > 0 else 0.0,
        "avg_embedding_time": final_emb_t / final_emb if final_emb > 0 else 0.0,
        "stage_compress":    stage_t.get("compress", 0.0) / n,
        "stage_segment":     stage_t.get("segment", 0.0) / n,
        "stage_llm_extract": stage_t.get("llm_extract", 0.0) / n,
        "stage_db_insert":   stage_t.get("db_insert", 0.0) / n,
    })

    current_running = False


# ============================================================
# FULL SWEEP
# ============================================================

async def run_sweep():
    global sweep_finished
    await _init_memory_for_sweep_thread()
    for c in CONFIG["concurrency_levels"]:
        await run_single_concurrency(c)
    sweep_finished = True


# ============================================================
# DASHBOARD — DESIGN SYSTEM
# ============================================================

BG         = "#0f1117"
CARD_BG    = "#1a1d27"
BORDER     = "#2a2d3e"
ACCENT     = "#6366f1"
ACCENT2    = "#22d3ee"
ACCENT3    = "#f59e0b"
ACCENT4    = "#10b981"
ACCENT_ERR = "#ef4444"
TEXT       = "#e2e8f0"
TEXT_DIM   = "#64748b"

PLOT_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    margin=dict(l=48, r=16, t=44, b=40),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showline=True),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showline=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
)


def make_scatter(x, y, color, name=None, mode="lines", dash=None):
    line_kw = dict(color=color, width=2)
    if dash:
        line_kw["dash"] = dash
    return go.Scatter(
        x=x, y=y, mode=mode, name=name or "",
        line=line_kw,
        marker=dict(color=color, size=7) if "markers" in mode else {},
    )


def card_graph(gid):
    return dcc.Graph(
        id=gid,
        config={"displayModeBar": False},
        style={"borderRadius": "12px", "overflow": "hidden"},
    )


def section(title, graphs, ncols=2):
    return html.Div([
        html.H3(title, style={
            "color": ACCENT2, "fontFamily": "Inter, sans-serif",
            "fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.1em",
            "textTransform": "uppercase", "margin": "0 0 12px 2px",
        }),
        html.Div(graphs, style={
            "display": "grid",
            "gridTemplateColumns": f"repeat({ncols}, 1fr)",
            "gap": "16px",
        }),
    ], style={"marginBottom": "36px"})


_PROVIDER_LABEL = {
    "ollama": f"Ollama ({os.getenv('OLLAMA_MODEL_NAME', 'unknown')})",
    "gemini": f"Gemini ({os.getenv('GEMINI_MODEL_NAME', 'unknown')})",
    "openai": f"OpenAI ({os.getenv('OPENAI_MODEL_NAME', 'unknown')})",
    "vllm":   f"vLLM ({os.getenv('VLLM_MODEL_NAME', 'unknown')})",
    "mock":   "Mock Provider (Simulated)",
}

app = Dash(__name__)
app.logger.setLevel(logging.ERROR)
app.title = f"LightMem Profiler — {CONFIG['provider'].capitalize()}"

app.layout = html.Div([

    # ── Header ──────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("⚡", style={"fontSize": "26px", "marginRight": "12px"}),
            html.Div([
                html.H1("LightMem Load Test", style={
                    "margin": 0, "fontSize": "20px", "fontWeight": "700",
                    "color": TEXT, "fontFamily": "Inter, sans-serif",
                }),
                html.P(
                    f"Provider: {_PROVIDER_LABEL[CONFIG['provider']]}  ·  "
                    f"{len(DATASET) if DATASET else 0} dataset turns  ·  "
                    f"concurrency sweep {CONFIG['concurrency_levels']}",
                    style={"margin": 0, "color": TEXT_DIM, "fontSize": "12px",
                           "fontFamily": "Inter, sans-serif"},
                ),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Button(
                "⬇ Download CSV",
                id="download_btn",
                n_clicks=0,
                style={
                    "display": "none",
                    "background": ACCENT, "color": "#fff",
                    "border": "none", "borderRadius": "8px",
                    "padding": "8px 18px", "fontSize": "13px",
                    "fontFamily": "Inter, sans-serif", "fontWeight": "600",
                    "cursor": "pointer", "marginRight": "12px",
                },
            ),
            dcc.Download(id="download_csv"),
            html.Div(id="status_badge"),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "18px 28px",
        "background": CARD_BG, "borderBottom": f"1px solid {BORDER}",
        "marginBottom": "28px",
    }),

    # ── Body ────────────────────────────────────────────────
    html.Div([

        section("Live — Current Concurrency Run", [
            card_graph("live_throughput"),
            card_graph("live_latency"),
            card_graph("live_llm"),
            card_graph("live_embedding"),
            card_graph("live_errors"),
        ], ncols=3),

        section("Sweep — Throughput, Latency & Errors", [
            card_graph("sweep_throughput"),
            card_graph("sweep_latency_percentiles"),
            card_graph("sweep_error_rate"),
        ], ncols=3),

        section("Sweep — API Call Rates & Avg Times", [
            card_graph("sweep_llm"),
            card_graph("sweep_embedding"),
            card_graph("avg_llm_time"),
            card_graph("avg_emb_time"),
        ], ncols=2),

        section("Sweep — Pipeline Stage Breakdown (avg sec/write, stacked)", [
            card_graph("stage_breakdown"),
        ], ncols=1),

    ], style={"padding": "0 28px 40px"}),

    dcc.Interval(id="interval", interval=1000, n_intervals=0),

], style={"backgroundColor": BG, "minHeight": "100vh"})


# ============================================================
# MAIN UPDATE CALLBACK
# ============================================================

@app.callback(
    [
        Output("live_throughput",           "figure"),
        Output("live_latency",              "figure"),
        Output("live_llm",                  "figure"),
        Output("live_embedding",            "figure"),
        Output("live_errors",               "figure"),
        Output("sweep_throughput",          "figure"),
        Output("sweep_latency_percentiles", "figure"),
        Output("sweep_error_rate",          "figure"),
        Output("sweep_llm",                 "figure"),
        Output("sweep_embedding",           "figure"),
        Output("avg_llm_time",              "figure"),
        Output("avg_emb_time",              "figure"),
        Output("stage_breakdown",           "figure"),
        Output("status_badge",              "children"),
        Output("download_btn",              "style"),
        Output("interval",                  "disabled"),
    ],
    [Input("interval", "n_intervals")],
)
def update_dashboard(n):
    ts   = list(live_timestamps)
    conc = [r["concurrency"] for r in sweep_results]

    # ── Live graphs ─────────────────────────────────────────
    def live_fig(y_data, color, title, yaxis_title, xaxis_title="elapsed (s)"):
        fig = go.Figure()
        fig.add_trace(make_scatter(ts, list(y_data), color))
        fig.update_layout(title=dict(text=title, font=dict(size=13, color=TEXT)),
                          yaxis_title=yaxis_title, xaxis_title=xaxis_title,
                          **PLOT_LAYOUT)
        return fig

    fig_lt = live_fig(live_throughput,    ACCENT,  f"Throughput  ·  C={current_concurrency}", "writes/sec")
    fig_ll = live_fig(live_latency,       ACCENT2, "Avg E2E Latency  ·  live",                "seconds")
    fig_lr = live_fig(live_llm_rate,      ACCENT3, "LLM Calls/sec  ·  live",                  "calls/sec")
    fig_le = live_fig(live_embedding_rate, ACCENT4, "Embedding Calls/sec  ·  live",            "calls/sec")
    fig_lerr = live_fig(live_errors_per_sec, ACCENT_ERR, "Errors/sec  ·  live",               "errors/sec")

    # ── Sweep: Throughput ────────────────────────────────────
    fig_st = go.Figure()
    fig_st.add_trace(make_scatter(conc, [r["throughput"] for r in sweep_results],
                                  ACCENT, "writes/sec", mode="lines+markers"))
    fig_st.update_layout(title=dict(text="Throughput vs Concurrency",
                                    font=dict(size=13, color=TEXT)),
                         yaxis_title="writes/sec", xaxis_title="concurrency",
                         **PLOT_LAYOUT)

    # ── Sweep: Latency percentiles ───────────────────────────
    fig_slp = go.Figure()
    for key, label, color, dash in [
        ("avg_latency", "mean",  ACCENT,     None),
        ("p50",         "p50",   ACCENT2,    None),
        ("p95",         "p95",   ACCENT3,    "dot"),
        ("p99",         "p99",   ACCENT_ERR, "dash"),
    ]:
        fig_slp.add_trace(make_scatter(
            conc, [r[key] for r in sweep_results], color, label,
            mode="lines+markers", dash=dash,
        ))
    fig_slp.update_layout(title=dict(text="E2E Latency Percentiles vs Concurrency",
                                     font=dict(size=13, color=TEXT)),
                          yaxis_title="seconds", xaxis_title="concurrency",
                          **PLOT_LAYOUT)

    # ── Sweep: Error rate ────────────────────────────────────
    fig_ser = go.Figure()
    fig_ser.add_trace(make_scatter(
        conc, [r["error_rate"] * 100 for r in sweep_results],
        ACCENT_ERR, "error %", mode="lines+markers",
    ))
    fig_ser.update_layout(title=dict(text="Error Rate vs Concurrency",
                                     font=dict(size=13, color=TEXT)),
                          yaxis_title="error rate (%)", xaxis_title="concurrency",
                          **PLOT_LAYOUT)

    # ── Sweep: LLM / embedding rates ─────────────────────────
    fig_sl = go.Figure()
    fig_sl.add_trace(make_scatter(conc, [r["llm_rate"] for r in sweep_results],
                                  ACCENT3, "LLM calls/sec", mode="lines+markers"))
    fig_sl.update_layout(title=dict(text="LLM Calls/sec vs Concurrency",
                                    font=dict(size=13, color=TEXT)),
                         yaxis_title="calls/sec", xaxis_title="concurrency",
                         **PLOT_LAYOUT)

    fig_se = go.Figure()
    fig_se.add_trace(make_scatter(conc, [r["embedding_rate"] for r in sweep_results],
                                  ACCENT4, "embed calls/sec", mode="lines+markers"))
    fig_se.update_layout(title=dict(text="Embedding Calls/sec vs Concurrency",
                                    font=dict(size=13, color=TEXT)),
                         yaxis_title="calls/sec", xaxis_title="concurrency",
                         **PLOT_LAYOUT)

    # ── Sweep: Avg call times ────────────────────────────────
    fig_alt = go.Figure()
    fig_alt.add_trace(make_scatter(conc, [r["avg_llm_time"] for r in sweep_results],
                                   ACCENT3, "avg LLM time", mode="lines+markers"))
    fig_alt.update_layout(title=dict(text="Avg Time/LLM Call vs Concurrency",
                                     font=dict(size=13, color=TEXT)),
                          yaxis_title="seconds", xaxis_title="concurrency",
                          **PLOT_LAYOUT)

    fig_aet = go.Figure()
    fig_aet.add_trace(make_scatter(conc, [r["avg_embedding_time"] for r in sweep_results],
                                   ACCENT4, "avg embed time", mode="lines+markers"))
    fig_aet.update_layout(title=dict(text="Avg Time/Embedding Call vs Concurrency",
                                     font=dict(size=13, color=TEXT)),
                          yaxis_title="seconds", xaxis_title="concurrency",
                          **PLOT_LAYOUT)

    # ── Sweep: Stage breakdown (stacked) ────────────────────
    fig_sb = go.Figure()
    stages = [
        ("stage_compress",    "Compression",  ACCENT),
        ("stage_segment",     "Segmentation", ACCENT2),
        ("stage_llm_extract", "LLM Extract",  ACCENT3),
        ("stage_db_insert",   "DB Insert",    ACCENT4),
    ]
    for key, label, color in stages:
        fig_sb.add_trace(go.Scatter(
            x=conc, y=[r[key] for r in sweep_results],
            mode="lines+markers", name=label,
            stackgroup="stages",
            line=dict(color=color, width=2),
        ))
    fig_sb.update_layout(
        title=dict(text="Pipeline Stage Breakdown vs Concurrency (avg sec/write, stacked)",
                   font=dict(size=13, color=TEXT)),
        yaxis_title="seconds / write", xaxis_title="concurrency",
        **PLOT_LAYOUT,
    )

    # ── Status badge ─────────────────────────────────────────
    if sweep_finished:
        badge = html.Span("✅  Sweep Complete", style={
            "fontSize": "12px", "fontFamily": "Inter, sans-serif",
            "padding": "5px 14px", "borderRadius": "20px", "fontWeight": "600",
            "backgroundColor": "#064e3b", "color": "#6ee7b7",
            "border": "1px solid #065f46",
        })
        btn_style = {
            "display": "inline-block", "background": ACCENT, "color": "#fff",
            "border": "none", "borderRadius": "8px", "padding": "8px 18px",
            "fontSize": "13px", "fontFamily": "Inter, sans-serif",
            "fontWeight": "600", "cursor": "pointer", "marginRight": "12px",
        }
    elif current_running:
        badge = html.Span(f"⚡  Running  ·  C={current_concurrency}", style={
            "fontSize": "12px", "fontFamily": "Inter, sans-serif",
            "padding": "5px 14px", "borderRadius": "20px", "fontWeight": "600",
            "backgroundColor": "#1e1b4b", "color": "#a5b4fc",
            "border": f"1px solid {ACCENT}",
        })
        btn_style = {"display": "none"}
    else:
        badge = html.Span("⏳  Preparing...", style={
            "fontSize": "12px", "fontFamily": "Inter, sans-serif",
            "padding": "5px 14px", "borderRadius": "20px", "fontWeight": "600",
            "backgroundColor": "#1c1917", "color": "#a3a3a3",
            "border": "1px solid #3f3f46",
        })
        btn_style = {"display": "none"}

    return (
        fig_lt, fig_ll, fig_lr, fig_le, fig_lerr,
        fig_st, fig_slp, fig_ser,
        fig_sl, fig_se,
        fig_alt, fig_aet,
        fig_sb,
        badge,
        btn_style,
        sweep_finished,
    )


# ============================================================
# CSV DOWNLOAD CALLBACK
# ============================================================

@app.callback(
    Output("download_csv", "data"),
    Input("download_btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    if not sweep_results:
        return None

    fieldnames = [
        "concurrency", "throughput", "avg_latency", "p50", "p95", "p99",
        "error_rate", "total_writes", "total_errors",
        "llm_rate", "embedding_rate", "avg_llm_time", "avg_embedding_time",
        "stage_compress", "stage_segment", "stage_llm_extract", "stage_db_insert",
    ]

    buf    = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in sweep_results:
        writer.writerow({k: round(v, 6) if isinstance(v, float) else v
                         for k, v in row.items()})

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    provider   = CONFIG["provider"]
    # Sanitise model name so it's safe to use in a filename
    model_slug = CONFIG["model_name"].replace("/", "-").replace(" ", "_")
    run_name   = CONFIG["run_name"]
    parts      = ["lightmem_sweep", provider, model_slug]
    if run_name:
        parts.append(run_name)
    parts.append(timestamp_str)
    return dict(
        content=buf.getvalue(),
        filename="_".join(parts) + ".csv",
    )


# ============================================================
# ENTRYPOINT
# ============================================================

def main():
    def _run_sweep_thread():
        asyncio.run(run_sweep())

    threading.Thread(target=_run_sweep_thread, daemon=True).start()
    app.run(host="0.0.0.0", port=CONFIG["dashboard_port"], debug=False)


if __name__ == "__main__":
    main()
