import asyncio
import time
import threading
import uuid
import datetime
from collections import deque
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from lightmem.memory.lightmem import LightMemory

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "concurrency_levels": [1, 2, 4, 8, 16, 32],
    "test_seconds": 20,
    "dashboard_port": 8051  # Different from Mem0 to avoid conflict
}

# ============ LightMem Configuration ============
your_ollama_model_name = "gemma3:27b-cloud"  # such as "llama3:latest"
your_ollama_host = "http://localhost:11434"  # default Ollama host is "http://localhost:11434"

# ============ Small Model Paths ============
LLMLINGUA_MODEL_PATH = "/Users/chinmaydandekar/Desktop/i/CS 598 - Systems for GenAI/lightmem-playground/llmlingua-2-bert-base-multilingual-cased-meetingbank"
EMBEDDING_MODEL_PATH = "/Users/chinmaydandekar/Desktop/i/CS 598 - Systems for GenAI/lightmem-playground/all-MiniLM-L6-v2"
QDRANT_DATA_DIR = "/Users/chinmaydandekar/Desktop/i/CS 598 - Systems for GenAI/lightmem-playground/qdrant_data"

def load_lightmem(collection_name):
    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": LLMLINGUA_MODEL_PATH,
                    "device_map": "cpu",
                    "use_llmlingua2": True,
                },
                "compress_config": {
                    "rate": 0.6
                }
            }
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {
            "model_name": "llmlingua-2",
        },
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "ollama",
            "configs": {
                "model": your_ollama_model_name,
                "host": your_ollama_host,
                "max_tokens": 16384,
            }
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": EMBEDDING_MODEL_PATH,
                "embedding_dims": 384,
                "model_kwargs": {"device": "cpu"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": f'{QDRANT_DATA_DIR}/{collection_name}',
            }
        },
        "update": "offline",
        "logging": {
            "level": "INFO",
            "file_enabled": True,
            "log_dir": "logs",
            "log_filename_prefix": "run",
            "console_enabled": True,
            "file_level": "DEBUG",
        }
    }
    lightmem_instance = LightMemory.from_config(config)
    return lightmem_instance

def reset_stats(mem):
    mem.token_stats = {
        "add_memory_calls": 0,
        "add_memory_prompt_tokens": 0,
        "add_memory_completion_tokens": 0,
        "add_memory_total_tokens": 0,
        "update_calls": 0,
        "update_prompt_tokens": 0,
        "update_completion_tokens": 0,
        "update_total_tokens": 0,
        "embedding_calls": 0,
        "embedding_total_tokens": 0,
        "summarize_calls": 0,
        "summarize_prompt_tokens": 0,
        "summarize_completion_tokens": 0,
        "summarize_total_tokens": 0,
    }
    if hasattr(mem, 'text_embedder') and hasattr(mem.text_embedder, 'reset_stats'):
        mem.text_embedder.reset_stats()


print("Initializing LightMem...")
collection_id = "load_test_" + str(uuid.uuid4().hex)[:8]
memory = load_lightmem(collection_id)
print("LightMem initialized.")

# ============================================================
# GLOBAL STATE
# ============================================================

current_concurrency = None
current_running = False
sweep_finished = False

live_throughput = deque(maxlen=500)
live_llm_rate = deque(maxlen=500)
live_embedding_rate = deque(maxlen=500)
live_timestamps = deque(maxlen=500)

sweep_results = []

metrics_lock = threading.Lock()
total_writes = 0


# ============================================================
# WORKERS
# ============================================================

def get_current_timestamp():
    now = datetime.datetime.now()
    weekday = now.strftime('%a')
    return now.strftime(f"%Y/%m/%d ({weekday}) %H:%M:%S")

async def write_memory(worker_id, i):
    messages = [
        {
            "role": "user",
            "content": f"user {worker_id} preference {i}. Actually I like apples more than bananas.",
            "time_stamp": get_current_timestamp()
        }
    ]
    await asyncio.to_thread(
        memory.add_memory,
        messages=messages,
        force_segment=True,   # ensure immediate processing
        force_extract=True    # ensure memory is extracted
    )

async def worker(worker_id, stop_event):
    global total_writes
    i = 0
    while not stop_event.is_set():
        try:
            await write_memory(worker_id, i)
            with metrics_lock:
                total_writes += 1
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            pass
        i += 1


# ============================================================
# SINGLE CONCURRENCY RUN
# ============================================================

async def run_single_concurrency(concurrency):
    global current_running, current_concurrency, total_writes

    print(f"\nRunning concurrency={concurrency}")

    current_concurrency = concurrency
    current_running = True

    reset_stats(memory)
    total_writes = 0

    live_throughput.clear()
    live_llm_rate.clear()
    live_embedding_rate.clear()
    live_timestamps.clear()

    stop_event = asyncio.Event()

    workers = [
        asyncio.create_task(worker(i, stop_event))
        for i in range(concurrency)
    ]

    start = time.perf_counter()
    last_sample = start
    last_writes = 0

    while time.perf_counter() - start < CONFIG["test_seconds"]:
        await asyncio.sleep(1)

        now = time.perf_counter()
        elapsed = now - start
        interval = now - last_sample

        with metrics_lock:
            writes_now = total_writes

        delta_writes = writes_now - last_writes
        throughput = delta_writes / interval

        stats = memory.get_token_statistics()

        llm_calls = stats.get("summary", {}).get("total_llm_calls", 0)
        emb_calls = stats.get("summary", {}).get("total_embedding_calls", 0)

        llm_rate = llm_calls / elapsed if elapsed > 0 else 0
        embedding_rate = emb_calls / elapsed if elapsed > 0 else 0

        live_timestamps.append(elapsed)
        live_throughput.append(throughput)
        live_llm_rate.append(llm_rate)
        live_embedding_rate.append(embedding_rate)

        last_sample = now
        last_writes = writes_now

    stop_event.set()
    await asyncio.gather(*workers)

    total_time = time.perf_counter() - start
    final_stats = memory.get_token_statistics()
    
    final_llm = final_stats.get("summary", {}).get("total_llm_calls", 0)
    final_emb = final_stats.get("summary", {}).get("total_embedding_calls", 0)

    sweep_results.append({
        "concurrency": concurrency,
        "throughput": total_writes / total_time,
        "llm_rate": final_llm / total_time,
        "embedding_rate": final_emb / total_time
    })

    current_running = False


# ============================================================
# FULL SWEEP
# ============================================================

async def run_sweep():
    global sweep_finished

    for c in CONFIG["concurrency_levels"]:
        await run_single_concurrency(c)

    sweep_finished = True


# ============================================================
# DASHBOARD
# ============================================================

app = Dash(__name__)

app.layout = html.Div([
    html.H2("LightMem Concurrency Sweep Harness"),

    html.Div(id="status"),

    dcc.Graph(id="live_throughput"),
    dcc.Graph(id="live_llm"),
    dcc.Graph(id="live_embedding"),

    dcc.Graph(id="sweep_throughput"),
    dcc.Graph(id="sweep_llm"),
    dcc.Graph(id="sweep_embedding"),

    dcc.Interval(id="interval", interval=1000, n_intervals=0)
])


@app.callback(
    [
        Output("live_throughput", "figure"),
        Output("live_llm", "figure"),
        Output("live_embedding", "figure"),
        Output("sweep_throughput", "figure"),
        Output("sweep_llm", "figure"),
        Output("sweep_embedding", "figure"),
        Output("status", "children"),
        Output("interval", "disabled")
    ],
    [Input("interval", "n_intervals")]
)
def update_dashboard(n):

    # ================= LIVE GRAPHS =================

    live_t_fig = go.Figure()
    live_t_fig.add_trace(go.Scatter(
        x=list(live_timestamps),
        y=list(live_throughput),
        mode="lines"
    ))
    live_t_fig.update_layout(title=f"Live Throughput (Concurrency={current_concurrency})")

    live_l_fig = go.Figure()
    live_l_fig.add_trace(go.Scatter(
        x=list(live_timestamps),
        y=list(live_llm_rate),
        mode="lines"
    ))
    live_l_fig.update_layout(title="Live LLM Calls/sec")

    live_e_fig = go.Figure()
    live_e_fig.add_trace(go.Scatter(
        x=list(live_timestamps),
        y=list(live_embedding_rate),
        mode="lines"
    ))
    live_e_fig.update_layout(title="Live Embedding Calls/sec")

    # ================= SWEEP GRAPHS =================

    concurrencies = [r["concurrency"] for r in sweep_results]

    sweep_t = go.Figure()
    sweep_t.add_trace(go.Scatter(
        x=concurrencies,
        y=[r["throughput"] for r in sweep_results],
        mode="lines+markers"
    ))
    sweep_t.update_layout(title="Throughput vs Concurrency")

    sweep_l = go.Figure()
    sweep_l.add_trace(go.Scatter(
        x=concurrencies,
        y=[r["llm_rate"] for r in sweep_results],
        mode="lines+markers"
    ))
    sweep_l.update_layout(title="LLM Calls/sec vs Concurrency")

    sweep_e = go.Figure()
    sweep_e.add_trace(go.Scatter(
        x=concurrencies,
        y=[r["embedding_rate"] for r in sweep_results],
        mode="lines+markers"
    ))
    sweep_e.update_layout(title="Embedding Calls/sec vs Concurrency")

    # ================= STATUS =================

    if sweep_finished:
        status = "✅ SWEEP COMPLETE"
    elif current_running:
        status = f"Running Concurrency {current_concurrency}"
    else:
        status = "Preparing next concurrency..."

    disable_interval = sweep_finished

    return (
        live_t_fig,
        live_l_fig,
        live_e_fig,
        sweep_t,
        sweep_l,
        sweep_e,
        status,
        disable_interval
    )


# ============================================================
# ENTRYPOINT
# ============================================================

def main():
    threading.Thread(target=lambda: asyncio.run(run_sweep()), daemon=True).start()
    app.run(host="0.0.0.0", port=CONFIG["dashboard_port"])


if __name__ == "__main__":
    main()
