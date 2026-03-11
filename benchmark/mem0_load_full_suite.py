import asyncio
import time
import threading
import numpy as np
from collections import deque
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from mem0 import Memory

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "concurrency_levels": [1, 2, 4, 8, 16, 32],
    "test_seconds": 20,
    "dashboard_port": 8050
}

MEM0_CONFIG = {
    "llm": {"provider": "mock", "config": {"mock_latency": 1}},
    "embedder": {
        "provider": "mock",
        "config": {"mock_latency": 1, "output_dimensionality": 1536}
    }
}

memory = Memory.from_config(MEM0_CONFIG)

# ============================================================
# GLOBAL STATE
# ============================================================

current_concurrency = None
current_running = False
sweep_finished = False

live_throughput = deque(maxlen=500)
live_sleep_ratio = deque(maxlen=500)
live_llm_rate = deque(maxlen=500)
live_embedding_rate = deque(maxlen=500)
live_timestamps = deque(maxlen=500)

sweep_results = []

metrics_lock = threading.Lock()
total_writes = 0


# ============================================================
# WORKERS
# ============================================================

async def write_memory(worker_id, i):
    await asyncio.to_thread(
        memory.add,
        [{"role": "user", "content": f"user {worker_id} preference {i}"}],
        user_id=str(worker_id)
    )

async def worker(worker_id, stop_event):
    global total_writes
    i = 0
    while not stop_event.is_set():
        try:
            await write_memory(worker_id, i)
            with metrics_lock:
                total_writes += 1
        except:
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

    memory.reset_stats()
    total_writes = 0

    live_throughput.clear()
    live_sleep_ratio.clear()
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

        stats = memory.get_stats()

        sleep_ratio = stats["sleep_counter"] / elapsed if elapsed > 0 else 0
        llm_rate = stats["llm_call_counter"] / elapsed if elapsed > 0 else 0
        embedding_rate = stats["embedding_call_counter"] / elapsed if elapsed > 0 else 0

        live_timestamps.append(elapsed)
        live_throughput.append(throughput)
        live_sleep_ratio.append(sleep_ratio)
        live_llm_rate.append(llm_rate)
        live_embedding_rate.append(embedding_rate)

        last_sample = now
        last_writes = writes_now

    stop_event.set()
    await asyncio.gather(*workers)

    total_time = time.perf_counter() - start
    final_stats = memory.get_stats()

    sweep_results.append({
        "concurrency": concurrency,
        "throughput": total_writes / total_time,
        "sleep_ratio": final_stats["sleep_counter"] / total_time,
        "llm_rate": final_stats["llm_call_counter"] / total_time,
        "embedding_rate": final_stats["embedding_call_counter"] / total_time
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
    html.H2("Mem0 Concurrency Sweep Harness"),

    html.Div(id="status"),

    dcc.Graph(id="live_throughput"),
    dcc.Graph(id="live_sleep"),
    dcc.Graph(id="live_llm"),
    dcc.Graph(id="live_embedding"),

    dcc.Graph(id="sweep_throughput"),
    dcc.Graph(id="sweep_sleep"),
    dcc.Graph(id="sweep_llm"),
    dcc.Graph(id="sweep_embedding"),

    dcc.Interval(id="interval", interval=1000, n_intervals=0)
])


@app.callback(
    [
        Output("live_throughput", "figure"),
        Output("live_sleep", "figure"),
        Output("live_llm", "figure"),
        Output("live_embedding", "figure"),
        Output("sweep_throughput", "figure"),
        Output("sweep_sleep", "figure"),
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

    live_s_fig = go.Figure()
    live_s_fig.add_trace(go.Scatter(
        x=list(live_timestamps),
        y=list(live_sleep_ratio),
        mode="lines"
    ))
    live_s_fig.update_layout(title="Live Sleep Ratio")

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

    sweep_s = go.Figure()
    sweep_s.add_trace(go.Scatter(
        x=concurrencies,
        y=[r["sleep_ratio"] for r in sweep_results],
        mode="lines+markers"
    ))
    sweep_s.update_layout(title="Sleep Ratio vs Concurrency")

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
        live_s_fig,
        live_l_fig,
        live_e_fig,
        sweep_t,
        sweep_s,
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
