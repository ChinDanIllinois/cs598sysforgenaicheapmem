import asyncio
import time
import os
import statistics
import threading
import numpy as np
from collections import deque
import psutil

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from mem0 import Memory

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "writes_per_worker": 100,
    "concurrency": 10,
    "warmup_seconds": 5,
    "test_seconds": 90,
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

test_finished = False

# ============================================================
# METRICS COLLECTOR
# ============================================================

class Metrics:
    def __init__(self, history=2000):
        self.latencies = []
        self.errors = {}
        self.total_writes = 0

        self.timestamps = deque(maxlen=history)
        self.throughput = deque(maxlen=history)

        self.lock = threading.Lock()

    def record(self, latency):
        with self.lock:
            self.latencies.append(latency)
            self.total_writes += 1

    def record_error(self, e):
        name = type(e).__name__
        with self.lock:
            self.errors[name] = self.errors.get(name, 0) + 1

    def snapshot(self):
        with self.lock:
            lat_copy = list(self.latencies)
            total = self.total_writes

        if not lat_copy:
            return {}

        arr = np.array(lat_copy)
        return {
            "count": total,
            "avg": arr.mean() * 1000,
            "p50": np.percentile(arr, 50) * 1000,
            "p95": np.percentile(arr, 95) * 1000,
            "p99": np.percentile(arr, 99) * 1000,
            "p999": np.percentile(arr, 99.9) * 1000,
        }

metrics = Metrics()

# ============================================================
# RESOURCE MONITOR
# ============================================================

class ResourceMonitor:
    def __init__(self, interval=0.5, history=2000):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.cpu = deque(maxlen=history)
        self.memory = deque(maxlen=history)
        self.timestamps = deque(maxlen=history)
        self.running = False

    def start(self):
        self.running = True
        self.process.cpu_percent()

        start_time = time.perf_counter()

        while self.running:
            now = time.perf_counter() - start_time
            cpu = self.process.cpu_percent()
            mem = self.process.memory_info().rss / (1024 * 1024)

            self.timestamps.append(now)
            self.cpu.append(cpu)
            self.memory.append(mem)

            time.sleep(self.interval)

    def stop(self):
        self.running = False

monitor = ResourceMonitor()

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
    i = 0
    while not stop_event.is_set():
        start = time.perf_counter()
        try:
            await write_memory(worker_id, i)
            latency = time.perf_counter() - start
            metrics.record(latency)
        except Exception as e:
            metrics.record_error(e)
        i += 1

async def run_load():
    global test_finished

    stop_event = asyncio.Event()

    workers = [
        asyncio.create_task(worker(i, stop_event))
        for i in range(CONFIG["concurrency"])
    ]

    await asyncio.sleep(CONFIG["test_seconds"])
    stop_event.set()
    await asyncio.gather(*workers)

    # STOP resource monitor
    monitor.stop()

    test_finished = True



# ============================================================
# DASHBOARD
# ============================================================

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Mem0 Production Load Harness"),
    html.Div(id="stats"),
    dcc.Graph(id="cpu"),
    dcc.Graph(id="memory"),
    dcc.Interval(id="interval", interval=1000, n_intervals=0)
])


@app.callback(
    [Output("cpu", "figure"),
     Output("memory", "figure"),
     Output("stats", "children"),
     Output("interval", "disabled")],
    [Input("interval", "n_intervals")]
)
def update_dashboard(n):

    cpu_fig = go.Figure()
    cpu_fig.add_trace(go.Scatter(
        x=list(monitor.timestamps),
        y=list(monitor.cpu),
        mode="lines"
    ))
    cpu_fig.update_layout(title="CPU %")

    mem_fig = go.Figure()
    mem_fig.add_trace(go.Scatter(
        x=list(monitor.timestamps),
        y=list(monitor.memory),
        mode="lines"
    ))
    mem_fig.update_layout(title="Memory (MB)")

    snap = metrics.snapshot()

    stats_text = ""
    if snap:
        stats_text = (
            f"Writes: {snap['count']} | "
            f"Avg: {snap['avg']:.2f} ms | "
            f"P95: {snap['p95']:.2f} ms | "
            f"P99: {snap['p99']:.2f} ms"
        )

    # 🔥 disable interval once test is done
    disable_interval = test_finished
    if test_finished:
        stats_text = "✅ TEST COMPLETE\n\n" + stats_text

    return cpu_fig, mem_fig, stats_text, disable_interval


# ============================================================
# ENTRYPOINT
# ============================================================

def main():
    # Start resource monitor
    threading.Thread(target=monitor.start, daemon=True).start()

    # Start load test in background
    threading.Thread(target=lambda: asyncio.run(run_load()), daemon=True).start()

    # Start dashboard
    app.run(host="0.0.0.0", port=CONFIG["dashboard_port"])

if __name__ == "__main__":
    main()
