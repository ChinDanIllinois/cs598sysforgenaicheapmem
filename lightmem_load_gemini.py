import asyncio
import time
import threading
import uuid
import datetime
import numpy as np
from collections import deque
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import os
import dotenv
dotenv.load_dotenv()

from lightmem.memory.lightmem import LightMemory

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "concurrency_levels": [1, 2, 4, 8, 16, 32],
    "test_seconds": 20,
    "dashboard_port": 8051,
}

# ============ LightMem Configuration ============
your_gemini_model_name = os.getenv("GEMINI_MODEL_NAME")
LLMLINGUA_MODEL_PATH = os.getenv("LLMLINGUA_MODEL_PATH")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
QDRANT_DATA_DIR = os.getenv("QDRANT_DATA_DIR")


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
                "compress_config": {"rate": 0.6},
            },
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {"model_name": "llmlingua-2"},
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "gemini",
            "configs": {
                "model": your_gemini_model_name,
                "api_key": os.getenv("GEMINI_API_KEY"),
                "max_tokens": 16384,
            },
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
                "path": f"{QDRANT_DATA_DIR}/{collection_name}",
            },
        },
        "update": "offline",
        "logging": {
            "level": "INFO",
            "file_enabled": True,
            "log_dir": "logs",
            "log_filename_prefix": "run",
            "console_enabled": True,
            "file_level": "DEBUG",
        },
    }
    return LightMemory.from_config(config)


def reset_stats(mem):
    mem.token_stats = {
        "add_memory_calls": 0,
        "add_memory_prompt_tokens": 0,
        "add_memory_completion_tokens": 0,
        "add_memory_total_tokens": 0,
        "add_memory_time": 0.0,
        "update_calls": 0,
        "update_prompt_tokens": 0,
        "update_completion_tokens": 0,
        "update_total_tokens": 0,
        "update_time": 0.0,
        "embedding_calls": 0,
        "embedding_total_tokens": 0,
        "embedding_time": 0.0,
        "summarize_calls": 0,
        "summarize_prompt_tokens": 0,
        "summarize_completion_tokens": 0,
        "summarize_total_tokens": 0,
        "summarize_time": 0.0,
        "stage_compress_time": 0.0,
        "stage_segment_time": 0.0,
        "stage_llm_extract_time": 0.0,
        "stage_db_insert_time": 0.0,
    }
    if hasattr(mem, "text_embedder") and hasattr(mem.text_embedder, "reset_stats"):
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

# Live time-series data
live_timestamps = deque(maxlen=500)
live_throughput = deque(maxlen=500)
live_latency = deque(maxlen=500)
live_llm_rate = deque(maxlen=500)
live_embedding_rate = deque(maxlen=500)

# All raw latency samples for the current concurrency run (for percentiles)
latency_samples = []
latency_samples_lock = threading.Lock()

# Sweep summary results
sweep_results = []

# Counters
metrics_lock = threading.Lock()
total_writes = 0
total_latency = 0.0
total_errors = 0
total_attempts = 0


# ============================================================
# WORKERS
# ============================================================


def get_current_timestamp():
    now = datetime.datetime.now()
    weekday = now.strftime("%a")
    return now.strftime(f"%Y/%m/%d ({weekday}) %H:%M:%S")


async def write_memory(worker_id, i):
    messages = [
        {
            "role": "user",
            "content": f"user {worker_id} preference {i}. Actually I like apples more than bananas.",
            "time_stamp": get_current_timestamp(),
        }
    ]
    await asyncio.to_thread(
        memory.add_memory,
        messages=messages,
        force_segment=True,
        force_extract=True,
    )


async def worker(worker_id, stop_event):
    global total_writes, total_latency, total_errors, total_attempts
    i = 0
    while not stop_event.is_set():
        start_t = time.perf_counter()
        with metrics_lock:
            total_attempts += 1
        try:
            await write_memory(worker_id, i)
            duration = time.perf_counter() - start_t
            with metrics_lock:
                total_writes += 1
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
    current_running = True

    reset_stats(memory)
    total_writes = 0
    total_latency = 0.0
    total_errors = 0
    total_attempts = 0

    with latency_samples_lock:
        latency_samples.clear()

    live_timestamps.clear()
    live_throughput.clear()
    live_latency.clear()
    live_llm_rate.clear()
    live_embedding_rate.clear()

    stop_event = asyncio.Event()
    workers = [asyncio.create_task(worker(i, stop_event)) for i in range(concurrency)]

    start = time.perf_counter()
    last_sample = start
    last_writes = 0
    last_latency = 0.0

    while time.perf_counter() - start < CONFIG["test_seconds"]:
        await asyncio.sleep(1)

        now = time.perf_counter()
        elapsed = now - start
        interval = now - last_sample

        with metrics_lock:
            writes_now = total_writes
            latency_now = total_latency

        delta_writes = writes_now - last_writes
        delta_latency = latency_now - last_latency
        throughput = delta_writes / interval
        avg_latency_sec = (delta_latency / delta_writes) if delta_writes > 0 else 0.0

        stats = memory.get_token_statistics()
        llm_calls = stats.get("summary", {}).get("total_llm_calls", 0)
        emb_calls = stats.get("summary", {}).get("total_embedding_calls", 0)
        llm_rate = llm_calls / elapsed if elapsed > 0 else 0
        embedding_rate = emb_calls / elapsed if elapsed > 0 else 0

        live_timestamps.append(elapsed)
        live_throughput.append(throughput)
        live_latency.append(avg_latency_sec)
        live_llm_rate.append(llm_rate)
        live_embedding_rate.append(embedding_rate)

        last_sample = now
        last_writes = writes_now
        last_latency = latency_now

    stop_event.set()
    await asyncio.gather(*workers)

    total_time = time.perf_counter() - start
    final_stats = memory.get_token_statistics()

    final_llm = final_stats.get("summary", {}).get("total_llm_calls", 0)
    final_llm_t = final_stats.get("summary", {}).get("total_llm_time", 0.0)
    final_emb = final_stats.get("summary", {}).get("total_embedding_calls", 0)
    final_emb_t = final_stats.get("summary", {}).get("total_embedding_time", 0.0)
    stage_t = final_stats.get("stage_timings", {})

    # Compute latency percentiles
    with latency_samples_lock:
        samples = list(latency_samples)

    if samples:
        p50 = float(np.percentile(samples, 50))
        p95 = float(np.percentile(samples, 95))
        p99 = float(np.percentile(samples, 99))
    else:
        p50 = p95 = p99 = 0.0

    # Error rate
    with metrics_lock:
        _attempts = total_attempts
        _errors = total_errors
        _writes = total_writes
    error_rate = _errors / _attempts if _attempts > 0 else 0.0

    # Avg stage times per completed write
    n = _writes if _writes > 0 else 1
    sweep_results.append({
        "concurrency": concurrency,
        "throughput": _writes / total_time,
        "avg_latency": total_latency / _writes if _writes > 0 else 0.0,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "error_rate": error_rate,
        "llm_rate": final_llm / total_time,
        "embedding_rate": final_emb / total_time,
        "avg_llm_time": final_llm_t / final_llm if final_llm > 0 else 0.0,
        "avg_embedding_time": final_emb_t / final_emb if final_emb > 0 else 0.0,
        # avg per-write stage seconds
        "stage_compress": stage_t.get("compress", 0.0) / n,
        "stage_segment": stage_t.get("segment", 0.0) / n,
        "stage_llm_extract": stage_t.get("llm_extract", 0.0) / n,
        "stage_db_insert": stage_t.get("db_insert", 0.0) / n,
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
# DASHBOARD — DESIGN SYSTEM
# ============================================================

BG = "#0f1117"
CARD_BG = "#1a1d27"
BORDER = "#2a2d3e"
ACCENT = "#6366f1"         # indigo
ACCENT2 = "#22d3ee"        # cyan
ACCENT3 = "#f59e0b"        # amber
ACCENT4 = "#10b981"        # emerald
ACCENT_ERR = "#ef4444"     # red
TEXT = "#e2e8f0"
TEXT_DIM = "#64748b"

PLOT_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    margin=dict(l=48, r=16, t=44, b=40),
    xaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER,
        tickcolor=BORDER,
        showline=True,
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER,
        tickcolor=BORDER,
        showline=True,
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=BORDER,
    ),
)


def make_scatter(x, y, color, name=None, mode="lines", dash=None):
    line_kwargs = dict(color=color, width=2)
    if dash:
        line_kwargs["dash"] = dash
    return go.Scatter(
        x=x, y=y, mode=mode, name=name or "",
        line=line_kwargs,
        marker=dict(color=color, size=7) if "markers" in mode else {},
    )


def card_graph(graph_id):
    return dcc.Graph(
        id=graph_id,
        config={"displayModeBar": False},
        style={"borderRadius": "12px", "overflow": "hidden"},
    )


def section(title, graphs, ncols=2):
    grid_style = {
        "display": "grid",
        "gridTemplateColumns": f"repeat({ncols}, 1fr)",
        "gap": "16px",
        "marginTop": "16px",
    }
    return html.Div([
        html.H3(title, style={
            "color": ACCENT2,
            "fontFamily": "Inter, sans-serif",
            "fontSize": "14px",
            "fontWeight": "600",
            "letterSpacing": "0.08em",
            "textTransform": "uppercase",
            "margin": "0 0 4px 2px",
        }),
        html.Div(graphs, style=grid_style),
    ], style={"marginBottom": "32px"})


app = Dash(__name__)
app.title = "LightMem Load Test Dashboard"

app.layout = html.Div([

    # ── Header ──────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("⚡", style={"fontSize": "28px", "marginRight": "12px"}),
            html.Div([
                html.H1("LightMem Load Test", style={
                    "margin": 0, "fontSize": "22px", "fontWeight": "700",
                    "color": TEXT, "fontFamily": "Inter, sans-serif",
                }),
                html.P("Real-time concurrency sweep dashboard", style={
                    "margin": 0, "color": TEXT_DIM, "fontSize": "13px",
                    "fontFamily": "Inter, sans-serif",
                }),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(id="status_badge", style={
            "fontSize": "13px",
            "fontFamily": "Inter, sans-serif",
            "padding": "6px 14px",
            "borderRadius": "20px",
            "fontWeight": "600",
        }),
    ], style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "padding": "20px 28px",
        "background": CARD_BG,
        "borderBottom": f"1px solid {BORDER}",
        "marginBottom": "28px",
    }),

    # ── Body ────────────────────────────────────────────────
    html.Div([

        section("Live — Current Concurrency Run", [
            card_graph("live_throughput"),
            card_graph("live_latency"),
            card_graph("live_llm"),
            card_graph("live_embedding"),
        ], ncols=2),

        section("Sweep Results — Throughput & Latency", [
            card_graph("sweep_throughput"),
            card_graph("sweep_latency_percentiles"),
            card_graph("sweep_error_rate"),
        ], ncols=3),

        section("Sweep Results — API Call Rates", [
            card_graph("sweep_llm"),
            card_graph("sweep_embedding"),
            card_graph("avg_llm_time"),
            card_graph("avg_emb_time"),
        ], ncols=2),

        section("Sweep Results — Pipeline Stage Breakdown (avg sec/write)", [
            card_graph("stage_breakdown"),
        ], ncols=1),

    ], style={"padding": "0 28px 40px"}),

    dcc.Interval(id="interval", interval=1000, n_intervals=0),

], style={"backgroundColor": BG, "minHeight": "100vh"})


# ============================================================
# CALLBACK
# ============================================================


@app.callback(
    [
        Output("live_throughput", "figure"),
        Output("live_latency", "figure"),
        Output("live_llm", "figure"),
        Output("live_embedding", "figure"),
        Output("sweep_throughput", "figure"),
        Output("sweep_latency_percentiles", "figure"),
        Output("sweep_error_rate", "figure"),
        Output("sweep_llm", "figure"),
        Output("sweep_embedding", "figure"),
        Output("avg_llm_time", "figure"),
        Output("avg_emb_time", "figure"),
        Output("stage_breakdown", "figure"),
        Output("status_badge", "children"),
        Output("status_badge", "style"),
        Output("interval", "disabled"),
    ],
    [Input("interval", "n_intervals")],
)
def update_dashboard(n):
    ts = list(live_timestamps)
    conc = [r["concurrency"] for r in sweep_results]

    def empty_fig(title):
        fig = go.Figure()
        fig.update_layout(title=dict(text=title, font=dict(size=13, color=TEXT)),
                          **PLOT_LAYOUT)
        return fig

    # ── Live: Throughput ────────────────────────────────────
    fig_lt = go.Figure()
    fig_lt.add_trace(make_scatter(ts, list(live_throughput), ACCENT,
                                  name="writes/sec"))
    fig_lt.update_layout(title=dict(
        text=f"Throughput  ·  concurrency={current_concurrency}",
        font=dict(size=13, color=TEXT)),
        yaxis_title="writes/sec", xaxis_title="elapsed (s)", **PLOT_LAYOUT)

    # ── Live: Latency ───────────────────────────────────────
    fig_ll = go.Figure()
    fig_ll.add_trace(make_scatter(ts, list(live_latency), ACCENT2,
                                  name="avg E2E (s)"))
    fig_ll.update_layout(title=dict(text="Avg E2E Latency  ·  live",
                                    font=dict(size=13, color=TEXT)),
                         yaxis_title="seconds", xaxis_title="elapsed (s)",
                         **PLOT_LAYOUT)

    # ── Live: LLM rate ──────────────────────────────────────
    fig_lr = go.Figure()
    fig_lr.add_trace(make_scatter(ts, list(live_llm_rate), ACCENT3,
                                  name="LLM calls/sec"))
    fig_lr.update_layout(title=dict(text="LLM Calls/sec  ·  live",
                                    font=dict(size=13, color=TEXT)),
                         yaxis_title="calls/sec", xaxis_title="elapsed (s)",
                         **PLOT_LAYOUT)

    # ── Live: Embedding rate ─────────────────────────────────
    fig_le = go.Figure()
    fig_le.add_trace(make_scatter(ts, list(live_embedding_rate), ACCENT4,
                                  name="embed calls/sec"))
    fig_le.update_layout(title=dict(text="Embedding Calls/sec  ·  live",
                                    font=dict(size=13, color=TEXT)),
                         yaxis_title="calls/sec", xaxis_title="elapsed (s)",
                         **PLOT_LAYOUT)

    # ── Sweep: Throughput ────────────────────────────────────
    fig_st = go.Figure()
    fig_st.add_trace(make_scatter(conc, [r["throughput"] for r in sweep_results],
                                  ACCENT, name="throughput", mode="lines+markers"))
    fig_st.update_layout(title=dict(text="Throughput vs Concurrency",
                                    font=dict(size=13, color=TEXT)),
                         yaxis_title="writes/sec", xaxis_title="concurrency",
                         **PLOT_LAYOUT)

    # ── Sweep: Latency percentiles ───────────────────────────
    fig_slp = go.Figure()
    fig_slp.add_trace(make_scatter(conc, [r["avg_latency"] for r in sweep_results],
                                   ACCENT, name="mean", mode="lines+markers"))
    fig_slp.add_trace(make_scatter(conc, [r["p50"] for r in sweep_results],
                                   ACCENT2, name="p50", mode="lines+markers"))
    fig_slp.add_trace(make_scatter(conc, [r["p95"] for r in sweep_results],
                                   ACCENT3, name="p95", mode="lines+markers",
                                   dash="dot"))
    fig_slp.add_trace(make_scatter(conc, [r["p99"] for r in sweep_results],
                                   ACCENT_ERR, name="p99", mode="lines+markers",
                                   dash="dash"))
    fig_slp.update_layout(title=dict(text="E2E Latency Percentiles vs Concurrency",
                                     font=dict(size=13, color=TEXT)),
                          yaxis_title="seconds", xaxis_title="concurrency",
                          **PLOT_LAYOUT)

    # ── Sweep: Error rate ────────────────────────────────────
    fig_ser = go.Figure()
    fig_ser.add_trace(make_scatter(conc, [r["error_rate"] * 100 for r in sweep_results],
                                   ACCENT_ERR, name="error %", mode="lines+markers"))
    fig_ser.update_layout(title=dict(text="Error Rate vs Concurrency",
                                     font=dict(size=13, color=TEXT)),
                          yaxis_title="error rate (%)", xaxis_title="concurrency",
                          **PLOT_LAYOUT)

    # ── Sweep: LLM call rate ─────────────────────────────────
    fig_sl = go.Figure()
    fig_sl.add_trace(make_scatter(conc, [r["llm_rate"] for r in sweep_results],
                                  ACCENT3, name="LLM calls/sec", mode="lines+markers"))
    fig_sl.update_layout(title=dict(text="LLM Calls/sec vs Concurrency",
                                    font=dict(size=13, color=TEXT)),
                         yaxis_title="calls/sec", xaxis_title="concurrency",
                         **PLOT_LAYOUT)

    # ── Sweep: Embedding call rate ───────────────────────────
    fig_se = go.Figure()
    fig_se.add_trace(make_scatter(conc, [r["embedding_rate"] for r in sweep_results],
                                  ACCENT4, name="embed calls/sec",
                                  mode="lines+markers"))
    fig_se.update_layout(title=dict(text="Embedding Calls/sec vs Concurrency",
                                    font=dict(size=13, color=TEXT)),
                         yaxis_title="calls/sec", xaxis_title="concurrency",
                         **PLOT_LAYOUT)

    # ── Sweep: Avg time per LLM call ─────────────────────────
    fig_alt = go.Figure()
    fig_alt.add_trace(make_scatter(conc, [r["avg_llm_time"] for r in sweep_results],
                                   ACCENT3, name="avg LLM time", mode="lines+markers"))
    fig_alt.update_layout(title=dict(text="Avg Time/LLM Call vs Concurrency",
                                     font=dict(size=13, color=TEXT)),
                          yaxis_title="seconds", xaxis_title="concurrency",
                          **PLOT_LAYOUT)

    # ── Sweep: Avg time per embedding call ───────────────────
    fig_aet = go.Figure()
    fig_aet.add_trace(make_scatter(conc, [r["avg_embedding_time"] for r in sweep_results],
                                   ACCENT4, name="avg embed time",
                                   mode="lines+markers"))
    fig_aet.update_layout(title=dict(text="Avg Time/Embedding Call vs Concurrency",
                                     font=dict(size=13, color=TEXT)),
                          yaxis_title="seconds", xaxis_title="concurrency",
                          **PLOT_LAYOUT)

    # ── Sweep: Stage breakdown (stacked area) ────────────────
    fig_sb = go.Figure()
    if sweep_results:
        stage_colors = [ACCENT, ACCENT2, ACCENT3, ACCENT4]
        stage_keys = ["stage_compress", "stage_segment", "stage_llm_extract", "stage_db_insert"]
        stage_labels = ["Compression", "Segmentation", "LLM Extract", "DB Insert"]
        for key, label, color in zip(stage_keys, stage_labels, stage_colors):
            fig_sb.add_trace(go.Scatter(
                x=conc,
                y=[r[key] for r in sweep_results],
                mode="lines+markers",
                name=label,
                stackgroup="stages",
                line=dict(color=color, width=2),
                fillcolor=color.replace(")", ", 0.25)").replace("rgb", "rgba")
                             if "rgb" in color else color,
            ))
    fig_sb.update_layout(
        title=dict(text="Pipeline Stage Time Breakdown (avg sec/write, stacked)",
                   font=dict(size=13, color=TEXT)),
        yaxis_title="seconds per write", xaxis_title="concurrency",
        **PLOT_LAYOUT,
    )

    # ── Status badge ─────────────────────────────────────────
    if sweep_finished:
        badge_text = "✅  Sweep Complete"
        badge_style = {
            "fontSize": "13px", "fontFamily": "Inter, sans-serif",
            "padding": "6px 14px", "borderRadius": "20px", "fontWeight": "600",
            "backgroundColor": "#064e3b", "color": "#6ee7b7",
            "border": "1px solid #065f46",
        }
    elif current_running:
        badge_text = f"⚡  Running  ·  concurrency={current_concurrency}"
        badge_style = {
            "fontSize": "13px", "fontFamily": "Inter, sans-serif",
            "padding": "6px 14px", "borderRadius": "20px", "fontWeight": "600",
            "backgroundColor": "#1e1b4b", "color": "#a5b4fc",
            "border": f"1px solid {ACCENT}",
        }
    else:
        badge_text = "⏳  Preparing..."
        badge_style = {
            "fontSize": "13px", "fontFamily": "Inter, sans-serif",
            "padding": "6px 14px", "borderRadius": "20px", "fontWeight": "600",
            "backgroundColor": "#1c1917", "color": "#a3a3a3",
            "border": "1px solid #3f3f46",
        }

    return (
        fig_lt, fig_ll, fig_lr, fig_le,
        fig_st, fig_slp, fig_ser,
        fig_sl, fig_se,
        fig_alt, fig_aet,
        fig_sb,
        badge_text, badge_style,
        sweep_finished,
    )


# ============================================================
# ENTRYPOINT
# ============================================================


def main():
    threading.Thread(target=lambda: asyncio.run(run_sweep()), daemon=True).start()
    app.run(host="0.0.0.0", port=CONFIG["dashboard_port"], debug=False)


if __name__ == "__main__":
    main()
