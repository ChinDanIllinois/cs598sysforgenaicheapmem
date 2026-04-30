"""
LightMem Multi-Tenant Load Test Harness
Simulates a realistic multi-tenant workload using LongMemEval dataset.
Events (historical session archiving and queries) are played back according to their timestamps.
Synced with configuration options from lightmem_profiler.py.
Includes a live Dash dashboard for real-time monitoring.

Streaming precompression support:
- --streaming-precompress enables LightMem pre_compress_streaming.
- For archive events, ingest_message_stream(...) is called before add_memory(...), so
  add_memory can reuse the precomputed compressed result if LightMem cache matching works.
- Per-event CSV includes stage_stream_precompute, stage_add_memory, stage_extract_wait.
- Throughput CSV includes stream_cache_hits, stream_cache_misses, stream_precompute_calls,
  and stream_precompute_time.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import time
import uuid
import threading
import traceback
from typing import Dict, Any

import dotenv
import numpy as np
import pandas as pd
from lightmem.memory.lightmem import LightMemory

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import flask.cli

# Suppress Logs
flask.cli.show_server_banner = lambda *args: None
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("flask").setLevel(logging.ERROR)
logging.getLogger("dash").setLevel(logging.ERROR)
logging.getLogger("lightmem").setLevel(logging.ERROR)
logging.getLogger("LightMemory").setLevel(logging.ERROR)

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
# GLOBAL STATE & METRICS
# ============================================================

class LoadTestMetrics:
    def __init__(self):
        self.results = []
        self.throughput_records = []
        self.total_completed = 0
        self.total_errors = 0
        self.total_events = 0
        self.active_archives = 0
        self.active_queries = 0
        self.queue_depth_history = []
        self.start_time = time.time()
        self.simulation_finished = False
        self.lock = threading.RLock()

    def record(self, event_type, user_id, latency, status="success", stage_timings=None):
        row = {
            "wall_time": time.time() - self.start_time,
            "type": event_type,
            "user_id": user_id,
            "latency": latency,
            "status": status,
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
        with self.lock:
            df = pd.DataFrame(self.results)
        df.to_csv(f"{filename_base}.csv", index=False)

        if self.throughput_records:
            df_tput = pd.DataFrame(self.throughput_records)
            df_tput.to_csv(f"{filename_base}_throughput.csv", index=False)
            print(f"Throughput stats saved to {filename_base}_throughput.csv")

        print(f"Raw event metrics saved to {filename_base}.csv")


GLOBAL_METRICS = LoadTestMetrics()


# ============================================================
# CLI ARGUMENT PARSING
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LightMem Multi-Tenant Multi-User Load Test simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Provider & Backend Config
    parser.add_argument(
        "--provider",
        required=True,
        choices=["ollama", "gemini", "openai", "vllm", "mock"],
        help="LLM backend provider.",
    )
    parser.add_argument("--llm-batch-size", type=int, default=1)
    parser.add_argument("--llm-batch-timeout", type=int, default=10)
    parser.add_argument("--vllm-adaptive-shaping", action="store_true")
    parser.add_argument("--vllm-metrics-url", type=str, default="")
    parser.add_argument("--vllm-embed-url", type=str, default="", help="vLLM endpoint for embedding model.")
    parser.add_argument("--rpm", type=float, default=0.0, metavar="RPM")
    parser.add_argument(
        "--use-server-compress",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
        help="Use external server for llmlingua-2 compression. Default: True.",
    )

    # Streaming precompression
    parser.add_argument(
        "--streaming-precompress",
        action="store_true",
        help="Enable LightMem streaming precompression before archive writes.",
    )
    parser.add_argument(
        "--disable-inline-stream-precompute",
        action="store_true",
        help=(
            "Enable pre_compress_streaming in config but do not explicitly call "
            "ingest_message_stream before add_memory. Mostly useful if another component "
            "already does streaming ingest."
        ),
    )

    # Multi-Tenant Simulation Control
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--time-scale", type=float, default=3600.0)
    parser.add_argument("--target-duration", type=float, default=0)
    parser.add_argument("--max-users", type=int, default=50)
    parser.add_argument("--max-sessions-per-user", type=int, default=10)
    parser.add_argument("--concurrency-limit", type=int, default=10)
    parser.add_argument("--skip-history", action="store_true")
    parser.add_argument("--skip-queries", action="store_true")
    parser.add_argument("--autonomous-sleep", action="store_true", help="Enable autonomous sleep detection and consolidation.")
    # Dataset Slicing
    parser.add_argument("--start-date", type=str, default="")
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--max-events", type=int, default=0)

    # Metadata & Logging
    parser.add_argument("--run-name", type=str, default="multitenant_run")
    parser.add_argument("--port", type=int, default=8052, help="Dashboard port.")

    return parser.parse_args()


# ============================================================
# DASHBOARD — DESIGN SYSTEM
# ============================================================

BG = "#0f1117"
CARD_BG = "#1a1d27"
BORDER = "#2a2d3e"
ACCENT = "#6366f1"
ACCENT2 = "#22d3ee"
ACCENT3 = "#f59e0b"
ACCENT4 = "#10b981"
ACCENT_ERR = "#ef4444"
TEXT = "#e2e8f0"
TEXT_DIM = "#64748b"

PLOT_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    margin=dict(l=48, r=16, t=44, b=40),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showline=True),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showline=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
)


def make_scatter(x, y, color, name=None, mode="lines", dash=None, fill=None):
    kw = dict(x=x, y=y, mode=mode, name=name or "", line=dict(color=color, width=2))
    if dash:
        kw["line"]["dash"] = dash
    if fill:
        kw["fill"] = fill
    return go.Scatter(**kw)


def card_graph(gid):
    return dcc.Graph(
        id=gid,
        config={"displayModeBar": False},
        style={"borderRadius": "12px", "overflow": "hidden"},
    )


def section(title, graphs, ncols=2):
    return html.Div(
        [
            html.H3(
                title,
                style={
                    "color": ACCENT2,
                    "fontSize": "11px",
                    "fontWeight": "700",
                    "letterSpacing": "0.1em",
                    "textTransform": "uppercase",
                    "margin": "0 0 12px 2px",
                },
            ),
            html.Div(
                graphs,
                style={
                    "display": "grid",
                    "gridTemplateColumns": f"repeat({ncols}, 1fr)",
                    "gap": "16px",
                },
            ),
        ],
        style={"marginBottom": "36px"},
    )


# ============================================================
# DASHBOARD SETUP
# ============================================================

def create_dash_app(requests_pathname_prefix: str):
    app = Dash(
        __name__,
        requests_pathname_prefix=requests_pathname_prefix,
        routes_pathname_prefix="/",
    )
    app.title = "LightMem Multi-Tenant Dashboard"

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("🏢", style={"fontSize": "26px", "marginRight": "12px"}),
                            html.Div(
                                [
                                    html.H1(
                                        "Multi-Tenant Load Test",
                                        style={"margin": 0, "fontSize": "20px", "color": TEXT},
                                    ),
                                    html.P(
                                        id="sim-info",
                                        style={"margin": 0, "color": TEXT_DIM, "fontSize": "12px"},
                                    ),
                                ]
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Div(id="status_badge"),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "padding": "18px 28px",
                    "background": CARD_BG,
                    "borderBottom": f"1px solid {BORDER}",
                    "marginBottom": "28px",
                },
            ),
            html.Div(
                [
                    section("Real-Time Load Performance", [card_graph("live_throughput"), card_graph("live_latency")], ncols=2),
                    section("System Mix & Scaling", [card_graph("event_mix"), card_graph("active_users")], ncols=2),
                    section("Pipeline & Backlog", [card_graph("stage_breakdown"), card_graph("live_backlog")], ncols=2),
                    section("Simulation Progress", [card_graph("sim_progress"), card_graph("recent_errors")], ncols=2),
                ],
                style={"padding": "0 28px 40px"},
            ),
            dcc.Interval(id="interval", interval=2000, n_intervals=0),
        ],
        style={"backgroundColor": BG, "minHeight": "100vh", "fontFamily": "Inter, sans-serif"},
    )

    @app.callback(
        [
            Output("live_throughput", "figure"),
            Output("live_latency", "figure"),
            Output("event_mix", "figure"),
            Output("active_users", "figure"),
            Output("stage_breakdown", "figure"),
            Output("live_backlog", "figure"),
            Output("sim_progress", "figure"),
            Output("recent_errors", "figure"),
            Output("status_badge", "children"),
            Output("sim-info", "children"),
        ],
        [Input("interval", "n_intervals")],
    )
    def update_metrics(n):
        with GLOBAL_METRICS.lock:
            data = list(GLOBAL_METRICS.results)
            tput_hist = list(GLOBAL_METRICS.throughput_records)
            backlog_hist = list(GLOBAL_METRICS.queue_depth_history)
            finished = GLOBAL_METRICS.simulation_finished
            total_comp = GLOBAL_METRICS.total_completed
            total_evs = GLOBAL_METRICS.total_events

        # 1. Throughput Figure
        fig_tput = go.Figure()
        if tput_hist:
            tx = [r["elapsed_sec"] for r in tput_hist]
            ty = [r["throughput_eps"] for r in tput_hist]
            fig_tput.add_trace(make_scatter(tx, ty, ACCENT, "EPS", fill="tozeroy"))
        fig_tput.update_layout(title="Throughput (Events Per Second)", **PLOT_LAYOUT)

        # 2. Latency Figure
        fig_lat = go.Figure()
        if data:
            df = pd.DataFrame(data)
            if not df.empty and "latency" in df.columns:
                df["win"] = (df["wall_time"] // 5) * 5
                win_groups = (
                    df.groupby("win")["latency"]
                    .agg([
                        lambda x: np.percentile(x, 50),
                        lambda x: np.percentile(x, 95),
                        lambda x: np.percentile(x, 99),
                    ])
                    .reset_index()
                )
                win_groups.columns = ["win", "p50", "p95", "p99"]
                fig_lat.add_trace(make_scatter(win_groups["win"], win_groups["p50"], ACCENT2, "P50 Latency"))
                fig_lat.add_trace(make_scatter(win_groups["win"], win_groups["p95"], ACCENT3, "P95 Latency", dash="dot"))
                fig_lat.add_trace(make_scatter(win_groups["win"], win_groups["p99"], ACCENT_ERR, "P99 Latency", dash="dash"))
        fig_lat.update_layout(title="Latency Trends (P50/P95/P99)", **PLOT_LAYOUT)

        # 3. Request Mix
        fig_mix = go.Figure()
        if data:
            df = pd.DataFrame(data)
            df["win"] = (df["wall_time"] // 5) * 5
            mix = df.groupby(["win", "type"]).size().unstack(fill_value=0).reset_index()
            if "archive" in mix.columns:
                fig_mix.add_trace(make_scatter(mix["win"], mix["archive"], ACCENT, "Archives (Writes)", fill="tozeroy"))
            if "query" in mix.columns:
                fig_mix.add_trace(make_scatter(mix["win"], mix["query"], ACCENT2, "Queries (Reads)", fill="tonexty"))
        fig_mix.update_layout(title="Request Mix (Writes vs Reads)", **PLOT_LAYOUT)

        # 4. Active Users
        fig_users = go.Figure()
        if data:
            df = pd.DataFrame(data)
            df["win"] = (df["wall_time"] // 5) * 5
            unique_users = []
            user_timeline = df.sort_values("wall_time")
            seen = set()
            current_win = -1
            for _, r in user_timeline.iterrows():
                seen.add(r["user_id"])
                w = (r["wall_time"] // 5) * 5
                if w > current_win:
                    unique_users.append((w, len(seen)))
                    current_win = w
            ux, uy = zip(*unique_users) if unique_users else ([], [])
            fig_users.add_trace(make_scatter(ux, uy, ACCENT4, "Total Unique Tenants", fill="tozeroy"))
        fig_users.update_layout(title="Tenant Saturation (Unique Users)", **PLOT_LAYOUT)

        # 5. Pipeline Stages
        fig_sb = go.Figure()
        if len(tput_hist) > 1:
            t_df = pd.DataFrame(tput_hist)
            t_df_delta = t_df.set_index("elapsed_sec").diff().reset_index()

            stages = [
                ("stage_compress_time", "Compress", ACCENT),
                ("stage_segment_time", "Segment", ACCENT2),
                ("stage_llm_extract_time", "LLM Extract", ACCENT3),
                ("stage_db_insert_time", "DB Insert", ACCENT4),
                ("stream_precompute_time", "Stream Precompute", "#a78bfa"),
            ]
            for k, label, color in stages:
                if k in t_df_delta.columns:
                    fig_sb.add_trace(go.Bar(name=label, x=t_df_delta["elapsed_sec"], y=t_df_delta[k], marker_color=color))
        fig_sb.update_layout(barmode="stack", title="Pipeline Stage Distribution (Seconds per Interval)", **PLOT_LAYOUT)

        # 6. Backlog Figure
        fig_backlog = go.Figure()
        if backlog_hist:
            bx = [r["elapsed_sec"] for r in backlog_hist]
            by_arch = [r["archives"] for r in backlog_hist]
            by_q = [r["queries"] for r in backlog_hist]
            fig_backlog.add_trace(make_scatter(bx, by_arch, ACCENT, "Queued Writes (Archives)", fill="tozeroy"))
            fig_backlog.add_trace(make_scatter(bx, by_q, ACCENT2, "Queued Reads (Queries)", fill="tonexty"))
        fig_backlog.update_layout(title="Active Backlog (In-Flight Requests)", **PLOT_LAYOUT)

        # 7. Simulation Progress
        fig_prog = go.Figure()
        if total_evs > 0:
            fig_prog.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=total_comp,
                    delta={"reference": total_evs, "position": "top", "increasing": {"color": ACCENT}},
                    title={"text": "Total Progress", "font": {"size": 14}},
                    gauge={
                        "axis": {"range": [None, total_evs], "tickwidth": 1, "tickcolor": TEXT_DIM},
                        "bar": {"color": ACCENT},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 2,
                        "bordercolor": BORDER,
                        "steps": [{"range": [0, total_evs], "color": CARD_BG}],
                        "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": total_comp},
                    },
                )
            )
        fig_prog.update_layout(title="Events Processed vs Total", height=300, **PLOT_LAYOUT)

        # 8. Recent Errors
        fig_err = go.Figure()
        if data:
            df = pd.DataFrame(data)
            err_df = df[df["status"] == "error"]
            if not err_df.empty:
                err_tail = err_df.tail(50)
                fig_err.add_trace(
                    go.Scatter(
                        x=err_tail.index,
                        y=err_tail["user_id"],
                        mode="markers",
                        marker=dict(color=ACCENT_ERR, size=12, symbol="x"),
                        name="Error",
                    )
                )
        fig_err.update_layout(title="Last 50 Errors (Event Timeline)", **PLOT_LAYOUT, xaxis_title="Event Index", yaxis_title="User ID")

        status = html.Span("✅ Finished", style={"color": "#6ee7b7"}) if finished else html.Span("⚡ Processing", style={"color": ACCENT2})
        info = f"Progress: {total_comp}/{total_evs} | Multi-Tenant Simulation v1.0"

        return fig_tput, fig_lat, fig_mix, fig_users, fig_sb, fig_backlog, fig_prog, fig_err, status, info

    return app


# ============================================================
# CORE LOGIC
# ============================================================

def parse_date(date_str: str) -> float:
    try:
        parts = date_str.split(" ")
        clean_str = f"{parts[0]} {parts[2]}"
        dt = datetime.datetime.strptime(clean_str, "%Y/%m/%d %H:%M")
        return dt.timestamp()
    except Exception:
        return time.time()


def load_events(data_path: str, args):
    if not data_path or not os.path.exists(data_path):
        return []

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    start_ts = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").timestamp() if args.start_date else 0.0
    end_ts = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").timestamp() if args.end_date else float("inf")

    all_events = []
    for item in data[: args.max_users]:
        uid = item["question_id"]

        if not args.skip_history:
            sessions = item.get("haystack_sessions", [])[: args.max_sessions_per_user]
            dates = item.get("haystack_dates", [])
            for session, d in zip(sessions, dates):
                ts = parse_date(d)
                if start_ts <= ts <= end_ts:
                    msg_tagged = [dict(m, time_stamp=d) for m in session]
                    all_events.append({"ts": ts, "type": "archive", "user_id": uid, "content": msg_tagged})

        if not args.skip_queries:
            q_date = item.get("question_date")
            if q_date:
                ts = parse_date(q_date)
                if start_ts <= ts <= end_ts:
                    all_events.append({"ts": ts, "type": "query", "user_id": uid, "content": item["question"]})

    all_events.sort(key=lambda x: x["ts"])
    return all_events[: args.max_events] if args.max_events > 0 else all_events


async def monitor_throughput(memory, metrics, stop_event):
    last_count = 0
    last_time = time.time()

    while not stop_event.is_set():
        await asyncio.sleep(2)
        now = time.time()

        try:
            all_stats = await asyncio.to_thread(memory.get_token_statistics)
        except Exception as e:
            print(f"Monitor failed to fetch token statistics: {repr(e)}")
            continue

        with metrics.lock:
            cur, err = metrics.total_completed, metrics.total_errors
            delta, dt = cur - last_count, now - last_time
            tput = delta / dt if dt > 0 else 0

            record = {
                "elapsed_sec": now - metrics.start_time,
                "throughput_eps": tput,
                "completed_so_far": cur,
                "errors_so_far": err,
                "archives": metrics.active_archives,
                "queries": metrics.active_queries,
                "total_backlog": metrics.active_archives + metrics.active_queries,
            }

            # LightMem stage timing stats.
            stage_stats = all_stats.get("stage_timings", {})
            mapping = {
                "stage_compress_time": "compress",
                "stage_segment_time": "segment",
                "stage_llm_extract_time": "llm_extract",
                "stage_db_insert_time": "db_insert",
            }
            for prof_k, lib_k in mapping.items():
                if lib_k in stage_stats:
                    record[prof_k] = stage_stats[lib_k]

            # Streaming precompression cache stats.
            stream_stats = getattr(memory, "streaming_precompress_stats", {}) or {}
            record["stream_cache_hits"] = stream_stats.get("cache_hits", 0)
            record["stream_cache_misses"] = stream_stats.get("cache_misses", 0)
            record["stream_precompute_calls"] = stream_stats.get("precompute_calls", 0)
            record["stream_precompute_time"] = stream_stats.get("precompute_time", 0.0)

            metrics.throughput_records.append(record)
            metrics.queue_depth_history.append(record)

            last_count, last_time = cur, now

            if stream_stats:
                print(
                    f"   >>> [{now - metrics.start_time:6.1f}s] "
                    f"T-Put: {tput:6.2f} eps | "
                    f"Backlog: {metrics.active_archives + metrics.active_queries:3} | "
                    f"Total: {cur:5} | "
                    f"Stream hits/misses: {stream_stats.get('cache_hits', 0)}/"
                    f"{stream_stats.get('cache_misses', 0)}"
                )
            else:
                print(
                    f"   >>> [{now - metrics.start_time:6.1f}s] "
                    f"T-Put: {tput:6.2f} eps | "
                    f"Backlog: {metrics.active_archives + metrics.active_queries:3} | "
                    f"Total: {cur:5}"
                )


async def run_simulation(events, args, memory, rate_limiter):
    sem = asyncio.Semaphore(args.concurrency_limit)
    first_ts, start_wall = events[0]["ts"], time.time()
    GLOBAL_METRICS.total_events = len(events)

    stop_mon = asyncio.Event()
    mon_task = asyncio.create_task(monitor_throughput(memory, GLOBAL_METRICS, stop_mon))

    tasks = []
    user_locks = {}

    async def run_event(event):
        uid = event["user_id"]
        if uid not in user_locks:
            user_locks[uid] = asyncio.Lock()

        async with user_locks[uid]:
            # Important: semaphore must wrap the actual LightMem work, not just the rate-limiter wait.
            async with sem:
                await rate_limiter.wait()

                st = time.perf_counter()
                is_archive = event["type"] == "archive"

                with GLOBAL_METRICS.lock:
                    if is_archive:
                        GLOBAL_METRICS.active_archives += 1
                    else:
                        GLOBAL_METRICS.active_queries += 1

                try:
                    stage_timings: Dict[str, float] = {}

                    if is_archive:
                        # Inline streaming precompute: precompute compression for the exact stream
                        # that will be sent to add_memory. This should produce cache hits if the
                        # LightMem streaming cache key matches add_memory's compression path.
                        if args.streaming_precompress and not args.disable_inline_stream_precompute:
                            t_pre = time.perf_counter()
                            await asyncio.to_thread(
                                memory.ingest_message_stream,
                                event["content"],
                            )
                            stage_timings["stream_precompute"] = time.perf_counter() - t_pre

                        t_add = time.perf_counter()
                        res = await asyncio.to_thread(
                            memory.add_memory,
                            messages=event["content"],
                            user_id=event["user_id"],
                            force_segment=True,
                            force_extract=True,
                        )
                        stage_timings["add_memory"] = time.perf_counter() - t_add

                        if isinstance(res, dict) and "extraction_future" in res:
                            t_extract_wait = time.perf_counter()
                            await asyncio.wrap_future(res["extraction_future"])
                            stage_timings["extract_wait"] = time.perf_counter() - t_extract_wait

                    else:
                        t_retrieve = time.perf_counter()
                        await asyncio.to_thread(
                            memory.retrieve_memory,
                            query=event["content"],
                            user_id=event["user_id"],
                        )
                        stage_timings["retrieve"] = time.perf_counter() - t_retrieve

                    with GLOBAL_METRICS.lock:
                        GLOBAL_METRICS.record(
                            event["type"],
                            event["user_id"],
                            time.perf_counter() - st,
                            "success",
                            stage_timings=stage_timings,
                        )

                except Exception as e:
                    print(f"Error in {event['type']} for user {event['user_id']}: {repr(e)}")
                    traceback.print_exc()
                    with GLOBAL_METRICS.lock:
                        GLOBAL_METRICS.record(
                            event["type"],
                            event["user_id"],
                            time.perf_counter() - st,
                            "error",
                        )

                finally:
                    with GLOBAL_METRICS.lock:
                        if is_archive:
                            GLOBAL_METRICS.active_archives -= 1
                        else:
                            GLOBAL_METRICS.active_queries -= 1

    for i, ev in enumerate(events):
        delay = start_wall + ((ev["ts"] - first_ts) / args.time_scale) - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(run_event(ev)))

    await asyncio.gather(*tasks)
    await asyncio.sleep(2)
    stop_mon.set()
    await mon_task
    GLOBAL_METRICS.simulation_finished = True

    model_name = "unknown"
    if hasattr(memory, "manager") and hasattr(memory.manager, "config"):
        model_name = getattr(memory.manager.config, "model", "unknown")
    model_short = str(model_name).split("/")[-1].replace(".", "-")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    params = [
        f"c{args.concurrency_limit}",
        f"b{args.llm_batch_size}",
        f"rpm{int(args.rpm)}",
        f"dur{int(args.target_duration)}",
    ]
    if args.vllm_adaptive_shaping:
        params.append("adaptive")
    if args.streaming_precompress:
        params.append("streampre")
    if args.disable_inline_stream_precompute:
        params.append("noinline")

    param_str = "_".join(params)
    filename = f"profiling_runs/{args.run_name}_{args.provider}_{model_short}_{param_str}_{run_id}"

    GLOBAL_METRICS.save(filename)


# ============================================================
# LIGHTMEM CONFIG
# ============================================================

def setup_lightmem(args):
    builder = _BUILDERS.get(args.provider)
    if builder is None:
        raise ValueError(f"Unsupported provider: {args.provider}")

    cfg = builder(args)

    # vLLM backend gets GPU for local models or remote endpoints.
    device = "cuda" if args.provider == "vllm" else "cpu"

    if args.provider == "vllm" and args.vllm_embed_url:
        embedder_cfg = {
            "model_name": "openai",
            "configs": {
                "model": os.getenv("EMBEDDING_MODEL_PATH"),
                "api_key": "EMPTY",
                "base_url": args.vllm_embed_url,
                "embedding_dims": 384,
            },
        }
    else:
        embedder_cfg = {
            "model_name": "huggingface",
            "configs": {
                "model": os.getenv("EMBEDDING_MODEL_PATH"),
                "embedding_dims": 384,
                "model_kwargs": {"device": device},
            },
        }

    collection_name = f"mt_{uuid.uuid4().hex[:8]}"
    qdrant_root = os.getenv("QDRANT_DATA_DIR") or "./qdrant_data"
    qdrant_path = os.path.join(qdrant_root, "multitenant", collection_name)

    return LightMemory.from_config(
        {
            "pre_compress": True,
            "pre_compress_streaming": args.streaming_precompress,
            "pre_compressor": {
                "model_name": "llmlingua-2",
                "configs": {
                    "llmlingua_config": {
                        "model_name": os.getenv("LLMLINGUA_MODEL_PATH"),
                        "device_map": device,
                        "use_llmlingua2": True,
                    },
                    "use_server": args.use_server_compress,
                    "server_url": "http://localhost:8090/compress",
                    "compress_config": {"rate": 0.6},
                },
            },
            "topic_segment": True,
            "precomp_topic_shared": True,
            "topic_segmenter": {
                "model_name": "llmlingua-2",
                "use_server": args.use_server_compress,
                "server_url": "http://localhost:8090/segment",
            },
            "messages_use": "user_only",
            "metadata_generate": True,
            "text_summary": True,
            "memory_manager": cfg,
            "extract_threshold": 0.1,
            "index_strategy": "embedding",
            "text_embedder": embedder_cfg,
            "retrieve_strategy": "embedding",
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": collection_name,
                    "embedding_model_dims": 384,
                    "path": qdrant_path,
                },
            },
            "update": "offline" if args.provider == "vllm" else "sync",
            "logging": {"level": "ERROR"},
            "extraction_concurrency": args.concurrency_limit,
            "autonomous_sleep": args.autonomous_sleep
        }
    )

_BUILDERS = {
    "ollama": lambda args: {
        "model_name": "ollama",
        "configs": {
            "model": os.getenv("OLLAMA_MODEL_NAME"),
            "host": os.getenv("OLLAMA_HOST"),
            "max_tokens": 16384,
        },
    },
    "gemini": lambda args: {
        "model_name": "gemini",
        "configs": {
            "model": os.getenv("GEMINI_MODEL_NAME"),
            "api_key": os.getenv("GEMINI_API_KEY"),
            "max_tokens": 16384,
        },
    },
    "vllm": lambda args: {
        "model_name": "vllm",
        "configs": {
            "model": os.getenv("VLLM_MODEL_NAME"),
            "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
            "max_tokens": 16384,
            "vllm_base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            "llm_batch_size": args.llm_batch_size,
            "llm_batch_timeout": args.llm_batch_timeout,
            "vllm_adaptive_shaping": args.vllm_adaptive_shaping,
            "vllm_metrics_url": args.vllm_metrics_url or None,
        },
    },
    "openai": lambda args: {
        "model_name": "openai",
        "configs": {
            "model": os.getenv("OPENAI_MODEL_NAME"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_tokens": 16384,
        },
    },
    "mock": lambda args: {
        "model_name": "mock",
        "configs": {"model": "mock-model"},
    },
}


# ============================================================
# MAIN
# ============================================================

def main_cli():
    args = parse_args()
    os.makedirs("profiling_runs", exist_ok=True)

    events = load_events(args.data_path or os.getenv("DATA_PATH"), args)
    print(f"Loaded {len(events)} events...")
    if not events:
        return

    if args.target_duration > 0:
        span = events[-1]["ts"] - events[0]["ts"]
        args.time_scale = span / args.target_duration if span > 0 else 1.0
        print(f"Target duration {args.target_duration}s -> Scale {args.time_scale:.2f}")

    print(f"Streaming precompression: {args.streaming_precompress}")
    print(f"Inline stream precompute: {args.streaming_precompress and not args.disable_inline_stream_precompute}")

    memory = setup_lightmem(args)
    limiter = AsyncRateLimiter(args.rpm)

    threading.Thread(
        target=lambda: asyncio.run(run_simulation(events, args, memory, limiter)),
        daemon=True,
    ).start()

    print(f"Dashboard serving at http://localhost:{args.port}")

    prefix_str = os.getenv("DASH_PROXY_PREFIX", "/").strip("/")
    requests_pathname_prefix = "/" + "/".join([prefix_str, str(args.port)]) if prefix_str else f"/{args.port}"
    requests_pathname_prefix = requests_pathname_prefix + "/"

    app = create_dash_app(requests_pathname_prefix=requests_pathname_prefix)
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main_cli()
