import dash
import os
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import requests
import time
from datetime import datetime
from collections import deque
from prometheus_client.parser import text_string_to_metric_families
import dotenv
dotenv.load_dotenv()
port = os.getenv("DASH_PORT", 8050)

# --- CONFIGURATION ---
VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL", "http://localhost:8000/metrics")
UPDATE_INTERVAL_MS = 2000  # 2 seconds
MAX_DATA_POINTS = 60  # ~2 minutes of history at 2s intervals

# --- STYLING (Glassmorphism & Neon) ---
COLORS = {
    'bg': '#0f172a',
    'card_bg': 'rgba(30, 41, 59, 0.7)',
    'accent': '#38bdf8',  # Sky blue
    'accent_secondary': '#818cf8', # Indigo
    'text': '#f1f5f9',
    'text_muted': '#94a3b8',
    'success': '#4ade80',
    'warning': '#fbbf24',
    'danger': '#f87171',
    'gpu_cache': '#22d3ee',
    'cpu_cache': '#f472b6',
}

EXTERNAL_STYLESHEETS = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap",
    "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
]

# --- METRICS DATA STORAGE ---
history = {
    'timestamps': deque(maxlen=MAX_DATA_POINTS),
    'throughput': deque(maxlen=MAX_DATA_POINTS),
    'running': deque(maxlen=MAX_DATA_POINTS),
    'waiting': deque(maxlen=MAX_DATA_POINTS),
}

last_metrics = {
    'prompt': 0, 'gen': 0, 'time': 0,
    'reg_count': 0, 'batch_count': 0
}

def get_metric_from_families(families, name, labels=None):
    """Finds a specific sample in families matching name and labels."""
    for family in families:
        for sample in family.samples:
            if sample.name == name:
                if labels:
                    if all(sample.labels.get(k) == v for k, v in labels.items()):
                        return sample.value
                else:
                    return sample.value
    return 0.0

def get_buckets_from_families(families, name, labels=None):
    """Extracts all buckets for a histogram matching name and labels."""
    buckets = []
    bucket_name = f"{name}_bucket"
    for family in families:
        for sample in family.samples:
            if sample.name == bucket_name:
                if labels:
                    if not all(sample.labels.get(k) == v for k, v in labels.items() if k != 'le'):
                        continue
                
                le = sample.labels.get('le', '+Inf')
                le_val = float('inf') if le == '+Inf' else float(le)
                buckets.append((le_val, float(sample.value)))
    return sorted(buckets)

def compute_histogram_dist(buckets):
    """Converts cumulative buckets to discrete distribution for bar charts."""
    if not buckets: return [], []
    x_labels, y_values, last_count = [], [], 0.0
    for le, count in buckets:
        label = f"<{le}" if le != float('inf') else "Outlier"
        x_labels.append(label)
        y_values.append(max(0.0, float(count) - last_count))
        last_count = float(count)
    return x_labels, y_values

def parse_vllm_metrics():
    """Professional parser using the official prometheus_client library."""
    try:
        response = requests.get(VLLM_METRICS_URL, timeout=1)
        if response.status_code != 200: return None
        
        families = list(text_string_to_metric_families(response.text))
        metrics = {}
        
        # Core Engine
        metrics['running'] = get_metric_from_families(families, "vllm:num_requests_running")
        metrics['waiting'] = get_metric_from_families(families, "vllm:num_requests_waiting")
        
        # Cache
        metrics['kv_cache'] = get_metric_from_families(families, "vllm:kv_cache_usage_perc")
        if metrics['kv_cache'] == 0:
            metrics['kv_cache'] = get_metric_from_families(families, "vllm:gpu_cache_usage_perc")
        
        # RAM
        metrics['rss_mem'] = get_metric_from_families(families, "process_resident_memory_bytes") / (1024**3)
        metrics['vms_mem'] = get_metric_from_families(families, "process_virtual_memory_bytes") / (1024**3)
            
        # Throughput
        metrics['prompt_tokens_total'] = get_metric_from_families(families, "vllm:prompt_tokens_total")
        metrics['generation_tokens_total'] = get_metric_from_families(families, "vllm:generation_tokens_total")
        
        # HTTP
        metrics['reg_count'] = get_metric_from_families(families, "http_requests_total", {"handler": "/v1/chat/completions"})
        metrics['batch_count'] = get_metric_from_families(families, "http_requests_total", {"handler": "/v1/chat/completions/batch"})
        metrics['reg_lat_sum'] = get_metric_from_families(families, "http_request_duration_seconds_sum", {"handler": "/v1/chat/completions"})
        metrics['reg_lat_count'] = get_metric_from_families(families, "http_request_duration_seconds_count", {"handler": "/v1/chat/completions"})
        
        # Histograms
        metrics['hist_ttft'] = get_buckets_from_families(families, "vllm:time_to_first_token_seconds")
        metrics['hist_itl'] = get_buckets_from_families(families, "vllm:inter_token_latency_seconds")
        metrics['hist_e2e'] = get_buckets_from_families(families, "vllm:e2e_request_latency_seconds")
                
        return metrics
    except Exception as e:
        print(f"Metrics collection failed: {e}")
        return None

# --- DASH APP ---
prefix_str = os.getenv("DASH_PROXY_PREFIX", "/").strip("/")
requests_pathname_prefix = "/" + "/".join([prefix_str, str(port)]) if prefix_str else f"/{port}/"

print(f"Serving Dash app at {requests_pathname_prefix}")

app = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS, requests_pathname_prefix=requests_pathname_prefix)

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>vLLM Real-time Dashboard</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            body {{
                background-color: {COLORS['bg']};
                color: {COLORS['text']};
                font-family: 'Inter', sans-serif;
                margin: 0;
                overflow-x: hidden;
            }}
            .glass-card {{
                background: {COLORS['card_bg']};
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            }}
            .stat-value {{
                font-size: 2.2rem;
                font-weight: 700;
                background: linear-gradient(to right, {COLORS['accent']}, {COLORS['accent_secondary']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .stat-label {{
                color: {COLORS['text_muted']};
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 4px;
            }}
            .header-dash {{
                padding: 1.5rem 0;
                background: radial-gradient(circle at top right, rgba(56, 189, 248, 0.1), transparent);
            }}
            .status-indicator {{
                height: 10px;
                width: 10px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 8px;
            }}
            .status-online {{ background-color: {COLORS['success']}; box-shadow: 0 0 10px {COLORS['success']}; }}
            .status-offline {{ background-color: {COLORS['danger']}; box-shadow: 0 0 10px {COLORS['danger']}; }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    dcc.Interval(id='interval-component', interval=UPDATE_INTERVAL_MS, n_intervals=0),
    
    # Header
    html.Div([
        html.Div([
            html.Div([
                html.H2("vLLM Engine Insights", className="mb-0 fw-bold"),
                html.P([
                    html.Span(id='status-dot', className="status-indicator status-offline"),
                    html.Span(id='status-text', children="Connecting...")
                ], className="mb-0 mt-1 small")
            ], className="col-md-6"),
            html.Div([
                html.P(f"Target: {VLLM_METRICS_URL}", className="text-end text-muted small mb-0")
            ], className="col-md-6 d-flex align-items-center justify-content-end")
        ], className="row g-0 container mx-auto px-4")
    ], className="header-dash mb-4"),

    html.Div([
        # Main Stats Row
        html.Div([
            html.Div([
                html.Div([
                    html.Div("Running Req", className="stat-label"),
                    html.Div(id="val-running", className="stat-value"),
                ], className="glass-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div("Queue Depth", className="stat-label"),
                    html.Div(id="val-waiting", className="stat-value"),
                ], className="glass-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div("Throughput (tok/s)", className="stat-label"),
                    html.Div(id="val-throughput", className="stat-value"),
                ], className="glass-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div("Process RAM (GB)", className="stat-label"),
                    html.Div(id="val-rss", className="stat-value"),
                ], className="glass-card")
            ], className="col-md-3"),
        ], className="row mb-4"),

        # Cache & Latency Summary
        html.Div([
            html.Div([
                html.Div([
                    html.Div("KV Cache Utilization", className="stat-label"),
                    html.Div(id="val-kv-perc", className="stat-value"),
                    html.Small("Portion of pre-allocated GPU blocks in use", className="text-muted", style={'fontSize': '0.6rem'})
                ], className="glass-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div("Reg Latency (avg)", className="stat-label"),
                    html.Div(id="val-reg-lat", className="stat-value"),
                ], className="glass-card")
            ], className="col-md-3"),
             html.Div([
                html.Div([
                    html.Div("Batch Traffic", className="stat-label"),
                    html.Div(id="val-batch-count", className="stat-value"),
                ], className="glass-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div("VMS Memory (GB)", className="stat-label"),
                    html.Div(id="val-vms", className="stat-value"),
                ], className="glass-card")
            ], className="col-md-3"),
        ], className="row mb-4"),

        # Charts Section
        html.Div([
            html.Div([
                html.Div([
                    html.H6("Throughput History", className="mb-3"),
                    dcc.Graph(id='graph-throughput', config={'displayModeBar': False}),
                ], className="glass-card")
            ], className="col-md-6"),
            html.Div([
                html.Div([
                    html.H6("Queue History (Running vs Waiting)", className="mb-3"),
                    dcc.Graph(id='graph-queue', config={'displayModeBar': False}),
                ], className="glass-card")
            ], className="col-md-6"),
        ], className="row mb-4"),

        html.Div([
             html.Div([
                html.Div([
                    html.H6("KV Cache Usage", className="mb-3"),
                    dcc.Graph(id='graph-cache', config={'displayModeBar': False}),
                ], className="glass-card h-100")
            ], className="col-md-4"),
            html.Div([
                html.Div([
                    html.H6("Time to First Token (TTFT) Distribution", className="mb-2"),
                    dcc.Graph(id='hist-ttft', config={'displayModeBar': False}),
                ], className="glass-card")
            ], className="col-md-8"),
        ], className="row mb-4"),

        # Histograms Section
        html.Div([
            html.Div([
                html.Div([
                    html.H6("Inter-Token Latency (ITL) Distribution", className="mb-2"),
                    dcc.Graph(id='hist-itl', config={'displayModeBar': False}),
                ], className="glass-card")
            ], className="col-md-6"),
            html.Div([
                html.Div([
                    html.H6("E2E Request Latency Distribution", className="mb-2"),
                    dcc.Graph(id='hist-e2e', config={'displayModeBar': False}),
                ], className="glass-card")
            ], className="col-md-6"),
        ], className="row")

    ], className="container px-4 pb-5")
])

@app.callback(
    [Output('status-dot', 'className'),
     Output('status-text', 'children'),
     Output('val-running', 'children'),
     Output('val-waiting', 'children'),
     Output('val-throughput', 'children'),
     Output('val-kv-perc', 'children'),
     Output('val-rss', 'children'),
     Output('val-vms', 'children'),
     Output('val-reg-lat', 'children'),
     Output('val-batch-count', 'children'),
     Output('graph-throughput', 'figure'),
     Output('graph-queue', 'figure'),
     Output('graph-cache', 'figure'),
     Output('hist-ttft', 'figure'),
     Output('hist-itl', 'figure'),
     Output('hist-e2e', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    global last_metrics
    metrics = parse_vllm_metrics()
    now = datetime.now()
    
    if not metrics:
        empty_fig = go.Figure().update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return ("status-indicator status-offline", "vLLM Offline", "0", "0", "0.0", "0%", "0.0G", "0.0G", "0.0s", "0",
                empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig)
    
    # Calculations
    current_time = time.time()
    dt = current_time - last_metrics['time']
    total_tok = metrics['prompt_tokens_total'] + metrics['generation_tokens_total']
    tput = ((total_tok - (last_metrics['prompt'] + last_metrics['gen'])) / dt) if (dt > 0 and last_metrics['time'] > 0) else 0.0
    
    last_metrics.update({
        'prompt': metrics['prompt_tokens_total'], 'gen': metrics['generation_tokens_total'],
        'time': current_time, 'reg_count': metrics['reg_count'], 'batch_count': metrics['batch_count']
    })

    reg_lat = (metrics['reg_lat_sum'] / metrics['reg_lat_count']) if metrics['reg_lat_count'] > 0 else 0
    history['timestamps'].append(now)
    history['throughput'].append(tput)
    history['running'].append(metrics['running'])
    history['waiting'].append(metrics['waiting'])

    # THROUGHPUT FIGURE
    fig_tput = go.Figure(go.Scatter(
        x=list(history['timestamps']), y=list(history['throughput']),
        mode='lines', line=dict(color=COLORS['accent'], width=3),
        fill='tozeroy', fillcolor='rgba(56, 189, 248, 0.1)'
    )).update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0), height=250, yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))

    # QUEUE FIGURE
    fig_queue = go.Figure()
    fig_queue.add_trace(go.Scatter(
        x=list(history['timestamps']), y=list(history['running']),
        mode='lines', name='Running', line=dict(color=COLORS['success'], width=2),
        fill='tozeroy', fillcolor='rgba(74, 222, 128, 0.1)'
    ))
    fig_queue.add_trace(go.Scatter(
        x=list(history['timestamps']), y=list(history['waiting']),
        mode='lines', name='Waiting', line=dict(color=COLORS['warning'], width=2),
        fill='tonexty', fillcolor='rgba(251, 191, 36, 0.1)'
    ))
    fig_queue.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           margin=dict(l=0, r=0, t=10, b=0), height=250, showlegend=True,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                           yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))

    # CACHE FIGURE
    fig_cache = go.Figure(go.Bar(
        x=['KV Cache'], y=[metrics['kv_cache']*100], marker_color=COLORS['gpu_cache'], width=0.4
    )).update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=20, b=20), height=250, yaxis=dict(range=[0, 100], gridcolor='rgba(255,255,255,0.05)'))

    # HISTOGRAM FIGURES
    def create_hist_fig(buckets, color, height=180):
        x, y = compute_histogram_dist(buckets)
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=color))
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         margin=dict(l=0, r=0, t=0, b=0), height=height, showlegend=False,
                         xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                         yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        return fig

    fig_ttft = create_hist_fig(metrics['hist_ttft'], COLORS['accent'], height=250)
    fig_itl = create_hist_fig(metrics['hist_itl'], COLORS['accent_secondary'])
    fig_e2e = create_hist_fig(metrics['hist_e2e'], COLORS['warning'])

    return (
        "status-indicator status-online", "vLLM Online",
        f"{int(metrics['running'])}", f"{int(metrics['waiting'])}", f"{tput:.1f}",
        f"{metrics['kv_cache']*100:.1f}%", f"{metrics['rss_mem']:.1f}G", f"{metrics['vms_mem']:.1f}G",
        f"{reg_lat:.2f}s", f"{int(metrics['batch_count'])}",
        fig_tput, fig_queue, fig_cache, fig_ttft, fig_itl, fig_e2e
    )

if __name__ == '__main__':
    print(f"Serving vLLM Dashboard at http://localhost:{requests_pathname_prefix}")
    app.run(debug=True, port=port)
