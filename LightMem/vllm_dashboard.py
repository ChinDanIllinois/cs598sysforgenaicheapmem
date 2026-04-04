import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import requests
import re
import time
from datetime import datetime
from collections import deque

# --- CONFIGURATION ---
VLLM_METRICS_URL = "http://localhost:8000/metrics"
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
    'running': deque(maxlen=MAX_DATA_POINTS),
    'waiting': deque(maxlen=MAX_DATA_POINTS),
    'gpu_cache': deque(maxlen=MAX_DATA_POINTS),
    'cpu_cache': deque(maxlen=MAX_DATA_POINTS),
    'throughput': deque(maxlen=MAX_DATA_POINTS),
    'reg_rps': deque(maxlen=MAX_DATA_POINTS),
    'batch_rps': deque(maxlen=MAX_DATA_POINTS),
    'reg_latency': deque(maxlen=MAX_DATA_POINTS),
    'batch_latency': deque(maxlen=MAX_DATA_POINTS),
}

last_metrics = {
    'prompt': 0, 'gen': 0, 'time': 0,
    'reg_count': 0, 'batch_count': 0
}

def get_metric_with_labels(name, labels, text):
    """Helper to extract Prometheus metrics with specific labels."""
    pattern = name + r'\{'
    for k, v in labels.items():
        pattern += r'.*?' + k + r'="' + v + r'"'
    pattern += r'.*?\} ([\d.e+-]+)'
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return 0.0

def parse_vllm_metrics():
    """Parses vLLM Prometheus metrics including HTTP handler stats."""
    try:
        response = requests.get(VLLM_METRICS_URL, timeout=1)
        if response.status_code != 200:
            return None
        
        text = response.text
        metrics = {}
        
        # Regex patterns for key vLLM engine metrics
        engine_patterns = {
            'running': r'vllm:num_requests_running ([\d.]+)',
            'waiting': r'vllm:num_requests_waiting ([\d.]+)',
            'swapped': r'vllm:num_requests_swapped ([\d.]+)',
            'gpu_cache': r'vllm:gpu_cache_usage_perc ([\d.]+)',
            'cpu_cache': r'vllm:cpu_cache_usage_perc ([\d.]+)',
            'prompt_tokens_total': r'vllm:prompt_tokens_total ([\d.]+)',
            'generation_tokens_total': r'vllm:generation_tokens_total ([\d.]+)',
        }
        
        for key, pattern in engine_patterns.items():
            match = re.search(pattern, text)
            metrics[key] = float(match.group(1)) if match else 0.0
        
        # HTTP Handler Metrics (Regular vs Batch)
        metrics['reg_count'] = get_metric_with_labels("http_requests_total", {"handler": "/v1/chat/completions", "status": "2xx"}, text)
        metrics['batch_count'] = get_metric_with_labels("http_requests_total", {"handler": "/v1/chat/completions/batch", "status": "2xx"}, text)
        
        metrics['reg_lat_sum'] = get_metric_with_labels("http_request_duration_seconds_sum", {"handler": "/v1/chat/completions"}, text)
        metrics['reg_lat_count'] = get_metric_with_labels("http_request_duration_seconds_count", {"handler": "/v1/chat/completions"}, text)
        
        metrics['batch_lat_sum'] = get_metric_with_labels("http_request_duration_seconds_sum", {"handler": "/v1/chat/completions/batch"}, text)
        metrics['batch_lat_count'] = get_metric_with_labels("http_request_duration_seconds_count", {"handler": "/v1/chat/completions/batch"}, text)
                
        return metrics
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None

# --- DASH APP ---
app = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)
server = app.server

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
                padding: 24px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }}
            .glass-card:hover {{
                border-color: {COLORS['accent']};
            }}
            .stat-value {{
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(to right, {COLORS['accent']}, {COLORS['accent_secondary']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .stat-label {{
                color: {COLORS['text_muted']};
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            .header-dash {{
                padding: 2rem 0;
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
                html.H1("vLLM Engine Insights", className="mb-0 fw-bold"),
                html.P([
                    html.Span(id='status-dot', className="status-indicator status-offline"),
                    html.Span(id='status-text', children="Connecting to vLLM...")
                ], className="mb-0 mt-2")
            ], className="col-md-6"),
            html.Div([
                html.P(f"Monitoring: {VLLM_METRICS_URL}", className="text-end text-muted small")
            ], className="col-md-6 d-flex align-items-center justify-content-end")
        ], className="row g-0 container mx-auto px-4")
    ], className="header-dash mb-4"),

    html.Div([
        # Row  summary Stats
        html.Div([
            html.Div([
                html.Div([
                    html.Div("Running Requests", className="stat-label"),
                    html.Div(id="val-running", children="0", className="stat-value"),
                ], className="glass-card h-100")
            ], className="col-md-4"),
            html.Div([
                html.Div([
                    html.Div("Waiting in Queue", className="stat-label"),
                    html.Div(id="val-waiting", children="0", className="stat-value"),
                ], className="glass-card h-100")
            ], className="col-md-4"),
            html.Div([
                html.Div([
                    html.Div("Tokens / Sec", className="stat-label"),
                    html.Div(id="val-throughput", children="0.0", className="stat-value"),
                ], className="glass-card h-100")
            ], className="col-md-4"),
        ], className="row mb-3"),

        html.Div([
            html.Div([
                html.Div([
                    html.Div("GPU Cache Usage", className="stat-label"),
                    html.Div(id="val-gpu-perc", children="0%", className="stat-value"),
                ], className="glass-card h-100")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div("Batch Traffic (Req)", className="stat-label"),
                    html.Div(id="val-batch-count", children="0", className="stat-value"),
                ], className="glass-card h-100")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div("Reg Latency (s)", className="stat-label"),
                    html.Div(id="val-reg-lat", children="0", className="stat-value"),
                ], className="glass-card h-100")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div("Batch Latency (s)", className="stat-label"),
                    html.Div(id="val-batch-lat", children="0", className="stat-value"),
                ], className="glass-card h-100")
            ], className="col-md-3"),
        ], className="row mb-4"),

        # Row 2: Charts
        html.Div([
            # Throughput History
            html.Div([
                html.Div([
                    html.H5("Throughput (Tokens/s)", className="mb-3 fw-bold"),
                    dcc.Graph(id='graph-throughput', config={'displayModeBar': False}),
                ], className="glass-card")
            ], className="col-md-8"),
            
            # Cache Gauges
            html.Div([
                html.Div([
                    html.H5("Memory Utilization", className="mb-3 fw-bold"),
                    dcc.Graph(id='graph-cache', config={'displayModeBar': False}),
                ], className="glass-card h-100")
            ], className="col-md-4"),
        ], className="row mb-4"),

        # Row 3: Detail Charts
        html.Div([
             html.Div([
                html.Div([
                    html.H5("HTTP Request Rate (Req/s)", className="mb-3 fw-bold"),
                    dcc.Graph(id='graph-rps', config={'displayModeBar': False}),
                ], className="glass-card")
            ], className="col-md-6"),
             html.Div([
                html.Div([
                    html.H5("Request Queue Depth", className="mb-3 fw-bold"),
                    dcc.Graph(id='graph-queue', config={'displayModeBar': False}),
                ], className="glass-card")
            ], className="col-md-6"),
        ], className="row")

    ], className="container px-4")
])

@app.callback(
    [Output('status-dot', 'className'),
     Output('status-text', 'children'),
     Output('val-running', 'children'),
     Output('val-waiting', 'children'),
     Output('val-throughput', 'children'),
     Output('val-gpu-perc', 'children'),
     Output('val-batch-count', 'children'),
     Output('val-reg-lat', 'children'),
     Output('val-batch-lat', 'children'),
     Output('graph-throughput', 'figure'),
     Output('graph-cache', 'figure'),
     Output('graph-queue', 'figure'),
     Output('graph-rps', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    global last_metrics
    
    metrics = parse_vllm_metrics()
    now = datetime.now()
    
    if not metrics:
        return ("status-indicator status-offline", "vLLM Offline", "0", "0", "0.0", "0%", "0", "0.0", "0.0", 
                go.Figure(), go.Figure(), go.Figure(), go.Figure())
    
    # Calculate Throughput and Rates
    current_time = time.time()
    dt = current_time - last_metrics['time']
    
    total_tokens = metrics['prompt_tokens_total'] + metrics['generation_tokens_total']
    tput = 0.0
    reg_rps = 0.0
    batch_rps = 0.0
    
    if dt > 0 and last_metrics['time'] > 0:
        tput = (total_tokens - (last_metrics['prompt'] + last_metrics['gen'])) / dt
        reg_rps = (metrics['reg_count'] - last_metrics['reg_count']) / dt
        batch_rps = (metrics['batch_count'] - last_metrics['batch_count']) / dt
        
    last_metrics = {
        'prompt': metrics['prompt_tokens_total'],
        'gen': metrics['generation_tokens_total'],
        'time': current_time,
        'reg_count': metrics['reg_count'],
        'batch_count': metrics['batch_count']
    }

    # Calculate Latencies (from cumulative sums)
    reg_lat = (metrics['reg_lat_sum'] / metrics['reg_lat_count']) if metrics['reg_lat_count'] > 0 else 0
    batch_lat = (metrics['batch_lat_sum'] / metrics['batch_lat_count']) if metrics['batch_lat_count'] > 0 else 0

    # Update history
    history['timestamps'].append(now)
    history['running'].append(metrics['running'])
    history['waiting'].append(metrics['waiting'])
    history['gpu_cache'].append(metrics['gpu_cache'] * 100)
    history['cpu_cache'].append(metrics['cpu_cache'] * 100)
    history['throughput'].append(tput)
    history['reg_rps'].append(reg_rps)
    history['batch_rps'].append(batch_rps)
    history['reg_latency'].append(reg_lat)
    history['batch_latency'].append(batch_lat)

    # Throughput Figure
    fig_tput = go.Figure()
    fig_tput.add_trace(go.Scatter(
        x=list(history['timestamps']), y=list(history['throughput']),
        mode='lines', name='Tokens/s',
        line=dict(color=COLORS['accent'], width=3),
        fill='tozeroy', fillcolor='rgba(56, 189, 248, 0.1)'
    ))
    fig_tput.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=0, r=0, t=10, b=0), height=300, yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))

    # RPS Figure (Regular vs Batch)
    fig_rps = go.Figure()
    fig_rps.add_trace(go.Scatter(
        x=list(history['timestamps']), y=list(history['reg_rps']),
        mode='lines', name='Regular RPS', line=dict(color=COLORS['accent'], width=2)
    ))
    fig_rps.add_trace(go.Scatter(
        x=list(history['timestamps']), y=list(history['batch_rps']),
        mode='lines', name='Batch RPS', line=dict(color=COLORS['accent_secondary'], width=2)
    ))
    fig_rps.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         margin=dict(l=0, r=0, t=10, b=0), height=200, yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))

    # Queue Figure
    fig_queue = go.Figure()
    fig_queue.add_trace(go.Scatter(
        x=list(history['timestamps']), y=list(history['running']),
        mode='lines', name='Running', stackgroup='one', line=dict(color=COLORS['success'], width=0)
    ))
    fig_queue.add_trace(go.Scatter(
        x=list(history['timestamps']), y=list(history['waiting']),
        mode='lines', name='Waiting', stackgroup='one', line=dict(color=COLORS['warning'], width=0)
    ))
    fig_queue.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           margin=dict(l=0, r=0, t=10, b=0), height=200, yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))

    # Cache Gauges
    fig_cache = go.Figure()
    fig_cache.add_trace(go.Bar(
        x=['GPU', 'CPU'], y=[metrics['gpu_cache']*100, metrics['cpu_cache']*100],
        marker_color=[COLORS['gpu_cache'], COLORS['cpu_cache']], width=[0.5, 0.5]
    ))
    fig_cache.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           margin=dict(l=20, r=20, t=10, b=20), height=300, yaxis=dict(range=[0, 100]))

    return (
        "status-indicator status-online", "vLLM Engine Online",
        f"{int(metrics['running'])}", f"{int(metrics['waiting'])}", f"{tput:.1f}",
        f"{metrics['gpu_cache']*100:.1f}%", f"{int(metrics['batch_count'])}",
        f"{reg_lat:.2f}s", f"{batch_lat:.2f}s",
        fig_tput, fig_cache, fig_queue, fig_rps
    )

if __name__ == '__main__':
    print("Starting vLLM Monitoring Dashboard on http://localhost:8050")
    print(f"Polling vLLM Metrics at: {VLLM_METRICS_URL}")
    app.run(debug=True, port=8050)
