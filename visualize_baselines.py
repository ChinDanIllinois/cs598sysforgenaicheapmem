import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

# Theme Constants from lightmem_multitenant_profiler.py
BG         = "#0f1117"
CARD_BG    = "#1a1d27"
BORDER     = "#2a2d3e"
ACCENT     = "#6366f1"
ACCENT2    = "#22d3ee"
ACCENT3    = "#f59e0b"
ACCENT4    = "#10b981"
ACCENT5    = "#8b5cf6"
ACCENT6    = "#9333ea"
ACCENT7    = "#e5e7eb"
ACCENT8    = "#db2777"
ACCENT_ERR = "#ef4444"
TEXT       = "#e2e8f0"
TEXT_DIM   = "#64748b"

# Color palette for comparison
COLORS = {
    "mem0": ACCENT,
    "baseline": ACCENT2,
    "sleep": ACCENT3,
    "batching": ACCENT4,
    "batch+sleep": ACCENT5,
    "fixed batch 2": ACCENT6,
    "fixed batch 16": ACCENT7,
    "baseline 2": ACCENT8
}

FILE_MAPPING = {
    "mem0": {
        "lat": "final_baselines/mem0_multitenant_run_vllm_c32_b1_rpm0_dur120_20260430_050852.csv",
        "tput": "final_baselines/mem0_multitenant_run_vllm_c32_b1_rpm0_dur120_20260430_050852_throughput.csv"
    },
    "baseline": {
        "lat": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b1_rpm0_dur120_20260430_043354_baseline_lat.csv",
        "tput": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b1_rpm0_dur120_20260430_043354_throughput_baseline.csv"
    },
    "sleep": {
        "lat": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b1_rpm0_dur120_20260430_051221_sleep_lat.csv",
        "tput": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b1_rpm0_dur120_20260430_051221_throughput_sleep.csv"
    },
    "batching": {
        "lat": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b16_rpm0_dur120_adaptive_20260430_045447.csv",
        "tput": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b16_rpm0_dur120_adaptive_20260430_045447_throughput.csv"
    },
    "batch+sleep": {
        "lat": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b16_rpm0_dur120_adaptive_20260430_054518_and_sleep_lat.csv",
        "tput": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b16_rpm0_dur120_adaptive_20260430_054518_throughput_and_sleep.csv"
    },
    "fixed batch 2": {
        "lat": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b2_rpm0_dur120_20260430_064151_fixed_batch_2.csv",
        "tput": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b2_rpm0_dur120_20260430_064151_throughput_fixed_batch_2.csv"
    },
    "fixed batch 16": {
        "lat": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b16_rpm0_dur120_20260430_065404_fixed_batch_16.csv",
        "tput": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b16_rpm0_dur120_20260430_065404_throughput_fixed_batch_16.csv"
    },
    "baseline 2": {
        "lat": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b1_rpm0_dur120_20260430_070359_baseline_2.csv",
        "tput": "final_baselines/multitenant_run_vllm_Qwen2-5-7B-Instruct_c64_b1_rpm0_dur120_20260430_070359_throughput_baseline_2.csv"
    }
}

PLOT_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showline=True, showgrid=False),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showline=True, showgrid=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(size=10)),
    margin=dict(l=50, r=20, t=50, b=50)
)

def load_data():
    data = {}
    for label, paths in FILE_MAPPING.items():
        if os.path.exists(paths['lat']) and os.path.exists(paths['tput']):
            df_lat = pd.read_csv(paths['lat'])
            df_tput = pd.read_csv(paths['tput'])
            data[label] = {'lat': df_lat, 'tput': df_tput}
        else:
            print(f"Warning: Files for {label} not found: {paths['tput']}")
    return data

def create_comparison_charts(data):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Throughput (EPS - Moving Avg)", "Latency (P50 - Moving Avg)", "Backlog Depth (Moving Avg)", "Cumulative Errors"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    for label, dfs in data.items():
        tput_df = dfs['tput'].copy()
        lat_df = dfs['lat'].copy()
        color = COLORS[label]

        # 1. Throughput
        tput_df['tput_ma'] = tput_df['throughput_eps'].rolling(window=5, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(x=tput_df['elapsed_sec'], y=tput_df['tput_ma'], 
                       name=label, line=dict(color=color, width=2), mode='lines',
                       legendgroup=label),
            row=1, col=1
        )

        # 2. Latency
        lat_df['win'] = (lat_df['wall_time'] // 5) * 5
        p50 = lat_df.groupby('win')['latency'].median().reset_index()
        p50['lat_ma'] = p50['latency'].rolling(window=5, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(x=p50['win'], y=p50['lat_ma'], 
                       name=label, line=dict(color=color, width=2), mode='lines',
                       legendgroup=label, showlegend=False),
            row=1, col=2
        )

        # 3. Backlog
        backlog_y = tput_df['total_backlog'] if 'total_backlog' in tput_df.columns else (tput_df['archives'] + tput_df['queries'])
        backlog_ma = backlog_y.rolling(window=5, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(x=tput_df['elapsed_sec'], y=backlog_ma, 
                       name=label, line=dict(color=color, width=2), mode='lines',
                       legendgroup=label, showlegend=False),
            row=2, col=1
        )

        # 4. Errors
        fig.add_trace(
            go.Scatter(x=tput_df['elapsed_sec'], y=tput_df['errors_so_far'], 
                       name=label, line=dict(color=color, width=2), mode='lines',
                       legendgroup=label, showlegend=False),
            row=2, col=2
        )

    fig.update_layout(height=900, title_text="Multi-Tenant Baseline Comparison (Smoothed)", **PLOT_LAYOUT)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_html("baseline_comparison.html")
    print("Saved baseline_comparison.html")

def create_breakdown_charts(data):
    # Stages to analyze
    stages = [
        ("stage_compress_time", "Compression"), 
        ("stage_segment_time", "Segmentation"), 
        ("stage_llm_extract_time", "LLM Extraction"), 
        ("stage_db_insert_time", "Database Insertion")
    ]
    
    # Create 2x2 grid for stage comparisons
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[s[1] for s in stages],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # For each stage, plot a line for each baseline run
    for i, (col, title) in enumerate(stages):
        row = (i // 2) + 1
        col_idx = (i % 2) + 1
        
        for label, dfs in data.items():
            if label == "mem0": continue # mem0 doesn't have these breakdown columns
            
            df = dfs['tput'].copy()
            if col not in df.columns: continue
            
            # Calculate delta stage time and delta completed to get average time per event in each interval
            df_diff = df.diff()
            df_diff = df_diff[df_diff['completed_so_far'] > 0] # Avoid division by zero
            
            avg_stage_time = df_diff[col] / df_diff['completed_so_far']
            
            # Smooth with moving average
            smoothed = avg_stage_time.rolling(window=5, min_periods=1).mean()
            
            fig.add_trace(
                go.Scatter(x=df.loc[smoothed.index, 'elapsed_sec'], y=smoothed, 
                           name=label, line=dict(color=COLORS[label], width=2), mode='lines',
                           legendgroup=label, showlegend=(i == 0)),
                row=row, col=col_idx
            )

    fig.update_layout(height=900, title_text="Latency Breakdown Comparison by Stage (Moving Avg)", **PLOT_LAYOUT)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_html("latency_breakdowns.html")
    print("Saved latency_breakdowns.html")

def main():
    data = load_data()
    if not data:
        print("No data found in final_baselines/")
        return
    
    create_comparison_charts(data)
    create_breakdown_charts(data)
    
    with open("report.html", "w") as f:
        f.write(f"""
        <html>
        <head>
            <title>Performance Baselines Report</title>
            <style>
                body {{ background-color: {BG}; color: {TEXT}; font-family: 'Inter', sans-serif; margin: 0; padding: 20px; }}
                h1 {{ color: {ACCENT2}; text-align: center; }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                iframe {{ border: none; width: 100%; height: 950px; margin-bottom: 40px; border-radius: 12px; background: {CARD_BG}; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Performance Baselines Analysis</h1>
                <iframe src="baseline_comparison.html"></iframe>
                <iframe src="latency_breakdowns.html"></iframe>
            </div>
        </body>
        </html>
        """)
    print("Generated report.html")

if __name__ == "__main__":
    main()
