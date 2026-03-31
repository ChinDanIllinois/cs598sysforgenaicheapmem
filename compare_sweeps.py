import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import argparse
import sys

# ============================================================
# DESIGN SYSTEM (Matching lightmem_profiler.py)
# ============================================================
BG         = "#0f1117"
CARD_BG    = "#1a1d27"
BORDER     = "#2a2d3e"
ACCENT     = "#6366f1"  # Gemini color (Primary)
ACCENT2    = "#22d3ee"  # Ollama color (Secondary)
ACCENT3    = "#f59e0b"
ACCENT4    = "#10b981"
ACCENT_ERR = "#ef4444"
TEXT       = "#e2e8f0"
TEXT_DIM   = "#64748b"

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    margin=dict(l=60, r=40, t=80, b=60),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showline=True, title="Concurrency"),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showline=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, x=0.02, y=0.98),
)

def create_comparison_graph(csv1_path, csv2_path, output_path="comparison_results.html"):
    # Load data
    df1 = pd.read_csv(csv1_path, skipinitialspace=True)
    df2 = pd.read_csv(csv2_path, skipinitialspace=True)
    
    # Strip whitespace from column names to prevent KeyErrors from padded CSVs
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Extract labels from filenames
    name1 = os.path.basename(csv1_path).replace("lightmem_sweep_", "").replace(".csv", "").replace("_", " ").title()
    name2 = os.path.basename(csv2_path).replace("lightmem_sweep_", "").replace(".csv", "").replace("_", " ").title()

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Throughput (writes/sec)", 
            "P95 Latency (seconds)",
            "Error Rate (%)",
            "Avg LLM Time (seconds)"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    metrics = [
        ("throughput", "Throughput (writes/sec)", 1, 1),
        ("p95", "P95 Latency (s)", 1, 2),
        ("error_rate", "Error Rate", 2, 1),
        ("avg_llm_time", "Avg LLM Time (s)", 2, 2)
    ]

    for col_name, title, row, col in metrics:
        # Trace for CSV 1
        fig.add_trace(
            go.Scatter(
                x=df1["concurrency"], y=df1[col_name],
                name=f"{name1}",
                mode="lines+markers",
                line=dict(color=ACCENT, width=3),
                marker=dict(size=8),
                legendgroup="group1",
                showlegend=(row == 1 and col == 1)
            ),
            row=row, col=col
        )
        # Trace for CSV 2
        fig.add_trace(
            go.Scatter(
                x=df2["concurrency"], y=df2[col_name],
                name=f"{name2}",
                mode="lines+markers",
                line=dict(color=ACCENT2, width=3),
                marker=dict(size=8),
                legendgroup="group2",
                showlegend=(row == 1 and col == 1)
            ),
            row=row, col=col
        )

    # Update layout
    fig.update_layout(
        **PLOT_LAYOUT,
        height=900,
        width=1200,
        title_text=f"LightMem Comparison: {name1} vs {name2}",
        title_x=0.5,
        title_font=dict(size=24, color=TEXT)
    )

    # Update axes for all subplots
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showline=True, title="Concurrency")
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showline=True)

    # Save to HTML
    fig.write_html(output_path)
    print(f"Comparison graph saved to: {output_path}")
    
    # Also show it if in a notebook/interactive environment
    # fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two LightMem profiling CSV sweeps.")
    parser.add_argument(
        "--csv1",
        type=str,
        default="/Users/chinmaydandekar/Desktop/i/CS 598 - Systems for GenAI/lightmem-playground/profiling_runs/lightmem_sweep_gemini_base.csv",
        help="Path to the first CSV sweep (e.g., Gemini baseline)."
    )
    parser.add_argument(
        "--csv2",
        type=str,
        default="/Users/chinmaydandekar/Desktop/i/CS 598 - Systems for GenAI/lightmem-playground/profiling_runs/lightmem_sweep_ollama_gemma_12b_base.csv",
        help="Path to the second CSV sweep (e.g., Ollama baseline)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_results.html",
        help="Path to save the output HTML graph."
    )

    args = parser.parse_args()
    
    if not os.path.exists(args.csv1):
        print(f"Error: CSV 1 not found: {args.csv1}")
        sys.exit(1)
    if not os.path.exists(args.csv2):
        print(f"Error: CSV 2 not found: {args.csv2}")
        sys.exit(1)
        
    create_comparison_graph(args.csv1, args.csv2, args.output)
