import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import argparse
import sys

# ============================================================
# DESIGN SYSTEM (Research / Academic Style)
# ============================================================
PLOT_LAYOUT = dict(
    template="plotly_white",
    font=dict(color="black", family="Inter, sans-serif", size=14),
    margin=dict(l=80, r=40, t=100, b=120), # Increased bottom margin for legend
    xaxis=dict(
        showgrid=True, gridcolor='rgba(0,0,0,0.1)', 
        linecolor='black', linewidth=1.5, mirror=True, 
        title="Concurrency", ticks='outside'
    ),
    yaxis=dict(
        showgrid=True, gridcolor='rgba(0,0,0,0.1)', 
        linecolor='black', linewidth=1.5, mirror=True,
        ticks='outside'
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.12,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.8)", 
        bordercolor='black', borderwidth=1,
        font=dict(size=11),
    ),
)

# High-contrast Research Palette
RESEARCH_COLORS = [
    "#1f77b4", # Blue
    "#d62728", # Red
    "#2ca02c", # Green
    "#ff7f0e", # Orange
    "#9467bd", # Purple
    "#8c564b", # Brown
]

def shorten_model_name(name):
    """Make model names more concise for research plots."""
    n = name.lower()
    if n.startswith("google-"): name = name[len("google-"):]
    elif n.startswith("mistralai-"): name = name[len("mistralai-"):]
    elif n.startswith("qwen-"): name = name[len("qwen-"):]
    
    # Remove common suffixes
    for suffix in ["-it", "-instruct-v0.3", "-instruct", "-instruct-v3"]:
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
    return name

def parse_filename(filepath):
    """
    Parse filename for model and run_name according to lightmem_profiler naming convention:
    lightmem_sweep_{provider}_{model}_{run_name}_{date}_{time}.csv
    """
    base = os.path.basename(filepath).replace(".csv", "")
    parts = base.split("_")
    
    # Defaults
    model = "Unknown"
    run_name = base
    
    if len(parts) >= 7:
        model = parts[3]
        run_name = parts[4]
    
    return model, run_name

def create_comparison_graph(csv_paths, identifiers, output_path="comparison_results.html"):
    # Load all datasets
    runs = []
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"Warning: Skipping missing file: {p}")
            continue
            
        df = pd.read_csv(p, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        
        model, run_name = parse_filename(p)
        batch_type = "Base"
        for identifier in identifiers:
            if identifier.lower() in run_name.lower():
                batch_type = identifier
                break
        
        runs.append({
            "model": model,
            "run_name": run_name,
            "batch_type": batch_type,
            "df": df,
            "path": p
        })

    if not runs:
        print("Error: No data to plot.")
        return

    # Group by model
    models = sorted(list(set(r["model"] for r in runs)))
    num_models = len(models)
    
    # Create subplots grid: 2x2 for Throughput, Latency, Error, LLM Time
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Throughput (writes/s) ↑", 
            "P95 Latency (s) ↓", 
            "Error Rate (%) ↓", 
            "Avg LLM Time (s) ↓"
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    metrics = [
        ("throughput",   1, 1),
        ("p95",          1, 2),
        ("error_rate",   2, 1),
        ("avg_llm_time", 2, 2)
    ]

    # Map models to colors
    color_map = {model: RESEARCH_COLORS[i % len(RESEARCH_COLORS)] for i, model in enumerate(models)}

    for m_idx, model in enumerate(models):
        model_runs = [r for r in runs if r["model"] == model]
        # Sort to ensure consistent line styles (Base then Batch)
        model_runs.sort(key=lambda x: 0 if x["batch_type"] == "Base" else 1)

        display_model = shorten_model_name(model)

        for run in model_runs:
            # If batch_type matches any of our custom identifiers, use dashed line
            dash = None
            if isinstance(identifiers, list):
                if any(i.lower() in run["batch_type"].lower() for i in identifiers):
                    dash = "dash"
            elif isinstance(identifiers, str) and identifiers.lower() in run["batch_type"].lower():
                dash = "dash"
            
            width = 3 if dash else 4
            color = color_map[model]
            
            for m_name, row, col in metrics:
                if m_name not in run["df"].columns:
                    continue
                
                y_data = run["df"][m_name]
                if m_name == "error_rate" and y_data.max() <= 1.0:
                    y_data = y_data * 100

                fig.add_trace(
                    go.Scatter(
                        x=run["df"]["concurrency"], 
                        y=y_data,
                        name=f"{display_model} ({run['batch_type']})",
                        mode="lines+markers",
                        line=dict(color=color, width=width, dash=dash),
                        marker=dict(size=7, symbol="circle" if not dash else "diamond"),
                        legendgroup=f"{model}",
                    ),
                    row=row, col=col
                )

    # Styling subplot titles
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font = dict(size=14, color="black", family="Inter, sans-serif")

    # Update layout
    fig.update_layout(
        **PLOT_LAYOUT,
        height=900,
        width=1300,
        title_text="LightMem Multi-Model Performance Comparison",
        title_x=0.5,
        title_font=dict(size=24, color="black", family="Inter, sans-serif")
    )

    # Save to HTML
    fig.write_html(output_path)
    print(f"Optimized research graph saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple LightMem profiling CSV sweeps in Concise Style.")
    parser.add_argument(
        "csvs",
        nargs="*",
        help="List of CSV files to compare."
    )

    parser.add_argument(
        "--identifiers",
        nargs="*",
        type=str,
        default=["Batch", "adaptive-batching", "with-batch"],
        help="Identifier in the run name to distinguish between runs."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_results.html",
        help="Path to save the output HTML graph."
    )

    args = parser.parse_args()
    
    if not args.csvs:
        print("No CSV files provided.")
        sys.exit(1)
        
    create_comparison_graph(args.csvs, args.identifiers, args.output)
