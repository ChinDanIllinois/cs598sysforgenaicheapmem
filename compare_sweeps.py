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

RUN_STYLES = [
    {"dash": None,        "marker": "circle",       "width": 4},
    {"dash": "dash",     "marker": "diamond",      "width": 3},
    {"dash": "dot",      "marker": "square",       "width": 3},
    {"dash": "dashdot",  "marker": "triangle-up",  "width": 3},
    {"dash": "longdash", "marker": "x",            "width": 3},
    {"dash": "longdashdot","marker": "star",      "width": 3},
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


def prettify_run_name(run_name):
    """Format run names for display without changing their identity."""
    if not run_name:
        return "run"
    return run_name.replace("_", " ").strip()


def normalize_key(text):
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def parse_filename(filepath):
    """
    Parse filename for model and run_name according to lightmem_profiler naming convention.

    Expected shape:
        lightmem_sweep_{provider}_{model}_{run_name}_{date}_{time}.csv

    The parser is tolerant of underscore-heavy model names by taking the final
    token before the timestamp as the run name and joining the remaining middle
    tokens as the model slug.
    """
    base = os.path.basename(filepath).replace(".csv", "")
    stem = base.removeprefix("lightmem_sweep_")
    parts = stem.split("_")

    meta = {
        "provider": "Unknown",
        "model": "Unknown",
        "run_name": base,
        "label": base,
    }

    if len(parts) < 4:
        return meta

    meta["provider"] = parts[0]

    # Strip trailing date/time if present.
    core = parts[1:]
    if len(core) >= 3 and len(core[-2]) == 8 and len(core[-1]) == 6 and core[-2].isdigit() and core[-1].isdigit():
        core = core[:-2]

    if not core:
        return meta

    if len(core) == 1:
        model, run_name = core[0], "run"
    else:
        model, run_name = "_".join(core[:-1]), core[-1]

    meta["model"] = model
    meta["run_name"] = run_name
    meta["label"] = f"{shorten_model_name(model)} ({prettify_run_name(run_name)})"
    return meta

def create_comparison_graph(csv_paths, output_path="comparison_results.html"):
    # Load all datasets
    runs = []
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"Warning: Skipping missing file: {p}")
            continue
            
        df = pd.read_csv(p, sep=",", skipinitialspace=True)  # type: ignore[arg-type]
        df.columns = df.columns.str.strip()
        
        meta = parse_filename(p)
        run_key = normalize_key(meta["run_name"])
        print(f"model: {meta['model']}, run_name: {meta['run_name']}")

        runs.append({
            "provider": meta["provider"],
            "model": meta["model"],
            "run_name": meta["run_name"],
            "label": meta["label"],
            "run_key": run_key,
            "df": df,
            "path": p
        })

    if not runs:
        print("Error: No data to plot.")
        return

    # Group by model
    models = sorted(list(set(r["model"] for r in runs)))
    run_names = sorted(list(set(r["run_name"] for r in runs)))
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
    # Map run names to appearance so the same experiment type looks consistent across models.
    style_map = {run_name: RUN_STYLES[i % len(RUN_STYLES)] for i, run_name in enumerate(run_names)}

    for m_idx, model in enumerate(models):
        model_runs = [r for r in runs if r["model"] == model]
        # Sort to ensure consistent line styles (Base then Batch)
        model_runs.sort(key=lambda x: x["run_name"])

        display_model = shorten_model_name(model)

        for run in model_runs:
            style = style_map[run["run_name"]]
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
                        name=f"{display_model} ({prettify_run_name(run['run_name'])})",
                        mode="lines+markers",
                        line=dict(color=color, width=style["width"], dash=style["dash"]),
                        marker=dict(size=7, symbol=style["marker"]),
                        legendgroup=f"{model}",
                        showlegend=(m_name == metrics[0][0]),
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
        "--output",
        type=str,
        default="comparison_results.html",
        help="Path to save the output HTML graph."
    )

    args = parser.parse_args()
    
    if not args.csvs:
        print("No CSV files provided.")
        sys.exit(1)
        
    create_comparison_graph(args.csvs, args.output)
