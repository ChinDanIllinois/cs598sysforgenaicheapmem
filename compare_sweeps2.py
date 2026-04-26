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
    margin=dict(l=80, r=40, t=100, b=120),
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
        font=dict(size=12),
    ),
)

RESEARCH_COLORS = [
    "#1f77b4", # Blue (Single-Tenant)
    "#d62728", # Red (Multi-Tenant)
    "#2ca02c", # Green
    "#ff7f0e", # Orange
]

def parse_metadata(filepath):
    """
    Extract Scenario (Single vs Multi) and Batch Status from filename.
    """
    base = os.path.basename(filepath).lower()
    
    # Batch Status
    is_batch = "with-batch" in base or "adaptive-batching" in base
    batch_label = "Batch" if is_batch else "Baseline"
    
    # Scenario
    is_multi = "multi-tenant" in base or "multitenant" in base
    scenario_label = "Multi-Tenant" if is_multi else "Single-Tenant"
    
    return scenario_label, batch_label

def create_comparison_graph(csv_paths, output_path="comparison_results2.html"):
    runs = []
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"Warning: Skipping missing file: {p}")
            continue
            
        df = pd.read_csv(p, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        
        scenario, batch_type = parse_metadata(p)
        
        runs.append({
            "scenario": scenario,
            "batch_type": batch_type,
            "df": df,
            "path": p
        })

    if not runs:
        print("Error: No data found.")
        return

    scenarios = sorted(list(set(r["scenario"] for r in runs)))
    
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

    # Map scenarios to colors
    color_map = {
        "Single-Tenant": RESEARCH_COLORS[0],
        "Multi-Tenant":  RESEARCH_COLORS[1]
    }

    for scenario in scenarios:
        scenario_runs = [r for r in runs if r["scenario"] == scenario]
        # Sort so Baseline is plotted first
        scenario_runs.sort(key=lambda x: 0 if x["batch_type"] == "Baseline" else 1)

        for run in scenario_runs:
            dash = "dash" if run["batch_type"] == "Batch" else None
            width = 3 if dash else 4
            color = color_map.get(scenario, RESEARCH_COLORS[2])
            
            # Unique group for each of the four lines
            trace_group = f"{scenario}_{run['batch_type']}"
            trace_name = f"{scenario} ({run['batch_type']})"

            for m_idx, (m_name, row, col) in enumerate(metrics):
                if m_name not in run["df"].columns:
                    continue
                
                y_data = run["df"][m_name]
                if m_name == "error_rate" and y_data.max() <= 1.0:
                    y_data = y_data * 100

                fig.add_trace(
                    go.Scatter(
                        x=run["df"]["concurrency"], 
                        y=y_data,
                        name=trace_name,
                        mode="lines+markers",
                        line=dict(color=color, width=width, dash=dash),
                        marker=dict(size=7, symbol="circle" if not dash else "diamond"),
                        legendgroup=trace_group,
                        showlegend=(row == 1 and col == 1) # Show in legend ONLY once
                    ),
                    row=row, col=col
                )

    # Styling
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font = dict(size=14, color="black")

    fig.update_layout(
        **PLOT_LAYOUT,
        height=900,
        width=1300,
        title_text="LightMem Comparison: Single-Tenant vs Multi-Tenant Adaptive Batching",
        title_x=0.5,
        title_font=dict(size=24, color="black")
    )

    fig.write_html(output_path)
    print(f"Comparison graph saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scenario-based LightMem comparison.")
    parser.add_argument("csvs", nargs="*", help="List of CSV files.")
    parser.add_argument("--output", type=str, default="comparison_results2.html")

    args = parser.parse_args()
    if not args.csvs:
        # Default to the files requested if none provided, or just exit
        sys.exit("Please provide CSV files.")
        
    create_comparison_graph(args.csvs, args.output)
