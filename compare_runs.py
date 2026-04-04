#!/usr/bin/env python3
"""
compare_benchmarks.py
Usage: python compare_benchmarks.py file1.csv file2.csv
Generates a self-contained HTML file comparing benchmark metrics across both CSVs.
"""

import csv
import sys
import os
import json
from pathlib import Path

# ── Color palette ─────────────────────────────────────────────────────────────
BG         = "#0f1117"
CARD_BG    = "#1a1d27"
BORDER     = "#2a2d3e"
ACCENT     = "#6366f1"
ACCENT2    = "#22d3ee"
ACCENT3    = "#f59e0b"
ACCENT4    = "#10b981"
ACCENT_ERR = "#ef4444"
TEXT       = "#e2e8f0"
TEXT_DIM   = "#64748b"


def load_csv(path: str) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    headers = list(rows[0].keys()) if rows else []
    return headers, rows


def rows_to_series(rows: list[dict]) -> dict[str, list]:
    """Convert list-of-dicts to dict-of-lists, casting values to float."""
    series: dict[str, list] = {}
    for row in rows:
        for k, v in row.items():
            series.setdefault(k, [])
            try:
                series[k].append(float(v))
            except (ValueError, TypeError):
                series[k].append(v)
    return series


# ── Chart definitions ──────────────────────────────────────────────────────────
CHARTS = [
    {
        "id": "throughput",
        "title": "Throughput",
        "subtitle": "requests / sec",
        "metrics": ["throughput"],
        "type": "line",
        "higher_is_better": True,
    },
    {
        "id": "latency",
        "title": "Latency",
        "subtitle": "milliseconds",
        "metrics": ["avg_latency", "p50", "p95", "p99"],
        "type": "line",
        "higher_is_better": False,
    },
    {
        "id": "error_rate",
        "title": "Error Rate",
        "subtitle": "fraction (0–1)",
        "metrics": ["error_rate"],
        "type": "bar",
        "higher_is_better": False,
    },
    {
        "id": "llm_embedding_rate",
        "title": "LLM & Embedding Rate",
        "subtitle": "ops / sec",
        "metrics": ["llm_rate", "embedding_rate"],
        "type": "line",
        "higher_is_better": True,
    },
    {
        "id": "avg_times",
        "title": "Avg LLM & Embedding Time",
        "subtitle": "seconds",
        "metrics": ["avg_llm_time", "avg_embedding_time"],
        "type": "line",
        "higher_is_better": False,
    },
    {
        "id": "stages",
        "title": "Pipeline Stage Latency",
        "subtitle": "seconds",
        "metrics": ["stage_compress", "stage_segment", "stage_llm_extract", "stage_db_insert"],
        "type": "line",
        "higher_is_better": False,
    },
]

METRIC_COLORS = [ACCENT, ACCENT2, ACCENT3, ACCENT4]


def build_chart_data(charts, name1, series1, name2, series2):
    """Return a JSON-serialisable structure consumed by Chart.js."""
    result = []
    for ch in charts:
        x_labels = [str(int(v)) for v in series1.get("concurrency", [])]

        datasets = []
        for i, metric in enumerate(ch["metrics"]):
            base_color = METRIC_COLORS[i % len(METRIC_COLORS)]
            # File 1 – solid line
            datasets.append({
                "label": f"{name1} – {metric}",
                "data": series1.get(metric, []),
                "borderColor": base_color,
                "backgroundColor": base_color + "33",
                "borderWidth": 2,
                "pointRadius": 4,
                "pointHoverRadius": 6,
                "fill": ch["type"] == "bar",
                "tension": 0.35,
                "file": 1,
            })
            # File 2 – dashed line
            datasets.append({
                "label": f"{name2} – {metric}",
                "data": series2.get(metric, []),
                "borderColor": base_color,
                "backgroundColor": base_color + "22",
                "borderWidth": 2,
                "borderDash": [6, 3],
                "pointRadius": 4,
                "pointStyle": "rectRot",
                "pointHoverRadius": 6,
                "fill": False,
                "tension": 0.35,
                "file": 2,
            })

        result.append({
            "id": ch["id"],
            "title": ch["title"],
            "subtitle": ch["subtitle"],
            "type": "bar" if ch["type"] == "bar" else "line",
            "labels": x_labels,
            "datasets": datasets,
            "higherIsBetter": ch["higher_is_better"],
        })
    return result


def build_summary_table(name1, series1, name2, series2):
    """Build per-concurrency summary rows for a quick comparison table."""
    rows = []
    conc_list = series1.get("concurrency", [])
    for i, c in enumerate(conc_list):
        def g(s, k): return s.get(k, [None] * (i + 1))[i]
        rows.append({
            "concurrency": int(c),
            "tput1": g(series1, "throughput"),
            "tput2": g(series2, "throughput"),
            "lat1": g(series1, "avg_latency"),
            "lat2": g(series2, "avg_latency"),
            "err1": g(series1, "error_rate"),
            "err2": g(series2, "error_rate"),
        })
    return rows


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Benchmark Comparison · {name1} vs {name2}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

  :root {{
    --bg:         {BG};
    --card:       {CARD_BG};
    --border:     {BORDER};
    --accent:     {ACCENT};
    --accent2:    {ACCENT2};
    --accent3:    {ACCENT3};
    --accent4:    {ACCENT4};
    --err:        {ACCENT_ERR};
    --text:       {TEXT};
    --dim:        {TEXT_DIM};
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    padding: 2rem 1.5rem 4rem;
  }}

  /* ── scanline texture ───────────────────── */
  body::before {{
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(255,255,255,.015) 2px,
      rgba(255,255,255,.015) 4px
    );
    pointer-events: none;
    z-index: 0;
  }}

  .wrap {{ position: relative; z-index: 1; max-width: 1280px; margin: 0 auto; }}

  /* ── header ─────────────────────────────── */
  header {{
    display: flex; flex-direction: column; gap: .4rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem; margin-bottom: 2.5rem;
  }}
  .header-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem; letter-spacing: .15em;
    color: var(--accent2); text-transform: uppercase;
  }}
  h1 {{
    font-size: clamp(1.4rem, 3vw, 2.4rem);
    font-weight: 800; line-height: 1.1;
  }}
  .vs-pill {{
    display: inline-flex; align-items: center; gap: .5rem;
    font-size: .85rem; margin-top: .3rem;
  }}
  .pill {{
    padding: .2rem .7rem; border-radius: 3px;
    font-family: 'JetBrains Mono', monospace; font-size: .75rem;
  }}
  .pill-a {{ background: {ACCENT}33; border: 1px solid {ACCENT}; color: {ACCENT}; }}
  .pill-b {{ background: {ACCENT2}22; border: 1px dashed {ACCENT2}; color: {ACCENT2}; }}
  .sep {{ color: var(--dim); font-size: .7rem; }}

  /* ── legend ─────────────────────────────── */
  .legend {{
    display: flex; flex-wrap: wrap; gap: .6rem 1.4rem;
    margin-bottom: 2rem; align-items: center;
    font-size: .78rem; color: var(--dim);
  }}
  .legend-item {{ display: flex; align-items: center; gap: .4rem; }}
  .legend-line {{
    width: 24px; height: 2px; border-radius: 1px;
  }}
  .legend-dashed {{
    width: 24px; height: 0;
    border-top: 2px dashed currentColor;
  }}

  /* ── grid ───────────────────────────────── */
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
    gap: 1.5rem;
  }}

  /* ── card ───────────────────────────────── */
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem 1.5rem 1.5rem;
    transition: border-color .2s;
  }}
  .card:hover {{ border-color: var(--accent); }}
  .card-header {{ margin-bottom: 1rem; }}
  .card-title {{
    font-size: 1rem; font-weight: 600; color: var(--text);
  }}
  .card-subtitle {{
    font-size: .72rem; color: var(--dim);
    font-family: 'JetBrains Mono', monospace;
    margin-top: .15rem;
  }}
  .chart-wrap {{ position: relative; height: 220px; }}

  /* ── summary table ──────────────────────── */
  .section-title {{
    font-size: 1.1rem; font-weight: 700;
    margin: 3rem 0 1rem;
    display: flex; align-items: center; gap: .6rem;
  }}
  .section-title::before {{
    content: ''; display: block;
    width: 4px; height: 1em; border-radius: 2px;
    background: var(--accent);
  }}

  table {{
    width: 100%; border-collapse: collapse;
    font-size: .8rem;
  }}
  th, td {{
    padding: .55rem .75rem; text-align: right;
    border-bottom: 1px solid var(--border);
  }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{
    font-family: 'JetBrains Mono', monospace;
    font-size: .68rem; letter-spacing: .06em;
    color: var(--dim); text-transform: uppercase;
    background: var(--card);
    position: sticky; top: 0;
  }}
  tr:hover td {{ background: rgba(255,255,255,.025); }}
  .better {{ color: var(--accent4); font-weight: 600; }}
  .worse  {{ color: var(--err);    font-weight: 600; }}
  .same   {{ color: var(--dim); }}
  .mono   {{ font-family: 'JetBrains Mono', monospace; }}

  /* ── footer ─────────────────────────────── */
  footer {{
    margin-top: 3rem; padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    font-size: .7rem; color: var(--dim);
    font-family: 'JetBrains Mono', monospace;
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: .4rem;
  }}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <span class="header-eyebrow">benchmark · analysis</span>
    <h1>Performance Comparison</h1>
    <div class="vs-pill">
      <span class="pill pill-a">{name1}</span>
      <span class="sep">vs</span>
      <span class="pill pill-b">{name2}</span>
    </div>
  </header>

  <div class="legend">
    <span style="color:var(--dim); font-size:.75rem; font-family:'JetBrains Mono',monospace;">LEGEND:</span>
    <div class="legend-item">
      <div class="legend-line" style="background:var(--accent)"></div>
      <span style="color:var(--accent)">{name1}</span>
      <span>— solid line</span>
    </div>
    <div class="legend-item">
      <div class="legend-dashed" style="color:var(--accent2); width:24px"></div>
      <span style="color:var(--accent2)">{name2}</span>
      <span>— dashed line</span>
    </div>
  </div>

  <div class="grid" id="charts"></div>

  <h2 class="section-title">Quick Comparison Table</h2>
  <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>Concurrency</th>
          <th>Throughput ({name1})</th>
          <th>Throughput ({name2})</th>
          <th>Avg Latency ({name1})</th>
          <th>Avg Latency ({name2})</th>
          <th>Error Rate ({name1})</th>
          <th>Error Rate ({name2})</th>
        </tr>
      </thead>
      <tbody id="summary-tbody"></tbody>
    </table>
  </div>

  <footer>
    <span>generated by compare_benchmarks.py</span>
    <span>{name1} · {name2}</span>
  </footer>

</div>

<script>
const CHARTS_DATA = {charts_json};
const SUMMARY     = {summary_json};
const NAME1 = "{name1}";
const NAME2 = "{name2}";

const COLORS = {{
  bg:     "{BG}",
  card:   "{CARD_BG}",
  border: "{BORDER}",
  accent: "{ACCENT}",
  a2:     "{ACCENT2}",
  a3:     "{ACCENT3}",
  a4:     "{ACCENT4}",
  err:    "{ACCENT_ERR}",
  text:   "{TEXT}",
  dim:    "{TEXT_DIM}",
}};

const defaultChartOptions = (subtitle) => ({{
  responsive: true,
  maintainAspectRatio: false,
  interaction: {{ mode: 'index', intersect: false }},
  plugins: {{
    legend: {{
      display: true,
      position: 'bottom',
      labels: {{
        color: COLORS.dim,
        font: {{ family: "'JetBrains Mono', monospace", size: 10 }},
        boxWidth: 20, boxHeight: 2, padding: 10,
        usePointStyle: false,
      }}
    }},
    tooltip: {{
      backgroundColor: COLORS.card,
      borderColor: COLORS.border,
      borderWidth: 1,
      titleColor: COLORS.text,
      bodyColor: COLORS.dim,
      titleFont: {{ family: "'Syne', sans-serif", weight: '600' }},
      bodyFont: {{ family: "'JetBrains Mono', monospace", size: 11 }},
      padding: 10,
    }}
  }},
  scales: {{
    x: {{
      title: {{ display: true, text: 'concurrency', color: COLORS.dim, font: {{ size: 10 }} }},
      grid: {{ color: COLORS.border + '60' }},
      ticks: {{ color: COLORS.dim, font: {{ family: "'JetBrains Mono', monospace", size: 10 }} }},
    }},
    y: {{
      title: {{ display: true, text: subtitle, color: COLORS.dim, font: {{ size: 10 }} }},
      grid: {{ color: COLORS.border + '60' }},
      ticks: {{ color: COLORS.dim, font: {{ family: "'JetBrains Mono', monospace", size: 10 }} }},
    }}
  }}
}});

// ── render charts ───────────────────────────────────────────────────────────
const container = document.getElementById('charts');

CHARTS_DATA.forEach(ch => {{
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `
    <div class="card-header">
      <div class="card-title">${{ch.title}}</div>
      <div class="card-subtitle">${{ch.subtitle}}</div>
    </div>
    <div class="chart-wrap"><canvas id="c-${{ch.id}}"></canvas></div>
  `;
  container.appendChild(card);

  const ctx = document.getElementById(`c-${{ch.id}}`).getContext('2d');

  // patch borderDash into each dataset
  const datasets = ch.datasets.map(ds => {{
    const d = {{ ...ds }};
    if (ds.borderDash) d.borderDash = ds.borderDash;
    return d;
  }});

  new Chart(ctx, {{
    type: ch.type,
    data: {{ labels: ch.labels, datasets }},
    options: defaultChartOptions(ch.subtitle),
  }});
}});

// ── summary table ────────────────────────────────────────────────────────────
const tbody = document.getElementById('summary-tbody');

const fmt = (v, digits=4) => v == null ? '—' : Number(v).toFixed(digits);

SUMMARY.forEach(row => {{
  const tputBetter = row.tput2 > row.tput1;   // higher is better
  const latBetter  = row.lat2  < row.lat1;    // lower is better
  const errBetter  = row.err2  < row.err1;    // lower is better

  const cls1 = (betterWhen2) => betterWhen2 ? 'worse'  : 'better';
  const cls2 = (betterWhen2) => betterWhen2 ? 'better' : 'worse';

  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td class="mono">${{row.concurrency}}</td>
    <td class="mono ${{cls1(tputBetter)}}">${{fmt(row.tput1)}}</td>
    <td class="mono ${{cls2(tputBetter)}}">${{fmt(row.tput2)}}</td>
    <td class="mono ${{cls1(latBetter)}}" >${{fmt(row.lat1)}}</td>
    <td class="mono ${{cls2(latBetter)}}" >${{fmt(row.lat2)}}</td>
    <td class="mono ${{cls1(errBetter)}}" >${{fmt(row.err1)}}</td>
    <td class="mono ${{cls2(errBetter)}}" >${{fmt(row.err2)}}</td>
  `;
  tbody.appendChild(tr);
}});
</script>
</body>
</html>
"""


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_benchmarks.py <file1.csv> <file2.csv>")
        sys.exit(1)

    path1, path2 = sys.argv[1], sys.argv[2]
    name1 = Path(path1).stem
    name2 = Path(path2).stem

    _, rows1 = load_csv(path1)
    _, rows2 = load_csv(path2)

    series1 = rows_to_series(rows1)
    series2 = rows_to_series(rows2)

    charts_data = build_chart_data(CHARTS, name1, series1, name2, series2)
    summary     = build_summary_table(name1, series1, name2, series2)

    html = HTML_TEMPLATE.format(
        name1=name1,
        name2=name2,
        charts_json=json.dumps(charts_data, indent=2),
        summary_json=json.dumps(summary, indent=2),
        BG=BG, CARD_BG=CARD_BG, BORDER=BORDER,
        ACCENT=ACCENT, ACCENT2=ACCENT2, ACCENT3=ACCENT3, ACCENT4=ACCENT4,
        ACCENT_ERR=ACCENT_ERR, TEXT=TEXT, TEXT_DIM=TEXT_DIM,
    )

    out_name = f"{name1}_vs_{name2}.html"
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✓ Report written to: {out_name}")


if __name__ == "__main__":
    main()