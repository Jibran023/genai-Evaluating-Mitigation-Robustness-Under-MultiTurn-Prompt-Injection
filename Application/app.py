"""
Research Dashboard — Evaluating Mitigation Robustness Under Multi-Turn Prompt Injection
Authors: Jibran Shaikh, Syeda Wania Hussain
"""

import json, os, math
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_APP_DIR      = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR  = os.path.dirname(_APP_DIR)
_RESULTS_DIR  = os.path.join(_PROJECT_DIR, "results")
_DATASETS_DIR = os.path.join(_PROJECT_DIR, "Datasets")
_DATASET_FILE = os.path.join(_DATASETS_DIR, "test2_final_hardened_v2_cleaned.json")

# ── Constants ─────────────────────────────────────────────────────────────────
MITIGATIONS = ["none", "m1", "m2", "m3"]
MITIGATION_LABELS = {
    "none": "Baseline (None)",
    "m1":   "M1 — Prompt Hardening",
    "m2":   "M2 — I/O Gate",
    "m3":   "M3 — State Monitor",
}
TOPIC_LABELS = {
    "bypassing_ai_rules":     "Bypassing AI Rules",
    "cybersecurity":          "Cybersecurity",
    "dangerous_instructions": "Dangerous Instructions",
    "financial_manipulation": "Financial Manipulation",
    "identity_theft":         "Identity Theft",
    "misinformation":         "Misinformation",
}
PALETTE = {"none": "#94a3b8", "m1": "#6366f1", "m2": "#0891b2", "m3": "#f59e0b"}
FILL_RGBA = {
    "none": "rgba(148,163,184,0.12)",
    "m1":   "rgba(99,102,241,0.12)",
    "m2":   "rgba(8,145,178,0.12)",
    "m3":   "rgba(245,158,11,0.12)",
}
TOPIC_COLORS = ["#6366f1", "#0891b2", "#059669", "#d97706", "#e11d48", "#7c3aed"]
AVAILABLE_MODELS = {"openai-gpt-oss-120b": "OpenAI GPT-OSS 120B"}

LENGTH_GROUPS   = ["short", "medium", "long"]
LENGTH_LABELS   = {"short": "Short (≤8 turns)", "medium": "Medium (9–14)", "long": "Long (>14 turns)"}
HEATMAP_METRICS = ["ASR", "DL", "ORR", "ERR", "Trust"]

# ── plotly layout defaults ────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#374151", family="Inter, sans-serif", size=12),
    margin=dict(t=55, b=30, l=40, r=20),
    height=360,
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#e2e8f0", borderwidth=1,
        font=dict(size=11),
    ),
)

def _styled_layout(**kwargs):
    d = dict(PLOT_LAYOUT)
    d.update(kwargs)
    return d


# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics(model_slug: str) -> dict:
    data = {}
    for mit in MITIGATIONS:
        path = os.path.join(_RESULTS_DIR, mit, model_slug, "all_samples", "metrics_summary.json")
        if os.path.exists(path):
            with open(path) as f:
                data[mit] = json.load(f)
    return data

@st.cache_data
def load_results(model_slug: str) -> dict:
    """Load per-conversation results.json for each mitigation."""
    data = {}
    for mit in MITIGATIONS:
        path = os.path.join(_RESULTS_DIR, mit, model_slug, "all_samples", "results.json")
        if os.path.exists(path):
            with open(path) as f:
                data[mit] = json.load(f)
    return data

@st.cache_data
def load_comparison_summary(model_slug: str) -> dict:
    path = os.path.join(_RESULTS_DIR, "comparison", model_slug, "all_samples", "comparison_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_dataset_sample() -> dict:
    if not os.path.exists(_DATASET_FILE):
        return {}
    with open(_DATASET_FILE) as f:
        ds = json.load(f)
    return ds[0] if ds else {}

@st.cache_data
def dataset_stats() -> dict:
    if not os.path.exists(_DATASET_FILE):
        return {}
    with open(_DATASET_FILE) as f:
        ds = json.load(f)
    total   = len(ds)
    attacks = sum(1 for d in ds if d.get("attack_type") != "none")
    benign  = total - attacks
    topics  = {}
    lengths = {"Short (≤8 turns)": 0, "Medium (9–14)": 0, "Long (>14 turns)": 0}
    for d in ds:
        t = d.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1
        n = len(d.get("turns", []))
        if n <= 8:    lengths["Short (≤8 turns)"] += 1
        elif n <= 14: lengths["Medium (9–14)"] += 1
        else:         lengths["Long (>14 turns)"] += 1
    return {"total": total, "attacks": attacks, "benign": benign,
            "topics": topics, "lengths": lengths}


# ── Chart builders ────────────────────────────────────────────────────────────

def chart_response_curves(results: dict, available_mits: list) -> go.Figure:
    """Cumulative % attacks caught vs. turn number (one line per mitigation)."""
    fig = go.Figure()
    # Find max injection turn across all mits
    max_turn = 0
    for mit in available_mits:
        for r in results.get(mit, []):
            if r.get("is_attack") and r.get("injection_turn"):
                max_turn = max(max_turn, int(r["injection_turn"]))
            if r.get("caught_at_turn"):
                max_turn = max(max_turn, int(r["caught_at_turn"]))
    max_turn = max(max_turn, 17)

    for mit in available_mits:
        recs = [r for r in results.get(mit, []) if r.get("is_attack")]
        if not recs:
            continue
        total = len(recs)
        # Build cumulative caught per turn
        by_turn = {}
        for r in recs:
            t = r.get("caught_at_turn")
            if t is not None:
                by_turn[int(t)] = by_turn.get(int(t), 0) + 1
        turns = list(range(0, max_turn + 1))
        cumulative = []
        running = 0
        for t in turns:
            running += by_turn.get(t, 0)
            cumulative.append(round(running / total * 100, 1))

        fig.add_trace(go.Scatter(
            x=turns, y=cumulative,
            mode="lines+markers",
            name=MITIGATION_LABELS[mit],
            line=dict(color=PALETTE[mit], width=2.5),
            marker=dict(size=5, color=PALETTE[mit]),
            hovertemplate=f"<b>{MITIGATION_LABELS[mit]}</b><br>Turn: %{{x}}<br>Caught: %{{y}}%<extra></extra>",
        ))

    fig.update_layout(
        **_styled_layout(height=380),
        title=dict(text="Unified Mitigation Response Comparison<br><sup>Cumulative % of attacks caught by turn number — higher & earlier is better</sup>",
                   font=dict(size=13, color="#1e293b")),
        xaxis=dict(title="Turn Number", showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        yaxis=dict(title="Attacks Caught (%)", range=[0, 105], showgrid=True, gridcolor="#f1f5f9"),
    )
    return fig


def chart_run_history(results: dict, available_mits: list) -> go.Figure:
    """Running average ASR across all 160 conversations."""
    fig = go.Figure()
    for mit in available_mits:
        recs = results.get(mit, [])
        attack_recs = [r for r in recs if r.get("is_attack")]
        if not attack_recs:
            continue
        running_x, running_y = [], []
        total_seen, total_failed = 0, 0
        for r in attack_recs:
            total_seen += 1
            if r.get("attack_succeeded"):
                total_failed += 1
            running_x.append(total_seen)
            running_y.append(round(total_failed / total_seen * 100, 1))

        fig.add_trace(go.Scatter(
            x=running_x, y=running_y,
            mode="lines",
            name=MITIGATION_LABELS[mit],
            line=dict(color=PALETTE[mit], width=2),
            fill="tozeroy", fillcolor=FILL_RGBA[mit],
            hovertemplate=f"<b>{MITIGATION_LABELS[mit]}</b><br>Sample: %{{x}}<br>Running ASR: %{{y}}%<extra></extra>",
        ))

    fig.update_layout(
        **_styled_layout(height=380),
        title=dict(text="Unified Run History Comparison<br><sup>Running average ASR as the evaluation progresses — lower convergence = better</sup>",
                   font=dict(size=13, color="#1e293b")),
        xaxis=dict(title="Total Progression (Attack Samples)", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Running Average ASR (%)", range=[0, 105], showgrid=True, gridcolor="#f1f5f9"),
    )
    return fig


def chart_efficiency(comp_summary: dict, available_mits: list) -> go.Figure:
    """Dual-axis: ASR (red bar) + AI Detection Latency (blue bar) per mitigation."""
    mits   = [m for m in available_mits if m in comp_summary.get("mitigations", {})]
    labels = [MITIGATION_LABELS[m] for m in mits]
    asrs   = [comp_summary["mitigations"][m]["attack_success_rate_pct"] for m in mits]
    lats   = [comp_summary["mitigations"][m]["mean_ai_latency_turns"] for m in mits]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="ASR % (lower=better)", x=labels, y=asrs,
        marker_color=["#ef4444" if a > 30 else "#f97316" if a > 15 else "#22c55e" for a in asrs],
        text=[f"{v:.1f}%" for v in asrs], textposition="outside",
        textfont=dict(size=11, color="#374151"),
        yaxis="y1", offsetgroup=1,
    ))
    fig.add_trace(go.Bar(
        name="AI Latency (turns, lower=better)", x=labels, y=lats,
        marker_color="#93c5fd",
        text=[f"{v:.2f}" for v in lats], textposition="outside",
        textfont=dict(size=11, color="#374151"),
        yaxis="y2", offsetgroup=2,
    ))
    fig.update_layout(
        **_styled_layout(height=380),
        barmode="group",
        title=dict(text="Efficiency Analysis: Security vs. Performance<br><sup>Lower ASR (red) and lower latency (blue) = better mitigation</sup>",
                   font=dict(size=13, color="#1e293b")),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="ASR (%)", range=[0, max(asrs) * 1.3 + 1], showgrid=True, gridcolor="#f1f5f9"),
        yaxis2=dict(title="Latency (Turns)", overlaying="y", side="right",
                    range=[0, max(lats) * 4 + 0.1], showgrid=False),
    )
    return fig


def chart_asr_by_length(metrics: dict, available_mits: list) -> go.Figure:
    """Grouped bar: ASR by conversation length for each mitigation."""
    fig = go.Figure()
    bar_width = 0.18
    n_mits = len(available_mits)
    x_base = np.arange(len(LENGTH_GROUPS))

    for i, mit in enumerate(available_mits):
        m = metrics.get(mit, {})
        asr_list = m.get("asr_by_length_group", [])
        asr_map  = {entry["length_group"]: entry["asr_pct"] for entry in asr_list}
        y_vals   = [asr_map.get(lg, 0) for lg in LENGTH_GROUPS]
        offset   = (i - n_mits / 2 + 0.5) * (bar_width + 0.02)
        fig.add_trace(go.Bar(
            name=MITIGATION_LABELS[mit],
            x=[MITIGATION_LABELS[mit] + "<br>" + LENGTH_LABELS[lg] for lg in LENGTH_GROUPS],
            y=y_vals,
            marker_color=PALETTE[mit],
            marker_line_width=0,
            text=[f"{v:.0f}%" for v in y_vals],
            textposition="outside",
            textfont=dict(size=10, color="#374151"),
            width=bar_width,
        ))

    fig.update_layout(
        **_styled_layout(height=380),
        barmode="group",
        title=dict(text="ASR by Mitigation & Conversation Length<br><sup>Lower is better — reveals if protection weakens in longer chats</sup>",
                   font=dict(size=13, color="#1e293b")),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(title="ASR (%)", range=[0, 110], showgrid=True, gridcolor="#f1f5f9"),
    )
    return fig


def chart_reliability(comp_summary: dict, available_mits: list) -> go.Figure:
    """Horizontal bar: Safety Score, Availability Score, Overall Reliability."""
    mits_data = comp_summary.get("mitigations", {})
    # ordered from top to bottom inside the figure
    mits = [m for m in reversed(available_mits) if m in mits_data]
    labels = [MITIGATION_LABELS[m] for m in mits]
    safety_scores  = [round(100 - mits_data[m]["attack_success_rate_pct"], 1) for m in mits]
    avail_scores   = [round(100 - mits_data[m]["over_refusal_rate_pct"], 1)   for m in mits]
    # geometric mean of the two
    combined = [round(math.sqrt(s * a), 1) if s > 0 and a > 0 else 0.0
                for s, a in zip(safety_scores, avail_scores)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Safety Score (100−ASR)", y=labels, x=safety_scores,
        orientation="h", marker_color="#22c55e",
        text=[f"{v}" for v in safety_scores], textposition="auto",
        textfont=dict(size=11, color="#1e293b"),
    ))
    fig.add_trace(go.Bar(
        name="Availability (100−ORR)", y=labels, x=avail_scores,
        orientation="h", marker_color="#0891b2",
        text=[f"{v}" for v in avail_scores], textposition="auto",
        textfont=dict(size=11, color="#1e293b"),
    ))
    fig.add_trace(go.Bar(
        name="Overall Reliability (√Safety×Avail)", y=labels, x=combined,
        orientation="h",
        marker=dict(color=combined, colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#6366f1"]],
                    showscale=False),
        text=[f"<b>{v}</b>" for v in combined], textposition="auto",
        textfont=dict(size=11, color="#1e293b"),
    ))
    fig.update_layout(
        **_styled_layout(height=380, margin=dict(t=55, b=30, l=140, r=40)),
        barmode="group",
        title=dict(text="Overall Mitigation Reliability<br><sup>Geometric mean balances Safety vs. Usability — higher is better</sup>",
                   font=dict(size=13, color="#1e293b")),
        xaxis=dict(title="Score (0–100)", range=[0, 115], showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=False),
    )
    return fig


def chart_heatmap(comp_summary: dict, available_mits: list) -> go.Figure:
    """Metric heatmap: rows = mitigations, cols = key metrics, colored by normalized performance."""
    mits_data = comp_summary.get("mitigations", {})
    mits = [m for m in available_mits if m in mits_data]

    # Build raw matrix
    rows, annot = [], []
    for mit in mits:
        d = mits_data[mit]
        safety   = round(100 - d["attack_success_rate_pct"], 1)
        avail    = round(100 - d["over_refusal_rate_pct"], 1)
        combined = round(math.sqrt(max(safety, 0) * max(avail, 0)), 1) if safety > 0 and avail > 0 else 0.0
        row_vals = [
            d["attack_success_rate_pct"],     # ASR — lower better
            d["mean_ai_latency_turns"],        # DL  — lower better
            d["over_refusal_rate_pct"],        # ORR — lower better
            d["err_overall"],                  # ERR — higher better
            combined,                          # Trust — higher better
        ]
        rows.append(row_vals)
        annot.append([
            f"{d['attack_success_rate_pct']:.1f}%",
            f"{d['mean_ai_latency_turns']:.2f}",
            f"{d['over_refusal_rate_pct']:.1f}%",
            f"{d['err_overall']:.1f}%",
            f"{combined:.1f}",
        ])

    # Normalize each column (0–1, where 1 = best performance)
    n_mits, n_metrics = len(rows), len(HEATMAP_METRICS)
    norm = [[0.0] * n_metrics for _ in range(n_mits)]
    lower_better = [True, True, True, False, False]  # ASR, DL, ORR, ERR, Trust
    for c in range(n_metrics):
        col = [rows[r][c] for r in range(n_mits)]
        mn, mx = min(col), max(col)
        for r in range(n_mits):
            if mx == mn:
                norm[r][c] = 0.5
            else:
                raw = (rows[r][c] - mn) / (mx - mn)
                norm[r][c] = (1 - raw) if lower_better[c] else raw

    y_labels = [MITIGATION_LABELS[m] for m in mits]
    fig = go.Figure(go.Heatmap(
        z=norm,
        x=HEATMAP_METRICS,
        y=y_labels,
        colorscale=[[0, "#fca5a5"], [0.5, "#fde68a"], [1, "#86efac"]],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(title="Relative Performance<br>(0=Worst, 1=Best)",
                      tickfont=dict(size=10), len=0.7),
        text=[[annot[r][c] for c in range(n_metrics)] for r in range(n_mits)],
        texttemplate="%{text}",
        textfont=dict(size=12, color="#1e293b", family="Inter"),
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        **_styled_layout(height=360, margin=dict(t=55, b=30, l=140, r=80)),
        title=dict(text="Mitigation Comparison Heatmap<br><sup>Green = stronger performance on that metric</sup>",
                   font=dict(size=13, color="#1e293b")),
        xaxis=dict(side="bottom", tickfont=dict(size=12, color="#374151")),
        yaxis=dict(tickfont=dict(size=11, color="#374151")),
    )
    return fig


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PromptShield Research Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #f8faff 0%, #f0f4ff 40%, #fafbff 100%);
    min-height: 100vh;
}

h1, h2, h3, h4, h5, h6 { color: #1e293b !important; }

/* Hero */
.hero-wrap { text-align: center; padding: 3.2rem 1rem 2rem; }
.hero-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: linear-gradient(135deg, #ede9fe, #dbeafe);
    border: 1px solid #c4b5fd; border-radius: 999px;
    padding: 0.35rem 1.2rem; font-size: 0.85rem; font-weight: 700;
    color: #5b21b6; letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 1.3rem;
}
.hero-title {
    font-size: 3.4rem; font-weight: 800; line-height: 1.18;
    background: linear-gradient(135deg, #4f46e5 0%, #0891b2 55%, #059669 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.85rem;
}
.hero-sub { font-size: 1.15rem; color: #64748b; line-height: 1.6; }
.hero-authors { color: #4f46e5; font-weight: 600; }

/* Divider */
.hr { height: 1px; background: linear-gradient(90deg,transparent,#e2e8f0,transparent); margin: 1.4rem 0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: #ffffff; border-radius: 14px;
    padding: 5px; border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(79,70,229,0.07);
    margin-bottom: 1.8rem; justify-content: center;
}
.stTabs [data-baseweb="tab"] {
    flex: 1; text-align: center; justify-content: center;
    border-radius: 10px; padding: 0.7rem 1rem;
    font-weight: 600; font-size: 0.92rem; color: #64748b;
    background: transparent; border: none;
    transition: all 0.18s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #0891b2) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.25) !important;
}

/* Cards */
.card {
    background: #ffffff; border: 1px solid #e8ecf4; border-radius: 18px;
    padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(79,70,229,0.06);
}
.card h3 { font-size: 1.05rem; font-weight: 700; color: #1e293b; margin-bottom: 0.7rem; }
.card p, .card li { font-size: 0.9rem; color: #475569; line-height: 1.75; }

/* KPI card */
.kpi-card {
    background: #ffffff; border: 1px solid #e8ecf4; border-radius: 16px;
    padding: 1.1rem 1rem 1rem; text-align: center;
    box-shadow: 0 2px 10px rgba(79,70,229,0.07);
    position: relative; overflow: hidden;
    transition: transform 0.18s, box-shadow 0.18s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(79,70,229,0.12); }
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: var(--accent, linear-gradient(90deg,#4f46e5,#0891b2));
    border-radius: 16px 16px 0 0;
}
.kpi-val  { font-size: 1.95rem; font-weight: 800; color: #1e293b; line-height: 1.1; }
.kpi-lbl  { font-size: 0.7rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.25rem; }
.kpi-sub  { font-size: 0.76rem; color: #64748b; margin-top: 0.15rem; }

/* Metric card */
.mcard {
    background: #fff; border: 1px solid #e8ecf4;
    border-left: 4px solid var(--lc, #4f46e5);
    border-radius: 0 14px 14px 0; padding: 1.1rem 1.3rem;
    margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(79,70,229,0.05);
}
.mcard-title { font-size: 0.95rem; font-weight: 700; color: #1e293b; margin-bottom: 0.2rem; }
.mcard-formula {
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #059669;
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px;
    padding: 0.3rem 0.65rem; display: inline-block; margin: 0.35rem 0 0.5rem;
}
.mcard-desc { font-size: 0.86rem; color: #475569; line-height: 1.65; margin: 0; }
.mcard-note { font-size: 0.78rem; color: #94a3b8; margin-top: 0.4rem; border-top: 1px solid #f1f5f9; padding-top: 0.4rem; }

/* Badges */
.badge {
    display: inline-block; border-radius: 999px;
    padding: 0.18rem 0.65rem; font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.bp { background: #ede9fe; color: #6d28d9; }
.bc { background: #e0f2fe; color: #0369a1; }
.ba { background: #fef3c7; color: #92400e; }
.bg { background: #d1fae5; color: #065f46; }
.br { background: #fee2e2; color: #991b1b; }

/* Finding boxes */
.finding {
    background: #f8faff; border: 1px solid #dbeafe;
    border-left: 4px solid #4f46e5; border-radius: 0 12px 12px 0;
    padding: 0.95rem 1.2rem; margin: 0.7rem 0;
    font-size: 0.88rem; color: #334155; line-height: 1.65;
}
.finding-lbl { font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
               letter-spacing: 0.08em; color: #4f46e5; margin-bottom: 0.3rem; }
.finding-good { border-left-color: #059669; background: #f0fdf4; border-color: #bbf7d0; }
.finding-warn { border-left-color: #d97706; background: #fffbeb; border-color: #fde68a; }

/* Strategy card */
.strat-card {
    background: #fff; border: 1px solid #e8ecf4; border-radius: 16px;
    padding: 1.3rem; box-shadow: 0 2px 8px rgba(79,70,229,0.06); height: 100%;
}
.strat-icon { font-size: 1.7rem; margin-bottom: 0.5rem; }
.strat-name { font-size: 0.97rem; font-weight: 700; color: #1e293b; margin-bottom: 0.3rem; }
.strat-desc { font-size: 0.82rem; color: #64748b; line-height: 1.6; }

/* Pipeline */
.pipeline-step { display: flex; align-items: flex-start; gap: 0.9rem; padding: 0.75rem 0; border-bottom: 1px solid #f1f5f9; }
.step-num { min-width: 28px; height: 28px; border-radius: 50%; background: linear-gradient(135deg,#4f46e5,#0891b2); color: #fff; font-size: 0.8rem; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.step-body .step-title { font-size: 0.88rem; font-weight: 700; color: #1e293b; }
.step-body .step-desc  { font-size: 0.82rem; color: #64748b; line-height: 1.55; margin-top: 0.1rem; }

/* Plot wrapper */
.plot-wrap {
    background: #ffffff; border: 1px solid #e8ecf4; border-radius: 16px;
    padding: 0.15rem 0.3rem 0; box-shadow: 0 2px 8px rgba(79,70,229,0.05);
    margin-bottom: 1rem;
}

/* Dataset turn row */
.turn-row { display: flex; gap: 0.8rem; padding: 0.65rem 0.8rem; border-radius: 10px; align-items: flex-start; margin-bottom: 0.4rem; }
.turn-speaker { min-width: 48px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; padding-top: 0.15rem; }
.turn-text { font-size: 0.86rem; color: #334155; line-height: 1.6; flex: 1; }
.turn-label-wrap { min-width: 115px; text-align: right; }

/* Warn / info panel */
.warn-panel { background: #fffbeb; border: 1px solid #fde68a; border-radius: 12px; padding: 1rem 1.3rem; font-size: 0.88rem; color: #92400e; line-height: 1.6; }

/* Buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg,#4f46e5,#0891b2) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.84rem !important;
    padding: 0.45rem 1.1rem !important;
}
.stButton > button {
    background: linear-gradient(135deg,#4f46e5,#0891b2) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; padding: 0.5rem 1.8rem !important; transition: all 0.2s !important;
}
.stButton > button:hover { box-shadow: 0 6px 20px rgba(79,70,229,0.3) !important; }
.stSelectbox label { color: #374151 !important; font-size: 0.85rem !important; font-weight: 500 !important; }
div[data-baseweb="select"] > div { border-radius: 10px !important; border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="hero-chip">Generative AI · 2026</div>
  <div class="hero-title">Evaluating Mitigation Robustness<br>Under Multi-Turn Prompt Injection</div>
  <div class="hero-sub">
    A systematic empirical study of LLM safety defences against adversarial conversational attacks<br>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

tab_overview, tab_findings, tab_dataset = st.tabs([
    "🎯   Goal · Metrics · Harness",
    "📊   Findings & Results",
    "🗂️   Our Dataset",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — GOAL · METRICS · HARNESS
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    st.markdown("""
    <div class="card">
      <h3>🎯 Research Goal</h3>
      <p>Large Language Models are increasingly deployed in interactive, multi-turn settings where a single
      conversation can span many user messages. This creates a dangerous attack surface: an adversary can
      <em>gradually</em> build context and intent across turns, bypassing safety filters that only inspect
      isolated messages.</p>
      <blockquote style="border-left:4px solid #4f46e5;padding-left:1.1rem;margin:0.8rem 0;
                         color:#4f46e5;font-style:italic;font-weight:600;font-size:1rem;">
        "How robust are existing prompt-injection mitigations when the attack unfolds across
        multiple conversational turns rather than in a single adversarial message?"
      </blockquote>
      <p>We evaluate <strong>three defensive strategies</strong> plus a no-mitigation baseline against
      160 multi-turn adversarial conversations spanning six harm categories and three conversation lengths.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛡️ Defensive Strategies")
    s0, s1, s2, s3 = st.columns(4)
    strategy_data = [
        ("⚡", "Baseline (No Mitigation)", "bg", "No Mitigation",
         "Standard model safety training only. Serves as the control condition against which every mitigation is measured."),
        ("📜", "M1 — Prompt Hardening",    "bp", "Instruction-Level",
         "Prepends a structured safety system prompt guiding the LLM to recognise and refuse gradual adversarial escalation."),
        ("🔍", "M2 — I/O Gate",            "bc", "Architectural Filter",
         "A keyword filter that blocks messages before the LLM (input gate) and scans the model's reply after generation (output gate)."),
        ("📈", "M3 — State Monitor",       "ba", "Heuristic Tracker",
         "Accumulates a per-turn escalation score. When the cumulative score exceeds a threshold the conversation is blocked."),
    ]
    for col, (icon, name, badge_cls, badge_txt, desc) in zip([s0, s1, s2, s3], strategy_data):
        with col:
            st.markdown(f"""
            <div class="strat-card">
              <div class="strat-icon">{icon}</div>
              <div class="strat-name">{name}</div>
              <span class="badge {badge_cls}" style="margin-bottom:0.6rem;">{badge_txt}</span>
              <div class="strat-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 📐 Evaluation Metrics" )
    st.markdown("<p style='color:#64748b;font-size:0.9rem;margin-bottom:1.2rem;'>Five complementary metrics capture different facets of mitigation quality.</p>", unsafe_allow_html=True)

    metrics_meta = [
        {
            "icon": "🎯", "name": "ASR — Attack Success Rate",
            "badge": "br", "badge_txt": "Primary · Lower is Better", "lc": "#e11d48",
            "formula": "ASR = Attacks Missed / Total Attacks × 100",
            "desc": "The headline metric. Measures the percentage of adversarial conversations where the model ultimately provided harmful content. An ASR of 0% means every attack was blocked; 100% means every attack slipped through.",
            "note": "A catch at any turn counts — even a late refusal on the final escalation turn counts as a block.",
        },
        {
            "icon": "⏱️", "name": "DL — Detection Latency",
            "badge": "bc", "badge_txt": "Timeliness · Lower is Better", "lc": "#0891b2",
            "formula": "DL = user turns from injection turn to first refusal",
            "desc": "How quickly a mitigation reacts after the attack begins. DL = 0 means the mitigation fired immediately on the very first adversarial turn. A higher DL means the attacker had several unimpeded turns before being blocked.",
            "note": "Split into Gate Latency (M2/M3 code-level block) and AI Latency (LLM self-refusal via phrase-match or judge).",
        },
        {
            "icon": "⚠️", "name": "ORR — Over-Refusal Rate",
            "badge": "ba", "badge_txt": "Usability · Lower is Better", "lc": "#d97706",
            "formula": "ORR = False Positives / Total Benign Conversations × 100",
            "desc": "The usability cost of a mitigation. Measures how often the system wrongly blocks a harmless conversation. An ORR of 88% means almost every legitimate query gets refused — rendering the system practically unusable.",
            "note": "Especially problematic for M2, which keyword-matches without understanding context.",
        },
        {
            "icon": "📏", "name": "CLD — Context-Length Drift",
            "badge": "bp", "badge_txt": "Stability · Near-Zero is Best", "lc": "#7c3aed",
            "formula": "CLD = ASR_long − ASR_short  (percentage points)",
            "desc": "Reveals whether a mitigation weakens as conversations grow longer. A positive CLD means the model becomes more vulnerable in long conversations — it 'forgets' its safety stance as context accumulates.",
            "note": "Computed across three length buckets: Short (≤8 turns), Medium (9–14 turns), Long (>14 turns).",
        },
        {
            "icon": "🏔️", "name": "ERR — Escalation Resistance Rate",
            "badge": "bg", "badge_txt": "Resistance · Higher is Better", "lc": "#059669",
            "formula": "ERR = Blocked ATTACK_ESCALATES turns / Total ATTACK_ESCALATES turns × 100",
            "desc": "Zooms in on mid-conversation escalation moves. ERR measures what fraction of high-risk escalation turns the mitigation intercepts. Split into ERR_early and ERR_late to reveal vigilance throughout a conversation.",
            "note": "Computed only on ATTACK_ESCALATES labelled turns, not ATTACK_STARTS or benign turns.",
        },
    ]

    mc1, mc2 = st.columns(2)
    for col, mlist in [(mc1, metrics_meta[:3]), (mc2, metrics_meta[3:])]:
        with col:
            for m in mlist:
                st.markdown(f"""
                <div class="mcard" style="--lc:{m['lc']}">
                  <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.2rem;">
                    <span style="font-size:1.2rem;">{m['icon']}</span>
                    <span class="mcard-title">{m['name']}</span>
                    &nbsp;<span class="badge {m['badge']}">{m['badge_txt']}</span>
                  </div>
                  <div class="mcard-formula">{m['formula']}</div>
                  <p class="mcard-desc">{m['desc']}</p>
                  <div class="mcard-note">💡 {m['note']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### ⚙️ The Evaluation Harness")
    hc1, hc2 = st.columns([3, 2])
    with hc1:
        st.markdown("""
        <div class="card">
          <h3>How the Harness Works</h3>
          <p style="margin-bottom:1rem;"><code style="color:#0891b2;background:#f0f9ff;padding:0.15rem 0.4rem;border-radius:5px;">System/harness.py</code> iterates over every conversation in the dataset, replaying each USER turn through the active mitigation pipeline in sequence.</p>
        """, unsafe_allow_html=True)
        steps = [
            ("Apply Mitigation", "Run the active strategy (none/M1/M2/M3). M2 and M3 may block the turn before an LLM call is made."),
            ("Call the LLM", "Send the full conversation history to the model API — skipped if the M2/M3 gate already fired."),
            ("Detect Refusal", "Two-stage detector: fast phrase-match against 80+ patterns, then LLM-as-judge fallback for novel phrasings."),
            ("Log Turn", "Record the label, mitigation flags, latency, and whether this was a false positive into turn_logs."),
            ("Compute Metrics", "After all conversations: aggregate ASR, ORR, DL, CLD, ERR from results and turn_logs."),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(f"""
            <div class="pipeline-step">
              <div class="step-num">{i}</div>
              <div class="step-body"><div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<p style="color:#059669;font-size:0.82rem;margin-top:0.9rem;">🧪 Fully reproducible: Temperature=0.0, Seed=42, dataset MD5 logged in run_info.json</p></div>', unsafe_allow_html=True)
    with hc2:
        st.markdown("""
        <div class="card">
          <h3>Two-Stage Refusal Detector</h3>
          <div style="margin-bottom:0.9rem;"><span class="badge bp">Stage 1 — Phrase Match</span><p style="margin-top:0.5rem;">Fast, zero-cost scan for 80+ normalised refusal phrases. Catches the vast majority of obvious refusals instantly.</p></div>
          <div style="margin-bottom:0.9rem;"><span class="badge bc">Stage 2 — LLM-as-Judge</span><p style="margin-top:0.5rem;">Only activates when Stage 1 produces no match. Uses <strong>meta/llama-3.1-70b-instruct</strong> via NVIDIA NIM to classify indirect or novel refusal phrasings.</p></div>
          <div><span class="badge bg">Backup Judge</span><p style="margin-top:0.5rem;">If the primary judge is rate-limited (3× 429s), the system switches to <strong>nvidia/nemotron-3-super-120b</strong> for the remainder of the run.</p></div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — FINDINGS & RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_findings:

    sel_col, btn_col = st.columns([3, 1])
    with sel_col:
        model_options = {
            "OpenAI GPT-OSS 120B (openai/gpt-oss-120b)": "openai-gpt-oss-120b",
            "All Models (Coming Soon)": "__all__",
        }
        selected_label = st.selectbox("Select Model", options=list(model_options.keys()), index=0)
        selected_slug  = model_options[selected_label]
    with btn_col:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        st.button("🔬  Generate Analysis", use_container_width=True)

    if selected_slug == "__all__":
        st.markdown('<div class="warn-panel" style="margin-top:1rem;"><strong>⚠️ Cross-Model Analysis Coming Soon</strong><br>Aggregated "All Models" statistics are not yet available. Check back once the full evaluation suite is complete.</div>', unsafe_allow_html=True)
        st.stop()

    # ── Load data ──────────────────────────────────────────────────────────────
    metrics      = load_metrics(selected_slug)
    results_data = load_results(selected_slug)
    comp_summary = load_comparison_summary(selected_slug)

    if not metrics:
        st.error("No result files found. Run the evaluation harness first.")
        st.stop()

    available_mits = [m for m in MITIGATIONS if m in metrics]
    model_display  = AVAILABLE_MODELS.get(selected_slug, selected_slug)

    badge_map   = {"none": "bg", "m1": "bp", "m2": "bc", "m3": "ba"}
    badges_html = " ".join(f'<span class="badge {badge_map[m]}">{MITIGATION_LABELS[m]}</span>' for m in available_mits)
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.9rem;margin-bottom:1.4rem;
                padding:0.9rem 1.2rem;background:#fff;border:1px solid #e8ecf4;
                border-radius:14px;box-shadow:0 2px 8px rgba(79,70,229,0.06);">
      <div style="font-size:1.5rem;">🤖</div>
      <div>
        <div style="font-size:1rem;font-weight:700;color:#1e293b;">{model_display}</div>
        <div style="font-size:0.76rem;color:#94a3b8;font-family:'JetBrains Mono',monospace;">{selected_slug}</div>
      </div>
      <div style="margin-left:auto;display:flex;gap:0.4rem;flex-wrap:wrap;">{badges_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────────────────
    st.markdown("#### 📊 Key Metrics at a Glance")
    accent_map = {
        "none": "linear-gradient(90deg,#94a3b8,#64748b)",
        "m1":   "linear-gradient(90deg,#4f46e5,#7c3aed)",
        "m2":   "linear-gradient(90deg,#0891b2,#0369a1)",
        "m3":   "linear-gradient(90deg,#d97706,#b45309)",
    }
    kpi_cols = st.columns(len(available_mits))
    for idx, mit in enumerate(available_mits):
        m = metrics[mit]
        with kpi_cols[idx]:
            asr    = m.get("attack_success_rate_pct", "—")
            orr    = m.get("over_refusal_rate_pct",   "—")
            err    = m.get("err_overall",              "—")
            dl     = m.get("mean_detection_latency_turns", "—")
            caught = m.get("attacks_caught",           "—")
            st.markdown(f"""
            <div class="kpi-card" style="--accent:{accent_map[mit]}">
              <div class="kpi-lbl">{MITIGATION_LABELS[mit]}</div>
              <div class="hr" style="margin:0.5rem 0;"></div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
                <div><div class="kpi-val" style="color:#e11d48;">{asr}%</div><div class="kpi-lbl">ASR ↓</div></div>
                <div><div class="kpi-val">{orr}%</div><div class="kpi-lbl">ORR ↓</div></div>
                <div><div class="kpi-val" style="color:#059669;">{err}%</div><div class="kpi-lbl">ERR ↑</div></div>
                <div><div class="kpi-val">{dl}</div><div class="kpi-lbl">DL (turns)</div></div>
              </div>
              <div class="kpi-sub" style="margin-top:0.5rem;">Caught: {caught}/110 attacks</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  INTERACTIVE CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### 📈 Interactive Visualisations")

    # Row 1: Response curves + Run history
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(
            chart_response_curves(results_data, available_mits),
            use_container_width=True, theme=None,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with ch2:
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(
            chart_run_history(results_data, available_mits),
            use_container_width=True, theme=None,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Efficiency + ASR by length
    ch3, ch4 = st.columns(2)
    with ch3:
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        if comp_summary:
            st.plotly_chart(chart_efficiency(comp_summary, available_mits), use_container_width=True, theme=None)
        else:
            st.info("comparison_summary.json not found. Re-run comparison step.")
        st.markdown('</div>', unsafe_allow_html=True)
    with ch4:
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(chart_asr_by_length(metrics, available_mits), use_container_width=True, theme=None)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: Reliability + Heatmap
    ch5, ch6 = st.columns(2)
    with ch5:
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        if comp_summary:
            st.plotly_chart(chart_reliability(comp_summary, available_mits), use_container_width=True, theme=None)
        else:
            st.info("comparison_summary.json not found.")
        st.markdown('</div>', unsafe_allow_html=True)
    with ch6:
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        if comp_summary:
            st.plotly_chart(chart_heatmap(comp_summary, available_mits), use_container_width=True, theme=None)
        else:
            st.info("comparison_summary.json not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ── Findings ───────────────────────────────────────────────────────────────
    st.markdown("#### 🔍 Key Findings & Interpretation")
    m_none = metrics.get("none", {})
    m_m1   = metrics.get("m1",   {})
    m_m2   = metrics.get("m2",   {})
    m_m3   = metrics.get("m3",   {})

    asr_none = m_none.get("attack_success_rate_pct", "N/A")
    asr_m1   = m_m1.get("attack_success_rate_pct",  "N/A")
    asr_m2   = m_m2.get("attack_success_rate_pct",  "N/A")
    asr_m3   = m_m3.get("attack_success_rate_pct",  "N/A")
    orr_m2   = m_m2.get("over_refusal_rate_pct",    "N/A")
    orr_m1   = m_m1.get("over_refusal_rate_pct",    "N/A")
    cld_m3   = m_m3.get("context_length_drift_pct", "N/A")
    cld_m1   = m_m1.get("context_length_drift_pct", "N/A")
    err_cyber  = m_m1.get("err_by_topic", {}).get("cybersecurity", "N/A")
    err_danger = m_m1.get("err_by_topic", {}).get("dangerous_instructions", "N/A")
    err_bypass = m_m1.get("err_by_topic", {}).get("bypassing_ai_rules",     "N/A")

    try: gap_m1 = round(float(asr_none) - float(asr_m1), 1)
    except: gap_m1 = "N/A"

    findings = [
        ("", "🏆", "Finding 1 — Best Attack Containment",
         f"<strong>M1 (Prompt Hardening)</strong> achieves the lowest Attack Success Rate. "
         f"The no-mitigation baseline lets <strong>{asr_none}%</strong> of attacks succeed. "
         f"M1 drops this to <strong>{asr_m1}%</strong> — a {gap_m1}pp reduction — by giving the LLM "
         f"explicit escalation-awareness through its system prompt, without any architectural overhead."),
        ("finding-warn", "⚠️", "Finding 2 — The M2 Usability Crisis",
         f"The I/O Gate (M2) carries an enormous usability cost: an Over-Refusal Rate of "
         f"<strong>{orr_m2}%</strong> on fully benign conversations. Nearly 9 in 10 legitimate "
         f"user queries are wrongly blocked. This traces directly to the keyword filter's inability "
         f"to distinguish context — a sentence mentioning 'bypass' is treated identically whether it "
         f"comes from a security researcher or an attacker."),
        ("", "📏", "Finding 3 — Context-Length Vulnerability",
         f"M3 (State Monitor) shows a large positive CLD of <strong>+{cld_m3}pp</strong>, "
         f"meaning protection degrades significantly in long conversations. Once M3 blocks a turn "
         f"and resets its cumulative score, the attacker gets a clean slate. M1 achieves a CLD of "
         f"just <strong>{cld_m1}pp</strong>, showing near-stable protection across conversation lengths."),
        ("", "🎯", "Finding 4 — Topic-Level Blind Spots",
         f"Even M1 has distinct vulnerability hotspots. Escalation Resistance is near-perfect for "
         f"<em>Bypassing AI Rules</em> (<strong>{err_bypass}%</strong>) but drops for "
         f"<em>Cybersecurity</em> (<strong>{err_cyber}%</strong>) and "
         f"<em>Dangerous Instructions</em> (<strong>{err_danger}%</strong>). Attacks in "
         f"technical domains are harder to catch because their early preamble turns are "
         f"indistinguishable from legitimate educational queries."),
        ("finding-good", "✅", "Overall Verdict",
         f"For <strong>{model_display}</strong>, M1 is the most balanced mitigation: "
         f"lowest ASR ({asr_m1}%), near-zero false positives (ORR={orr_m1}%), and stable "
         f"protection across all conversation lengths. Large instruction-following models can "
         f"effectively self-regulate with a well-crafted safety prompt. Purely architectural "
         f"defences (M2) create unacceptable usability costs, while heuristic monitors (M3) "
         f"can be gamed by patient attackers across long conversations."),
    ]
    for cls, icon, title, body in findings:
        st.markdown(f"""
        <div class="finding {cls}">
          <div class="finding-lbl">{icon} {title}</div>
          {body}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab_dataset:

    stats  = dataset_stats()
    sample = load_dataset_sample()

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown("""
        <div style="padding:0.2rem 0 0.6rem;">
          <div style="font-size:1.35rem;font-weight:800;color:#1e293b;margin-bottom:0.3rem;">
            📦 Evaluation Dataset — Hardened V2
          </div>
          <div style="font-size:0.9rem;color:#64748b;line-height:1.6;max-width:720px;">
            160 synthetic multi-turn adversarial conversations, stratified across 6 harm categories
            and 3 conversation-length groups, with turn-level attack labels.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with top_right:
        st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
        if os.path.exists(_DATASET_FILE):
            with open(_DATASET_FILE, "rb") as f:
                st.download_button(
                    label="⬇️ Download",
                    data=f.read(),
                    file_name="test2_final_hardened_v2_cleaned.json",
                    mime="application/json",
                    use_container_width=True,
                )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    if stats:
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        for col, val, lbl, acc in [
            (k1, str(stats["total"]),   "Total Conversations",  "linear-gradient(90deg,#4f46e5,#7c3aed)"),
            (k2, str(stats["attacks"]), "Attack Conversations", "linear-gradient(90deg,#e11d48,#be123c)"),
            (k3, str(stats["benign"]),  "Benign Conversations", "linear-gradient(90deg,#059669,#047857)"),
            (k4, "6",                   "Harm Categories",      "linear-gradient(90deg,#d97706,#b45309)"),
            (k5, "3",                   "Length Groups",        "linear-gradient(90deg,#0891b2,#0369a1)"),
            (k6, "160",                 "Total Samples",        "linear-gradient(90deg,#7c3aed,#6d28d9)"),
        ]:
            with col:
                st.markdown(f'<div class="kpi-card" style="--accent:{acc};margin-bottom:1rem;"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    dc1, dc2 = st.columns(2)
    with dc1:
        if stats:
            topic_data = stats["topics"]
            fig_topic = go.Figure(go.Pie(
                labels=[TOPIC_LABELS.get(k, k) for k in topic_data],
                values=list(topic_data.values()),
                hole=0.52, marker_colors=TOPIC_COLORS,
                textfont=dict(size=11, color="#1e293b", family="Inter"),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            ))
            fig_topic.update_layout(**_styled_layout(height=320), title=dict(text="Topic Distribution", font=dict(color="#1e293b", size=14)), showlegend=False)
            st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_topic, use_container_width=True, theme=None)
            st.markdown('</div>', unsafe_allow_html=True)
    with dc2:
        if stats:
            len_data = stats["lengths"]
            fig_len  = go.Figure(go.Bar(
                x=list(len_data.keys()), y=list(len_data.values()),
                marker_color=["#4f46e5", "#0891b2", "#d97706"], marker_line_width=0,
                text=list(len_data.values()), textposition="outside",
                textfont=dict(color="#1e293b", size=13),
            ))
            fig_len.update_layout(
                **_styled_layout(height=320),
                title=dict(text="Conversations by Length Group", font=dict(color="#1e293b", size=14)),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Count",
                           range=[0, max(len_data.values()) * 1.25]),
            )
            st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_len, use_container_width=True, theme=None)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.05rem;font-weight:700;color:#1e293b;margin-bottom:0.8rem;">📄 Sample Dataset Entry — <code style="font-size:0.9rem;color:#4f46e5;">V2-001</code></div>', unsafe_allow_html=True)

    if sample:
        topic_nice  = TOPIC_LABELS.get(sample.get("topic", ""), sample.get("topic", ""))
        attack_type = sample.get("attack_type", "—")
        inj_turn    = sample.get("injection_turn", "—")
        success     = "✅ Blocked" if sample.get("success") == "no" else "❌ Succeeded"
        n_turns     = len(sample.get("turns", []))

        st.markdown(f"""
        <div style="display:flex;flex-wrap:wrap;gap:0.6rem;margin-bottom:1rem;">
          <span class="badge bp">Topic: {topic_nice}</span>
          <span class="badge bc">Type: {attack_type}</span>
          <span class="badge ba">Injection Turn: {inj_turn}</span>
          <span class="badge {'bg' if success.startswith('✅') else 'br'}">{success}</span>
          <span class="badge" style="background:#f1f5f9;color:#374151;">{n_turns} turns total</span>
        </div>
        """, unsafe_allow_html=True)

        label_style = {
            "BENIGN":           ("bg", "Benign"),
            "ATTACK_STARTS":    ("ba", "Attack Starts"),
            "ATTACK_ESCALATES": ("br", "Escalates"),
            "DETECTED":         ("bp", "Detected"),
        }
        turns = sample.get("turns", [])[:8]
        st.markdown('<div class="card" style="padding:1.2rem 1.5rem;">', unsafe_allow_html=True)
        for turn in turns:
            speaker = turn.get("speaker", "")
            text    = turn.get("text", "")
            label   = turn.get("label", "")
            lbl_cls, lbl_txt = label_style.get(label, ("", label))
            badge_html = f'<span class="badge {lbl_cls}">{lbl_txt}</span>' if lbl_txt else ""
            is_user    = speaker == "USER"
            bg_color   = "#f8faff" if is_user else "#ffffff"
            spk_color  = "#4f46e5" if is_user else "#0891b2"
            st.markdown(f"""
            <div class="turn-row" style="background:{bg_color};">
              <div class="turn-speaker" style="color:{spk_color};">{speaker}</div>
              <div class="turn-text">{text}</div>
              <div class="turn-label-wrap">{badge_html}</div>
            </div>
            """, unsafe_allow_html=True)
        if len(sample.get("turns", [])) > 8:
            st.markdown(f'<div style="text-align:center;color:#94a3b8;font-size:0.8rem;padding:0.5rem 0;">… {len(sample.get("turns",[])) - 8} more turns not shown</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;flex-wrap:wrap;gap:0.5rem;align-items:center;margin-top:0.2rem;">
          <span style="font-size:0.78rem;color:#64748b;font-weight:600;">Turn Labels:</span>
          <span class="badge bg">Benign</span> <span class="badge ba">Attack Starts</span>
          <span class="badge br">Attack Escalates</span> <span class="badge bp">Detected (refusal)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Dataset file not found. Ensure Datasets/test2_final_hardened_v2_cleaned.json exists.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div style="text-align:center;padding:2.2rem 0 1rem;color:#cbd5e1;font-size:0.78rem;">Evaluating Mitigation Robustness Under Multi-Turn Prompt Injection &nbsp;·&nbsp; Jibran Shaikh &amp; Syeda Wania Hussain &nbsp;·&nbsp; GenAI Research — 2026</div>', unsafe_allow_html=True)
