"""
plots.py
========
All Matplotlib visualisations.  Every function accepts the data it needs
and a plots_dir path.  Nothing is computed here — call metrics.py first,
then pass the results in.

Functions
---------
plot_asr_by_topic(results, plots_dir)
plot_detection_latency(results, mean_dl, plots_dir)
plot_context_length_drift(cld_rows, cld_val, plots_dir)
plot_failure_breakdown(failures, plots_dir)
plot_tvc(tvc_metrics, plots_dir)
plot_err(err_metrics, plots_dir)
plot_rcs(rcs_metrics, plots_dir)
plot_mitigation_comparison(comparison_data, plots_dir)
plot_comparison_radar(summary_table, plots_dir)
plot_tradeoff_scatter(summary_table, plots_dir)
plot_latency_efficiency_bubble(summary_table, plots_dir)
save_all(results, mean_dl, cld_rows, cld_val, failures, plots_dir,
         tvc_metrics, err_metrics, rcs_metrics)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
AMBER  = "#D97706"
GRAY   = "#6B7280"
PURPLE = "#7C3AED"
TEAL   = "#0D9488"

MITIGATION_COLORS = {
    "none" : GRAY,
    "m1"   : BLUE,
    "m2"   : TEAL,
    "m3"   : PURPLE,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure(path: str):
    os.makedirs(path, exist_ok=True)


# ── 1. ASR by topic ───────────────────────────────────────────────────────────

def plot_asr_by_topic(results: list[dict], plots_dir: str):
    _ensure(plots_dir)
    df = pd.DataFrame(results)
    attack_df = df[df["is_attack"]]
    if attack_df.empty:
        return

    topic_df = (
        attack_df.groupby("topic")
        .apply(lambda x: round(x["attack_succeeded"].sum() / len(x) * 100, 1),
               include_groups=False)
        .reset_index(name="asr")
        .sort_values("asr", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, max(4, len(topic_df) * 0.55)))
    colors = [RED if v >= 50 else BLUE for v in topic_df["asr"]]
    bars = ax.barh(topic_df["topic"], topic_df["asr"], color=colors)
    line_50 = ax.axvline(50, color=GRAY, linestyle="--", linewidth=1, label="50 % threshold")
    ax.set_xlabel("Attack Success Rate (%)")
    ax.set_title("Attack Success Rate by Topic", fontsize=11)
    ax.set_xlim(0, 110)
    
    legend_elements = [
        Patch(facecolor=RED, label='≥ 50% (Vulnerable)'),
        Patch(facecolor=BLUE, label='< 50% (More Secure)'),
        line_50
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right', title="Key")
    for bar, val in zip(bars, topic_df["asr"]):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "asr_by_topic.png"), dpi=150)
    plt.close()


# ── 2. Detection latency distribution ────────────────────────────────────────

def plot_detection_latency(results: list[dict], mean_dl: float, plots_dir: str):
    _ensure(plots_dir)
    latencies = [
        r["detection_latency"] for r in results
        if r.get("detection_latency") is not None
    ]
    if not latencies:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(latencies, bins=range(0, max(latencies) + 2),
            color=BLUE, edgecolor="white", alpha=0.85, align="left")
    ax.axvline(mean_dl, color=RED, linestyle="--", linewidth=1.5,
               label=f"Mean DL = {mean_dl} turns")
    ax.set_xlabel("Detection Latency (turns after injection)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Detection Latency")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "detection_latency_dist.png"), dpi=150)
    plt.close()


# ── 3. Context-length drift bar chart ────────────────────────────────────────

def plot_context_length_drift(
    cld_rows: list[dict],
    cld_val:  float | str,
    plots_dir: str,
):
    _ensure(plots_dir)
    if not cld_rows:
        return

    df = pd.DataFrame(cld_rows)
    order = ["short", "medium", "long"]
    df["length_group"] = pd.Categorical(df["length_group"], categories=order, ordered=True)
    df = df.sort_values("length_group")

    bar_colors = [
        GREEN if v < 10 else (AMBER if v < 30 else RED)
        for v in df["asr_pct"]
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(df["length_group"], df["asr_pct"], color=bar_colors, width=0.5)
    ax.set_ylabel("ASR (%)")
    ax.set_xlabel("Conversation Length Group")
    ax.set_title(
        f"Context-Length Drift  (CLD = {cld_val}%)\n"
        "ASR_long − ASR_short",
        fontsize=10,
    )
    ax.set_ylim(0, 110)

    legend_elements = [
        Patch(facecolor=GREEN, label='< 10% (Low ASR)'),
        Patch(facecolor=AMBER, label='10-30% (Medium ASR)'),
        Patch(facecolor=RED, label='> 30% (High ASR)')
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left', title="Severity Key")

    for bar, val in zip(bars, df["asr_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "context_length_drift.png"), dpi=150)
    plt.close()


# ── 4. Failure breakdown ──────────────────────────────────────────────────────

def plot_failure_breakdown(failures: list[dict], plots_dir: str):
    _ensure(plots_dir)
    f_missed = sum(1 for x in failures if x["failure_type"] == "MISSED_ATTACK")
    f_fp     = sum(1 for x in failures if x["failure_type"] == "FALSE_POSITIVE")
    if f_missed + f_fp == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Missed Attacks", "False Positives"], [f_missed, f_fp],
           color=[RED, BLUE], width=0.4)
    ax.set_ylabel("Count")
    ax.set_title("Failure Analysis: Missed Attacks vs False Positives")
    for i, v in enumerate([f_missed, f_fp]):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "failure_breakdown.png"), dpi=150)
    plt.close()


# ── 5. Topic Vulnerability Consistency (TVC) ──────────────────────────────────

def plot_tvc(tvc_metrics: dict, plots_dir: str):
    """
    Horizontal bar chart of per-topic ASR with TVC score annotated.
    Lower bar = lower ASR = better for that topic.
    """
    _ensure(plots_dir)
    tvc_by_topic = tvc_metrics.get("tvc_by_topic", {})
    tvc_score    = tvc_metrics.get("tvc_score")

    if not tvc_by_topic:
        return

    topics = sorted(tvc_by_topic.keys())
    asrs   = [tvc_by_topic[t]["asr_pct"] for t in topics]
    ns     = [tvc_by_topic[t]["n"]       for t in topics]

    # Sort ascending so best topics appear at top of horizontal chart
    paired = sorted(zip(topics, asrs, ns), key=lambda x: x[1])
    topics, asrs, ns = zip(*paired) if paired else ([], [], [])

    fig, ax = plt.subplots(figsize=(9, max(4, len(topics) * 0.55)))
    colors = [RED if v >= 50 else (AMBER if v >= 25 else GREEN) for v in asrs]
    bars   = ax.barh(list(topics), list(asrs), color=colors)
    line_50 = ax.axvline(50, color=GRAY, linestyle="--", linewidth=1, alpha=0.7,
               label="50% ASR threshold")
    ax.set_xlabel("Attack Success Rate (%)")
    tvc_label = f"{tvc_score:.3f}" if tvc_score is not None else "N/A"
    ax.set_title(
        f"Topic Vulnerability Consistency (TVC = {tvc_label})\n"
        "Per-topic ASR — lower & more uniform = higher TVC",
        fontsize=10,
    )
    ax.set_xlim(0, 115)
    
    legend_elements = [
        Patch(facecolor=GREEN, label='< 25% (Strong)'),
        Patch(facecolor=AMBER, label='25-50% (Moderate)'),
        Patch(facecolor=RED, label='≥ 50% (Vulnerable)'),
        line_50
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right', title="Key")
    for bar, val, n in zip(bars, asrs, ns):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val}%  (n={n})", va="center", fontsize=8)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels([t.replace("_", " ").title() for t in topics])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "tvc_by_topic.png"), dpi=150)
    plt.close()


# ── 6. Escalation Resistance Rate (ERR) ──────────────────────────────────────

def plot_err(err_metrics: dict, plots_dir: str):
    """
    Two panels:
      Left  — ERR overall + by mitigation (grouped bar)
      Right — Early vs Late ERR split (shows degradation deeper in conversation)
    """
    _ensure(plots_dir)
    err_overall = err_metrics.get("err_overall")
    err_by_mit  = err_metrics.get("err_by_mitigation", {})
    err_early   = err_metrics.get("err_early")
    err_late    = err_metrics.get("err_late")

    if err_overall is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: ERR by mitigation ───────────────────────────────────────────────
    mits  = sorted(err_by_mit.keys())
    vals  = [err_by_mit[m] if err_by_mit[m] is not None else 0.0 for m in mits]
    cols  = [MITIGATION_COLORS.get(m, BLUE) for m in mits]
    bars1 = ax1.bar(mits, vals, color=cols, width=0.5)
    ax1.axhline(err_overall, color=RED, linestyle="--", linewidth=1.5,
                label=f"Overall ERR = {err_overall}%")
    ax1.set_ylim(0, 110)
    ax1.set_xlabel("Mitigation")
    ax1.set_ylabel("ERR (%)")
    ax1.set_title("Escalation Resistance Rate by Mitigation\n(% of ATTACK_ESCALATES turns blocked)",
                  fontsize=10)
    ax1.legend(fontsize=8)
    for bar, val in zip(bars1, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val}%",
                 ha="center", fontsize=9)

    # ── Right: Early vs Late ERR ───────────────────────────────────────────────
    early_val = err_early if err_early is not None else 0.0
    late_val  = err_late  if err_late  is not None else 0.0
    categories = ["Early Escalation", "Late Escalation"]
    values     = [early_val, late_val]
    bar_colors = [GREEN if early_val >= late_val else AMBER, RED if late_val < early_val else TEAL]
    bars2 = ax2.bar(categories, values, color=bar_colors, width=0.45)
    ax2.axhline(err_overall, color=GRAY, linestyle="--", linewidth=1,
                alpha=0.8, label=f"Overall ERR = {err_overall}%")
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("ERR (%)")
    ax2.set_title(
        "ERR: Early vs Late Escalation\n"
        "(degradation = late ERR < early ERR)",
        fontsize=10,
    )
    ax2.legend(fontsize=8)
    for bar, val in zip(bars2, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val}%",
                 ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "err_by_mitigation.png"), dpi=150)
    plt.close()


def plot_err_by_topic(err_metrics: dict, plots_dir: str):
    """Horizontal bar chart of ERR broken down by topic."""
    _ensure(plots_dir)
    err_by_topic = err_metrics.get("err_by_topic", {})
    if not err_by_topic:
        return

    topics = sorted(err_by_topic.keys(), key=lambda t: err_by_topic[t] or 0)
    vals   = [err_by_topic[t] if err_by_topic[t] is not None else 0.0 for t in topics]
    cols   = [GREEN if v >= 70 else (AMBER if v >= 40 else RED) for v in vals]

    fig, ax = plt.subplots(figsize=(9, max(4, len(topics) * 0.55)))
    bars = ax.barh(list(topics), vals, color=cols)
    ax.set_xlabel("ERR (%)")
    ax.set_title("Escalation Resistance Rate by Topic\n(% of ATTACK_ESCALATES turns blocked)\n"
                 "higher = better", fontsize=10)
    ax.set_xlim(0, 115)
    
    legend_elements = [
        Patch(facecolor=GREEN, label='≥ 70% (Strong Resistance)'),
        Patch(facecolor=AMBER, label='40-70% (Moderate)'),
        Patch(facecolor=RED, label='< 40% (Weak Resistance)')
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right', title="ERR Quality")

    for bar, val in zip(bars, vals):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=8)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels([t.replace("_", " ").title() for t in topics])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "err_by_topic.png"), dpi=150)
    plt.close()


# ── 7. Refusal Consistency Score (RCS) ───────────────────────────────────────

def plot_rcs(rcs_metrics: dict, plots_dir: str):
    """
    Bar chart of RCS per mitigation.
    RCS = 1.0 → mitigation never reset after firing (perfect consistency).
    RCS = 0.0 → mitigation always reset (completely inconsistent blocking).
    """
    _ensure(plots_dir)
    rcs_by_mit = rcs_metrics.get("rcs_by_mitigation", {})
    rcs_score  = rcs_metrics.get("rcs_score")

    # Filter to mitigations where the metric is defined (not None)
    mits = sorted(m for m in rcs_by_mit if rcs_by_mit[m] is not None)
    if not mits:
        return

    vals  = [rcs_by_mit[m] for m in mits]
    cols  = [GREEN if v >= 0.8 else (AMBER if v >= 0.5 else RED) for v in vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(mits, vals, color=cols, width=0.45)
    if rcs_score is not None:
        ax.axhline(rcs_score, color=GRAY, linestyle="--", linewidth=1.5,
                   label=f"Overall RCS = {rcs_score:.3f}")
    ax.axhline(1.0, color=GREEN, linestyle=":", linewidth=1, alpha=0.5,
               label="Perfect = 1.0")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Mitigation")
    ax.set_ylabel("RCS (0 – 1)")
    ax.set_title(
        "Refusal Consistency Score (RCS) by Mitigation\n"
        "higher = more sustained blocking after first refusal",
        fontsize=10,
    )
    
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        Patch(facecolor=GREEN, label='≥ 0.8 (Consistent)'),
        Patch(facecolor=AMBER, label='0.5-0.8 (Moderate)'),
        Patch(facecolor=RED, label='< 0.5 (Inconsistent)')
    ])
    ax.legend(handles=handles, fontsize=8, loc='lower left', title="Key")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rcs_by_mitigation.png"), dpi=150)
    plt.close()


# ── 8. Mitigation comparison (grouped bar) ────────────────────────────────────
# comparison_data format:
#   [
#     {"mitigation": "none", "length_group": "short", "asr_pct": 20.0},
#     {"mitigation": "m1",   "length_group": "short", "asr_pct": 10.0},
#     ...
#   ]

def plot_mitigation_comparison(comparison_data: list[dict], plots_dir: str):
    _ensure(plots_dir)
    if not comparison_data:
        return

    df = pd.DataFrame(comparison_data)
    mitigations  = df["mitigation"].unique().tolist()
    length_groups = ["short", "medium", "long"]
    length_groups = [g for g in length_groups if g in df["length_group"].values]

    x     = np.arange(len(length_groups))
    width = 0.8 / max(len(mitigations), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, mit in enumerate(mitigations):
        vals = []
        for grp in length_groups:
            row = df[(df["mitigation"] == mit) & (df["length_group"] == grp)]
            vals.append(float(row["asr_pct"].values[0]) if not row.empty else 0.0)
        offset = (i - len(mitigations) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=mit.upper(),
                      color=MITIGATION_COLORS.get(mit, GRAY))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    f"{v:.0f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in length_groups])
    ax.set_xlabel("Conversation Length Group")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("ASR by Mitigation & Conversation Length\n(lower is better)")
    ax.set_ylim(0, 110)
    ax.legend(title="Mitigation", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "mitigation_comparison.png"), dpi=150)
    plt.close()


# ── 9. Comparison Radar Chart (Spider Plot) ──────────────────────────────────

def plot_comparison_radar(summary_table: dict, plots_dir: str):
    """
    A normalized spider chart comparing all mitigations across key metrics.
    Each metric is mapped to a 0.0 (Worst) -> 1.0 (Best) scale.
    """
    _ensure(plots_dir)
    if not summary_table:
        return

    metrics = [
        "Security (1-ASR)",
        "Latency (1-DL)",
        "Usability (1-ORR)",
        "TVC",
        "ERR",
        "RCS"
    ]
    num_vars = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Custom styling for a "premium" look
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.xticks(angles[:-1], metrics, color=GRAY, size=10)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color=GRAY, size=8)
    plt.ylim(0, 1.1)

    for m, m_data in summary_table.items():
        if not m_data or m == "none" and len(summary_table) > 1:
            # We still want 'none' usually as a baseline, but if it has NO data skip
            if not m_data: continue
        
        # Normalization Logic
        asr = m_data.get("attack_success_rate_pct", 100)
        dl  = m_data.get("mean_detection_latency_turns", 5)
        orr = m_data.get("over_refusal_rate_pct", 100)
        tvc = m_data.get("tvc_score") or 0.0
        err = m_data.get("err_overall") or 0.0
        rcs = m_data.get("rcs_score") or 0.0

        # Map to 0-1 where 1 is better
        values = [
            1.0 - (asr / 100),
            max(0.0, 1.0 - (dl / 5)), # Assume 5 turns is the "ceiling" for bad latency
            1.0 - (orr / 100),
            tvc,
            err / 100,
            rcs
        ]
        values += values[:1] # Close circle

        color = MITIGATION_COLORS.get(m, BLUE)
        ax.plot(angles, values, color=color, linewidth=2, label=m.upper())
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Mitigation Comparison: Robustness Fingerprint\n(closer to outer edge = better)", 
                 va='bottom', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "mitigation_radar.png"), dpi=200, bbox_inches='tight')
    plt.close()


# ── 10. Security vs Usability Trade-off (Scatter) ─────────────────────────────

def plot_tradeoff_scatter(summary_table: dict, plots_dir: str):
    _ensure(plots_dir)
    if not summary_table:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    for m, m_data in summary_table.items():
        if not m_data: continue
        
        security  = 100 - m_data.get("attack_success_rate_pct", 100)
        usability = 100 - m_data.get("over_refusal_rate_pct", 0)
        color     = MITIGATION_COLORS.get(m, BLUE)
        
        ax.scatter(security, usability, s=200, color=color, label=m.upper(), 
                   edgecolor='white', linewidth=1.5, alpha=0.9, zorder=3)
        ax.text(security + 1, usability + 1, m.upper(), fontsize=9, fontweight='bold')

    # Draw targets
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.axhline(90, color=GRAY, linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(90, color=GRAY, linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlabel("Security Index (100 - ASR %)")
    ax.set_ylabel("Usability Index (100 - ORR %)")
    ax.set_title("The Pareto Frontier: Security vs. Usability", fontsize=12, fontweight='bold')
    
    # Annotations
    ax.text(5, 5, "Poor Performance", color=RED, alpha=0.4, fontsize=10)
    ax.text(85, 95, "Goal Region", color=GREEN, alpha=0.6, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "security_usability_tradeoff.png"), dpi=200)
    plt.close()


# ── 11. Latency Efficiency (Bubble Chart) ───────────────────────────────────

def plot_latency_efficiency_bubble(summary_table: dict, plots_dir: str):
    _ensure(plots_dir)
    if not summary_table:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    
    for m, m_data in summary_table.items():
        if not m_data: continue
        
        latency = m_data.get("mean_ai_latency_turns", 0)
        asr     = m_data.get("attack_success_rate_pct", 100)
        orr     = m_data.get("over_refusal_rate_pct", 0)
        color   = MITIGATION_COLORS.get(m, BLUE)
        
        # Bubble size = ORR (bigger bubble = more disruptive)
        # Scale it up for visibility
        size = max(100, orr * 50) 
        
        ax.scatter(latency, asr, s=size, color=color, label=m.upper(), 
                   edgecolor='black', linewidth=1.5, alpha=0.7, zorder=3)
        ax.text(latency + 0.05, asr + 2, m.upper(), fontsize=9, fontweight='bold')

    ax.set_xlabel("Mean Latency (Turns)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Efficiency Analysis: Latency vs. Security\n(Bubble size = Over-Refusal disruption)", 
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis() # Lower ASR (better) is at top
    ax.set_ylim(110, -10) 
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "latency_efficiency.png"), dpi=200)
    plt.close()


# ── 12. Simplified Comparison Heatmap ──────────────────────────────────────────

def plot_comparison_heatmap(summary_table: dict, plots_dir: str):
    """
    A color-coded grid showing all metrics for all mitigations.
    Green = Good performance, Red = Poor performance.
    """
    _ensure(plots_dir)
    if not summary_table:
        return

    mits    = sorted(summary_table.keys())
    # Define metrics and whether HIGHER or LOWER is better
    metrics_info = [
        ("ASR",  "attack_success_rate_pct", False),
        ("DL",   "mean_detection_latency_turns", False),
        ("ORR",  "over_refusal_rate_pct", False),
        ("TVC",  "tvc_score", True),
        ("ERR",  "err_overall", True),
        ("RCS",  "rcs_score", True)
    ]
    
    labels = [m[0] for m in metrics_info] + ["Trust"]
    data   = []
    for m in mits:
        row = []
        for _, key, _ in metrics_info:
            val = summary_table[m].get(key)
            row.append(val if val is not None else np.nan)
        
        # Calculate Reliability/Trust Score for Heatmap (Geometric Mean of Safety and Availability)
        safety       = 100 - summary_table[m].get("attack_success_rate_pct", 100)
        availability = 100 - summary_table[m].get("over_refusal_rate_pct", 0)
        trust        = np.sqrt(max(0, safety) * max(0, availability))
        row.append(trust)
        data.append(row)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(11, len(mits) * 1.2 + 2))
    
    # Normalize data for coloring (0 = Worst, 1 = Best)
    norm_data = np.zeros_like(data)
    for i in range(len(labels)):
        col = data[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) == 0: continue
        
        c_min, c_max = np.nanmin(valid), np.nanmax(valid)
        
        # Special case for the "Trust" column — we always want it normalized against 0-100
        if labels[i] == "Trust":
            norm_data[:, i] = col / 100.0
            continue

        if c_max == c_min:
            norm_data[:, i] = 0.5
        else:
            # Check higher_is_better from metrics_info
            if i < len(metrics_info):
                higher_is_better = metrics_info[i][2]
                if higher_is_better:
                    norm_data[:, i] = (col - c_min) / (c_max - c_min)
                else:
                    norm_data[:, i] = (c_max - col) / (c_max - c_min)
            else:
                norm_data[:, i] = col / 100.0

    im = ax.imshow(norm_data, cmap="RdYlGn", aspect="auto", alpha=0.6)
    
    for i in range(len(mits)):
        for j in range(len(labels)):
            val = data[i, j]
            txt = f"{val:.2f}" if (not np.isnan(val) and val < 2) else f"{val:.1f}"
            if np.isnan(val): txt = "N/A"
            ax.text(j, i, txt, ha="center", va="center", 
                    fontweight="bold", color="black", fontsize=10)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_yticks(np.arange(len(mits)))
    ax.set_yticklabels([m.upper() for m in mits], fontweight="bold")
    
    ax.set_title("Mitigation Comparison Heatmap\n(Green = Stronger Performance | Trust = Combined Metrics)", 
                 pad=20, fontsize=13, fontweight="bold")
    
    plt.colorbar(im, ax=ax, label="Relative Performance (Worst -> Best)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "mitigation_heatmap.png"), dpi=200)
    plt.close()


# ── 13. Simplified Efficiency Bars (ASR vs Latency) ───────────────────────────

def plot_efficiency_bars(summary_table: dict, plots_dir: str):
    _ensure(plots_dir)
    if not summary_table:
        return

    mits = sorted(summary_table.keys())
    asrs = [summary_table[m].get("attack_success_rate_pct", 0) for m in mits]
    lats = [summary_table[m].get("mean_ai_latency_turns", 0) for m in mits]
    
    x = np.arange(len(mits))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    rects1 = ax1.bar(x - width/2, asrs, width, label='Attack Success Rate (%)', color=RED, alpha=0.7)
    ax1.set_ylabel('ASR (%)', color=RED, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.tick_params(axis='y', labelcolor=RED)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, lats, width, label='Mean Latency (Turns)', color=BLUE, alpha=0.7)
    ax2.set_ylabel('Latency (Turns)', color=BLUE, fontweight="bold")
    ax2.set_ylim(0, max(lats + [2]) * 1.2)
    ax2.tick_params(axis='y', labelcolor=BLUE)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in mits], fontweight="bold")
    ax1.set_title("Efficiency Analysis: Security vs. Performance\n(Lower bars are better for both)", 
                  fontsize=12, fontweight="bold")
    
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1, ax1)
    autolabel(rects2, ax2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, "efficiency_comparison_bars.png"), dpi=200)
    plt.close()


# ── 14. Simplified Trade-off Bars (Safety, Availability, Overall) ──────────────

def plot_tradeoff_bars(summary_table: dict, plots_dir: str):
    """
    Combined comparison of Safety, Availability, and Overall Reliability.
    Uses Geometric Mean for Overall Reliability to punish extreme failures.
    """
    _ensure(plots_dir)
    if not summary_table:
        return

    mits = sorted(summary_table.keys())
    
    safety       = []
    availability = []
    overall      = []
    
    for m in mits:
        s = 100 - summary_table[m].get("attack_success_rate_pct", 100)
        a = 100 - summary_table[m].get("over_refusal_rate_pct", 0)
        # Force non-negative for geo-mean
        s_val = max(0.1, s) 
        a_val = max(0.1, a)
        o = np.sqrt(s_val * a_val)
        
        safety.append(s)
        availability.append(a)
        overall.append(round(o, 1))
    
    y = np.arange(len(mits))
    height = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.barh(y + height, safety,       height, label='Safety Score (100-ASR)', color=GREEN, alpha=0.8)
    ax.barh(y,          availability, height, label='Availability Score (100-ORR)', color=TEAL, alpha=0.8)
    ax.barh(y - height, overall,      height, label='OVERALL RELIABILITY', color=BLUE, alpha=0.9)
    
    ax.set_yticks(y)
    ax.set_yticklabels([m.upper() for m in mits], fontweight="bold")
    ax.set_xlabel("Score (0 - 100) | Higher is Better", fontweight="bold")
    ax.set_title("Overall Mitigation Reliability\n(Geometric Mean balances Safety vs. Service)", 
                 fontsize=13, fontweight="bold")
    
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.set_xlim(0, 115)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Value labels
    for i in range(len(mits)):
        ax.text(safety[i] + 1,       i + height, f"{safety[i]:.1f}", va='center', fontsize=9)
        ax.text(availability[i] + 1, i,          f"{availability[i]:.1f}", va='center', fontsize=9)
        ax.text(overall[i] + 1,      i - height, f"{overall[i]:.1f}", va='center', fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "tradeoff_comparison_bars.png"), dpi=200)
    plt.close()


# ── 15. Unified Comparison: Response Curves ──────────────────────────────────

def plot_comparison_response_curves(all_results: dict[str, list[dict]], plots_dir: str):
    """
    Unified line chart showing Cumulative Detection Rate by Turn for ALL mitigations.
    """
    _ensure(plots_dir)
    if not all_results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We'll plot up to turn 20 or the max seen
    max_turn = 10 
    
    for mit_name, results in sorted(all_results.items()):
        attack_results = [r for r in results if r["is_attack"]]
        if not attack_results:
            continue
            
        total_attacks = len(attack_results)
        catches = [r["caught_at_turn"] for r in attack_results if r.get("caught_at_turn") is not None]
        
        if catches:
            max_turn = max(max_turn, max(catches))
        
        # Calculate curve
        limit = max(max_turn, 10)
        x = range(0, limit + 1)
        y = []
        for t in x:
            caught_count = sum(1 for c in catches if c <= t)
            y.append(caught_count / total_attacks * 100)
            
        color = MITIGATION_COLORS.get(mit_name.lower(), GRAY)
        ax.plot(x, y, label=mit_name.upper(), color=color, marker="o", linewidth=2, markersize=4, alpha=0.8)

    ax.set_xlabel("Turn Number (Interaction Step)", fontweight="bold")
    ax.set_ylabel("Attacks Caught (Cumulative %)", fontweight="bold")
    ax.set_title("Unified Mitigation Response Comparison\n(Who catches attacks the fastest?)", 
                 fontsize=13, fontweight="bold")
    
    ax.set_ylim(-5, 105)
    ax.set_xticks(range(0, max_turn + 1))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Mitigation", frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comparison_response_curves.png"), dpi=200)
    plt.close()


# ── 16. Unified Comparison: Progression History ──────────────────────────────

def plot_comparison_run_history(all_results: dict[str, list[dict]], plots_dir: str):
    """
    Unified line chart showing Running Average ASR for ALL mitigations.
    Uses a global X-axis (1 to 130) for full context.
    """
    _ensure(plots_dir)
    if not all_results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    max_count = 0
    
    for mit_name, results in sorted(all_results.items()):
        # Sort results by ID for consistency
        sorted_res = sorted(results, key=lambda x: x.get("id", ""))
        max_count = max(max_count, len(sorted_res))
        
        running_asr = []
        a_succ = 0
        a_total = 0
        last_asr = 0.0
        
        for r in sorted_res:
            if r["is_attack"]:
                a_total += 1
                if r["attack_succeeded"]:
                    a_succ += 1
                last_asr = (a_succ / a_total * 100)
            
            # Record at every step (attack or benign) to stay in sync
            running_asr.append(last_asr)
        
        if running_asr:
            color = MITIGATION_COLORS.get(mit_name.lower(), GRAY)
            ax.plot(range(1, len(running_asr) + 1), running_asr, 
                    label=f"{mit_name.upper()}", color=color, linewidth=2, alpha=0.7)

    ax.set_xlabel(f"Total Progression (Samples 1 to {max_count})", fontweight="bold")
    ax.set_ylabel("Running Average ASR (%)", fontweight="bold")
    ax.set_title("Unified Run History Comparison\n(Across All Available Samples)", 
                 fontsize=13, fontweight="bold")
    
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="Mitigation", loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comparison_run_history.png"), dpi=200)
    plt.close()


def plot_run_history(results_list: list[dict], plots_dir: str):
    """
    Line chart showing the Running Average ASR and ORR across the samples.
    Shows the history of how the metrics stabilized during the run.
    Uses a unified X-axis for ALL 130 samples.
    """
    _ensure(plots_dir)
    if not results_list:
        return

    # Sort results by ID to ensure a consistent "progression"
    sorted_results = sorted(results_list, key=lambda x: x.get("id", ""))
    
    running_asr = []
    running_orr = []
    
    a_succ = 0
    a_total = 0
    b_fp = 0
    b_total = 0
    
    # Track the last known averages to fill the lines across all X points
    last_asr = 0.0
    last_orr = 0.0
    
    for r in sorted_results:
        if r["is_attack"]:
            a_total += 1
            if r["attack_succeeded"]:
                a_succ += 1
            last_asr = (a_succ / a_total * 100) if a_total > 0 else 0.0
        else:
            b_total += 1
            if r["caught_at_turn"] is not None:
                b_fp += 1
            last_orr = (b_fp / b_total * 100) if b_total > 0 else 0.0
        
        running_asr.append(last_asr)
        running_orr.append(last_orr)

    fig, ax = plt.subplots(figsize=(10, 5))
    
    steps = range(1, len(sorted_results) + 1)
    ax.plot(steps, running_asr, color=RED,  label="Running Average ASR (%)", linewidth=2)
    ax.plot(steps, running_orr, color=TEAL, label="Running Average ORR (%)", linewidth=2, linestyle="--")

    ax.set_xlabel(f"Total Progression (Samples 1 to {len(sorted_results)})", fontweight="bold")
    ax.set_ylabel("Metric Value (%)", fontweight="bold")
    ax.set_title("Full Experiment Progression History\n(ASR and ORR Stability Analysis)", fontsize=12, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "run_progression_history.png"), dpi=200)
    plt.close()


# ── 16. Detection Curve (Turn-by-Turn) ──────────────────────────────────────

def plot_detection_curve(results_list: list[dict], plots_dir: str):
    """
    Line chart showing Cumulative Detection Rate by Turn.
    Visualizes the 'History' of how quickly attacks are caught.
    """
    _ensure(plots_dir)
    attack_results = [r for r in results_list if r["is_attack"]]
    if not attack_results:
        return

    total_attacks = len(attack_results)
    # Find max turns across these samples
    max_turn = 0
    catches = []
    for r in attack_results:
        cat = r.get("caught_at_turn")
        if cat is not None:
            max_turn = max(max_turn, cat)
            catches.append(cat)
    
    # We also want to look at max turn in the dataset if possible, 
    # but 10 or 20 is standard.
    limit = max(max_turn, 10)
    
    x = range(0, limit + 1)
    y = []
    
    for t in x:
        # What % of attacks were caught AT or BEFORE turn t?
        caught_count = sum(1 for c in catches if c <= t)
        y.append(caught_count / total_attacks * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color=BLUE, marker="o", linewidth=2.5, markersize=5, label="Cumulative Detection (%)")
    
    ax.set_xlabel("Turn Number (Interaction Step)")
    ax.set_ylabel("Attacks Caught (%)")
    ax.set_title("Mitigation Response Curve\n(Cumulative Detection Probability by Turn)")
    ax.set_ylim(-5, 105)
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    
    # Add horizontal line for final success
    final_val = y[-1]
    ax.axhline(final_val, color=GRAY, linestyle="--", alpha=0.5)
    ax.text(limit, final_val + 2, f"Final: {final_val:.1f}%", ha="right", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "mitigation_response_curve.png"), dpi=150)
    plt.close()


# ── Convenience wrapper: generate all single-run plots at once ────────────────

def save_all(
    results:     list[dict],
    mean_dl:     float,
    cld_rows:    list[dict],
    cld_val:     float | str,
    failures:    list[dict],
    plots_dir:   str,
    tvc_metrics: dict | None = None,
    err_metrics: dict | None = None,
    rcs_metrics: dict | None = None,
):
    plot_asr_by_topic(results, plots_dir)
    plot_detection_latency(results, mean_dl, plots_dir)
    plot_context_length_drift(cld_rows, cld_val, plots_dir)
    plot_failure_breakdown(failures, plots_dir)
    plot_run_history(results, plots_dir)
    plot_detection_curve(results, plots_dir)

    if tvc_metrics:
        plot_tvc(tvc_metrics, plots_dir)
    if err_metrics:
        plot_err(err_metrics, plots_dir)
        plot_err_by_topic(err_metrics, plots_dir)
    if rcs_metrics:
        plot_rcs(rcs_metrics, plots_dir)

    print(f"Plots saved to {plots_dir}/")
    print("  asr_by_topic.png")
    print("  detection_latency_dist.png")
    print("  context_length_drift.png")
    print("  failure_breakdown.png")
    print("  run_progression_history.png")
    print("  mitigation_response_curve.png")
