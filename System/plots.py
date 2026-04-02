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
plot_mitigation_comparison(comparison_data, plots_dir)
plot_adt_heatmap(adt_data, plots_dir)
save_all(results, mean_dl, cld_rows, cld_val, failures, plots_dir)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

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
    ax.axvline(50, color=GRAY, linestyle="--", linewidth=1, label="50 % threshold")
    ax.set_xlabel("Attack Success Rate (%)")
    ax.set_title("Attack Success Rate by Topic\n(red ≥ 50 %)", fontsize=11)
    ax.set_xlim(0, 110)
    ax.legend(fontsize=8)
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


# ── 5. Mitigation comparison (grouped bar) ────────────────────────────────────
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


# ── 6. ADT heatmap ────────────────────────────────────────────────────────────
# adt_data format:
#   {
#     "m1": {"cybersecurity": 15.0, "financial_manipulation": 30.0, ...},
#     "m2": {...},
#     "m3": {...},
#   }

def plot_adt_heatmap(adt_data: dict, plots_dir: str):
    _ensure(plots_dir)
    if not adt_data:
        return

    mitigations = list(adt_data.keys())
    topics      = sorted({t for m in adt_data.values() for t in m})
    matrix      = np.array(
        [[adt_data[m].get(t, 0.0) for t in topics] for m in mitigations],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(max(6, len(topics) * 1.2), max(3, len(mitigations) * 0.8)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="ASR on unseen attacks (%)")

    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels([t.replace("_", " ").title() for t in topics],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(mitigations)))
    ax.set_yticklabels([m.upper() for m in mitigations])
    ax.set_title("Attack–Defence Transferability\n(cell = ASR on that unseen topic)", fontsize=10)

    for i in range(len(mitigations)):
        for j in range(len(topics)):
            ax.text(j, i, f"{matrix[i, j]:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > 60 else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "adt_heatmap.png"), dpi=150)
    plt.close()


# ── Convenience wrapper: generate all single-run plots at once ────────────────

def save_all(
    results:   list[dict],
    mean_dl:   float,
    cld_rows:  list[dict],
    cld_val:   float | str,
    failures:  list[dict],
    plots_dir: str,
):
    plot_asr_by_topic(results, plots_dir)
    plot_detection_latency(results, mean_dl, plots_dir)
    plot_context_length_drift(cld_rows, cld_val, plots_dir)
    plot_failure_breakdown(failures, plots_dir)
    print(f"Plots saved to {plots_dir}/")
    print("  asr_by_topic.png")
    print("  detection_latency_dist.png")
    print("  context_length_drift.png")
    print("  failure_breakdown.png")
