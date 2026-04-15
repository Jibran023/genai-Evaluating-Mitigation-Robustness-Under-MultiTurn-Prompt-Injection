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
    ax.axvline(50, color=GRAY, linestyle="--", linewidth=1, alpha=0.7,
               label="50% ASR threshold")
    ax.set_xlabel("Attack Success Rate (%)")
    tvc_label = f"{tvc_score:.3f}" if tvc_score is not None else "N/A"
    ax.set_title(
        f"Topic Vulnerability Consistency (TVC = {tvc_label})\n"
        "Per-topic ASR — lower & more uniform = higher TVC",
        fontsize=10,
    )
    ax.set_xlim(0, 115)
    ax.legend(fontsize=8)
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
    ax.legend(fontsize=8)
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
    if tvc_metrics:
        print("  tvc_by_topic.png")
    if err_metrics:
        print("  err_by_mitigation.png")
        print("  err_by_topic.png")
    if rcs_metrics:
        print("  rcs_by_mitigation.png")
