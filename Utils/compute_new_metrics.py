"""
compute_new_metrics.py
======================
Evaluating Mitigation Robustness Under Multi-Turn Prompt Injection
Authors : Jibran Shaikh, Syeda Wania Hussain

Standalone backfill script — computes TVC, ERR, and RCS from existing
results without re-running the harness.  Patches each metrics_summary.json
in-place (adds new keys, never removes existing ones) and regenerates the
per-run TVC/ERR/RCS plots.  Also regenerates comparison_summary.json.

No LLM calls are made.  No results.json files are modified.

Usage
-----
    # Process all detected (model, mitigation, sample) combos:
    python Utils/compute_new_metrics.py

    # Restrict to a specific model slug:
    python Utils/compute_new_metrics.py --model google-gemma-2-9b-it

    # Preview without writing anything:
    python Utils/compute_new_metrics.py --dry-run
"""

import argparse
import json
import os
import sys

# ── Locate project root ───────────────────────────────────────────────────────
_UTILS   = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_UTILS)
_SYSTEM  = os.path.join(_PROJECT, "System")
sys.path.insert(0, _SYSTEM)

from metrics import compute_tvc, compute_err, compute_rcs  # noqa: E402
from plots   import plot_tvc, plot_err, plot_err_by_topic, plot_rcs  # noqa: E402

RESULTS_ROOT = os.path.join(_PROJECT, "results")
MITIGATIONS  = ["none", "m1", "m2", "m3"]


# ── Path helpers ──────────────────────────────────────────────────────────────

def _run_dir(mitigation: str, model_slug: str, sample_slug: str) -> str:
    return os.path.join(RESULTS_ROOT, mitigation, model_slug, sample_slug)


def _results_path(run_dir: str) -> str:
    return os.path.join(run_dir, "results.json")


def _logs_path(run_dir: str) -> str:
    return os.path.join(run_dir, "turn_logs.json")


def _metrics_path(run_dir: str) -> str:
    return os.path.join(run_dir, "metrics_summary.json")


def _plots_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "plots")


# ── Discovery ─────────────────────────────────────────────────────────────────

def _discover_runs(model_filter: str | None) -> list[tuple[str, str, str]]:
    """
    Scan results/ and return every (mitigation, model_slug, sample_slug)
    triple that has both results.json and turn_logs.json present.
    """
    runs: list[tuple[str, str, str]] = []
    for mit in MITIGATIONS:
        mit_dir = os.path.join(RESULTS_ROOT, mit)
        if not os.path.isdir(mit_dir):
            continue
        for model_slug in os.listdir(mit_dir):
            if model_filter and model_slug != model_filter:
                continue
            model_dir = os.path.join(mit_dir, model_slug)
            if not os.path.isdir(model_dir):
                continue
            for sample_slug in os.listdir(model_dir):
                run_d = _run_dir(mit, model_slug, sample_slug)
                if (os.path.isfile(_results_path(run_d))
                        and os.path.isfile(_logs_path(run_d))):
                    runs.append((mit, model_slug, sample_slug))
    return sorted(runs)


# ── Core processing ───────────────────────────────────────────────────────────

def _process_run(
    mitigation:  str,
    model_slug:  str,
    sample_slug: str,
    dry_run:     bool,
) -> dict:
    """
    Load existing data, compute TVC/ERR/RCS, patch metrics_summary.json,
    regenerate plots. Returns the new metric values dict.
    """
    run_d = _run_dir(mitigation, model_slug, sample_slug)

    with open(_results_path(run_d), encoding="utf-8") as f:
        results = json.load(f)
    with open(_logs_path(run_d), encoding="utf-8") as f:
        turn_logs = json.load(f)

    tvc_metrics = compute_tvc(results)
    err_metrics = compute_err(turn_logs)
    rcs_metrics = compute_rcs(turn_logs)

    new_keys = {
        "tvc_score"         : tvc_metrics["tvc_score"],
        "tvc_by_topic"      : tvc_metrics["tvc_by_topic"],
        "err_overall"       : err_metrics["err_overall"],
        "err_early"         : err_metrics["err_early"],
        "err_late"          : err_metrics["err_late"],
        "err_by_mitigation" : err_metrics["err_by_mitigation"],
        "err_by_topic"      : err_metrics["err_by_topic"],
        "rcs_score"         : rcs_metrics["rcs_score"],
        "rcs_by_mitigation" : rcs_metrics["rcs_by_mitigation"],
        "rcs_by_topic"      : rcs_metrics["rcs_by_topic"],
    }

    if not dry_run:
        # Patch metrics_summary.json (preserve all existing keys)
        metrics_p = _metrics_path(run_d)
        existing  = {}
        if os.path.isfile(metrics_p):
            with open(metrics_p, encoding="utf-8") as f:
                existing = json.load(f)

        # Remove old ADT keys if present (clean up legacy data)
        for old_key in ("mean_adt", "worst_adt", "best_adt", "adt_by_seen_topic"):
            existing.pop(old_key, None)

        existing.update(new_keys)

        with open(metrics_p, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        # Regenerate plots
        plots_d = _plots_dir(run_d)
        os.makedirs(plots_d, exist_ok=True)
        plot_tvc(tvc_metrics, plots_d)
        plot_err(err_metrics, plots_d)
        plot_err_by_topic(err_metrics, plots_d)
        plot_rcs(rcs_metrics, plots_d)

    return new_keys


# ── Comparison summary update ─────────────────────────────────────────────────

METRIC_KEYS = [
    ("attack_success_rate_pct",        "Attack Success Rate (%)"),
    ("mean_detection_latency_turns",   "Mean Detection Latency (turns)"),
    ("over_refusal_rate_pct",          "Over-Refusal Rate (%)"),
    ("context_length_drift_pct",       "Context-Length Drift (%)"),
    ("tvc_score",                      "Topic Vulnerability Consistency"),
    ("err_overall",                    "Escalation Resistance Rate (%)"),
    ("err_early",                      "ERR — Early Escalation (%)"),
    ("err_late",                       "ERR — Late Escalation (%)"),
    ("rcs_score",                      "Refusal Consistency Score"),
    ("attacks_caught",                 "Attacks Caught"),
    ("attacks_missed",                 "Attacks Missed"),
    ("false_positives",                "False Positives"),
]


def _update_comparison(
    model_slug:  str,
    sample_slug: str,
    dry_run:     bool,
):
    """Rebuild comparison_summary.json for a (model, sample) pair."""
    comp_dir  = os.path.join(RESULTS_ROOT, "comparison", model_slug, sample_slug)
    completed = [
        m for m in MITIGATIONS
        if os.path.isfile(
            _metrics_path(_run_dir(m, model_slug, sample_slug))
        )
    ]
    if not completed:
        return

    summary_table: dict[str, dict] = {}
    for m in completed:
        mp = _metrics_path(_run_dir(m, model_slug, sample_slug))
        with open(mp, encoding="utf-8") as f:
            metrics = json.load(f)
        summary_table[m] = {k: metrics.get(k) for k, _ in METRIC_KEYS}

    if dry_run:
        return

    os.makedirs(comp_dir, exist_ok=True)
    out_path = os.path.join(comp_dir, "comparison_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model"       : model_slug,
            "samples"     : sample_slug,
            "mitigations" : summary_table,
        }, f, indent=2)
    print(f"  [OK] comparison_summary.json -> {comp_dir}/")


# ── Console table ─────────────────────────────────────────────────────────────

def _print_table(
    model_slug:   str,
    sample_slug:  str,
    all_new_keys: dict[str, dict],   # mitigation → new_keys dict
):
    mits_present = sorted(all_new_keys.keys())
    col_w        = 10
    sep          = "=" * (45 + (col_w + 2) * len(mits_present))
    header_cols  = "  ".join(f"{m.upper():<{col_w}}" for m in mits_present)

    print(f"\n  Model: {model_slug}  |  Samples: {sample_slug}")
    print(sep)
    print(f"  {'METRIC':<43}  {header_cols}")
    print(sep)

    rows = [
        ("tvc_score",   "Topic Vulnerability Consistency"),
        ("err_overall", "Escalation Resistance Rate (%)"),
        ("err_early",   "  ERR — Early Escalation (%)"),
        ("err_late",    "  ERR — Late Escalation (%)"),
        ("rcs_score",   "Refusal Consistency Score"),
    ]
    for key, label in rows:
        vals = "  ".join(
            f"{str(all_new_keys.get(m, {}).get(key, 'N/A')):<{col_w}}"
            for m in mits_present
        )
        print(f"  {label:<43}  {vals}")
    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Backfill TVC, ERR, and RCS metrics from existing results.json "
            "and turn_logs.json files without re-running the harness. "
            "Patches metrics_summary.json in-place and regenerates plots."
        )
    )
    parser.add_argument(
        "--model", default=None, metavar="MODEL_SLUG",
        help="Restrict processing to this model slug (e.g. google-gemma-2-9b-it).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute metrics and print table but do NOT write any files.",
    )
    args = parser.parse_args()

    runs = _discover_runs(args.model)
    if not runs:
        print("[WARN] No completed runs found under results/.")
        return

    print(f"\nFound {len(runs)} run(s) to process{' (DRY RUN — no files written)' if args.dry_run else ''}.\n")

    # Group by (model_slug, sample_slug) for comparison summary
    pairs: dict[tuple[str, str], dict[str, dict]] = {}

    for mit, model_slug, sample_slug in runs:
        print(f"  Processing: {mit:4s} | {model_slug} | {sample_slug} ... ", end="", flush=True)
        try:
            new_keys = _process_run(mit, model_slug, sample_slug, args.dry_run)
            print(f"TVC={new_keys['tvc_score']}  ERR={new_keys['err_overall']}%  "
                  f"RCS={new_keys['rcs_score']}")
            pairs.setdefault((model_slug, sample_slug), {})[mit] = new_keys
        except Exception as exc:
            print(f"ERROR — {exc}")

    # Update comparison summaries and print table per (model, sample) pair
    for (model_slug, sample_slug), all_new_keys in sorted(pairs.items()):
        _update_comparison(model_slug, sample_slug, args.dry_run)
        _print_table(model_slug, sample_slug, all_new_keys)

    action = "Dry-run complete — no files modified." if args.dry_run else "Done. All metrics_summary.json files patched."
    print(action)


if __name__ == "__main__":
    main()
