"""
run_all_mitigations.py
======================
Evaluating Mitigation Robustness Under Multi-Turn Prompt Injection
Authors : Jibran Shaikh, Syeda Wania Hussain

Runs the evaluation harness for every mitigation in sequence, then
produces cross-mitigation comparison plots.

Output structure
----------------
results/
  <mitigation>/
    <model_slug>/
      <sample_slug>/     ← "20", "30", "all_samples", …
        results.json
        turn_logs.json
        failure_analysis.json
        metrics_summary.json
        run_info.json
        plots/

  comparison/
    <model_slug>/
      <sample_slug>/
        mitigation_comparison.png
        adt_heatmap.png
        comparison_summary.json

Usage
-----
    # All 4 mitigations, 20 samples, NVIDIA model
    python Utils/run_all_mitigations.py --provider nvidia --model meta/llama-3.1-8b-instruct --limit 20

    # Full dataset (stored under all_samples/)
    python Utils/run_all_mitigations.py --provider nvidia --model meta/llama-3.1-8b-instruct

    # Skip mitigations already done
    python Utils/run_all_mitigations.py --provider nvidia --model meta/llama-3.1-8b-instruct --limit 20 --skip none m1

    # Only (re)generate comparison plots for a specific run
    python Utils/run_all_mitigations.py --plots-only --model meta/llama-3.1-8b-instruct --limit 20

    # Plots for ALL detected (model, sample) combinations
    python Utils/run_all_mitigations.py --plots-only --model all
"""

import argparse
import json
import os
import subprocess
import sys

# ── Locate project root (this script lives in Utils/) ────────────────────────
_UTILS   = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_UTILS)
_SYSTEM  = os.path.join(_PROJECT, "System")

sys.path.insert(0, _SYSTEM)

from plots  import plot_mitigation_comparison, plot_adt_heatmap  # noqa: E402
import config as _cfg                                            # noqa: E402

MITIGATIONS  = ["none", "m1", "m2", "m3"]
RESULTS_ROOT = os.path.join(_PROJECT, "results")
HARNESS      = os.path.join(_SYSTEM, "harness.py")

DEFAULT_MODEL = _cfg.MODEL
DEFAULT_SLUG  = _cfg.MODEL_SLUG


# ── Path helpers ──────────────────────────────────────────────────────────────

def _sample_slug(max_samples: int | None) -> str:
    """Convert a --limit value to the directory name used in results/."""
    return str(max_samples) if max_samples is not None else "all_samples"


def _comparison_dir(model_slug: str, sample_slug: str) -> str:
    return os.path.join(RESULTS_ROOT, "comparison", model_slug, sample_slug)


def _result_file(mitigation: str, model_slug: str, sample_slug: str) -> str:
    return os.path.join(
        RESULTS_ROOT, mitigation, model_slug, sample_slug, "results.json"
    )


def _metrics_file(mitigation: str, model_slug: str, sample_slug: str) -> str:
    return os.path.join(
        RESULTS_ROOT, mitigation, model_slug, sample_slug, "metrics_summary.json"
    )


# ── Model / sample discovery ──────────────────────────────────────────────────

def _detect_model_sample_pairs() -> list[tuple[str, str]]:
    """
    Scan results/none/<model_slug>/<sample_slug>/results.json and return
    every (model_slug, sample_slug) pair that has completed results.
    Used when --plots-only is given without --model.
    """
    base = os.path.join(RESULTS_ROOT, "none")
    if not os.path.isdir(base):
        return []
    pairs: list[tuple[str, str]] = []
    for model_entry in os.listdir(base):
        model_path = os.path.join(base, model_entry)
        if not os.path.isdir(model_path):
            continue
        for sample_entry in os.listdir(model_path):
            rfile = os.path.join(model_path, sample_entry, "results.json")
            if os.path.isfile(rfile):
                pairs.append((model_entry, sample_entry))
    return sorted(pairs)


# ── Harness runner ────────────────────────────────────────────────────────────

def _run_harness(
    mitigation:  str,
    model:       str | None = None,
    provider:    str | None = None,
    max_samples: int | None = None,
) -> bool:
    """
    Invoke harness.py as a subprocess with env vars controlling behaviour.
    Streams output live to the console.
    Returns True on success, False on failure.
    """
    env = {**os.environ, "EVAL_MITIGATION": mitigation}

    if provider:
        env["EVAL_PROVIDER"] = provider

    if model:
        env["EVAL_MODEL"] = model

    if max_samples is not None:
        env["EVAL_MAX_SAMPLES"] = str(max_samples)

    slug       = (model or DEFAULT_MODEL).replace("/", "-")
    prov_note  = f" [{provider or 'groq'}]"
    limit_note = f" | first {max_samples} samples" if max_samples else ""
    print(f"\n{'='*65}")
    print(f"  MITIGATION: {mitigation.upper():6s}  MODEL: {slug}{prov_note}{limit_note}")
    print(f"{'='*65}\n")

    result = subprocess.run(
        [sys.executable, HARNESS],
        env=env,
        cwd=_PROJECT,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Harness exited with code {result.returncode} "
              f"for mitigation '{mitigation}'.")
        return False
    return True


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_results(
    mitigation: str, model_slug: str, sample_slug: str
) -> list[dict] | None:
    path = _result_file(mitigation, model_slug, sample_slug)
    if not os.path.exists(path):
        print(f"[WARN] Results not found: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_metrics(
    mitigation: str, model_slug: str, sample_slug: str
) -> dict | None:
    path = _metrics_file(mitigation, model_slug, sample_slug)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Comparison plots & summary ────────────────────────────────────────────────

def build_comparison_plots(
    mitigations_done: list[str],
    model_slug:       str,
    sample_slug:      str,
):
    """
    Load each completed run's results.json and generate:
      1. Grouped ASR bar chart across length groups x mitigations
      2. ADT heatmap (ASR per topic per mitigation)
      3. comparison_summary.json + console table
    All outputs go to results/comparison/<model_slug>/<sample_slug>/.
    """
    comp_dir = _comparison_dir(model_slug, sample_slug)
    os.makedirs(comp_dir, exist_ok=True)

    all_results: dict[str, list[dict]] = {}
    for m in mitigations_done:
        rows = _load_results(m, model_slug, sample_slug)
        if rows:
            all_results[m] = rows

    if not all_results:
        print(f"[WARN] No results found for model '{model_slug}' "
              f"(samples: {sample_slug}) — skipping comparison plots.")
        return

    # ── 1. Mitigation comparison (grouped bar) ────────────────────────────────
    comparison_data = []
    for m, rows in all_results.items():
        df_groups: dict[str, list] = {}
        for r in rows:
            if r.get("is_attack"):
                g = r.get("length_group", "short")
                df_groups.setdefault(g, []).append(r)
        for grp, grp_rows in df_groups.items():
            total     = len(grp_rows)
            succeeded = sum(1 for r in grp_rows if r.get("attack_succeeded"))
            asr_pct   = round(succeeded / total * 100, 1) if total else 0.0
            comparison_data.append({
                "mitigation"   : m,
                "length_group" : grp,
                "asr_pct"      : asr_pct,
            })

    plot_mitigation_comparison(comparison_data, comp_dir)
    print(f"  [OK] mitigation_comparison.png  →  {comp_dir}/")

    # ── 2. ADT heatmap (ASR per topic per mitigation) ─────────────────────────
    adt_data: dict[str, dict[str, float]] = {}
    for m, rows in all_results.items():
        topic_buckets: dict[str, list] = {}
        for r in rows:
            if r.get("is_attack"):
                topic_buckets.setdefault(r.get("topic", "unknown"), []).append(r)
        adt_data[m] = {
            topic: round(
                sum(1 for r in trows if r.get("attack_succeeded"))
                / len(trows) * 100, 1
            )
            for topic, trows in topic_buckets.items()
        }

    plot_adt_heatmap(adt_data, comp_dir)
    print(f"  [OK] adt_heatmap.png            →  {comp_dir}/")

    # ── 3. Side-by-side summary JSON ─────────────────────────────────────────
    METRIC_KEYS = [
        ("attack_success_rate_pct",       "Attack Success Rate (%)"),
        ("mean_detection_latency_turns",   "Mean Detection Latency (turns)"),
        ("over_refusal_rate_pct",          "Over-Refusal Rate (%)"),
        ("context_length_drift_pct",       "Context-Length Drift (%)"),
        ("attacks_caught",                 "Attacks Caught"),
        ("attacks_missed",                 "Attacks Missed"),
        ("false_positives",                "False Positives"),
    ]

    summary_table: dict[str, dict] = {}
    for m in mitigations_done:
        metrics = _load_metrics(m, model_slug, sample_slug)
        if metrics:
            summary_table[m] = {k: metrics.get(k) for k, _ in METRIC_KEYS}

    out_path = os.path.join(comp_dir, "comparison_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model"       : model_slug,
            "samples"     : sample_slug,
            "mitigations" : summary_table,
        }, f, indent=2)
    print(f"  [OK] comparison_summary.json    →  {comp_dir}/")

    # ── Console table ─────────────────────────────────────────────────────────
    col_w       = 8
    header_cols = "  ".join(f"{m.upper():<{col_w}}" for m in mitigations_done)
    sep         = "=" * (42 + (col_w + 2) * len(mitigations_done))

    print(f"\n  Model: {model_slug}  |  Samples: {sample_slug}")
    print(sep)
    print(f"  {'METRIC':<40}  {header_cols}")
    print(sep)
    for key, label in METRIC_KEYS:
        vals = "  ".join(
            f"{str(summary_table.get(m, {}).get(key, 'N/A')):<{col_w}}"
            for m in mitigations_done
        )
        print(f"  {label:<40}  {vals}")
    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run all mitigation evaluations for a given model and generate "
            "comparison plots.  Results are isolated by (model, sample_count) "
            "so different runs never overwrite each other."
        )
    )
    parser.add_argument(
        "--skip", nargs="+", metavar="MITIGATION",
        choices=MITIGATIONS, default=[],
        help="Mitigations to skip (e.g. --skip none m1 if already run).",
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help=(
            "Skip harness runs; only (re)generate comparison plots from "
            "existing results. Combine with --limit to target a specific "
            "sample-count run. Use --model all to regenerate for everything."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help=(
            "Cap each run to N conversations (stratified sampling preserves "
            "attack/benign and short/medium/long ratios). "
            "Results land in results/<mit>/<model>/<N>/. "
            "Omitting --limit uses the full dataset → all_samples/."
        ),
    )
    parser.add_argument(
        "--model", default=None, metavar="MODEL_NAME",
        help=(
            f"Model to use (default: {DEFAULT_MODEL}). "
            "Use 'all' with --plots-only to generate plots for every "
            "detected (model, samples) combination."
        ),
    )
    parser.add_argument(
        "--provider", default=None, metavar="PROVIDER",
        choices=["groq", "gemini", "nvidia"],
        help="API provider: groq | gemini | nvidia (default: groq).",
    )
    args = parser.parse_args()

    # ── Resolve slugs ─────────────────────────────────────────────────────────
    requested_model = args.model or DEFAULT_MODEL
    model_slug      = requested_model.replace("/", "-")
    s_slug          = _sample_slug(args.limit)   # "20", "30", or "all_samples"

    # ── Announce run parameters ───────────────────────────────────────────────
    if not args.plots_only:
        print(f"\nProvider  : {args.provider or 'groq'}")
        print(f"Model     : {requested_model}")
        print(f"Samples   : {s_slug}")
        if args.limit:
            print(f"Limit     : {args.limit} conversations per mitigation "
                  "(stratified — preserves dataset composition)")
        print(f"Skip      : {args.skip or 'none'}\n")

    # ── Run harness for each mitigation ──────────────────────────────────────
    if not args.plots_only:
        to_run = [m for m in MITIGATIONS if m not in args.skip]
        failed = []
        for mitigation in to_run:
            ok = _run_harness(
                mitigation,
                provider    = args.provider,
                model       = requested_model if args.model else None,
                max_samples = args.limit,
            )
            if not ok:
                failed.append(mitigation)

        if failed:
            print(f"\n[WARN] The following mitigations failed: {failed}")
            print("       Their results may be incomplete or missing.\n")
    else:
        print("\n[--plots-only] Skipping harness runs, loading existing results.\n")

    # ── Generate comparison plots ─────────────────────────────────────────────
    if args.plots_only and (args.model or "").lower() == "all":
        # Generate plots for every (model, sample) pair found on disk
        pairs = _detect_model_sample_pairs()
        if not pairs:
            print("[WARN] No completed runs found under results/none/.")
            return
        print(f"Detected (model, samples) pairs: {pairs}\n")
        for slug, ss in pairs:
            completed = [m for m in MITIGATIONS
                         if os.path.exists(_result_file(m, slug, ss))]
            if completed:
                print(f"\n--- Model: {slug}  |  Samples: {ss} ---")
                build_comparison_plots(completed, slug, ss)

    elif args.plots_only and not args.model:
        # Auto-detect: if exactly one pair found, use it; otherwise list them
        pairs = _detect_model_sample_pairs()
        if len(pairs) == 1:
            slug, ss = pairs[0]
            print(f"[Auto-detect] Using model: {slug}  samples: {ss}\n")
            completed = [m for m in MITIGATIONS
                         if os.path.exists(_result_file(m, slug, ss))]
            build_comparison_plots(completed, slug, ss)
        elif len(pairs) > 1:
            print(f"[INFO] Multiple (model, samples) pairs found:")
            for p in pairs:
                print(f"       model={p[0]}  samples={p[1]}")
            print("\n       Re-run with --model <name> [--limit N] "
                  "or --model all\n")
        else:
            print("[WARN] No completed runs found. Run the harness first.")

    else:
        # Normal case: single model + sample slug
        completed = [m for m in MITIGATIONS
                     if os.path.exists(_result_file(m, model_slug, s_slug))]
        if completed:
            print(f"\nGenerating comparison plots for: {completed}  "
                  f"(model: {model_slug}  samples: {s_slug})")
            build_comparison_plots(completed, model_slug, s_slug)
        else:
            print(f"\n[WARN] No completed runs found for "
                  f"model='{model_slug}'  samples='{s_slug}'.")

    print("\nDone. All outputs in:")
    print(f"  {RESULTS_ROOT}/")
    print(f"  {_comparison_dir(model_slug, s_slug)}/\n")


if __name__ == "__main__":
    main()
