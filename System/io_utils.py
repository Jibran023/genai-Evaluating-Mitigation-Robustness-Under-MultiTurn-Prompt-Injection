"""
io_utils.py
===========
File I/O utilities: saving results, checkpoints, and reproducibility notes.
No computation happens here — only reading and writing.

Functions
---------
load_dataset(path)
stratified_sample(dataset, n, seed, short_max, medium_max)
save_checkpoint(out_dir, results, turn_logs, failures)
save_metrics(out_dir, summary, cld_rows, cld_val)
save_run_info(out_dir, cfg_snapshot, summary)
file_md5(path)
print_summary(summary, cld_rows, cld_val)
"""

import json
import math
import os
import sys
import hashlib
import datetime
import platform
import random as _rnd


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversations from {path}")
    return data


# ── Stratified sample ───────────────────────────────────────────────────────────────────

def stratified_sample(
    dataset:    list[dict],
    n:          int,
    seed:       int = 42,
    short_max:  int = 8,
    medium_max: int = 14,
) -> list[dict]:
    """
    Return n conversations sampled proportionally across all (is_attack, length)
    strata, so the sample mirrors the composition of the full dataset.

    Example: if the full set is 75% attack / 25% benign and 40% short, then a
    sample of 20 will contain ~15 attack and ~5 benign, and ~8 short convos.

    The sample is deterministic for a given seed and always sorted by id.
    Returns the full dataset unchanged if n >= len(dataset).
    """
    if n >= len(dataset):
        return list(dataset)

    def _bucket(convo: dict) -> str:
        nt = len(convo.get("turns", []))
        if nt <= short_max:
            return "short"
        if nt <= medium_max:
            return "medium"
        return "long"

    def _is_attack(convo: dict) -> bool:
        return convo.get("attack_type", "none") != "none"

    # Build strata dict: {(is_attack, topic, bucket): [convo, ...]}
    strata: dict[tuple, list] = {}
    for c in dataset:
        key = (_is_attack(c), c.get("topic"), _bucket(c))
        strata.setdefault(key, []).append(c)

    total  = len(dataset)
    keys   = sorted(strata.keys())

    # Proportional floor allocation
    allocs = {k: math.floor(len(strata[k]) / total * n) for k in keys}
    assigned = sum(allocs.values())

    # Distribute leftover slots to strata with the largest fractional remainders
    remainders = sorted(
        keys,
        key=lambda k: (len(strata[k]) / total * n) - allocs[k],
        reverse=True,
    )
    for k in remainders[: n - assigned]:
        allocs[k] += 1

    # Sample within each stratum (deterministic via seeded RNG)
    rng = _rnd.Random(seed)
    sampled: list[dict] = []
    for k in keys:
        take = min(allocs[k], len(strata[k]))
        if take > 0:
            sampled.extend(rng.sample(strata[k], take))

    # Sort by id so order is stable across runs
    sampled.sort(key=lambda c: c.get("id", ""))
    return sampled


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    out_dir:   str,
    results:   list[dict],
    turn_logs: list[dict],
    failures:  list[dict],
):
    """
    Persist everything collected so far.
    Called after every conversation so a crash loses at most one run.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(out_dir, "turn_logs.json"), "w") as f:
        json.dump(turn_logs, f, indent=2)

    with open(os.path.join(out_dir, "failure_analysis.json"), "w") as f:
        json.dump(
            {
                "summary": {
                    "total_failures" : len(failures),
                    "missed_attacks" : sum(1 for x in failures if x["failure_type"] == "MISSED_ATTACK"),
                    "false_positives": sum(1 for x in failures if x["failure_type"] == "FALSE_POSITIVE"),
                },
                "failures": failures,
            },
            f,
            indent=2,
        )

    print(f"  [checkpoint] {len(results)} conversations saved -> {out_dir}/")


def load_checkpoint(out_dir: str) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load results, turn logs, and failures from an existing checkpoint.
    Returns (results, turn_logs, failures).
    """
    res_path  = os.path.join(out_dir, "results.json")
    log_path  = os.path.join(out_dir, "turn_logs.json")
    fail_path = os.path.join(out_dir, "failure_analysis.json")

    results   = []
    turn_logs = []
    failures  = []

    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as f:
            results = json.load(f)

    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            turn_logs = json.load(f)

    if os.path.exists(fail_path):
        with open(fail_path, encoding="utf-8") as f:
            data = json.load(f)
            failures = data.get("failures", [])

    return results, turn_logs, failures


# ── Metrics summary + CLD ─────────────────────────────────────────────────────

def save_metrics(
    out_dir:  str,
    summary:  dict,
    cld_rows: list[dict],
    cld_val:  float | str,
):
    os.makedirs(out_dir, exist_ok=True)
    payload = {**summary, "context_length_drift_pct": cld_val, "asr_by_length_group": cld_rows}
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)


# ── Reproducibility record ────────────────────────────────────────────────────

def save_run_info(out_dir: str, cfg_snapshot: dict, summary: dict):
    """
    cfg_snapshot should be a plain dict of whatever config values you want
    to record (model, temperature, mitigation, dataset path, etc.).
    """
    os.makedirs(out_dir, exist_ok=True)
    run_info = {
        "run_timestamp" : datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "platform"      : platform.platform(),
        **cfg_snapshot,
        "metrics"       : summary,
    }
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)


# ── File fingerprint ──────────────────────────────────────────────────────────

def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(
    summary:  dict,
    cld_rows: list[dict],
    cld_val:  float | str,
):
    print("\n========== RESULTS ==========")
    print(f"Total conversations   : {summary['total_conversations']}")
    print(f"Attack conversations  : {summary['attack_conversations']}")
    print(f"Benign conversations  : {summary['benign_conversations']}")
    print(f"Attacks caught        : {summary['attacks_caught']}")
    print(f"Attacks missed        : {summary['attacks_missed']}")
    print(f"Attack Success Rate   : {summary['attack_success_rate_pct']}%")
    print(f"Mean Detection Latency: {summary['mean_detection_latency_turns']} turns")
    print(f"Over-Refusal Rate     : {summary['over_refusal_rate_pct']}%")
    print(f"Context-Length Drift  : {cld_val}%")
    print(f"False positives       : {summary['false_positives']}")
    if "mean_adt" in summary:
        print(f"Mean ADT              : {summary['mean_adt']}%")
        print(f"Worst-Case ADT        : {summary['worst_adt']}%")
        print(f"Best-Case ADT         : {summary['best_adt']}%")
    print("=============================\n")
    if cld_rows:
        print("ASR by length group:")
        for r in cld_rows:
            print(f"  {r['length_group']:8s}: {r['asr_pct']}%  (n={r['n']})")
        print()
    if "adt_by_seen_topic" in summary:
        print("ADT by seen topic:")
        for seen_topic, stats in summary["adt_by_seen_topic"].items():
            print(
                f"  {seen_topic:25s}: seen={stats['asr_seen']}%  "
                f"unseen={stats['asr_unseen']}%  ADT={stats['adt']}%  "
                f"({stats['interpretation']})"
            )
        print()
