"""
combine_datasets.py
===================
Merges separate short / medium / long dataset files into one
full_dataset.json that the harness expects.

Usage
-----
Put this script in your project root (same level as the datasets/ folder),
then run:

    python combine_datasets.py

Input files (edit paths below if yours are named differently):
    datasets/short_conversations.json
    datasets/medium_conversations.json
    datasets/long_conversations.json

Output:
    datasets/full_dataset.json
"""

import json
import os

# ── Edit these if your filenames are different ────────────────────────────────
INPUT_FILES = [
    "Datasets/short_context_length_dataset.json",
    "Datasets/medium_context_length_dataset.json",
    "Datasets/long_context_length_dataset.json",
]
OUTPUT_FILE = "Datasets/full_dataset.json"
# ─────────────────────────────────────────────────────────────────────────────

combined = []
seen_ids = set()
duplicates = []

for path in INPUT_FILES:
    if not os.path.exists(path):
        print(f"  [SKIP] Not found: {path}")
        continue

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # handle both a plain list and a dict wrapping a list
    if isinstance(data, dict):
        # e.g. {"conversations": [...]}  — grab the first list value
        data = next(v for v in data.values() if isinstance(v, list))

    before = len(combined)
    for convo in data:
        cid = convo.get("id", "")
        if cid in seen_ids:
            duplicates.append(cid)
            continue
        seen_ids.add(cid)
        combined.append(convo)

    added = len(combined) - before
    print(f"  [OK] {path:45s} → {added} conversations added  (total so far: {len(combined)})")

if duplicates:
    print(f"\n  [WARN] Skipped {len(duplicates)} duplicate IDs: {duplicates}")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print(f"\nDone. {len(combined)} conversations written to {OUTPUT_FILE}")

# ── Quick sanity check ────────────────────────────────────────────────────────
attack_count = sum(1 for c in combined if c.get("attack_type", "none") != "none")
benign_count = len(combined) - attack_count

short_count  = sum(1 for c in combined if len(c["turns"]) <= 8)
medium_count = sum(1 for c in combined if 9 <= len(c["turns"]) <= 14)
long_count   = sum(1 for c in combined if len(c["turns"]) > 14)

print(f"\nDataset breakdown (bucketed by TOTAL turns, USER + AI):")
print(f"  Attack conversations : {attack_count}")
print(f"  Benign conversations : {benign_count}")
print(f"  Short  (≤8  total turns) : {short_count}")
print(f"  Medium (9–14 total turns): {medium_count}")
print(f"  Long   (>14 total turns) : {long_count}")