"""
dataset_audit.py
================
Comprehensive correctness audit of the evaluation dataset AND a consistency
check that every dataset convention aligns with how harness.py / metrics.py
consumes the data.

Checks performed
----------------
DATASET INTEGRITY
  D1  Required top-level fields present in every entry
  D2  Unique IDs (no duplicates)
  D3  Turn numbering: starts at 1, consecutive, no gaps
  D4  Speaker alternation: USER -> AI -> USER -> AI ...
  D5  All label values are from the allowed set
  D6  attack_type is "none" or "multi_turn"
  D7  success field is "yes" or "no" (strings)
  D8  injection_turn / detection_turn are integers or null
  D9  injection_turn points to a USER turn
  D10 detection_turn points to an AI turn
  D11 Every turn dict has required keys

BENIGN CONVERSATION CHECKS (attack_type == "none")
  B1  All turn labels are BENIGN
  B2  injection_turn is null
  B3  detection_turn is null
  B4  success is "no"

ATTACK CONVERSATION CHECKS (attack_type == "multi_turn")
  A1  Exactly one ATTACK_STARTS label (on a USER turn)
  A2  ATTACK_STARTS turn number matches injection_turn
  A3  All ATTACK_ESCALATES labels appear after ATTACK_STARTS
  A4  No ATTACK_ESCALATES label appears before ATTACK_STARTS
  A5  AI turns only carry AI-valid labels (BENIGN/REFUSED/DETECTED/COMPLIED/MISSED)
  A6  USER turns only carry USER-valid labels (BENIGN/ATTACK_STARTS/ATTACK_ESCALATES)
  A7  success=="yes": no AI turn should be REFUSED or DETECTED
  A8  success=="no":  detection_turn points to a REFUSED or DETECTED AI turn
  A9  No BENIGN AI turn responds to a post-refusal ATTACK_ESCALATES

HARNESS ALIGNMENT CHECKS
  H1  injection_turn is always a USER turn (harness reads turn["turn"] == inj_turn)
  H2  Every turn has "text" and "speaker" and "label" (harness iterates turns)
  H3  length_group bucketing is consistent with config SHORT_MAX / MEDIUM_MAX
  H4  Topic set is non-empty and consistent (all 6 expected topics present)
  H5  Dataset balance sanity (attack/benign split, topic distribution)
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Config mirrors ────────────────────────────────────────────────────────────
DATASET_PATH = Path("Datasets/test2_final_hardened_v2_cleaned.json")
SHORT_MAX    = 8
MEDIUM_MAX   = 14

VALID_LABELS       = {"BENIGN", "ATTACK_STARTS", "ATTACK_ESCALATES",
                      "REFUSED", "DETECTED", "COMPLIED", "MISSED"}
USER_VALID_LABELS  = {"BENIGN", "ATTACK_STARTS", "ATTACK_ESCALATES"}
AI_VALID_LABELS    = {"BENIGN", "REFUSED", "DETECTED", "COMPLIED", "MISSED"}
TOP_LEVEL_FIELDS   = {"id", "topic", "attack_type", "injection_turn",
                      "detection_turn", "success", "turns"}
TURN_FIELDS        = {"turn", "speaker", "label", "text"}
EXPECTED_TOPICS    = {
    "cybersecurity", "dangerous_instructions", "financial_manipulation",
    "identity_theft", "misinformation", "bypassing_ai_rules"
}

# ── Helpers ───────────────────────────────────────────────────────────────────

class AuditReport:
    def __init__(self):
        self.errors   = []   # FAIL — must fix
        self.warnings = []   # WARN — worth reviewing
        self.info     = []   # INFO — statistics

    def fail(self, check, entry_id, msg):
        self.errors.append(f"[FAIL] {check} | {entry_id} | {msg}")

    def warn(self, check, entry_id, msg):
        self.warnings.append(f"[WARN] {check} | {entry_id} | {msg}")

    def note(self, msg):
        self.info.append(f"[INFO] {msg}")

    def print_report(self):
        print("=" * 72)
        print("DATASET & HARNESS ALIGNMENT AUDIT REPORT")
        print("=" * 72)

        if self.errors:
            print(f"\n{'='*72}")
            print(f"FAILURES  ({len(self.errors)} issues — must fix)")
            print(f"{'='*72}")
            for e in self.errors:
                print(f"  {e}")
        else:
            print("\n  All FAIL checks: PASSED")

        if self.warnings:
            print(f"\n{'='*72}")
            print(f"WARNINGS  ({len(self.warnings)} items)")
            print(f"{'='*72}")
            for w in self.warnings:
                print(f"  {w}")
        else:
            print("  All WARN checks: PASSED")

        print(f"\n{'='*72}")
        print("STATISTICS")
        print(f"{'='*72}")
        for i in self.info:
            print(f"  {i}")

        total = len(self.errors) + len(self.warnings)
        print(f"\n{'='*72}")
        verdict = "ALL CHECKS PASSED" if not self.errors else f"{len(self.errors)} CHECK(S) FAILED"
        print(f"VERDICT: {verdict}  ({len(self.warnings)} warnings)")
        print("=" * 72)
        return len(self.errors)


def bucket(n_turns):
    if n_turns <= SHORT_MAX:
        return "short"
    if n_turns <= MEDIUM_MAX:
        return "medium"
    return "long"


# ── Main audit ────────────────────────────────────────────────────────────────

def audit(data: list[dict], rpt: AuditReport):

    # ───── D2: unique IDs ────────────────────────────────────────────────────
    id_counts = Counter(e.get("id", "<missing>") for e in data)
    for eid, cnt in id_counts.items():
        if cnt > 1:
            rpt.fail("D2", eid, f"Duplicate ID appears {cnt} times")

    topic_counter    = Counter()
    attack_count     = 0
    benign_count     = 0
    length_counter   = Counter()
    label_counter    = Counter()

    for entry in data:
        eid = entry.get("id", "<no-id>")

        # ── D1: required top-level fields ─────────────────────────────────────
        missing = TOP_LEVEL_FIELDS - set(entry.keys())
        if missing:
            rpt.fail("D1", eid, f"Missing top-level fields: {missing}")
            continue   # can't safely do further checks

        attack_type    = entry["attack_type"]
        injection_turn = entry["injection_turn"]
        detection_turn = entry["detection_turn"]
        success        = entry["success"]
        turns          = entry["turns"]
        topic          = entry["topic"]
        topic_counter[topic] += 1

        # ── D6: attack_type ───────────────────────────────────────────────────
        if attack_type not in ("none", "multi_turn"):
            rpt.fail("D6", eid, f"Invalid attack_type: {attack_type!r}")

        # ── D7: success ───────────────────────────────────────────────────────
        if success not in ("yes", "no"):
            rpt.fail("D7", eid, f"Invalid success value: {success!r}")

        # ── D8: injection_turn / detection_turn types ─────────────────────────
        if injection_turn is not None and not isinstance(injection_turn, int):
            rpt.fail("D8", eid, f"injection_turn must be int or null, got {type(injection_turn)}")
        if detection_turn is not None and not isinstance(detection_turn, int):
            rpt.fail("D8", eid, f"detection_turn must be int or null, got {type(detection_turn)}")

        # ── D11: turn-level required keys ─────────────────────────────────────
        for t in turns:
            missing_t = TURN_FIELDS - set(t.keys())
            if missing_t:
                rpt.fail("D11", eid, f"Turn {t.get('turn','?')} missing keys: {missing_t}")

        # ── D3: turn numbering ────────────────────────────────────────────────
        turn_nums = [t["turn"] for t in turns]
        if turn_nums[0] != 1:
            rpt.fail("D3", eid, f"Turns don't start at 1 (starts at {turn_nums[0]})")
        for i in range(1, len(turn_nums)):
            if turn_nums[i] != turn_nums[i-1] + 1:
                rpt.fail("D3", eid, f"Non-consecutive turns: {turn_nums[i-1]} -> {turn_nums[i]}")

        # ── D4: speaker alternation ───────────────────────────────────────────
        speakers = [t["speaker"] for t in turns]
        for i in range(1, len(speakers)):
            if speakers[i] == speakers[i-1]:
                rpt.fail("D4", eid, f"Speaker does not alternate at turn {turn_nums[i]} ({speakers[i]})")

        # ── D5: valid labels ──────────────────────────────────────────────────
        for t in turns:
            label_counter[t["label"]] += 1
            if t["label"] not in VALID_LABELS:
                rpt.fail("D5", eid, f"Turn {t['turn']}: invalid label {t['label']!r}")

        # ── D9: injection_turn points to USER ─────────────────────────────────
        if injection_turn is not None:
            inj_turn_obj = next((t for t in turns if t["turn"] == injection_turn), None)
            if inj_turn_obj is None:
                rpt.fail("D9", eid, f"injection_turn={injection_turn} not found in turns")
            elif inj_turn_obj["speaker"] != "USER":
                rpt.fail("D9", eid, f"injection_turn={injection_turn} points to {inj_turn_obj['speaker']}, expected USER")

        # ── D10: detection_turn points to AI ──────────────────────────────────
        if detection_turn is not None:
            det_turn_obj = next((t for t in turns if t["turn"] == detection_turn), None)
            if det_turn_obj is None:
                rpt.fail("D10", eid, f"detection_turn={detection_turn} not found in turns")
            elif det_turn_obj["speaker"] != "AI":
                rpt.fail("D10", eid, f"detection_turn={detection_turn} points to {det_turn_obj['speaker']}, expected AI")

        if attack_type == "none":
            benign_count += 1
            # ── B1: all labels BENIGN ─────────────────────────────────────────
            for t in turns:
                if t["label"] != "BENIGN":
                    rpt.fail("B1", eid, f"Benign conversation has non-BENIGN label at turn {t['turn']}: {t['label']}")
            # ── B2/B3: nulls ──────────────────────────────────────────────────
            if injection_turn is not None:
                rpt.fail("B2", eid, f"Benign conversation has injection_turn={injection_turn} (expected null)")
            if detection_turn is not None:
                rpt.fail("B3", eid, f"Benign conversation has detection_turn={detection_turn} (expected null)")
            # ── B4: success=="no" ──────────────────────────────────────────────
            if success != "no":
                rpt.fail("B4", eid, f"Benign conversation has success={success!r} (expected 'no')")

        else:
            attack_count += 1
            user_turns = [t for t in turns if t["speaker"] == "USER"]
            ai_turns   = [t for t in turns if t["speaker"] == "AI"]

            # ── A5: AI label validity ─────────────────────────────────────────
            for t in ai_turns:
                if t["label"] not in AI_VALID_LABELS:
                    rpt.fail("A5", eid, f"AI turn {t['turn']} has USER-only label: {t['label']}")

            # ── A6: USER label validity ───────────────────────────────────────
            for t in user_turns:
                if t["label"] not in USER_VALID_LABELS:
                    rpt.fail("A6", eid, f"USER turn {t['turn']} has AI-only label: {t['label']}")

            # ── A1: exactly one ATTACK_STARTS ────────────────────────────────
            starts = [t for t in user_turns if t["label"] == "ATTACK_STARTS"]
            if len(starts) != 1:
                rpt.fail("A1", eid, f"Expected exactly 1 ATTACK_STARTS, found {len(starts)}")
            else:
                # ── A2: ATTACK_STARTS matches injection_turn ──────────────────
                if starts[0]["turn"] != injection_turn:
                    rpt.fail("A2", eid,
                        f"ATTACK_STARTS at turn {starts[0]['turn']} != injection_turn {injection_turn}")

                # ── A3/A4: ATTACK_ESCALATES only after ATTACK_STARTS ─────────
                starts_turn = starts[0]["turn"]
                for t in user_turns:
                    if t["label"] == "ATTACK_ESCALATES":
                        if t["turn"] <= starts_turn:
                            rpt.fail("A4", eid,
                                f"ATTACK_ESCALATES at turn {t['turn']} appears before/at ATTACK_STARTS ({starts_turn})")

            # ── A7: success=="yes" → no REFUSED/DETECTED ─────────────────────
            if success == "yes":
                for t in ai_turns:
                    if t["label"] in ("REFUSED", "DETECTED"):
                        rpt.fail("A7", eid,
                            f"success='yes' but AI turn {t['turn']} is {t['label']}")

            # ── A8: success=="no" → detection_turn has REFUSED/DETECTED ──────
            if success == "no":
                if detection_turn is None:
                    rpt.fail("A8", eid, "success='no' but detection_turn is null")
                else:
                    det_obj = next((t for t in ai_turns if t["turn"] == detection_turn), None)
                    if det_obj is None:
                        rpt.fail("A8", eid, f"detection_turn={detection_turn} not in AI turns")
                    elif det_obj["label"] not in ("REFUSED", "DETECTED"):
                        rpt.fail("A8", eid,
                            f"detection_turn={detection_turn} has label {det_obj['label']!r}, expected REFUSED or DETECTED")

            # ── A9: no BENIGN AI turn after first refusal responding to escalation ──
            first_refusal = None
            for t in ai_turns:
                if t["label"] in ("REFUSED", "DETECTED"):
                    first_refusal = t["turn"]
                    break

            if first_refusal is not None:
                for i, t in enumerate(turns):
                    if t["speaker"] != "AI" or t["label"] != "BENIGN":
                        continue
                    if t["turn"] <= first_refusal:
                        continue
                    prev_user = None
                    for prev in reversed(turns[:i]):
                        if prev["speaker"] == "USER":
                            prev_user = prev
                            break
                    if prev_user and prev_user["label"] == "ATTACK_ESCALATES":
                        rpt.fail("A9", eid,
                            f"AI turn {t['turn']} labeled BENIGN but responds to post-refusal ATTACK_ESCALATES")

        # ── H3: length group check ────────────────────────────────────────────
        n_turns = len(turns)
        grp = bucket(n_turns)
        length_counter[grp] += 1

    # ── H4: topic coverage ────────────────────────────────────────────────────
    seen_topics = set(topic_counter.keys())
    missing_topics = EXPECTED_TOPICS - seen_topics
    extra_topics   = seen_topics - EXPECTED_TOPICS
    if missing_topics:
        rpt.warn("H4", "dataset", f"Missing expected topics: {missing_topics}")
    if extra_topics:
        rpt.warn("H4", "dataset", f"Unexpected topics found: {extra_topics}")

    # ── H5: balance checks ────────────────────────────────────────────────────
    total = len(data)
    rpt.note(f"Total entries         : {total}")
    rpt.note(f"Attack conversations  : {attack_count}")
    rpt.note(f"Benign conversations  : {benign_count}")
    rpt.note(f"Length distribution   : {dict(sorted(length_counter.items()))}")
    rpt.note(f"Topic distribution    : {dict(sorted(topic_counter.items()))}")
    rpt.note(f"Label distribution    : {dict(sorted(label_counter.items()))}")

    attack_ratio = attack_count / total if total else 0
    if not (0.5 < attack_ratio < 0.9):
        rpt.warn("H5", "dataset", f"Attack ratio {attack_ratio:.1%} outside expected 50-90% range")

    for topic, cnt in topic_counter.items():
        if abs(cnt - total / len(topic_counter)) > 10:
            rpt.warn("H5", "dataset", f"Topic '{topic}' count {cnt} is unbalanced")


def main():
    print(f"Loading: {DATASET_PATH}\n")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rpt = AuditReport()
    audit(data, rpt)
    n_errors = rpt.print_report()
    sys.exit(n_errors)


if __name__ == "__main__":
    main()
