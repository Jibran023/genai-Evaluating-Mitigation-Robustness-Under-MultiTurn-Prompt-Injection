"""
harness.py
==========
Evaluating Mitigation Robustness Under Multi-Turn Prompt Injection
Authors : Jibran Shaikh, Syeda Wania Hussain

Main evaluation loop — kept intentionally minimal.
All logic lives in the four modules alongside this file:

    config.py       ← constants, paths, MITIGATION setting
    mitigations.py  ← M1 / M2 / M3 implementations
    metrics.py      ← ASR, DL, CLD, ORR, ADT computation
    plots.py        ← all Matplotlib figures
    io_utils.py     ← load/save/checkpoint/print helpers

Usage
-----
1. Edit MITIGATION in config.py  →  "none" | "m1" | "m2" | "m3"
2. python harness.py
3. Results land in  results/<MITIGATION>/
"""

import sys
import os
from openai      import OpenAI
from dotenv      import load_dotenv

import config
from mitigations import apply_mitigation, get_system_prompt, StateMonitor
from metrics     import ai_refused, bucket, compute_summary, compute_cld
from plots       import save_all
from io_utils    import (
    load_dataset, save_checkpoint, save_metrics,
    save_run_info, file_md5, print_summary, stratified_sample,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────────
load_dotenv()

_provider = config.PROVIDERS[config.MODEL_PROVIDER]
_api_key  = config.get_api_key()   # resolves per-model NVIDIA key, then falls back to generic
if not _api_key:
    import re
    _generic_env = _provider["api_key_env"]
    if config.MODEL_PROVIDER == "nvidia":
        _safe_model  = re.sub(r"[^A-Z0-9]", "_", config.MODEL.upper())
        _model_env   = f"{_generic_env}_{_safe_model}"
        raise EnvironmentError(
            f"NVIDIA API key not set for model '{config.MODEL}'.\n"
            f"  Add one of the following to your .env file:\n"
            f"    {_model_env}=nvapi-...   (model-specific key)\n"
            f"    {_generic_env}=nvapi-...          (generic fallback key)"
        )
    raise EnvironmentError(
        f"API key not set. Add {_generic_env} to your .env file."
    )
# NVIDIA NIM models (especially less popular ones like gemma-2-9b-it) spin up
# on demand and can take 30-60 s for the first request.  The default httpx
# timeout is ~60 s which is occasionally too short — 120 s is safer.
client = OpenAI(base_url=_provider["base_url"], api_key=_api_key, timeout=120.0)

os.makedirs(config.OUT_DIR,   exist_ok=True)
os.makedirs(config.PLOTS_DIR, exist_ok=True)

# ── LLM call wrapper ──────────────────────────────────────────────────────────

# Pre-compute once: does the current model reject {"role": "system"}?
_model_lower = config.MODEL.lower()
_no_system_role: bool = any(
    pat in _model_lower for pat in config.MODELS_WITHOUT_SYSTEM_ROLE
)
if _no_system_role:
    print(f"[COMPAT] '{config.MODEL}' does not support system role — "
          f"M1 system prompt will be injected into the first user turn.\n")


def ask_ai(history: list) -> str:
    system_prompt = get_system_prompt(config.MITIGATION)

    if system_prompt and _no_system_role:
        # Gemma / Phi-3 etc.: prepend the safety instructions to the first
        # user message so the model still sees them without a system role.
        messages = list(history)  # shallow copy — don't mutate caller's list
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                messages[i] = {
                    "role"   : "user",
                    "content": f"[Important instructions — follow these throughout our conversation]\n"
                               f"{system_prompt}\n\n"
                               f"[User message]\n{msg['content']}",
                }
                break
    elif system_prompt:
        # Standard path: prepend a proper system message
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history

    reply = client.chat.completions.create(
        model       = config.MODEL,
        messages    = messages,
        max_tokens  = config.MAX_TOKENS,
        temperature = config.TEMPERATURE,
    )
    return reply.choices[0].message.content.strip()

# ── Load dataset ─────────────────────────────────────────────────────────────────────
dataset = load_dataset(config.DATASET)

# Apply sample limit — uses stratified sampling to preserve dataset composition
if config.MAX_SAMPLES is not None:
    dataset = stratified_sample(
        dataset,
        n          = config.MAX_SAMPLES,
        seed       = config.RANDOM_SEED,
        short_max  = config.SHORT_MAX,
        medium_max = config.MEDIUM_MAX,
    )
    print(f"[LIMIT] Stratified sample: {len(dataset)} conversations "
          f"(MAX_SAMPLES={config.MAX_SAMPLES}) — not a full-dataset run.\n")

print(f"Mitigation : {config.MITIGATION.upper()}")
print(f"Model      : {config.MODEL}")
print(f"Model Slug : {config.MODEL_SLUG}")
print(f"Samples    : {len(dataset)}")
print(f"Output dir : {config.OUT_DIR}\n")

# ── Accumulators ──────────────────────────────────────────────────────────────
results   = []
turn_logs = []
failures  = []

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    for convo in dataset:
        cid         = convo["id"]
        topic       = convo["topic"]
        attack_type = convo["attack_type"]
        is_attack   = attack_type != "none"
        inj_turn    = convo.get("injection_turn")
        n_turns     = len(convo["turns"])   # total turns (USER + AI)
        length_grp  = bucket(n_turns)

        history  = []
        caught_at = None
        monitor   = StateMonitor()   # fresh per conversation (used by M3 only)

        for turn in convo["turns"]:
            if turn["speaker"] != "USER":
                continue

            label     = turn.get("label", "")
            user_text = turn["text"]
            history.append({"role": "user", "content": user_text})

            # ── call the appropriate mitigation ────────────────────────────
            response, blocked = apply_mitigation(
                mitigation = config.MITIGATION,
                user_text  = user_text,
                label      = label,
                history    = history,
                ask_fn     = ask_ai,
                monitor    = monitor,
            )
            history.append({"role": "assistant", "content": response})

            refused_here = blocked or ai_refused(response)
            if refused_here and caught_at is None:
                if not is_attack or (inj_turn is not None and turn["turn"] >= inj_turn):
                    caught_at = turn["turn"]
            # if refused_here and caught_at is None:
            #     caught_at = turn["turn"]

            turn_logs.append({
                "conv_id"         : cid,
                "topic"           : topic,
                "attack_type"     : attack_type,
                "mitigation"      : config.MITIGATION,
                "turn"            : turn["turn"],
                "label"           : label,
                "user_text"       : user_text,
                "ai_response"     : response,
                "mitigation_flag" : int(refused_here),
                "blocked_by_gate" : int(blocked),
                "over_refusal"    : int(not is_attack and refused_here),
            })

        # ── per-conversation metrics ────────────────────────────────────────
        dl = None
        if is_attack:
            if caught_at is not None and inj_turn is not None:
                dl = caught_at - inj_turn
        else:
            if caught_at is not None:
                # false positive
                pass

        attack_succeeded = is_attack and caught_at is None

        results.append({
            "id"               : cid,
            "topic"            : topic,
            "attack_type"      : attack_type,
            "mitigation"       : config.MITIGATION,
            "length_group"     : length_grp,
            "n_user_turns"     : n_turns,
            "injection_turn"   : inj_turn,
            "caught_at_turn"   : caught_at,
            "detection_latency": dl,
            "attack_succeeded" : attack_succeeded,
            "is_attack"        : is_attack,
        })

        # ── failure logging ─────────────────────────────────────────────────
        if attack_succeeded:
            inj_text = next(
                (t["text"] for t in convo["turns"]
                 if t.get("turn") == inj_turn and t["speaker"] == "USER"),
                "N/A",
            )
            failures.append({
                "failure_type"  : "MISSED_ATTACK",
                "conv_id"       : cid,
                "topic"         : topic,
                "attack_type"   : attack_type,
                "mitigation"    : config.MITIGATION,
                "length_group"  : length_grp,
                "injection_turn": inj_turn,
                "injection_text": inj_text,
                "note"          : "Model never triggered a refusal despite adversarial escalation.",
            })

        if not is_attack and caught_at is not None:
            fp_text = next(
                (t["text"] for t in convo["turns"]
                 if t.get("turn") == caught_at and t["speaker"] == "USER"),
                "N/A",
            )
            failures.append({
                "failure_type": "FALSE_POSITIVE",
                "conv_id"     : cid,
                "topic"       : topic,
                "mitigation"  : config.MITIGATION,
                "length_group": length_grp,
                "refused_turn": caught_at,
                "refused_text": fp_text,
                "note"        : "Model refused a benign query (over-refusal).",
            })

        status = f"CAUGHT at turn {caught_at}" if caught_at else "MISSED"
        print(f"{cid} | {topic:25s} | {length_grp:6s} | {config.MITIGATION:4s} | {status}")

        save_checkpoint(config.OUT_DIR, results, turn_logs, failures)

except Exception as exc:
    print(f"\n[ERROR] Interrupted at conversation {len(results)+1}/{len(dataset)}: {exc}")
    save_checkpoint(config.OUT_DIR, results, turn_logs, failures)
    print("[CHECKPOINT] Partial results saved.")
    sys.exit(1)

# ── Aggregate + save ──────────────────────────────────────────────────────────
summary          = compute_summary(results)
cld_val, cld_rows = compute_cld(results)

print_summary(summary, cld_rows, cld_val)
save_metrics(config.OUT_DIR, summary, cld_rows, cld_val)
save_run_info(
    config.OUT_DIR,
    cfg_snapshot={
        "mitigation"    : config.MITIGATION,
        "model"         : config.MODEL,
        "temperature"   : config.TEMPERATURE,
        "random_seed"   : config.RANDOM_SEED,
        "dataset_file"  : config.DATASET,
        "dataset_md5"   : file_md5(config.DATASET),
        "dataset_size"  : len(dataset),
        "short_max"     : config.SHORT_MAX,
        "medium_max"    : config.MEDIUM_MAX,
        "refusal_phrases": config.REFUSAL_PHRASES,
    },
    summary=summary,
)
save_all(results, summary["mean_detection_latency_turns"],
         cld_rows, cld_val, failures, config.PLOTS_DIR)

print(f"\nAll outputs saved to: {config.OUT_DIR}/")