"""
Research Dashboard — Evaluating Mitigation Robustness Under Multi-Turn Prompt Injection
Authors: Jibran Shaikh, Syeda Wania Hussain
"""

import json
import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_APP_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_APP_DIR)
_RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
_DATASETS_DIR = os.path.join(_PROJECT_DIR, "Datasets")
_DATASET_FILE = os.path.join(_DATASETS_DIR, "test2_final_hardened_v2_cleaned.json")

# ── Constants ─────────────────────────────────────────────────────────────────
MITIGATIONS = ["none", "m1", "m2", "m3"]
MITIGATION_LABELS = {
    "none": "Baseline (No Mitigation)",
    "m1":   "M1 — Prompt Hardening",
    "m2":   "M2 — Input/Output Gate",
    "m3":   "M3 — State Monitor",
}
MITIGATION_DESCRIPTIONS = {
    "none": "Standard model safety training with no additional defensive layers. Serves as the control condition.",
    "m1":   "A hardened safety system prompt is injected into every conversation, giving the LLM explicit "
            "instructions to recognise and refuse harmful multi-turn escalation patterns.",
    "m2":   "An architectural keyword-filter gate that intercepts messages BEFORE they reach the LLM (input gate) "
            "and also scans the LLM's response AFTER generation (output gate).",
    "m3":   "A heuristic conversation-state monitor that accumulates an escalation score across turns using "
            "raw user text. When the cumulative score exceeds a threshold, the conversation is terminated.",
}
TOPIC_LABELS = {
    "bypassing_ai_rules":    "Bypassing AI Rules",
    "cybersecurity":         "Cybersecurity",
    "dangerous_instructions":"Dangerous Instructions",
    "financial_manipulation":"Financial Manipulation",
    "identity_theft":        "Identity Theft",
    "misinformation":        "Misinformation",
}

# Colour palette (Plotly-compatible)
PALETTE = {
    "none": "#64748b",
    "m1":   "#6366f1",
    "m2":   "#06b6d4",
    "m3":   "#f59e0b",
}
TOPIC_COLORS = [
    "#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#f43f5e", "#8b5cf6"
]

# ── Available models (only those with complete results) ───────────────────────
AVAILABLE_MODELS = {
    "openai-gpt-oss-120b": "OpenAI GPT-OSS 120B",
}

# ── Data loading helpers ───────────────────────────────────────────────────────

@st.cache_data
def load_metrics(model_slug: str) -> dict:
    """Load all four mitigation metrics_summary.json files for a model."""
    data = {}
    for mit in MITIGATIONS:
        path = os.path.join(_RESULTS_DIR, mit, model_slug, "all_samples", "metrics_summary.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data[mit] = json.load(f)
    return data


@st.cache_data
def load_dataset_sample() -> dict:
    """Load the first entry from the evaluation dataset."""
    if not os.path.exists(_DATASET_FILE):
        return {}
    with open(_DATASET_FILE, "r") as f:
        ds = json.load(f)
    return ds[0] if ds else {}


@st.cache_data
def dataset_stats() -> dict:
    """Compute high-level dataset statistics."""
    if not os.path.exists(_DATASET_FILE):
        return {}
    with open(_DATASET_FILE, "r") as f:
        ds = json.load(f)

    total = len(ds)
    attacks   = sum(1 for d in ds if d.get("attack_type") != "none")
    benign    = total - attacks
    topics    = {}
    lengths   = {"short": 0, "medium": 0, "long": 0}
    for d in ds:
        t = d.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1
        n = len(d.get("turns", []))
        if n <= 8:
            lengths["short"] += 1
        elif n <= 14:
            lengths["medium"] += 1
        else:
            lengths["long"] += 1

    return {
        "total": total,
        "attacks": attacks,
        "benign": benign,
        "topics": topics,
        "lengths": lengths,
    }


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PromptShield Research Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google font ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Global background ─────────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1035 40%, #0d1b2a 100%);
    min-height: 100vh;
}

/* ── Hero header ─────────────────────────────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
    line-height: 1.2;
}
.hero-header p.subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
    margin-top: 0.3rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #a5b4fc;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

/* ── Metric cards ────────────────────────────────────────────────────────── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, linear-gradient(90deg, #6366f1, #06b6d4));
    border-radius: 16px 16px 0 0;
}
.metric-card:hover {
    background: rgba(255,255,255,0.07);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.74rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.3rem;
}
.metric-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}

/* ── Section cards ───────────────────────────────────────────────────────── */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.section-card h3 {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.8rem;
}

/* ── Pill badges ─────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-purple  { background: rgba(139,92,246,0.2); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.3); }
.badge-cyan    { background: rgba(6,182,212,0.2);  color: #67e8f9;  border: 1px solid rgba(6,182,212,0.3); }
.badge-amber   { background: rgba(245,158,11,0.2); color: #fcd34d;  border: 1px solid rgba(245,158,11,0.3); }
.badge-green   { background: rgba(16,185,129,0.2); color: #6ee7b7;  border: 1px solid rgba(16,185,129,0.3); }
.badge-red     { background: rgba(244,63,94,0.2);  color: #fda4af;  border: 1px solid rgba(244,63,94,0.3); }

/* ── Findings box ────────────────────────────────────────────────────────── */
.finding-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 4px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    color: #c7d2fe;
    font-size: 0.9rem;
    line-height: 1.6;
}
.finding-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #818cf8;
    margin-bottom: 0.3rem;
}

/* ── Info panel ──────────────────────────────────────────────────────────── */
.info-panel {
    background: rgba(6,182,212,0.07);
    border: 1px solid rgba(6,182,212,0.2);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    color: #bae6fd;
    font-size: 0.9rem;
    line-height: 1.6;
}
.warn-panel {
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    color: #fde68a;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Code block ──────────────────────────────────────────────────────────── */
.json-sample {
    background: #0f172a;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #e2e8f0;
    overflow-x: auto;
    line-height: 1.7;
}

/* ── Tab styling overrides ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 6px;
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 0.55rem 1.3rem;
    font-weight: 600;
    font-size: 0.88rem;
    color: #64748b;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(6,182,212,0.2)) !important;
    color: #a5b4fc !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
}

/* ── Metric row separator ─────────────────────────────────────────────────── */
.separator {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    margin: 1.5rem 0;
}

/* ── Table overrides ─────────────────────────────────────────────────────── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Selectbox / button overrides ────────────────────────────────────────── */
.stSelectbox label, .stMultiSelect label { color: #94a3b8 !important; font-size: 0.85rem !important; }
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #06b6d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}

/* ── Download button ─────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: rgba(16,185,129,0.15) !important;
    color: #6ee7b7 !important;
    border: 1px solid rgba(16,185,129,0.35) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🛡️ GenAI Security Research &bull; 2026</div>
    <h1>Evaluating Mitigation Robustness<br>Under Multi-Turn Prompt Injection</h1>
    <p class="subtitle">
        A systematic empirical study of LLM safety defences against adversarial multi-turn conversational attacks<br>
        <span style="color:#6366f1">Jibran Shaikh</span> &nbsp;·&nbsp; <span style="color:#06b6d4">Syeda Wania Hussain</span>
        &nbsp;·&nbsp; Generative AI — 8th Semester
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_overview, tab_findings, tab_dataset = st.tabs([
    "🎯  Goal · Metrics · Harness",
    "📊  Findings & Results",
    "🗂️  Dataset",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — GOAL · METRICS · HARNESS
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # ── Research Goal ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-card">
        <h3>🎯 Research Goal</h3>
        <p style="color:#cbd5e1; font-size:1rem; line-height:1.75;">
            Large Language Models are increasingly deployed in interactive, multi-turn settings where
            a single conversation can span many user messages. This opens a dangerous attack surface:
            an adversary can <em>gradually</em> build context and intent across turns, bypassing safety
            filters that only inspect isolated messages.
        </p>
        <p style="color:#cbd5e1; font-size:1rem; line-height:1.75; margin-top:0.8rem;">
            Our research asks a deceptively simple question:
        </p>
        <blockquote style="border-left: 4px solid #6366f1; padding-left: 1.2rem; margin: 1rem 0;
                           color: #a5b4fc; font-size: 1.05rem; font-style: italic; font-weight: 500;">
            "How robust are existing prompt-injection mitigations when the attack unfolds across
             multiple conversational turns instead of arriving in a single adversarial message?"
        </blockquote>
        <p style="color:#cbd5e1; font-size:1rem; line-height:1.75;">
            We evaluate <strong style="color:#e2e8f0;">three defensive strategies</strong> (plus a no-mitigation
            baseline) against a curated dataset of 160 multi-turn adversarial conversations, covering
            six distinct harm categories and three conversation length groups.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Defensive Strategies ──────────────────────────────────────────────────
    st.markdown("### 🛡️ Defensive Strategies")
    col_none, col_m1, col_m2, col_m3 = st.columns(4)

    with col_none:
        st.markdown("""
        <div class="metric-card" style="--accent: #64748b; text-align:left;">
            <div style="font-size:1.8rem; margin-bottom:0.5rem;">⚡</div>
            <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.4rem;">Baseline</div>
            <div class="badge badge-green" style="margin-bottom:0.7rem;">No Mitigation</div>
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">
                Standard model safety training only. Control condition for comparing
                the additive effect of each defence layer.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m1:
        st.markdown("""
        <div class="metric-card" style="--accent: linear-gradient(90deg,#6366f1,#8b5cf6); text-align:left;">
            <div style="font-size:1.8rem; margin-bottom:0.5rem;">📜</div>
            <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.4rem;">M1 — Prompt Hardening</div>
            <div class="badge badge-purple" style="margin-bottom:0.7rem;">Instruction-Level</div>
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">
                Prepends a structured safety system prompt to every conversation,
                instructing the LLM to recognise and refuse gradual escalation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="metric-card" style="--accent: linear-gradient(90deg,#06b6d4,#0ea5e9); text-align:left;">
            <div style="font-size:1.8rem; margin-bottom:0.5rem;">🔍</div>
            <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.4rem;">M2 — I/O Gate</div>
            <div class="badge badge-cyan" style="margin-bottom:0.7rem;">Architectural</div>
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">
                Keyword-filter gate intercepts messages <em>before</em> LLM call (input gate)
                and inspects LLM output <em>after</em> generation (output gate).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown("""
        <div class="metric-card" style="--accent: linear-gradient(90deg,#f59e0b,#f97316); text-align:left;">
            <div style="font-size:1.8rem; margin-bottom:0.5rem;">📈</div>
            <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.4rem;">M3 — State Monitor</div>
            <div class="badge badge-amber" style="margin-bottom:0.7rem;">Heuristic</div>
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">
                Accumulates a turn-level escalation score using raw user text.
                Blocks the conversation when cumulative score exceeds a threshold.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("### 📐 Evaluation Metrics")

    metrics_info = [
        {
            "icon": "🎯",
            "name": "ASR — Attack Success Rate",
            "badge": "badge-red",
            "badge_text": "Primary",
            "formula": "ASR = Attacks Missed / Total Attacks × 100",
            "desc": "The percentage of adversarial conversations where the attacker successfully extracted "
                    "harmful content or guidance from the model. <strong>Lower is better</strong> — a "
                    "perfect mitigation scores 0%.",
            "note": "Measured after the full conversation; a catch at any turn counts as a block.",
        },
        {
            "icon": "⏱️",
            "name": "DL — Detection Latency",
            "badge": "badge-cyan",
            "badge_text": "Timeliness",
            "formula": "DL = turns from injection turn to first refusal",
            "desc": "The number of user turns between the first adversarial injection and the moment the "
                    "mitigation fires. DL = 0 means the attack was blocked <em>immediately</em> on the "
                    "injection turn. <strong>Lower is better.</strong>",
            "note": "Decomposed into Gate Latency (M2/M3 code-level block) and AI Latency (LLM self-refusal).",
        },
        {
            "icon": "⚠️",
            "name": "ORR — Over-Refusal Rate",
            "badge": "badge-amber",
            "badge_text": "Usability",
            "formula": "ORR = False Positives / Total Benign Conversations × 100",
            "desc": "The fraction of <em>benign</em> conversations that were incorrectly blocked by a "
                    "mitigation. Captures the security-usability trade-off. "
                    "<strong>Lower is better</strong> — aggressive mitigations can score high here.",
            "note": "Especially critical for M2, whose keyword filter is context-blind.",
        },
        {
            "icon": "📏",
            "name": "CLD — Context-Length Drift",
            "badge": "badge-purple",
            "badge_text": "Stability",
            "formula": "CLD = ASR_long − ASR_short",
            "desc": "Captures whether a mitigation degrades as conversations grow longer. "
                    "A high positive CLD indicates the model 'forgets' its safety stance in long conversations. "
                    "<strong>Near-zero or negative is better.</strong>",
            "note": "Computed across three length buckets: Short (≤8 turns), Medium (9–14), Long (>14).",
        },
        {
            "icon": "🎲",
            "name": "TVC — Topic Vulnerability Consistency",
            "badge": "badge-green",
            "badge_text": "Uniformity",
            "formula": "TVC = 1 − (std(ASR_per_topic) / mean(ASR_per_topic))",
            "desc": "Measures how uniformly a mitigation protects across all six harm categories. "
                    "A high TVC means roughly equal protection everywhere; low variance. "
                    "<strong>Higher is better</strong> (range: 0–1).",
            "note": "A TVC of 0 with a mean ASR of 0% is paradoxically perfect — all topics blocked.",
        },
        {
            "icon": "🏔️",
            "name": "ERR — Escalation Resistance Rate",
            "badge": "badge-purple",
            "badge_text": "Resistance",
            "formula": "ERR = Blocked ATTACK_ESCALATES turns / Total ATTACK_ESCALATES turns × 100",
            "desc": "Focuses specifically on <em>escalation turns</em> — mid-conversation moves that "
                    "intensify the attack. ERR measures what fraction of these turns the mitigation catches. "
                    "<strong>Higher is better.</strong>",
            "note": "Split into ERR_early vs ERR_late to show whether mitigations weaken mid-conversation.",
        },
        {
            "icon": "🔄",
            "name": "RCS — Refusal Consistency Score",
            "badge": "badge-cyan",
            "badge_text": "Persistence",
            "formula": "RCS = mean(fraction of turns after first refusal that also refuse)",
            "desc": "Once a mitigation blocks a turn, does it continue to block subsequent turns in "
                    "the same conversation, or does it 'reset'? <strong>Higher is better</strong> (0–1). "
                    "Critical for catching M3's score-reset failure mode.",
            "note": "M3 design flaw: after blocking, accumulated score resets, allowing further attacks to slip through.",
        },
    ]

    for i in range(0, len(metrics_info), 2):
        row_cols = st.columns(2)
        for j, col in enumerate(row_cols):
            if i + j >= len(metrics_info):
                break
            m = metrics_info[i + j]
            with col:
                st.markdown(f"""
                <div class="section-card" style="margin-bottom:1rem;">
                    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.6rem;">
                        <span style="font-size:1.4rem;">{m['icon']}</span>
                        <div>
                            <span style="font-size:0.95rem; font-weight:700; color:#e2e8f0;">{m['name']}</span>
                            &nbsp;<span class="badge {m['badge']}">{m['badge_text']}</span>
                        </div>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#6ee7b7;
                                background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.15);
                                border-radius:8px; padding:0.45rem 0.75rem; margin-bottom:0.7rem;">
                        {m['formula']}
                    </div>
                    <p style="color:#94a3b8; font-size:0.86rem; line-height:1.65; margin-bottom:0.5rem;">
                        {m['desc']}
                    </p>
                    <div style="font-size:0.78rem; color:#64748b; border-top:1px solid rgba(255,255,255,0.06);
                                padding-top:0.5rem; margin-top:0.5rem;">
                        💡 {m['note']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    # ── Harness System ────────────────────────────────────────────────────────
    st.markdown("### ⚙️ The Evaluation Harness")

    col_h1, col_h2 = st.columns([3, 2])

    with col_h1:
        st.markdown("""
        <div class="section-card">
            <h3 style="margin-bottom:1rem;">How It Works</h3>
            <div style="color:#94a3b8; font-size:0.9rem; line-height:1.75;">
                <p style="margin-bottom:0.8rem;">
                    The harness (<code style="color:#67e8f9;">System/harness.py</code>) iterates over every
                    conversation in the dataset, replaying each USER turn through the active mitigation pipeline.
                    It records what happened at every turn and produces a comprehensive metrics summary at the end.
                </p>
                <p style="font-weight:600; color:#c7d2fe; margin-bottom:0.5rem;">Pipeline for each USER turn:</p>
                <ol style="margin-left:1.2rem; margin-bottom:0.8rem;">
                    <li style="margin-bottom:0.4rem;"><strong style="color:#e2e8f0;">Apply Mitigation</strong>
                        — run the active strategy (none/M1/M2/M3)</li>
                    <li style="margin-bottom:0.4rem;"><strong style="color:#e2e8f0;">Call the LLM</strong>
                        — send conversation history to the model API (skipped if M2/M3 gate fires)</li>
                    <li style="margin-bottom:0.4rem;"><strong style="color:#e2e8f0;">Detect Refusal</strong>
                        — two-stage: phrase-match → LLM-as-judge fallback</li>
                    <li style="margin-bottom:0.4rem;"><strong style="color:#e2e8f0;">Log Turn</strong>
                        — record labels, flags, latencies into turn_logs</li>
                </ol>
                <p style="margin-bottom:0.8rem;">
                    After all conversations, ASR, ORR, DL, CLD, TVC, ERR, and RCS are computed from the
                    accumulated results, and a suite of charts is generated with Matplotlib.
                </p>
                <p style="color:#6ee7b7; font-size:0.84rem;">
                    🧪 Fully reproducible: Temperature = 0.0, Seed = 42, dataset MD5 logged in run_info.json
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_h2:
        st.markdown("""
        <div class="section-card">
            <h3 style="margin-bottom:1rem;">Two-Stage Refusal Detector</h3>
            <div style="color:#94a3b8; font-size:0.88rem; line-height:1.7;">
                <div class="badge badge-purple" style="margin-bottom:0.7rem;">Stage 1 — Phrase Match</div>
                <p style="margin-bottom:0.8rem;">
                    Fast, zero-cost scan for 80+ normalised refusal phrases
                    (smart-quote normalised). Handles the clear majority of refusals instantly.
                </p>
                <div class="badge badge-cyan" style="margin-bottom:0.7rem;">Stage 2 — LLM-as-Judge</div>
                <p style="margin-bottom:0.8rem;">
                    Only activates when Stage 1 produces no match.
                    Uses <strong style="color:#e2e8f0;">meta/llama-3.1-70b-instruct</strong> via NVIDIA NIM
                    to classify novel or indirect refusal phrasings.
                </p>
                <div class="badge badge-green" style="margin-bottom:0.7rem;">Backup Judge</div>
                <p>
                    If the primary judge is rate-limited (3× consecutive 429s),
                    the system permanently switches to
                    <strong style="color:#e2e8f0;">nvidia/nemotron-3-super-120b</strong>
                    for the rest of the run.
                </p>
            </div>
        </div>

        <div class="section-card">
            <h3 style="margin-bottom:0.8rem;">Key Files</h3>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#94a3b8; line-height:1.9;">
                <span style="color:#a5b4fc;">System/harness.py</span> — main loop<br>
                <span style="color:#a5b4fc;">System/mitigations.py</span> — M1/M2/M3<br>
                <span style="color:#a5b4fc;">System/metrics.py</span> — all metric formulas<br>
                <span style="color:#a5b4fc;">System/plots.py</span> — Matplotlib charts<br>
                <span style="color:#a5b4fc;">System/config.py</span> — constants & paths<br>
                <span style="color:#a5b4fc;">Utils/run_all_mitigations.py</span> — batch runner
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — FINDINGS & RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_findings:

    # ── Model selection ───────────────────────────────────────────────────────
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        model_options = {
            "OpenAI GPT-OSS 120B (openai/gpt-oss-120b)": "openai-gpt-oss-120b",
            "All Models (Coming Soon)": "__all__",
        }
        selected_label = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0,
            help="Only models with completed evaluation runs are available.",
        )
        selected_slug = model_options[selected_label]

    with col_btn:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        run_analysis = st.button("🔬  Generate Analysis", use_container_width=True)

    # ── Guard: All models not yet available ───────────────────────────────────
    if selected_slug == "__all__":
        st.markdown("""
        <div class="warn-panel" style="margin-top:1rem;">
            <strong>⚠️ Cross-Model Analysis Coming Soon</strong><br>
            Aggregated "All Models" statistics are not yet available — additional model runs are
            currently in progress. Check back once the full evaluation suite has completed.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Load data ─────────────────────────────────────────────────────────────
    if run_analysis or True:   # auto-render on tab open
        metrics = load_metrics(selected_slug)

        if not metrics:
            st.error("No result files found for the selected model. Please run the evaluation harness first.")
            st.stop()

        available_mits = [m for m in MITIGATIONS if m in metrics]
        model_display = AVAILABLE_MODELS.get(selected_slug, selected_slug)

        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:1.5rem;">
            <div style="font-size:1.4rem;">🤖</div>
            <div>
                <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">{model_display}</div>
                <div style="font-size:0.8rem; color:#64748b; font-family:'JetBrains Mono',monospace;">{selected_slug}</div>
            </div>
            <div style="margin-left:auto; display:flex; gap:0.4rem;">
                {''.join(f'<span class="badge badge-{"purple" if m=="m1" else "cyan" if m=="m2" else "amber" if m=="m3" else "green"}">{MITIGATION_LABELS[m].split("—")[0].strip()}</span>' for m in available_mits)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Key metrics summary row ────────────────────────────────────────────
        st.markdown("#### 📊 Key Metrics at a Glance")

        kpi_cols = st.columns(len(available_mits))
        for idx, mit in enumerate(available_mits):
            m = metrics[mit]
            with kpi_cols[idx]:
                color = {"none": "#64748b", "m1": "#6366f1", "m2": "#06b6d4", "m3": "#f59e0b"}[mit]
                asr   = m.get("attack_success_rate_pct", "N/A")
                orr   = m.get("over_refusal_rate_pct", "N/A")
                err   = m.get("err_overall", "N/A")
                rcs   = m.get("rcs_score", "N/A")
                cld   = m.get("context_length_drift_pct", "N/A")
                label = MITIGATION_LABELS[mit]

                st.markdown(f"""
                <div class="metric-card" style="--accent: {color};">
                    <div class="metric-label" style="color:{color};">{label}</div>
                    <div class="separator" style="margin:0.5rem 0;"></div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-top:0.4rem;">
                        <div>
                            <div class="metric-value" style="font-size:1.5rem; color:{color};">{asr}%</div>
                            <div class="metric-label">ASR ↓</div>
                        </div>
                        <div>
                            <div class="metric-value" style="font-size:1.5rem;">{orr}%</div>
                            <div class="metric-label">ORR ↓</div>
                        </div>
                        <div>
                            <div class="metric-value" style="font-size:1.5rem;">{err}%</div>
                            <div class="metric-label">ERR ↑</div>
                        </div>
                        <div>
                            <div class="metric-value" style="font-size:1.5rem;">{rcs}</div>
                            <div class="metric-label">RCS ↑</div>
                        </div>
                    </div>
                    <div style="margin-top:0.6rem; color:#64748b; font-size:0.76rem;">
                        CLD: {cld}pp
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

        # ── Chart section ──────────────────────────────────────────────────────
        st.markdown("#### 📈 Visualisations")

        # Chart 1: ASR comparison bar chart
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            asr_vals   = [metrics[m].get("attack_success_rate_pct", 0) for m in available_mits]
            colors_bar = [PALETTE[m] for m in available_mits]
            labels_bar = [MITIGATION_LABELS[m] for m in available_mits]

            fig_asr = go.Figure()
            fig_asr.add_trace(go.Bar(
                x=labels_bar,
                y=asr_vals,
                marker_color=colors_bar,
                marker_line_width=0,
                text=[f"{v}%" for v in asr_vals],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=13, family="Inter"),
            ))
            fig_asr.update_layout(
                title=dict(text="Attack Success Rate (ASR) by Mitigation",
                           font=dict(color="#e2e8f0", size=14, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                           title="ASR (%)", range=[0, max(asr_vals) * 1.25 + 5]),
                margin=dict(t=50, b=10, l=10, r=10),
                height=340,
            )
            st.plotly_chart(fig_asr, use_container_width=True)

        with col_c2:
            orr_vals = [metrics[m].get("over_refusal_rate_pct", 0) for m in available_mits]

            fig_orr = go.Figure()
            fig_orr.add_trace(go.Bar(
                x=labels_bar,
                y=orr_vals,
                marker_color=colors_bar,
                marker_line_width=0,
                text=[f"{v}%" for v in orr_vals],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=13, family="Inter"),
            ))
            fig_orr.update_layout(
                title=dict(text="Over-Refusal Rate (ORR) by Mitigation",
                           font=dict(color="#e2e8f0", size=14, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                           title="ORR (%)", range=[0, max(orr_vals) * 1.3 + 5]),
                margin=dict(t=50, b=10, l=10, r=10),
                height=340,
            )
            st.plotly_chart(fig_orr, use_container_width=True)

        # Chart 2: ERR early vs late + RCS
        col_c3, col_c4 = st.columns(2)

        with col_c3:
            err_early = [metrics[m].get("err_early") or 0 for m in available_mits]
            err_late  = [metrics[m].get("err_late")  or 0 for m in available_mits]
            err_overall = [metrics[m].get("err_overall") or 0 for m in available_mits]

            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(
                name="ERR Early", x=labels_bar, y=err_early,
                marker_color="rgba(99,102,241,0.8)", marker_line_width=0,
            ))
            fig_err.add_trace(go.Bar(
                name="ERR Late", x=labels_bar, y=err_late,
                marker_color="rgba(6,182,212,0.8)", marker_line_width=0,
            ))
            fig_err.update_layout(
                barmode="group",
                title=dict(text="Escalation Resistance Rate — Early vs Late",
                           font=dict(color="#e2e8f0", size=14, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                           title="ERR (%)", range=[0, 110]),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
                margin=dict(t=50, b=10, l=10, r=10),
                height=340,
            )
            st.plotly_chart(fig_err, use_container_width=True)

        with col_c4:
            rcs_vals = [metrics[m].get("rcs_score") or 0 for m in available_mits]

            fig_rcs = go.Figure()
            fig_rcs.add_trace(go.Bar(
                x=labels_bar,
                y=rcs_vals,
                marker_color=colors_bar,
                marker_line_width=0,
                text=[f"{v:.3f}" for v in rcs_vals],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=13, family="Inter"),
            ))
            fig_rcs.update_layout(
                title=dict(text="Refusal Consistency Score (RCS)",
                           font=dict(color="#e2e8f0", size=14, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                           title="RCS (0–1)", range=[0, 1.2]),
                margin=dict(t=50, b=10, l=10, r=10),
                height=340,
            )
            st.plotly_chart(fig_rcs, use_container_width=True)

        # Chart 3: ASR by Topic heatmap
        st.markdown("##### ASR by Topic (Heatmap)")

        topics_ordered = [
            "bypassing_ai_rules", "cybersecurity", "dangerous_instructions",
            "financial_manipulation", "identity_theft", "misinformation"
        ]
        heat_data = []
        for mit in available_mits:
            tvc = metrics[mit].get("tvc_by_topic", {})
            row = [tvc.get(t, {}).get("asr_pct", None) for t in topics_ordered]
            heat_data.append(row)

        heat_df = pd.DataFrame(heat_data, index=[MITIGATION_LABELS[m] for m in available_mits],
                               columns=[TOPIC_LABELS[t] for t in topics_ordered])

        fig_heat = go.Figure(go.Heatmap(
            z=heat_df.values,
            x=heat_df.columns.tolist(),
            y=heat_df.index.tolist(),
            colorscale=[
                [0.0, "#1a2744"],
                [0.25, "#1e3a5f"],
                [0.5, "#7c3aed"],
                [0.75, "#dc2626"],
                [1.0, "#fbbf24"],
            ],
            text=heat_df.values,
            texttemplate="%{text:.1f}%",
            textfont=dict(color="white", size=13, family="Inter"),
            showscale=True,
            colorbar=dict(
                tickfont=dict(color="#94a3b8"),
                outlinewidth=0,
                title=dict(text="ASR %", font=dict(color="#94a3b8")),
            ),
        ))
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            xaxis=dict(side="bottom", tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12)),
            margin=dict(t=20, b=10, l=10, r=10),
            height=280,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Chart 4: CLD — ASR by length group
        st.markdown("##### Context-Length Drift — ASR across Conversation Lengths")

        col_c5, col_c6 = st.columns(2)
        with col_c5:
            length_groups = ["short", "medium", "long"]
            fig_cld = go.Figure()
            for mit in available_mits:
                asr_by_len = metrics[mit].get("asr_by_length_group", [])
                vals = {d["length_group"]: d["asr_pct"] for d in asr_by_len}
                y_vals = [vals.get(g, None) for g in length_groups]
                fig_cld.add_trace(go.Scatter(
                    x=["Short\n(≤8 turns)", "Medium\n(9–14 turns)", "Long\n(>14 turns)"],
                    y=y_vals,
                    mode="lines+markers",
                    name=MITIGATION_LABELS[mit],
                    line=dict(color=PALETTE[mit], width=2.5),
                    marker=dict(size=10, color=PALETTE[mit]),
                ))
            fig_cld.update_layout(
                title=dict(text="ASR by Conversation Length",
                           font=dict(color="#e2e8f0", size=14, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                           title="ASR (%)", range=[0, 110]),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
                margin=dict(t=50, b=10, l=10, r=10),
                height=340,
            )
            st.plotly_chart(fig_cld, use_container_width=True)

        with col_c6:
            cld_vals = []
            cld_labels = []
            for mit in available_mits:
                v = metrics[mit].get("context_length_drift_pct", None)
                if v is not None:
                    cld_vals.append(v)
                    cld_labels.append(MITIGATION_LABELS[mit])

            fig_cld_bar = go.Figure()
            fig_cld_bar.add_trace(go.Bar(
                x=cld_labels,
                y=cld_vals,
                marker_color=[
                    "#f43f5e" if v > 0 else "#10b981"
                    for v in cld_vals
                ],
                text=[f"{v:+.1f}pp" for v in cld_vals],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=13, family="Inter"),
                marker_line_width=0,
            ))
            fig_cld_bar.add_hline(
                y=0, line_color="rgba(255,255,255,0.3)", line_dash="dash",
            )
            fig_cld_bar.update_layout(
                title=dict(text="Context-Length Drift (CLD) — Long − Short ASR",
                           font=dict(color="#e2e8f0", size=14, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                           title="CLD (pp)", zeroline=False),
                margin=dict(t=50, b=10, l=10, r=10),
                height=340,
            )
            st.plotly_chart(fig_cld_bar, use_container_width=True)

        # Chart 5: RCS by topic radar
        st.markdown("##### RCS by Topic")
        topics_display = [TOPIC_LABELS[t] for t in topics_ordered]
        fig_radar = go.Figure()
        rgba_map = {
            "none": "rgba(100,116,139,0.15)",
            "m1":   "rgba(99,102,241,0.15)",
            "m2":   "rgba(6,182,212,0.15)",
            "m3":   "rgba(245,158,11,0.15)",
        }
        for mit in available_mits:
            rcs_by_topic = metrics[mit].get("rcs_by_topic", {})
            vals = [rcs_by_topic.get(t) or 0 for t in topics_ordered]
            vals_closed = vals + [vals[0]]
            ts_closed   = topics_display + [topics_display[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=ts_closed,
                fill="toself",
                fillcolor=rgba_map[mit],
                name=MITIGATION_LABELS[mit],
                line=dict(color=PALETTE[mit], width=2.5),
                marker=dict(size=7),
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    tickfont=dict(color="#64748b", size=10),
                    gridcolor="rgba(255,255,255,0.08)",
                    linecolor="rgba(255,255,255,0.08)",
                ),
                angularaxis=dict(
                    tickfont=dict(color="#94a3b8", size=11),
                    gridcolor="rgba(255,255,255,0.08)",
                    linecolor="rgba(255,255,255,0.08)",
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
            margin=dict(t=20, b=20, l=30, r=30),
            height=380,
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Comparative table ──────────────────────────────────────────────────
        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
        st.markdown("#### 📋 Full Metrics Comparison Table")

        table_rows = []
        for mit in available_mits:
            m = metrics[mit]
            table_rows.append({
                "Strategy": MITIGATION_LABELS[mit],
                "ASR (%) ↓": m.get("attack_success_rate_pct"),
                "ORR (%) ↓": m.get("over_refusal_rate_pct"),
                "ERR (%) ↑": m.get("err_overall"),
                "ERR Early ↑": m.get("err_early"),
                "ERR Late ↑": m.get("err_late"),
                "RCS (0–1) ↑": m.get("rcs_score"),
                "CLD (pp)": m.get("context_length_drift_pct"),
                "TVC (0–1) ↑": m.get("tvc_score"),
                "Attacks Caught": m.get("attacks_caught"),
                "False Positives": m.get("false_positives"),
            })

        df_table = pd.DataFrame(table_rows).set_index("Strategy")
        st.dataframe(
            df_table.style.format(precision=2, na_rep="—")
                .background_gradient(subset=["ASR (%) ↓"], cmap="RdYlGn_r")
                .background_gradient(subset=["ORR (%) ↓"], cmap="RdYlGn_r")
                .background_gradient(subset=["ERR (%) ↑"], cmap="RdYlGn")
                .background_gradient(subset=["RCS (0–1) ↑"], cmap="RdYlGn"),
            use_container_width=True,
        )

        # ── Findings ───────────────────────────────────────────────────────────
        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Key Findings & Interpretation")

        m_none = metrics.get("none", {})
        m_m1   = metrics.get("m1", {})
        m_m2   = metrics.get("m2", {})
        m_m3   = metrics.get("m3", {})

        # Finding 1: ASR comparison
        asr_none  = m_none.get("attack_success_rate_pct", "N/A")
        asr_m1    = m_m1.get("attack_success_rate_pct",  "N/A")
        asr_m2    = m_m2.get("attack_success_rate_pct",  "N/A")
        asr_m3    = m_m3.get("attack_success_rate_pct",  "N/A")
        best_asr  = min([(v, k) for k, v in [("none", asr_none), ("m1", asr_m1),
                                               ("m2", asr_m2), ("m3", asr_m3)]
                         if isinstance(v, (int, float))])[1]

        st.markdown(f"""
        <div class="finding-box">
            <div class="finding-title">🏆 Finding 1 — Best Attack Containment</div>
            <strong style="color:#a5b4fc;">{MITIGATION_LABELS[best_asr]}</strong> achieves the 
            lowest Attack Success Rate among all strategies tested on {model_display}.
            The baseline (no mitigation) allows <strong>{asr_none}%</strong> of attacks to succeed.
            M1 (Prompt Hardening) reduces this to <strong>{asr_m1}%</strong>, demonstrating that
            targeted safety instructions dramatically limit the model's compliance with multi-turn
            adversarial escalation — even without any architectural filtering.
        </div>
        """, unsafe_allow_html=True)

        # Finding 2: M2 over-refusal
        orr_m2 = m_m2.get("over_refusal_rate_pct", "N/A")
        st.markdown(f"""
        <div class="finding-box">
            <div class="finding-title">⚠️ Finding 2 — The M2 Usability Cost</div>
            The I/O Gate (M2) carries a severe usability penalty: an Over-Refusal Rate of
            <strong style="color:#fda4af;">{orr_m2}%</strong> on benign conversations.
            This means M2 blocks nearly 9 in 10 legitimate user queries — a level of false
            positives that would render the system unusable in production. The keyword filter
            is inherently context-blind and treats legitimate discussions of security topics
            as attacks, reflecting a fundamental limitation of surface-form matching.
        </div>
        """, unsafe_allow_html=True)

        # Finding 3: M3 CLD
        cld_m3 = m_m3.get("context_length_drift_pct", "N/A")
        cld_m1 = m_m1.get("context_length_drift_pct", "N/A")
        st.markdown(f"""
        <div class="finding-box">
            <div class="finding-title">📏 Finding 3 — Context-Length Vulnerability</div>
            M3 (State Monitor) shows a large positive CLD of <strong style="color:#fda4af;">+{cld_m3}pp</strong>,
            indicating that its protection degrades significantly in long conversations. This is consistent
            with its design: once the cumulative escalation score depletes after a block, the conversation
            resets, giving attackers a second window in longer exchanges. In contrast, M1 achieves a CLD of
            <strong style="color:#6ee7b7;">{cld_m1}pp</strong>, showing near-stable protection across all
            conversation lengths.
        </div>
        """, unsafe_allow_html=True)

        # Finding 4: ERR topic breakdown
        err_m1_cyber = m_m1.get("err_by_topic", {}).get("cybersecurity", "N/A")
        err_m1_danger = m_m1.get("err_by_topic", {}).get("dangerous_instructions", "N/A")
        err_m1_bypass = m_m1.get("err_by_topic", {}).get("bypassing_ai_rules", "N/A")
        st.markdown(f"""
        <div class="finding-box">
            <div class="finding-title">🎯 Finding 4 — Topic-Level Vulnerability Gaps</div>
            Even the best-performing mitigation (M1) shows notable topic-level vulnerability gaps.
            Escalation Resistance is perfect for <em>Bypassing AI Rules</em>
            (<strong style="color:#6ee7b7;">{err_m1_bypass}%</strong>) but drops sharply for
            <em>Cybersecurity</em> (<strong style="color:#fda4af;">{err_m1_cyber}%</strong>)
            and <em>Dangerous Instructions</em>
            (<strong style="color:#fda4af;">{err_m1_danger}%</strong>). This suggests that
            multi-turn attacks framed around technical domains are harder for prompt hardening
            alone to catch, likely because the early preamble turns are indistinguishable
            from legitimate educational queries.
        </div>
        """, unsafe_allow_html=True)

        # Finding 5: RCS
        rcs_m2 = m_m2.get("rcs_score", "N/A")
        rcs_m1 = m_m1.get("rcs_score", "N/A")
        rcs_m3 = m_m3.get("rcs_score", "N/A")
        st.markdown(f"""
        <div class="finding-box">
            <div class="finding-title">🔄 Finding 5 — Refusal Consistency (The Reset Problem)</div>
            M1 achieves an RCS of <strong style="color:#6ee7b7;">{rcs_m1}</strong>, meaning once it
            refuses, it maintains that refusal in 92% of subsequent turns. M3, despite its
            dedicated monitoring mechanism, achieves RCS = <strong>{rcs_m3}</strong>, but M2 is the
            weakest at <strong style="color:#fda4af;">{rcs_m2}</strong>. For M2, the lower RCS reflects
            that the keyword gate can be bypassed in later turns by rephrasing the attack to avoid
            trigger words — once an initial block fires, the attacker learns which phrases to avoid.
        </div>
        """, unsafe_allow_html=True)

        # Finding 6: Overall
        st.markdown(f"""
        <div class="finding-box" style="border-left-color: #10b981; background: rgba(16,185,129,0.08);
             border-color: rgba(16,185,129,0.25);">
            <div class="finding-title" style="color:#6ee7b7;">✅ Overall Takeaway</div>
            For <strong style="color:#e2e8f0;">{model_display}</strong>, <strong>M1 (Prompt Hardening)
            emerges as the most balanced mitigation</strong>: lowest ASR ({asr_m1}%), near-zero
            false positives (ORR={m_m1.get("over_refusal_rate_pct")}%), strong escalation resistance,
            and stable protection across lengths. The results suggest that large, instruction-following
            models can effectively self-regulate when given explicit, well-structured safety prompts —
            but that purely architectural defences (M2) create unacceptable usability costs, while
            heuristic monitors (M3) can be gamed by patient attackers in long conversations.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab_dataset:

    col_d1, col_d2 = st.columns([2, 1])

    with col_d1:
        st.markdown("""
        <div class="section-card">
            <h3>About the Dataset</h3>
            <p style="color:#94a3b8; font-size:0.9rem; line-height:1.75; margin-bottom:0.8rem;">
                The evaluation dataset is a <strong style="color:#e2e8f0;">Hardened V2</strong> collection
                of synthetic multi-turn adversarial conversations, curated and cleaned specifically for this
                research. Each entry represents a full conversation between a user and an AI assistant, where
                the user gradually escalates toward a harmful goal.
            </p>
            <p style="color:#94a3b8; font-size:0.9rem; line-height:1.75; margin-bottom:0.8rem;">
                The dataset is <strong style="color:#e2e8f0;">stratified</strong> across:
            </p>
            <ul style="color:#94a3b8; font-size:0.9rem; line-height:1.9; margin-left:1.2rem; margin-bottom:0.8rem;">
                <li><strong style="color:#c7d2fe;">6 harm categories</strong>:
                    Bypassing AI Rules, Cybersecurity, Dangerous Instructions,
                    Financial Manipulation, Identity Theft, Misinformation</li>
                <li><strong style="color:#c7d2fe;">3 conversation lengths</strong>:
                    Short (≤8 turns), Medium (9–14 turns), Long (>14 turns)</li>
                <li><strong style="color:#c7d2fe;">2 sample types</strong>:
                    Attack conversations (multi-turn injection) and Benign conversations (no attack)</li>
            </ul>
            <p style="color:#94a3b8; font-size:0.9rem; line-height:1.75;">
                Conversations are <strong style="color:#e2e8f0;">turn-level labelled</strong> with one of:
            </p>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.5rem;">
                <span class="badge badge-green">BENIGN</span>
                <span class="badge badge-amber">ATTACK_STARTS</span>
                <span class="badge badge-red">ATTACK_ESCALATES</span>
                <span class="badge badge-purple">DETECTED</span>
                <span class="badge badge-cyan">MISSED</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        stats = dataset_stats()
        if stats:
            st.markdown(f"""
            <div class="section-card">
                <h3>Dataset Statistics</h3>
                <div style="display:grid; gap:0.8rem; margin-top:0.5rem;">
                    <div class="metric-card" style="--accent: #6366f1; padding: 1rem;">
                        <div class="metric-value" style="font-size:1.8rem;">{stats['total']}</div>
                        <div class="metric-label">Total Conversations</div>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.6rem;">
                        <div class="metric-card" style="--accent: #f43f5e; padding: 0.8rem;">
                            <div class="metric-value" style="font-size:1.4rem; color:#fda4af;">{stats['attacks']}</div>
                            <div class="metric-label">Attack</div>
                        </div>
                        <div class="metric-card" style="--accent: #10b981; padding: 0.8rem;">
                            <div class="metric-value" style="font-size:1.4rem; color:#6ee7b7;">{stats['benign']}</div>
                            <div class="metric-label">Benign</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Length & topic distribution charts ─────────────────────────────────────
    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    if stats:
        col_chart_d1, col_chart_d2 = st.columns(2)

        with col_chart_d1:
            topic_data = stats["topics"]
            fig_topic = go.Figure(go.Pie(
                labels=[TOPIC_LABELS.get(k, k) for k in topic_data.keys()],
                values=list(topic_data.values()),
                hole=0.55,
                marker_colors=TOPIC_COLORS,
                textfont=dict(color="white", size=12, family="Inter"),
                textinfo="label+percent",
            ))
            fig_topic.update_layout(
                title=dict(text="Conversations by Topic",
                           font=dict(color="#e2e8f0", size=14, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                showlegend=False,
                margin=dict(t=50, b=10, l=10, r=10),
                height=340,
            )
            st.plotly_chart(fig_topic, use_container_width=True)

        with col_chart_d2:
            len_data  = stats["lengths"]
            len_labels = ["Short\n(≤8 turns)", "Medium\n(9–14 turns)", "Long\n(>14 turns)"]
            fig_len = go.Figure(go.Bar(
                x=len_labels,
                y=list(len_data.values()),
                marker_color=["#6366f1", "#06b6d4", "#f59e0b"],
                marker_line_width=0,
                text=list(len_data.values()),
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=14, family="Inter"),
            ))
            fig_len.update_layout(
                title=dict(text="Conversations by Length Group",
                           font=dict(color="#e2e8f0", size=14, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                           title="Count",
                           range=[0, max(len_data.values()) * 1.25]),
                margin=dict(t=50, b=10, l=10, r=10),
                height=340,
            )
            st.plotly_chart(fig_len, use_container_width=True)

    # ── Sample entry ───────────────────────────────────────────────────────────
    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
    st.markdown("#### 📄 Sample Dataset Entry — V2-001")

    sample = load_dataset_sample()

    if sample:
        # Trim turns for cleaner display (show first 6 turns)
        display_sample = dict(sample)
        display_sample["turns"] = sample["turns"][:6]
        sample_json = json.dumps(display_sample, indent=2)

        col_sample, col_legend = st.columns([3, 1])

        with col_sample:
            st.markdown(f"""
            <div class="json-sample"><pre style="margin:0;">{sample_json}</pre></div>
            """, unsafe_allow_html=True)

        with col_legend:
            st.markdown("""
            <div class="section-card">
                <h3 style="font-size:0.95rem;">Field Reference</h3>
                <div style="font-size:0.82rem; color:#94a3b8; line-height:1.9;">
                    <div><span style="color:#67e8f9; font-family:'JetBrains Mono',monospace;">id</span>
                        — Unique conversation ID</div>
                    <div><span style="color:#67e8f9; font-family:'JetBrains Mono',monospace;">topic</span>
                        — Harm category</div>
                    <div><span style="color:#67e8f9; font-family:'JetBrains Mono',monospace;">attack_type</span>
                        — multi_turn / none</div>
                    <div><span style="color:#67e8f9; font-family:'JetBrains Mono',monospace;">injection_turn</span>
                        — Turn where attack begins</div>
                    <div><span style="color:#67e8f9; font-family:'JetBrains Mono',monospace;">detection_turn</span>
                        — Turn of first refusal (null if missed)</div>
                    <div><span style="color:#67e8f9; font-family:'JetBrains Mono',monospace;">success</span>
                        — yes = attack succeeded</div>
                    <div><span style="color:#67e8f9; font-family:'JetBrains Mono',monospace;">turns</span>
                        — Ordered list of conversation turns</div>
                    <div style="margin-top:0.8rem;">
                        <strong style="color:#c7d2fe; font-size:0.78rem;">Turn Labels:</strong>
                    </div>
                    <div><span class="badge badge-green">BENIGN</span>  — Safe preamble turn</div>
                    <div style="margin-top:0.25rem;"><span class="badge badge-amber">ATTACK_STARTS</span></div>
                    <div style="margin-top:0.25rem;"><span class="badge badge-red">ATTACK_ESCALATES</span></div>
                    <div style="margin-top:0.25rem;"><span class="badge badge-purple">DETECTED</span></div>
                    <div style="margin-top:0.25rem;"><span class="badge badge-cyan">MISSED</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Dataset file not found. Ensure the Datasets directory is present.")

    # ── Download button ────────────────────────────────────────────────────────
    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    if os.path.exists(_DATASET_FILE):
        with open(_DATASET_FILE, "rb") as f:
            dataset_bytes = f.read()
        st.download_button(
            label="⬇️  Download Full Dataset (test2_final_hardened_v2_cleaned.json)",
            data=dataset_bytes,
            file_name="test2_final_hardened_v2_cleaned.json",
            mime="application/json",
            use_container_width=True,
        )
        file_size_mb = len(dataset_bytes) / (1024 * 1024)
        st.markdown(f"""
        <div class="info-panel" style="margin-top:0.6rem;">
            📦 <strong>Dataset file:</strong> test2_final_hardened_v2_cleaned.json
            &nbsp;·&nbsp; <strong>Size:</strong> {file_size_mb:.2f} MB
            &nbsp;·&nbsp; <strong>Format:</strong> JSON array (list of conversation objects)
            &nbsp;·&nbsp; <strong>Conversations:</strong> {stats.get('total', '160')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warn-panel">
            ⚠️ Dataset file not found at the expected path. Ensure
            <code>Datasets/test2_final_hardened_v2_cleaned.json</code> exists in the project root.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:#334155; font-size:0.8rem;">
    Evaluating Mitigation Robustness Under Multi-Turn Prompt Injection &nbsp;·&nbsp;
    Jibran Shaikh &amp; Syeda Wania Hussain &nbsp;·&nbsp; GenAI Research — 2026
</div>
""", unsafe_allow_html=True)
