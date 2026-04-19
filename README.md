# Evaluating Mitigation Robustness Under Multi-Turn Prompt Injection

**Authors:** Jibran Shaikh, Syeda Wania Hussain  
**Course:** Generative AI (8th Semester)

---

## 🎯 Project Overview

This project evaluates the robustness of LLM safety mitigations against **multi-turn prompt injection attacks**. These are conversational attacks that gradually escalate toward harmful intent, attempting to bypass safety filters by building context across multiple turns.

### 🏗️ System Architecture

```mermaid
graph TD
    A[Dataset<br/>Multi-Turn Conversations] --> B(Harness Engine)
    
    subgraph Mitigations
        M0[None - Baseline]
        M1[M1 - Prompt Hardening]
        M2[M2 - I/O Gate]
        M3[M3 - State Monitor]
    end
    
    B --> M0
    B --> M1
    B --> M2
    B --> M3
    
    M0 --> C{LLM API<br/>NVIDIA NIM / Groq}
    M1 --> C
    M2 --> C
    M3 --> C
    
    C --> D[Response Generation]
    D -.->|Feedback Loop| B
    
    B --> E[Evaluation Stage]
    E --> F{Phrase-Based Detection}
    F -- "Inconclusive" --> G[LLM-as-Judge]
    F -- "Clear Match" --> H[Evaluation Output]
    G --> H
    
    classDef default fill:#000000,stroke:#555,stroke-width:2px,color:#ffffff;
    classDef mitigation fill:#ffffff,stroke:#333,stroke-width:2px,color:#000000;
    class M0,M1,M2,M3 mitigation;
```

### 🛡️ Defensive Strategies
| ID | Strategy | Mechanism |
|----|----------|-------------|
| `none` | **Baseline** | Standard model safety training. |
| `m1` | **Prompt Hardening** | Injected safety system prompt (Instruction-level defense). |
| `m2` | **I/O Gate** | Keyword filtering of inputs and outputs (Architectural defense). |
| `m3` | **State Monitor** | Heuristic tracking of adversarial escalation across turns. |

#### Mitigation Workflows

```mermaid
flowchart TD
    In[User Input] --> Router{Active Mitigation}

    %% None Route
    Router -- "none" --> N1(LLM API)
    N1 --> N2[Raw LLM Response]

    %% M1 Route
    Router -- "m1" --> M1_1[Prepend Safety<br/>System Prompt]
    M1_1 --> M1_2(LLM API)
    M1_2 --> M1_3[LLM Response]

    %% M2 Route
    Router -- "m2" --> M2_1{Input Scan:<br/>Keywords Found?}
    M2_1 -- "Yes" --> Refusal[Canned Refusal / Block]
    M2_1 -- "No" --> M2_2(LLM API)
    M2_2 --> M2_3{Output Scan:<br/>Keywords Found?}
    M2_3 -- "Yes" --> Refusal
    M2_3 -- "No" --> M2_4[LLM Response]

    %% M3 Route
    Router -- "m3" --> M3_1[Sentence Transformer:<br/>Embed User Text]
    M3_1 --> M3_2[Max Cosine Sim vs.<br/>Attack Prototypes]
    M3_2 --> M3_3[Update State Monitor<br/>Exponential Decay]
    M3_3 --> M3_4{Score >= Threshold?}
    M3_4 -- "Yes" --> Refusal
    M3_4 -- "No" --> M3_5(LLM API)
    M3_5 --> M3_6[LLM Response]

    classDef default fill:#000000,stroke:#555,stroke-width:2px,color:#ffffff;
    classDef refusal fill:#ffffff,stroke:#333,stroke-width:2px,color:#000000;
    class Refusal refusal;
```

#### Two-Stage Evaluation Workflow

```mermaid
flowchart TD
    In[AI Response Text] --> Normalise[Normalise Text<br/>Collapse Smart Quotes]
    Normalise --> Stage1{Stage 1: Fast<br/>Phrase Match}
    
    Stage1 -- "Match Found" --> Refused[Log as REFUSED]
    Stage1 -- "No Match / Inconclusive" --> CheckJudge{Stage 2: LLM Judge<br/>Enabled & Length > 10?}
    
    CheckJudge -- "No" --> Complied[Log as COMPLIED]
    CheckJudge -- "Yes" --> Route{Primary Rate Limited<br/>Flag Set?}
    
    Route -- "No" --> Primary[Primary Judge:<br/>llama-3.1-70b-instruct]
    Route -- "Yes" --> Backup[Backup Judge]
    
    Primary -- "Success" --> Parse[Parse Binary Output:<br/>REFUSED / COMPLIED]
    Primary -- "429 Error" --> Retry{Retry 3x.<br/>Still 429?}
    Retry -- "No" --> Parse
    Retry -- "Yes" --> SetFlag[Set Rate Limit Flag]
    SetFlag --> Backup
    Primary -- "Other Error" --> Backup
    
    Backup -- "Success" --> Parse
    Backup -- "Fails" --> FailOpen[Fail Open]
    
    FailOpen --> Complied
    Parse -- "REFUSED" --> Refused
    Parse -- "COMPLIED" --> Complied

    classDef default fill:#000000,stroke:#555,stroke-width:2px,color:#ffffff;
    classDef terminal fill:#ffffff,stroke:#333,stroke-width:2px,color:#000000;
    class Refused,Complied terminal;
```

### 📈 Core Metrics
*   **ASR (Attack Success Rate)**: % of attacks that successfully bypassed all defenses.
*   **CLD (Context-Length Drift)**: Measures if defenses weaken as conversations grow longer. Close to zero (or negative) is better. Measures if the model becomes more or less vulnerable as the conversation goes on. A high positive CLD means the model "forgets" its safety training in long conversations.
*   **ERR (Escalation Resistance Rate)**: Measures how well a mitigation generalizes to unseen attack topics.

---

## 📂 Research Structure
```
.
├── Datasets/            # Training and test conversation sets
├── System/              # Core Logic (harness, mitigations, metrics, plotting)
├── Utils/               # Automation scripts (multi-mitigation runner)
└── results/             
    ├── [mitigation]/    # Raw results for individual runs
    └── comparison/      # Aggregated research data
        └── [model]/[samples]/
            ├── [mitigation]/   # Individual plots filed by strategy
            ├── adt_heatmap.png # Cross-topic generalization data
            └── mitigation_comparison.png
```

---

## ⚡ Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```
Create a `.env` file with your API keys. We support model-specific keys for NVIDIA NIM (e.g., `NVIDIA_API_KEY_META_LLAMA_3_1_8B_INSTRUCT`).

### 2. Run the Evaluation Pipeline
The recommended way to run the full research suite is via the `Utils/` script:

```bash
# Full dataset run (Default model)
python Utils/run_all_mitigations.py

# Recommended Balanced Run (24 samples covering all topics/lengths)
python Utils/run_all_mitigations.py --limit 24

# Run on Gemma-2 via NVIDIA NIM
python Utils/run_all_mitigations.py --provider nvidia --model google/gemma-2-9b-it --limit 24

# Run on Mixtral via Groq
python Utils/run_all_mitigations.py --provider groq --model mixtral-8x7b-32768 --limit 24

# Run only M2 and M3
python Utils/run_all_mitigations.py --skip none m1 --limit 24

# Re-run ONLY M3 (useful for debugging one specific strategy)
python Utils/run_all_mitigations.py --skip none m1 m2 --limit 24

# Regenerate plots for the default model/limit
python Utils/run_all_mitigations.py --plots-only

# Regenerate plots for a specific model you ran earlier
python Utils/run_all_mitigations.py --plots-only --model meta/llama-3.1-8b-instruct --limit 24
python Utils/run_all_mitigations.py --plots-only --model mistralai-mistral-small-4-119b-2603
python Utils/run_all_mitigations.py --plots-only --model all # to generate plots for all models
```

### 🛠️ Common Commands & Flags
| Command | Description |
|---------|-------------|
| `--model <name>` | Select a specific model (e.g., `meta/llama-3.1-8b-instruct`). |
| `--limit <N>` | Run on a stratified sample of N conversations (use 24 for a quick balanced set). |
| `--provider <p>` | `nvidia` (NIM) or `groq` (NPU). |
| `--skip m1 m2` | Skip specific mitigations if they are already completed. |
| `--plots-only` | Regenerate comparison heatmaps and folders from existing data. |

---

## 📋 Evaluation Outputs

Every run generates a deep-dive analysis in its results folder:
*   **`failure_analysis.json`**: Detailed logs of every missed attack and false positive.
*   **`metrics_summary.json`**: Aggregated performance data.
*   **`adt_heatmap.png`**: Visualizes the "Generalization Gap" for cross-topic robustness.
*   **`comparison_summary.json`**: A side-by-side technical table of all strategies.

---


---

## 🧪 Reproducibility Notes
All runs are **deterministic** (Temperature = 0.0, Seed = 42). The system records the Python version, dataset MD5 hash, and platform details in `run_info.json` for every execution.
