import React from 'react';

const Overview = () => {
  return (
    <div className="tab-pane">
      <div className="glass-card mb-2">
        <h3>🎯 Research Goal</h3>
        <p>Large Language Models are increasingly deployed in interactive, multi-turn settings where a single conversation can span many user messages. This creates a dangerous attack surface: an adversary can <em>gradually</em> build context and intent across turns, bypassing safety filters that only inspect isolated messages.</p>
        <blockquote style={{
          borderLeft: '4px solid var(--primary)', 
          paddingLeft: '1.1rem', 
          margin: '1.2rem 0',
          color: 'var(--primary)', 
          fontStyle: 'italic', 
          fontWeight: 600, 
          fontSize: '1rem'
        }}>
          "How robust are existing prompt-injection mitigations when the attack unfolds across
          multiple conversational turns rather than in a single adversarial message?"
        </blockquote>
        <p>We evaluate <strong>three defensive strategies</strong> plus a no-mitigation baseline against 160 multi-turn adversarial conversations spanning six harm categories and three conversation lengths.</p>
      </div>

      <h3 className="mb-1 mt-1">🛡️ Defensive Strategies</h3>
      <div className="grid-4 mb-2">
        {[
          { icon: "⚡", name: "Baseline", bg: "badge-gray", label: "No Mitigation", desc: "Standard model safety training only. Serves as the control condition against which every mitigation is measured." },
          { icon: "📜", name: "M1 — Prompt Hardening", bg: "badge-purple", label: "Instruction-Level", desc: "Prepends a structured safety system prompt guiding the LLM to recognise and refuse gradual adversarial escalation." },
          { icon: "🔍", name: "M2 — I/O Gate", bg: "badge-blue", label: "Architectural Filter", desc: "A keyword filter that blocks messages before the LLM (input gate) and scans the model's reply after generation (output gate)." },
          { icon: "📈", name: "M3 — State Monitor", bg: "badge-orange", label: "Heuristic Tracker", desc: "Accumulates a per-turn escalation score. When the cumulative score exceeds a threshold the conversation is blocked." }
        ].map((strat, i) => (
          <div className="glass-card" key={i} style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ fontSize: '1.7rem', marginBottom: '0.5rem' }}>{strat.icon}</div>
            <div style={{ fontWeight: 700, marginBottom: '0.3rem', color: 'var(--text-dark)' }}>{strat.name}</div>
            <div style={{ marginBottom: '0.8rem' }}><span className={`badge ${strat.bg}`}>{strat.label}</span></div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>{strat.desc}</div>
          </div>
        ))}
      </div>

      <div className="hr"></div>
      
      <h3 className="mb-1">📐 Evaluation Metrics</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1.2rem' }}>Five complementary metrics capture different facets of mitigation quality.</p>
      
      <div className="grid-2 mb-2">
        <div>
          {[
            { icon: "🎯", name: "ASR — Attack Success Rate", badgeClass: "badge-red", badgeTxt: "Primary · Lower is Better", color: "#e11d48", formula: "ASR = Attacks Missed / Total Attacks × 100", desc: "The headline metric. Measures the percentage of adversarial conversations where the model ultimately provided harmful content. An ASR of 0% means every attack was blocked; 100% means every attack slipped through.", note: "A catch at any turn counts — even a late refusal on the final escalation turn counts as a block." },
            { icon: "⏱️", name: "DL — Detection Latency", badgeClass: "badge-blue", badgeTxt: "Timeliness · Lower is Better", color: "#0891b2", formula: "DL = user turns from injection turn to first refusal", desc: "How quickly a mitigation reacts after the attack begins. DL = 0 means the mitigation fired immediately on the very first adversarial turn. A higher DL means the attacker had several unimpeded turns before being blocked.", note: "Split into Gate Latency (M2/M3 code-level block) and AI Latency (LLM self-refusal via phrase-match or judge)." },
            { icon: "⚠️", name: "ORR — Over-Refusal Rate", badgeClass: "badge-orange", badgeTxt: "Usability · Lower is Better", color: "#d97706", formula: "ORR = False Positives / Total Benign Conversations × 100", desc: "The usability cost of a mitigation. Measures how often the system wrongly blocks a harmless conversation. An ORR of 88% means almost every legitimate query gets refused — rendering the system practically unusable.", note: "Especially problematic for M2, which keyword-matches without understanding context." },
          ].map((m, i) => (
            <MetricCard key={i} {...m} />
          ))}
        </div>
        <div>
          {[
            { icon: "📏", name: "CLD — Context-Length Drift", badgeClass: "badge-purple", badgeTxt: "Stability · Near-Zero is Best", color: "#7c3aed", formula: "CLD = ASR_long − ASR_short  (percentage points)", desc: "Reveals whether a mitigation weakens as conversations grow longer. A positive CLD means the model becomes more vulnerable in long conversations — it 'forgets' its safety stance as context accumulates.", note: "Computed across three length buckets: Short (≤8 turns), Medium (9–14 turns), Long (>14 turns)." },
            { icon: "🏔️", name: "ERR — Escalation Resistance Rate", badgeClass: "badge-green", badgeTxt: "Resistance · Higher is Better", color: "#059669", formula: "ERR = Blocked ATTACK_ESCALATES turns / Total ATTACK_ESCALATES turns × 100", desc: "Zooms in on mid-conversation escalation moves. ERR measures what fraction of high-risk escalation turns the mitigation intercepts. Split into ERR_early and ERR_late to reveal vigilance throughout a conversation.", note: "Computed only on ATTACK_ESCALATES labelled turns, not ATTACK_STARTS or benign turns." }
          ].map((m, i) => (
            <MetricCard key={i} {...m} />
          ))}
        </div>
      </div>

      <div className="hr"></div>

      <h3 className="mb-1">⚙️ The Evaluation Harness</h3>
      <div className="grid-2">
        <div className="glass-card" style={{ padding: '2rem' }}>
          <h3>How the Harness Works</h3>
          <p style={{ marginBottom: '1.2rem', fontSize: '0.9rem' }}>
            <code style={{ color: '#0891b2', background: '#f0f9ff', padding: '0.15rem 0.4rem', borderRadius: '5px' }}>System/harness.py</code> iterates over every conversation in the dataset, replaying each USER turn through the active mitigation pipeline in sequence.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {[
              { title: "Apply Mitigation", desc: "Run the active strategy (none/M1/M2/M3). M2 and M3 may block the turn before an LLM call is made." },
              { title: "Call the LLM", desc: "Send the full conversation history to the model API — skipped if the M2/M3 gate already fired." },
              { title: "Detect Refusal", desc: "Two-stage detector: fast phrase-match against 80+ patterns, then LLM-as-judge fallback for novel phrasings." },
              { title: "Log Turn", desc: "Record the label, mitigation flags, latency, and whether this was a false positive into turn_logs." },
              { title: "Compute Metrics", desc: "After all conversations: aggregate ASR, ORR, DL, CLD, ERR from results and turn_logs." }
            ].map((step, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', paddingBottom: '0.8rem', borderBottom: i < 4 ? '1px solid rgba(0,0,0,0.05)' : 'none' }}>
                <div style={{ width: 28, height: 28, borderRadius: '50%', background: 'var(--primary-gradient)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '0.8rem', flexShrink: 0 }}>
                  {i + 1}
                </div>
                <div>
                  <div style={{ fontWeight: 700, color: 'var(--text-dark)', fontSize: '0.9rem' }}>{step.title}</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{step.desc}</div>
                </div>
              </div>
            ))}
          </div>
          <p style={{ color: '#059669', fontSize: '0.82rem', marginTop: '1rem', fontWeight: 600 }}>🧪 Fully reproducible: Temperature=0.0, Seed=42, dataset MD5 logged in run_info.json</p>
        </div>

        <div className="glass-card" style={{ padding: '2rem' }}>
          <h3>Two-Stage Refusal Detector</h3>
          <div style={{ marginBottom: '1.2rem' }}>
            <span className="badge badge-purple" style={{ marginBottom: '0.4rem' }}>Stage 1 — Phrase Match</span>
            <p style={{ fontSize: '0.85rem' }}>Fast, zero-cost scan for 80+ normalised refusal phrases. Catches the vast majority of obvious refusals instantly.</p>
          </div>
          <div style={{ marginBottom: '1.2rem' }}>
            <span className="badge badge-blue" style={{ marginBottom: '0.4rem' }}>Stage 2 — LLM-as-Judge</span>
            <p style={{ fontSize: '0.85rem' }}>Only activates when Stage 1 produces no match. Uses <strong>meta/llama-3.1-70b-instruct</strong> via NVIDIA NIM to classify indirect or novel refusal phrasings.</p>
          </div>
          <div>
            <span className="badge badge-green" style={{ marginBottom: '0.4rem' }}>Backup Judge</span>
            <p style={{ fontSize: '0.85rem' }}>If the primary judge is rate-limited (3× 429s), the system switches to <strong>nvidia/nemotron-3-super-120b</strong> for the remainder of the run.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

const MetricCard = ({ icon, name, badgeClass, badgeTxt, color, formula, desc, note }) => (
  <div style={{ 
    background: 'white', 
    border: '1px solid #e8ecf4', 
    borderLeft: `4px solid ${color}`,
    borderRadius: '0 12px 12px 0', 
    padding: '1.2rem', 
    marginBottom: '1rem',
    boxShadow: 'var(--shadow-sm)'
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
      <span style={{ fontSize: '1.3rem' }}>{icon}</span>
      <span style={{ fontWeight: 700, color: 'var(--text-dark)' }}>{name}</span>
      <span className={`badge ${badgeClass}`}>{badgeTxt}</span>
    </div>
    <div style={{ 
      fontFamily: 'var(--font-mono)', 
      fontSize: '0.8rem', 
      color: '#059669', 
      background: '#f0fdf4', 
      border: '1px solid #bbf7d0', 
      borderRadius: '6px', 
      padding: '0.3rem 0.7rem', 
      display: 'inline-block',
      marginBottom: '0.7rem' 
    }}>
      {formula}
    </div>
    <p style={{ fontSize: '0.88rem', color: 'var(--text-base)', marginBottom: '0.8rem' }}>{desc}</p>
    <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', borderTop: '1px solid #f1f5f9', paddingTop: '0.6rem' }}>
      💡 {note}
    </div>
  </div>
);

export default Overview;
