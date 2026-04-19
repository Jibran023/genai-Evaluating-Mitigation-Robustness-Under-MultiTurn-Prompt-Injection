import React from 'react';
import { getDataset } from '../utils/dataStore';
import { DatasetPieChart, DatasetBarChart } from './Charts';

const TOPIC_LABELS = {
  "cybersecurity": "Cybersecurity",
  "dangerous_instructions": "Dangerous Instructions",
  "bypassing_ai_rules": "Bypassing AI Rules",
  "misinformation": "Misinformation",
  "hate_speech": "Hate Speech",
  "social_engineering": "Social Engineering"
};

const Dataset = () => {
  const data = getDataset();
  
  if (!data || data.length === 0) {
    return <div className="warn-panel mt-1"><strong>⚠️ Dataset Not Found</strong><br/>Ensure Datasets/test2_final_hardened_v2_cleaned.json is present.</div>;
  }

  // Calculate stats
  const stats = {
    total: data.length,
    attacks: data.filter(d => d.attack_type && d.attack_type !== "none").length,
    benign: data.filter(d => !d.attack_type || d.attack_type === "none").length,
    topics: {},
    lengths: { "Short (<=8)": 0, "Medium (9-14)": 0, "Long (>14)": 0 }
  };

  data.forEach(d => {
    if (d.topic) {
      stats.topics[d.topic] = (stats.topics[d.topic] || 0) + 1;
    }
    const l = d.turns ? d.turns.length : 0;
    if (l <= 8) stats.lengths["Short (<=8)"]++;
    else if (l <= 14) stats.lengths["Medium (9-14)"]++;
    else stats.lengths["Long (>14)"]++;
  });

  const sample = data[0]; // Take first conversation as sample

  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = "test2_final_hardened_v2_cleaned.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="tab-pane">
      <div className="glass-card mb-2" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--text-dark)', marginBottom: '0.3rem' }}>
            📦 Evaluation Dataset — Hardened V2
          </div>
          <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.6, maxWidth: 720 }}>
            160 synthetic multi-turn adversarial conversations, stratified across 6 harm categories and 3 conversation-length groups, with turn-level attack labels.
          </div>
        </div>
        <button className="btn-primary" onClick={handleDownload} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexShrink: 0 }}>
          ⬇️ Download Dataset
        </button>
      </div>

      <div className="hr"></div>

      <div className="grid-6 mb-2" style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '1rem' }}>
        {[
          { v: stats.total, l: "Total Convs", c: "linear-gradient(90deg,#4f46e5,#7c3aed)" },
          { v: stats.attacks, l: "Attack Convs", c: "linear-gradient(90deg,#e11d48,#be123c)" },
          { v: stats.benign, l: "Benign Convs", c: "linear-gradient(90deg,#059669,#047857)" },
          { v: 6, l: "Harm Categories", c: "linear-gradient(90deg,#d97706,#b45309)" },
          { v: 3, l: "Length Groups", c: "linear-gradient(90deg,#0891b2,#0369a1)" },
          { v: 160, l: "Total Samples", c: "linear-gradient(90deg,#7c3aed,#6d28d9)" }
        ].map((k, i) => (
          <div key={i} className="glass-card" style={{ padding: '1rem', textAlign: 'center', borderTop: `4px solid transparent`, borderImage: `${k.c} 1` }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-dark)' }}>{k.v}</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>{k.l}</div>
          </div>
        ))}
      </div>

      <div className="grid-2 mb-2">
        <div className="glass-card"><DatasetPieChart dataObj={stats.topics} /></div>
        <div className="glass-card"><DatasetBarChart lenData={stats.lengths} /></div>
      </div>

      <div className="hr"></div>

      <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-dark)', marginBottom: '0.8rem' }}>
        📄 Sample Dataset Entry — <code style={{ fontSize: '0.9rem', color: 'var(--primary)' }}>V2-001</code>
      </div>

      {sample && (
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem', marginBottom: '1rem' }}>
            <span className="badge badge-purple">Topic: {TOPIC_LABELS[sample.topic] || sample.topic}</span>
            <span className="badge badge-blue">Type: {sample.attack_type || '—'}</span>
            <span className="badge badge-orange">Injection Turn: {sample.injection_turn || '—'}</span>
            <span className="badge badge-red">{sample.success === "no" ? "✅ Blocked" : "❌ Succeeded"}</span>
            <span className="badge badge-gray">{sample.turns?.length || 0} turns total</span>
          </div>

          <div className="glass-card" style={{ padding: '1.2rem 1.5rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            {sample.turns?.slice(0, 8).map((t, i) => {
              const lblMap = {
                "BENIGN": { c: "badge-green", t: "Benign" },
                "ATTACK_STARTS": { c: "badge-orange", t: "Attack Starts" },
                "ATTACK_ESCALATES": { c: "badge-red", t: "Escalates" },
                "DETECTED": { c: "badge-purple", t: "Detected" }
              };
              const l = lblMap[t.label];
              
              return (
                <div key={i} className={`turn-row ${t.speaker === 'USER' ? 'turn-user' : 'turn-model'}`}>
                  <div className="turn-speaker">{t.speaker}</div>
                  <div className="turn-content">{t.text}</div>
                  {l && <div style={{ minWidth: 100, textAlign: 'right' }}><span className={`badge ${l.c}`}>{l.t}</span></div>}
                </div>
              );
            })}
            {sample.turns?.length > 8 && (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', padding: '0.5rem 0' }}>
                … {sample.turns.length - 8} more turns not shown
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default Dataset;
