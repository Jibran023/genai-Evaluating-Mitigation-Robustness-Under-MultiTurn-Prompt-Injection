import React, { useState, useEffect } from 'react';
import { getAvailableModels, getMetricsForModel, getComparisonSummary, loadResultsData } from '../utils/dataStore';
import { 
  ResponseCurvesChart, 
  RunHistoryChart, 
  EfficiencyChart, 
  AsrByLengthChart, 
  ReliabilityChart, 
  HeatmapChart 
} from './Charts';
import CrossModelAnalysis from './CrossModelAnalysis';

const MITIGATION_LABELS = {
  "none": "Baseline (None)",
  "m1": "M1 — Prompt Hardening",
  "m2": "M2 — I/O Gate",
  "m3": "M3 — State Monitor"
};

const MITIGATIONS = ["none", "m1", "m2", "m3"];
const BADGE_MAP = { "none": "badge-gray", "m1": "badge-purple", "m2": "badge-blue", "m3": "badge-orange" };

const Findings = () => {
  const models = getAvailableModels();
  const [selectedSlug, setSelectedSlug] = useState(models[0] || "");
  const [resultsData, setResultsData] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedSlug) return;
    setLoading(true);
    loadResultsData(selectedSlug).then(data => {
      setResultsData(data);
      setLoading(false);
    });
  }, [selectedSlug]);

  if (!selectedSlug) {
    return <div className="warn-panel mt-1"><strong>⚠️ No Models Found</strong><br/>Ensure the results directory contains data.</div>;
  }

  const metrics = getMetricsForModel(selectedSlug);
  const compSummary = getComparisonSummary(selectedSlug);
  const availableMits = MITIGATIONS.filter(m => metrics[m]);
  


  const gapM1 = metrics.none && metrics.m1 ? 
    (metrics.none.attack_success_rate_pct - metrics.m1.attack_success_rate_pct).toFixed(1) : "N/A";

  return (
    <div className="tab-pane">
      <div className="select-wrapper">
        <div style={{ fontWeight: 600 }}>Select Model:</div>
        <select value={selectedSlug} onChange={e => setSelectedSlug(e.target.value)}>
          {models.map(m => (
            <option key={m} value={m}>{m === 'openai-gpt-oss-120b' ? 'OpenAI GPT-OSS 120B' : m}</option>
          ))}
          <option value="__all__">All Models (Cross-Model Analysis)</option>
        </select>
        <button className="btn-primary" onClick={() => window.dispatchEvent(new Event('resize'))}>🔬 Refresh Layout</button>
      </div>

      {selectedSlug === '__all__' ? (
        <CrossModelAnalysis />
      ) : availableMits.length === 0 ? (
        <div className="warn-panel mt-1">No result files found for this model.</div>
      ) : (
        <>
          <div className="glass-card mb-2" style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1rem 1.5rem' }}>
            <div style={{ fontSize: '2rem' }}>🤖</div>
            <div>
              <div style={{ fontWeight: 800, fontSize: '1.2rem', color: 'var(--text-dark)' }}>
                {selectedSlug === 'openai-gpt-oss-120b' ? 'OpenAI GPT-OSS 120B' : selectedSlug}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{selectedSlug}</div>
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {availableMits.map(m => (
                <span key={m} className={`badge ${BADGE_MAP[m]}`}>{MITIGATION_LABELS[m]}</span>
              ))}
            </div>
          </div>

          <h3 className="mb-1">📊 Key Metrics at a Glance</h3>
          <div className={`grid-${availableMits.length} mb-2`}>
            {availableMits.map(mit => {
              const m = metrics[mit];
              return (
                <div key={mit} className="glass-card kpi-card" style={{ '--kpi-color': `var(--color-${mit === 'none' ? 'none' : mit})` }}>
                  <div className="kpi-label" style={{ marginBottom: '0.8rem' }}>{MITIGATION_LABELS[mit]}</div>
                  <div className="grid-2" style={{ gap: '0.5rem' }}>
                    <div><div className="kpi-val" style={{ color: '#e11d48' }}>{m.attack_success_rate_pct ?? '—'}%</div><div className="kpi-label">ASR ↓</div></div>
                    <div><div className="kpi-val">{m.over_refusal_rate_pct ?? '—'}%</div><div className="kpi-label">ORR ↓</div></div>
                    <div><div className="kpi-val" style={{ color: '#059669' }}>{m.err_overall ?? '—'}%</div><div className="kpi-label">ERR ↑</div></div>
                    <div><div className="kpi-val">{m.mean_detection_latency_turns ?? '—'}</div><div className="kpi-label">DL (turns)</div></div>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '1rem' }}>Caught: {m.attacks_caught ?? '—'}/110 attacks</div>
                </div>
              );
            })}
          </div>

          <div className="hr"></div>

          <h3 className="mb-1">📈 Interactive Visualisations</h3>
          {!loading ? (
            <>
              {/* Row 1 */}
              <div className="grid-2 mb-2">
                <div className="glass-card"><ResponseCurvesChart results={resultsData} mits={availableMits} /></div>
                <div className="glass-card"><RunHistoryChart results={resultsData} mits={availableMits} /></div>
              </div>

              {/* Row 2 */}
              <div className="grid-2 mb-2">
                <div className="glass-card">
                  {compSummary ? <EfficiencyChart comp={compSummary} mits={availableMits} /> : <div>Data missing</div>}
                </div>
                <div className="glass-card">
                  <AsrByLengthChart metrics={metrics} mits={availableMits} />
                </div>
              </div>

              {/* Row 3 */}
              <div className="grid-2 mb-2">
                <div className="glass-card">
                  {compSummary ? <ReliabilityChart comp={compSummary} mits={availableMits} /> : <div>Data missing</div>}
                </div>
                <div className="glass-card">
                  {compSummary ? <HeatmapChart comp={compSummary} mits={availableMits} /> : <div>Data missing</div>}
                </div>
              </div>
            </>
          ) : (
            <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>Loading heavy chart data...</div>
          )}

          <div className="hr"></div>

          <h3 className="mb-1">🔍 Key Findings & Interpretation</h3>
          {[
            { icon: "🏆", bg: "white", bc: "var(--color-m1)", title: "Finding 1 — Best Attack Containment", body: `<strong>M1 (Prompt Hardening)</strong> achieves the lowest Attack Success Rate. The no-mitigation baseline lets <strong>${metrics.none?.attack_success_rate_pct}%</strong> of attacks succeed. M1 drops this to <strong>${metrics.m1?.attack_success_rate_pct}%</strong> — a ${gapM1}pp reduction — by giving the LLM explicit escalation-awareness through its system prompt, without any architectural overhead.` },
            { icon: "⚠️", bg: "#fffbeb", bc: "#d97706", title: "Finding 2 — The M2 Usability Crisis", body: `The I/O Gate (M2) carries an enormous usability cost: an Over-Refusal Rate of <strong>${metrics.m2?.over_refusal_rate_pct}%</strong> on fully benign conversations. Nearly 9 in 10 legitimate user queries are wrongly blocked. This traces directly to the keyword filter's inability to distinguish context.` },
            { icon: "📏", bg: "white", bc: "var(--color-m3)", title: "Finding 3 — Context-Length Vulnerability", body: `M3 (State Monitor) shows a large positive CLD of <strong>+${metrics.m3?.context_length_drift_pct}pp</strong>, meaning protection degrades significantly in long conversations. Once M3 blocks a turn and resets its cumulative score, the attacker gets a clean slate.` },
            { icon: "✅", bg: "#f0fdf4", bc: "#059669", title: "Overall Verdict", body: `For OpenAI GPT-OSS 120B, M1 is the most balanced mitigation: lowest ASR (${metrics.m1?.attack_success_rate_pct}%), near-zero false positives, and stable protection across all conversation lengths.` }
          ].map((f, i) => (
            <div key={i} style={{ background: f.bg, border: `1px solid rgba(0,0,0,0.1)`, borderLeft: `4px solid ${f.bc}`, padding: '1rem 1.5rem', borderRadius: '0 12px 12px 0', marginBottom: '1rem', fontSize: '0.9rem' }}>
              <div style={{ fontWeight: 700, color: f.bc, marginBottom: '0.3rem', textTransform: 'uppercase', fontSize: '0.75rem', letterSpacing: '0.05em' }}>{f.icon} {f.title}</div>
              <div dangerouslySetInnerHTML={{ __html: f.body }} style={{ color: 'var(--text-base)', lineHeight: 1.6 }} />
            </div>
          ))}
        </>
      )}
    </div>
  );
};

export default Findings;
