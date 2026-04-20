import React, { useMemo, useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import PlotlyFactory from 'react-plotly.js/factory';
import { getAllModelsComparisons } from '../utils/dataStore';

import googleLogo from '../assets/google_logo.png';
import mistralLogo from '../assets/mistral_logo.png';
import openaiLogo from '../assets/openai_logo.png';
import qwenLogo from '../assets/qwen_logo.png';
import moonshotLogo from '../assets/moonshot_logo.png';

const createPlot = PlotlyFactory.default || PlotlyFactory;
const Plot = createPlot(Plotly);

// ── Constants ────────────────────────────────────────────────────────────────
const MODEL_META = {
  'google-gemma-3-27b-it':               { label: 'Gemma 3 27B',   short: 'Gemma',   logo: googleLogo,   color: '#4285f4' },
  'mistralai-mistral-small-4-119b-2603': { label: 'Mistral Small', short: 'Mistral', logo: mistralLogo,  color: '#f97316' },
  'openai-gpt-oss-120b':                 { label: 'GPT-OSS 120B',  short: 'GPT-OSS', logo: openaiLogo,   color: '#10b981' },
  'qwen-qwen3-next-80b-a3b-instruct':    { label: 'Qwen3 80B',     short: 'Qwen3',   logo: qwenLogo,     color: '#8b5cf6' },
  'moonshotai-kimi-k2-instruct':         { label: 'Kimi K2',       short: 'Kimi',    logo: moonshotLogo, color: '#ec4899' },
};

const MIT_META = {
  none: { label: 'Baseline',          short: 'None', color: '#94a3b8' },
  m1:   { label: 'M1: Prompt Harden', short: 'M1',   color: '#6366f1' },
  m2:   { label: 'M2: I/O Gate',      short: 'M2',   color: '#0891b2' },
  m3:   { label: 'M3: State Monitor', short: 'M3',   color: '#f59e0b' },
};

const MITS = ['none', 'm1', 'm2', 'm3'];
const MIT_LABELS = ['Baseline', 'M1: Prompt Harden', 'M2: I/O Gate', 'M3: State Monitor'];

const base = {
  font: { family: 'Inter, sans-serif', color: '#1e293b', size: 12 },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  hoverlabel: { namelength: -1 },
};

// ── Helpers ──────────────────────────────────────────────────────────────────
const modelLabel = (slug) => MODEL_META[slug]?.label ?? slug;
const modelColor = (slug) => MODEL_META[slug]?.color ?? '#64748b';
const mitColor   = (id)   => MIT_META[id]?.color ?? '#64748b';

function gmean(a, b) {
  if (a > 0 && b > 0) return parseFloat(Math.sqrt(a * b).toFixed(1));
  return 0;
}

// ── Shared UI Components ──────────────────────────────────────────────────────

/** Color-coded range pill */
const RangePill = ({ color, label, range }) => (
  <span style={{
    display: 'inline-flex', alignItems: 'center', gap: '5px',
    background: `${color}15`, border: `1px solid ${color}40`,
    borderRadius: '999px', padding: '2px 10px', fontSize: '0.75rem',
    fontWeight: 600, color,
  }}>
    <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, display: 'inline-block' }} />
    {label}: {range}
  </span>
);

/** Formula & interpretation panel shown below each chart */
const MetricInfoPanel = ({ formula, description, unit, ranges }) => {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: '0.75rem', borderTop: '1px solid #f1f5f9', paddingTop: '0.75rem' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: '6px',
          background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'inherit',
          fontSize: '0.8rem', fontWeight: 600, color: '#64748b', padding: 0,
        }}
      >
        <span style={{ fontSize: '1rem' }}>{open ? '▾' : '▸'}</span>
        {open ? 'Hide' : 'Show'} formula & interpretation
      </button>
      {open && (
        <div style={{
          marginTop: '0.6rem', padding: '0.9rem 1.1rem',
          background: 'linear-gradient(135deg, #f8fafc, #f0f4ff)',
          border: '1px solid #e2e8f0', borderRadius: '10px',
          fontSize: '0.82rem', color: '#334155',
        }}>
          {/* Formula */}
          <div style={{ marginBottom: '0.6rem' }}>
            <span style={{ fontWeight: 700, color: '#4f46e5', textTransform: 'uppercase', fontSize: '0.68rem', letterSpacing: '0.06em' }}>Formula</span>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', background: '#1e293b', color: '#e2e8f0', padding: '6px 12px', borderRadius: '6px', marginTop: '4px', fontSize: '0.8rem', lineHeight: 1.6 }}>
              {formula}
            </div>
          </div>
          {/* Unit & Description */}
          <div style={{ marginBottom: '0.7rem', lineHeight: 1.6 }}>
            {unit && <span style={{ fontWeight: 700, color: '#0891b2' }}>Unit: </span>}
            {unit && <span style={{ fontStyle: 'italic' }}>{unit} · </span>}
            {description}
          </div>
          {/* Ranges */}
          {ranges && (
            <div>
              <span style={{ fontWeight: 700, color: '#475569', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Value guide: </span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '5px' }}>
                {ranges.map((r, i) => <RangePill key={i} {...r} />)}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

/** Logo legend row — shows provider logo + model color swatch + name */
const ModelLegend = ({ models }) => (
  <div style={{
    display: 'flex', flexWrap: 'wrap', gap: '10px', justifyContent: 'center',
    padding: '8px 0 4px', marginTop: '-4px',
  }}>
    {models.map(slug => {
      const meta = MODEL_META[slug];
      const color = modelColor(slug);
      return (
        <div key={slug} style={{
          display: 'flex', alignItems: 'center', gap: '6px',
          background: 'white', border: '1px solid #e2e8f0',
          borderRadius: '999px', padding: '4px 12px 4px 7px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.06)', fontSize: '0.8rem', fontWeight: 600, color: '#1e293b',
        }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: color, flexShrink: 0 }} />
          {meta?.logo && <img src={meta.logo} alt="" style={{ width: 14, height: 14, objectFit: 'contain' }} />}
          {meta?.label ?? slug}
        </div>
      );
    })}
  </div>
);

// ── BAR CHART COMPONENTS ──────────────────────────────────────────────────────

const AsrGroupedChart = ({ allData, models }) => {
  const data = useMemo(() => MITS.map(mit => ({
    name: MIT_META[mit].label,
    x: models.map(modelLabel),
    y: models.map(m => allData[m]?.mitigations?.[mit]?.attack_success_rate_pct ?? null),
    type: 'bar', marker: { color: mitColor(mit) },
    text: models.map(m => { const v = allData[m]?.mitigations?.[mit]?.attack_success_rate_pct; return v != null ? `${v}%` : ''; }),
    textposition: 'outside',
    hovertemplate: '<b>%{x}</b><br>ASR: %{y:.1f}%<extra>%{fullData.name}</extra>',
  })), [allData, models]);

  return (
    <>
      <Plot data={data} layout={{
        ...base, barmode: 'group',
        title: { text: 'Attack Success Rate (%) by Model & Mitigation<br><sup>Lower is better — percentage of attacks that bypassed the mitigation</sup>', x: 0.5, xanchor: 'center' },
        yaxis: { title: 'ASR (%)', range: [0, 115], showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        xaxis: { title: 'Model', showgrid: false, automargin: true },
        legend: { orientation: 'h', y: -0.28, x: 0.5, xanchor: 'center' },
        margin: { t: 90, b: 110, l: 65, r: 20 }, showlegend: true,
      }} useResizeHandler style={{ width: '100%', height: '420px' }} config={{ responsive: true, displayModeBar: false }} />
      <ModelLegend models={models} />
      <MetricInfoPanel
        formula={'ASR (%) = (Attacks Succeeded / Total Attack Samples) × 100'}
        unit="Percentage (0–100%)"
        description="Fraction of adversarial conversations where the attacker successfully extracted a harmful response. A mitigation with low ASR effectively blocked most attacks."
        ranges={[
          { color: '#10b981', label: 'Good', range: '< 20%' },
          { color: '#f59e0b', label: 'Moderate', range: '20–50%' },
          { color: '#ef4444', label: 'Poor', range: '> 50%' },
        ]}
      />
    </>
  );
};

const AsrReductionChart = ({ allData, models }) => {
  const data = useMemo(() => ['m1', 'm2', 'm3'].map(mit => ({
    name: MIT_META[mit].label,
    x: models.map(modelLabel),
    y: models.map(m => {
      const bA = allData[m]?.mitigations?.none?.attack_success_rate_pct ?? null;
      const mA = allData[m]?.mitigations?.[mit]?.attack_success_rate_pct ?? null;
      return (bA == null || mA == null) ? null : parseFloat((bA - mA).toFixed(1));
    }),
    type: 'bar', marker: { color: mitColor(mit) },
    text: models.map(m => {
      const bA = allData[m]?.mitigations?.none?.attack_success_rate_pct ?? null;
      const mA = allData[m]?.mitigations?.[mit]?.attack_success_rate_pct ?? null;
      if (bA == null || mA == null) return '';
      const d = bA - mA; return `${d >= 0 ? '+' : ''}${d.toFixed(1)} pp`;
    }),
    textposition: 'outside',
    hovertemplate: '<b>%{x}</b><br>ASR Reduction: %{y:.1f} percentage points<extra>%{fullData.name}</extra>',
  })), [allData, models]);

  return (
    <>
      <Plot data={data} layout={{
        ...base, barmode: 'group',
        title: { text: 'ASR Reduction vs Baseline (percentage points)<br><sup>Higher is better — how many points of ASR the mitigation eliminated</sup>', x: 0.5, xanchor: 'center' },
        yaxis: { title: 'ASR Reduction (percentage points)', showgrid: true, gridcolor: '#f1f5f9', automargin: true, zeroline: true, zerolinecolor: '#cbd5e1' },
        xaxis: { title: 'Model', showgrid: false, automargin: true },
        legend: { orientation: 'h', y: -0.28, x: 0.5, xanchor: 'center' },
        margin: { t: 90, b: 110, l: 85, r: 20 }, showlegend: true,
      }} useResizeHandler style={{ width: '100%', height: '420px' }} config={{ responsive: true, displayModeBar: false }} />
      <ModelLegend models={models} />
      <MetricInfoPanel
        formula={'ASR Reduction (pp) = Baseline ASR (%) − Mitigation ASR (%)'}
        unit="Percentage points (pp) — difference between two percentages"
        description="How much lower the Attack Success Rate became after applying the mitigation compared to the no-mitigation baseline. Positive values = mitigation helped. Negative = mitigation made things worse."
        ranges={[
          { color: '#10b981', label: 'Strong gain', range: '> 40 pp' },
          { color: '#f59e0b', label: 'Moderate', range: '10–40 pp' },
          { color: '#ef4444', label: 'Minimal / harmful', range: '< 10 pp' },
        ]}
      />
    </>
  );
};

const OrrGroupedChart = ({ allData, models }) => {
  const data = useMemo(() => MITS.map(mit => ({
    name: MIT_META[mit].label,
    x: models.map(modelLabel),
    y: models.map(m => allData[m]?.mitigations?.[mit]?.over_refusal_rate_pct ?? null),
    type: 'bar', marker: { color: mitColor(mit) },
    text: models.map(m => { const v = allData[m]?.mitigations?.[mit]?.over_refusal_rate_pct; return v != null ? `${v}%` : ''; }),
    textposition: 'outside',
    hovertemplate: '<b>%{x}</b><br>ORR: %{y:.1f}%<extra>%{fullData.name}</extra>',
  })), [allData, models]);

  return (
    <>
      <Plot data={data} layout={{
        ...base, barmode: 'group',
        title: { text: 'Over-Refusal Rate (%) by Model & Mitigation<br><sup>Lower is better — how often the mitigation wrongly blocked benign queries</sup>', x: 0.5, xanchor: 'center' },
        yaxis: { title: 'ORR (%)', range: [0, 115], showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        xaxis: { title: 'Model', showgrid: false, automargin: true },
        legend: { orientation: 'h', y: -0.28, x: 0.5, xanchor: 'center' },
        margin: { t: 90, b: 110, l: 65, r: 20 }, showlegend: true,
      }} useResizeHandler style={{ width: '100%', height: '420px' }} config={{ responsive: true, displayModeBar: false }} />
      <ModelLegend models={models} />
      <MetricInfoPanel
        formula={'ORR (%) = (Benign Queries Wrongly Blocked / Total Benign Queries) × 100'}
        unit="Percentage (0–100%)"
        description="Measures the usability cost of a mitigation. A high ORR means the model is too aggressive — it blocks legitimate user requests. An ideal mitigation has both low ASR and low ORR."
        ranges={[
          { color: '#10b981', label: 'Good', range: '< 15%' },
          { color: '#f59e0b', label: 'Moderate', range: '15–40%' },
          { color: '#ef4444', label: 'Poor (over-aggressive)', range: '> 40%' },
        ]}
      />
    </>
  );
};

const ErrGroupedChart = ({ allData, models }) => {
  const data = useMemo(() => MITS.map(mit => ({
    name: MIT_META[mit].label,
    x: models.map(modelLabel),
    y: models.map(m => allData[m]?.mitigations?.[mit]?.err_overall ?? null),
    type: 'bar', marker: { color: mitColor(mit) },
    text: models.map(m => { const v = allData[m]?.mitigations?.[mit]?.err_overall; return v != null ? `${v}%` : ''; }),
    textposition: 'outside',
    hovertemplate: '<b>%{x}</b><br>ERR: %{y:.1f}%<extra>%{fullData.name}</extra>',
  })), [allData, models]);

  return (
    <>
      <Plot data={data} layout={{
        ...base, barmode: 'group',
        title: { text: 'Escalation Resistance Rate (%) by Model & Mitigation<br><sup>Higher is better — attacks caught early, before they escalate</sup>', x: 0.5, xanchor: 'center' },
        yaxis: { title: 'ERR (%)', range: [0, 115], showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        xaxis: { title: 'Model', showgrid: false, automargin: true },
        legend: { orientation: 'h', y: -0.28, x: 0.5, xanchor: 'center' },
        margin: { t: 90, b: 110, l: 65, r: 20 }, showlegend: true,
      }} useResizeHandler style={{ width: '100%', height: '420px' }} config={{ responsive: true, displayModeBar: false }} />
      <ModelLegend models={models} />
      <MetricInfoPanel
        formula={'ERR (%) = (Attacks Caught in Early Turns / Total Attacks) × 100\n\n"Early" = detected before the attacker fully escalated the injection'}
        unit="Percentage (0–100%)"
        description="Unlike ASR which measures final outcome, ERR rewards mitigations that intercept the attack early in the conversation before much damage is done. High ERR = proactive defence."
        ranges={[
          { color: '#10b981', label: 'Good (proactive)', range: '> 50%' },
          { color: '#f59e0b', label: 'Moderate', range: '20–50%' },
          { color: '#ef4444', label: 'Reactive / poor', range: '< 20%' },
        ]}
      />
    </>
  );
};

const ReliabilityBarChart = ({ allData, models }) => {
  const data = useMemo(() => MITS.map(mit => {
    const scores = models.map(m => {
      const d = allData[m]?.mitigations?.[mit];
      if (!d) return null;
      const safety = 100 - (d.attack_success_rate_pct ?? 0);
      const avail  = 100 - (d.over_refusal_rate_pct  ?? 0);
      return gmean(safety, avail);
    });
    return {
      name: MIT_META[mit].label,
      x: models.map(modelLabel), y: scores, type: 'bar',
      marker: { color: mitColor(mit) },
      text: scores.map(v => v != null ? `${v}` : ''),
      textposition: 'outside',
      hovertemplate: '<b>%{x}</b><br>Reliability: %{y:.1f} / 100<extra>%{fullData.name}</extra>',
    };
  }), [allData, models]);

  return (
    <>
      <Plot data={data} layout={{
        ...base, barmode: 'group',
        title: { text: 'Overall Mitigation Reliability Score (0–100)<br><sup>√(Safety × Availability) — balances security with usability</sup>', x: 0.5, xanchor: 'center' },
        yaxis: { title: 'Reliability Score (0–100)', range: [0, 115], showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        xaxis: { title: 'Model', showgrid: false, automargin: true },
        legend: { orientation: 'h', y: -0.28, x: 0.5, xanchor: 'center' },
        margin: { t: 90, b: 110, l: 65, r: 20 }, showlegend: true,
      }} useResizeHandler style={{ width: '100%', height: '420px' }} config={{ responsive: true, displayModeBar: false }} />
      <ModelLegend models={models} />
      <MetricInfoPanel
        formula={'Safety     = 100 − ASR (%)\nAvailability = 100 − ORR (%)\nReliability  = √(Safety × Availability)   [Geometric Mean]'}
        unit="Score on a 0–100 scale"
        description="The geometric mean of Safety and Availability penalises extreme imbalances. A mitigation that blocks all attacks but also blocks all legitimate queries (ORR=100%) scores 0. The geometric mean forces both dimensions to be good simultaneously."
        ranges={[
          { color: '#10b981', label: 'Good (balanced)', range: '> 70' },
          { color: '#f59e0b', label: 'Moderate', range: '40–70' },
          { color: '#ef4444', label: 'Poor (imbalanced)', range: '< 40' },
        ]}
      />
    </>
  );
};

const CldGroupedChart = ({ allData, models }) => {
  const data = useMemo(() => MITS.map(mit => ({
    name: MIT_META[mit].label,
    x: models.map(modelLabel),
    y: models.map(m => allData[m]?.mitigations?.[mit]?.context_length_drift_pct ?? null),
    type: 'bar', marker: { color: mitColor(mit) },
    text: models.map(m => {
      const v = allData[m]?.mitigations?.[mit]?.context_length_drift_pct;
      return v != null ? `${v > 0 ? '+' : ''}${v} pp` : '';
    }),
    textposition: 'outside',
    hovertemplate: '<b>%{x}</b><br>CLD: %{y:.1f} percentage points<extra>%{fullData.name}</extra>',
  })), [allData, models]);

  return (
    <>
      <Plot data={data} layout={{
        ...base, barmode: 'group',
        title: { text: 'Context-Length Drift (CLD) in percentage points<br><sup>Closer to 0 is better — large deviation = protection weakens in longer conversations</sup>', x: 0.5, xanchor: 'center' },
        yaxis: { title: 'CLD (percentage points)', showgrid: true, gridcolor: '#f1f5f9', automargin: true, zeroline: true, zerolinecolor: '#94a3b8' },
        xaxis: { title: 'Model', showgrid: false, automargin: true },
        legend: { orientation: 'h', y: -0.28, x: 0.5, xanchor: 'center' },
        margin: { t: 90, b: 110, l: 85, r: 20 }, showlegend: true,
      }} useResizeHandler style={{ width: '100%', height: '420px' }} config={{ responsive: true, displayModeBar: false }} />
      <ModelLegend models={models} />
      <MetricInfoPanel
        formula={'CLD (pp) = ASR_long_conversations (%) − ASR_short_conversations (%)\n\npp = percentage points (difference between two percentages)'}
        unit="Percentage points (pp) — a pp is the arithmetic difference between two percentages (e.g. 60% − 40% = 20 pp)"
        description="Measures how much the ASR changes between short and long conversations. A positive CLD means protection degrades in longer chats (attacker benefits from more turns). A negative CLD means the mitigation improves over long conversations."
        ranges={[
          { color: '#10b981', label: 'Stable', range: '−10 to +10 pp' },
          { color: '#f59e0b', label: 'Moderate drift', range: '±10 to ±25 pp' },
          { color: '#ef4444', label: 'High drift (unstable)', range: '> |25 pp|' },
        ]}
      />
    </>
  );
};

// ── SCATTER — trajectory plot (one line per model, no label overlap) ──────────
const SecurityUsabilityScatter = ({ allData, models }) => {
  const MIT_SYMBOLS = { none: 'circle', m1: 'square', m2: 'diamond', m3: 'cross' };
  const MIT_LABEL = { none: 'Baseline', m1: 'M1', m2: 'M2', m3: 'M3' };

  const data = useMemo(() => {
    // One trace per model — four points connected by a dotted line showing trajectory
    return models.map(slug => {
      const color = modelColor(slug);
      const pts = MITS.map(mit => ({
        x: allData[slug]?.mitigations?.[mit]?.over_refusal_rate_pct ?? null,
        y: allData[slug]?.mitigations?.[mit]?.attack_success_rate_pct ?? null,
        mit,
      })).filter(p => p.x != null && p.y != null);

      return {
        name: modelLabel(slug),
        x: pts.map(p => p.x),
        y: pts.map(p => p.y),
        text: pts.map(p => MIT_LABEL[p.mit]),
        mode: 'lines+markers',
        line: { color, width: 1.5, dash: 'dot' },
        marker: {
          color: pts.map(p => p.mit === 'none' ? '#ffffff' : color),
          size: pts.map(p => p.mit === 'none' ? 11 : 13),
          symbol: pts.map(p => MIT_SYMBOLS[p.mit]),
          line: { color, width: 2.5 },
        },
        hovertemplate:
          `<b>${modelLabel(slug)}</b><br>` +
          'Stage: <b>%{text}</b><br>' +
          'ORR (usability cost): %{x:.1f}%<br>' +
          'ASR (attack success): %{y:.1f}%' +
          '<extra></extra>',
      };
    });
  }, [allData, models]);

  // Legend explaining marker shapes for mitigations
  const mitShapeLegend = MITS.map((m, i) => ({
    name: MIT_META[m].label,
    x: [null], y: [null],
    mode: 'markers',
    marker: { symbol: ['circle','square','diamond','cross'][i], color: '#94a3b8', size: 10, line: { color: '#64748b', width: 1.5 } },
    showlegend: true, hoverinfo: 'skip',
  }));

  const shapes = [{
    type: 'rect', xref: 'x', yref: 'y',
    x0: 0, y0: 0, x1: 20, y1: 20,
    fillcolor: 'rgba(16,185,129,0.08)',
    line: { color: 'rgba(16,185,129,0.35)', dash: 'dot', width: 1.5 },
  }];
  const annotations = [{
    x: 10, y: 10, xref: 'x', yref: 'y',
    text: '✅ Ideal Zone', showarrow: false,
    font: { size: 11, color: '#059669', family: 'Inter' },
  }];

  return (
    <>
      <Plot
        data={[...data, ...mitShapeLegend]}
        layout={{
          ...base,
          title: { text: 'Security vs Usability Trade-off — Model Trajectories<br><sup>Each line = one model moving from Baseline → M1 → M2 → M3. Lower-left = ideal.</sup>', x: 0.5, xanchor: 'center' },
          xaxis: { title: 'Over-Refusal Rate / ORR (%) ↓', range: [-5, 108], showgrid: true, gridcolor: '#f1f5f9', automargin: true, zeroline: false },
          yaxis: { title: 'Attack Success Rate / ASR (%) ↓', range: [-5, 108], showgrid: true, gridcolor: '#f1f5f9', automargin: true, zeroline: false },
          legend: { orientation: 'h', y: -0.28, x: 0.5, xanchor: 'center', tracegroupgap: 5 },
          margin: { t: 95, b: 120, l: 70, r: 20 },
          showlegend: true, shapes, annotations,
        }}
        useResizeHandler
        style={{ width: '100%', height: '500px' }}
        config={{ responsive: true, displayModeBar: false }}
      />
      {/* HTML marker-shape legend */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '16px', flexWrap: 'wrap', fontSize: '0.78rem', color: '#475569', marginTop: '-4px', marginBottom: '4px' }}>
        {MITS.map((m, i) => (
          <span key={m} style={{ display: 'flex', alignItems: 'center', gap: '5px', fontWeight: 500 }}>
            <span style={{ fontSize: '1rem' }}>{['⬤', '■', '◆', '✕'][i]}</span>
            {MIT_META[m].label}
          </span>
        ))}
      </div>
      <ModelLegend models={models} />
      <MetricInfoPanel
        formula={'X-axis (ORR): (Benign Queries Blocked / Total Benign) × 100\nY-axis (ASR): (Attacks Succeeded / Total Attacks) × 100\n\nIdeal zone: ORR < 20% AND ASR < 20%'}
        unit="Both axes in Percentage (0–100%)"
        description="Each model traces a path across 4 points (Baseline → M1 → M2 → M3). Dots moving toward the lower-left corner show that a mitigation improves both security and usability. Lines going right indicate usability cost; lines going down indicate security gain."
        ranges={[
          { color: '#10b981', label: 'Ideal zone', range: 'ASR<20% & ORR<20%' },
          { color: '#f59e0b', label: 'Acceptable trade-off', range: 'One axis < 40%' },
          { color: '#ef4444', label: 'Poor balance', range: 'Both > 50%' },
        ]}
      />
    </>
  );
};

// ── HEATMAP — per-mitigation with correct colorbar labelling ─────────────────
const CrossModelHeatmap = ({ allData, models, activeMit }) => {
  const { z, annotations, xLabels, yLabels } = useMemo(() => {
    const metrics = [
      { key: 'attack_success_rate_pct',  label: 'ASR (%)',        lowerBetter: true  },
      { key: 'over_refusal_rate_pct',    label: 'ORR (%)',        lowerBetter: true  },
      { key: 'err_overall',              label: 'ERR (%)',        lowerBetter: false },
      { key: 'mean_ai_latency_turns',    label: 'Latency (turns)', lowerBetter: true  },
      { key: 'context_length_drift_pct', label: 'CLD (% pts)',    lowerBetter: null  },
      { key: 'rcs_score',               label: 'RCS (0–1)',       lowerBetter: false },
    ];
    const xLabels = metrics.map(m => m.label);
    const yLabels = models.map(modelLabel);
    const rawMatrix = models.map(model =>
      metrics.map(m => allData[model]?.mitigations?.[activeMit]?.[m.key] ?? null)
    );
    const annots = models.map((model, ri) =>
      metrics.map((m, ci) => {
        const v = rawMatrix[ri][ci];
        if (v == null) return 'N/A';
        if (m.key === 'rcs_score')              return v.toFixed(3);
        if (m.key === 'mean_ai_latency_turns')  return `${v.toFixed(2)} turns`;
        if (m.key === 'context_length_drift_pct') return `${v > 0 ? '+' : ''}${v} pp`;
        return `${v}%`;
      })
    );

    const normMatrix = rawMatrix.map(r => r.map(() => 0));
    metrics.forEach((m, ci) => {
      const col = rawMatrix.map(r => r[ci]).filter(v => v != null);
      if (col.length === 0) return;
      const mn = Math.min(...col), mx = Math.max(...col);
      rawMatrix.forEach((row, ri) => {
        const v = row[ci];
        if (v == null) { normMatrix[ri][ci] = 0.5; return; }
        if (mx === mn) { normMatrix[ri][ci] = 0.5; return; }
        const norm = (v - mn) / (mx - mn);
        if (m.lowerBetter === true)  normMatrix[ri][ci] = 1 - norm;
        else if (m.lowerBetter === false) normMatrix[ri][ci] = norm;
        else normMatrix[ri][ci] = 1 - Math.abs(v) / Math.max(...col.map(Math.abs), 1);
      });
    });
    return { z: normMatrix, annotations: annots, xLabels, yLabels };
  }, [allData, models, activeMit]);

  return (
    <>
      <Plot
        data={[{
          z, x: xLabels, y: yLabels,
          type: 'heatmap',
          colorscale: [[0, '#fca5a5'], [0.5, '#fde68a'], [1, '#86efac']],
          zmin: 0, zmax: 1,
          showscale: true,
          colorbar: {
            title: { text: 'Relative<br>Perf.', side: 'right' },
            thickness: 14, len: 0.85,
            tickvals: [0, 0.5, 1],
            ticktext: ['Worst<br>in column', 'Middle', 'Best<br>in column'],
            tickfont: { size: 10 },
          },
          text: annotations,
          texttemplate: '%{text}',
          textfont: { size: 11, color: '#1e293b', family: 'Inter' },
          hovertemplate: '<b>%{y}</b><br>%{x}: <b>%{text}</b><br><i>Color = rank within this metric column</i><extra></extra>',
        }]}
        layout={{
          ...base,
          title: { text: `Metric Heatmap — ${MIT_META[activeMit]?.label}<br><sup>Cell values = raw data. Color = relative rank within each column (green=best, red=worst)</sup>`, x: 0.5, xanchor: 'center' },
          xaxis: { title: 'Metric', side: 'bottom', automargin: true },
          yaxis: { title: 'Model', automargin: true, autorange: 'reversed' },
          margin: { t: 90, b: 80, l: 160, r: 100 },
        }}
        useResizeHandler
        style={{ width: '100%', height: '370px' }}
        config={{ responsive: true, displayModeBar: false }}
      />
      <MetricInfoPanel
        formula={'Color scale normalisation (per column):\n  norm = (value − col_min) / (col_max − col_min)\n  For lower-better metrics: flipped to 1 − norm\n  For CLD: 1 − |value| / max(|values|)  [closest to 0 = best]'}
        unit="Raw values shown in cells; color represents rank within that metric column only"
        description="The colorbar (0–1) does NOT represent the actual metric values — it represents where that model ranks among all models for that specific metric. Green (score 1.0) = best model for that metric; Red (score 0) = worst model. Comparing colors across columns shows which model performs most consistently well."
        ranges={[
          { color: '#10b981', label: 'Best in column', range: 'Score → 1.0 (darkest green)' },
          { color: '#f59e0b', label: 'Middle of pack', range: 'Score ≈ 0.5 (yellow)' },
          { color: '#ef4444', label: 'Worst in column', range: 'Score → 0.0 (darkest red)' },
        ]}
      />
    </>
  );
};

// ── LINE / TREND CHARTS ───────────────────────────────────────────────────────

const AsrTrendLineChart = ({ allData, models }) => {
  const data = useMemo(() => models.map(slug => {
    const color = modelColor(slug);
    return {
      name: modelLabel(slug),
      x: MIT_LABELS,
      y: MITS.map(m => allData[slug]?.mitigations?.[m]?.attack_success_rate_pct ?? null),
      mode: 'lines+markers',
      line: { color, width: 2.5, shape: 'spline' },
      marker: { color, size: 9, symbol: 'circle', line: { color: 'white', width: 2 } },
      hovertemplate: `<b>${modelLabel(slug)}</b><br>%{x}<br>ASR: <b>%{y:.1f}%</b><extra></extra>`,
    };
  }), [allData, models]);

  return (
    <>
      <Plot data={data} layout={{
        ...base,
        title: { text: 'ASR (%) Trend Across Mitigation Stages<br><sup>Each line = one model — downward slope = effective protection</sup>', x: 0.5, xanchor: 'center' },
        yaxis: { title: 'Attack Success Rate (%)', range: [-3, 105], showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        xaxis: { title: 'Mitigation Strategy', showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        legend: { orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center' },
        margin: { t: 90, b: 100, l: 65, r: 20 }, showlegend: true,
      }} useResizeHandler style={{ width: '100%', height: '430px' }} config={{ responsive: true, displayModeBar: false }} />
      <ModelLegend models={models} />
      <MetricInfoPanel
        formula={'ASR (%) = (Attacks Succeeded / Total Attack Samples) × 100\nTracked at 4 stages: Baseline (no mitigation) → M1 → M2 → M3'}
        unit="Percentage (0–100%)"
        description="Tracks how the Attack Success Rate evolves as we apply each mitigation. A steeply falling line means that mitigation strongly suppresses attacks for that model. A flat line means the mitigation had minimal effect."
        ranges={[
          { color: '#10b981', label: 'Good', range: '< 20%' },
          { color: '#f59e0b', label: 'Moderate', range: '20–50%' },
          { color: '#ef4444', label: 'Poor', range: '> 50%' },
        ]}
      />
    </>
  );
};

const ReliabilityTrendLineChart = ({ allData, models }) => {
  const data = useMemo(() => models.map(slug => {
    const color = modelColor(slug);
    return {
      name: modelLabel(slug),
      x: MIT_LABELS,
      y: MITS.map(m => {
        const d = allData[slug]?.mitigations?.[m];
        if (!d) return null;
        return gmean(100 - (d.attack_success_rate_pct ?? 0), 100 - (d.over_refusal_rate_pct ?? 0));
      }),
      mode: 'lines+markers',
      line: { color, width: 2.5, shape: 'spline' },
      marker: { color, size: 9, symbol: 'diamond', line: { color: 'white', width: 2 } },
      hovertemplate: `<b>${modelLabel(slug)}</b><br>%{x}<br>Reliability: <b>%{y:.1f} / 100</b><extra></extra>`,
    };
  }), [allData, models]);

  return (
    <>
      <Plot data={data} layout={{
        ...base,
        title: { text: 'Reliability Score (0–100) Trend Across Mitigations<br><sup>√(Safety×Availability) — upward slope = mitigation improves balance</sup>', x: 0.5, xanchor: 'center' },
        yaxis: { title: 'Reliability Score (0–100)', range: [0, 105], showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        xaxis: { title: 'Mitigation Strategy', showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        legend: { orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center' },
        margin: { t: 90, b: 100, l: 65, r: 20 }, showlegend: true,
      }} useResizeHandler style={{ width: '100%', height: '430px' }} config={{ responsive: true, displayModeBar: false }} />
      <ModelLegend models={models} />
      <MetricInfoPanel
        formula={'Safety      = 100 − ASR (%)\nAvailability = 100 − ORR (%)\nReliability  = √(Safety × Availability)   [Geometric Mean, 0–100]'}
        unit="Score on a 0–100 scale"
        description="Tracks the combined security-usability score across mitigations. A rising line means the mitigation improves the balance. Note that M2 often has a low reliability despite low ASR because its ORR is very high (over-aggressive blocking), which brings down the geometric mean."
        ranges={[
          { color: '#10b981', label: 'Good', range: '> 70' },
          { color: '#f59e0b', label: 'Moderate', range: '40–70' },
          { color: '#ef4444', label: 'Poor', range: '< 40' },
        ]}
      />
    </>
  );
};

const ErrTrendLineChart = ({ allData, models }) => {
  const data = useMemo(() => models.map(slug => {
    const color = modelColor(slug);
    return {
      name: modelLabel(slug),
      x: MIT_LABELS,
      y: MITS.map(m => allData[slug]?.mitigations?.[m]?.err_overall ?? null),
      mode: 'lines+markers',
      line: { color, width: 2.5, shape: 'spline' },
      marker: { color, size: 9, symbol: 'square', line: { color: 'white', width: 2 } },
      hovertemplate: `<b>${modelLabel(slug)}</b><br>%{x}<br>ERR: <b>%{y:.1f}%</b><extra></extra>`,
    };
  }), [allData, models]);

  return (
    <>
      <Plot data={data} layout={{
        ...base,
        title: { text: 'ERR (%) Trend Across Mitigation Stages<br><sup>Escalation Resistance Rate — upward slope = better early interception</sup>', x: 0.5, xanchor: 'center' },
        yaxis: { title: 'ERR (%)', range: [-3, 105], showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        xaxis: { title: 'Mitigation Strategy', showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        legend: { orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center' },
        margin: { t: 90, b: 100, l: 65, r: 20 }, showlegend: true,
      }} useResizeHandler style={{ width: '100%', height: '430px' }} config={{ responsive: true, displayModeBar: false }} />
      <ModelLegend models={models} />
      <MetricInfoPanel
        formula={'ERR (%) = (Attacks Caught Before Full Escalation / Total Attacks) × 100'}
        unit="Percentage (0–100%)"
        description="Shows how early each model's mitigation kicks in. A rising line across mitigations means each successive strategy catches attacks progressively earlier, before the adversarial context has fully built up."
        ranges={[
          { color: '#10b981', label: 'Good (proactive)', range: '> 50%' },
          { color: '#f59e0b', label: 'Moderate', range: '20–50%' },
          { color: '#ef4444', label: 'Reactive / poor', range: '< 20%' },
        ]}
      />
    </>
  );
};

const ModelProfileLineChart = ({ allData, models }) => {
  const [selectedModel, setSelectedModel] = React.useState(models[0] || '');
  const METRICS = [
    { key: 'attack_success_rate_pct', label: 'ASR % (↓ lower=better)', color: '#ef4444', dash: 'solid',   symbol: 'circle'  },
    { key: 'over_refusal_rate_pct',   label: 'ORR % (↓ lower=better)', color: '#f59e0b', dash: 'dot',     symbol: 'square'  },
    { key: 'err_overall',             label: 'ERR % (↑ higher=better)', color: '#10b981', dash: 'dashdot', symbol: 'diamond' },
  ];

  const data = useMemo(() => {
    if (!selectedModel) return [];
    return METRICS.map(m => ({
      name: m.label,
      x: MIT_LABELS,
      y: MITS.map(mit => allData[selectedModel]?.mitigations?.[mit]?.[m.key] ?? null),
      mode: 'lines+markers',
      line: { color: m.color, width: 2.5, dash: m.dash, shape: 'spline' },
      marker: { color: m.color, size: 9, symbol: m.symbol, line: { color: 'white', width: 2 } },
      hovertemplate: `%{x}<br><b>${m.label.split('(')[0].trim()}: %{y:.1f}%</b><extra></extra>`,
    }));
  }, [allData, selectedModel]);

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <span style={{ fontWeight: 600, fontSize: '0.9rem', color: '#1e293b' }}>Select Model:</span>
        {models.map(slug => {
          const meta = MODEL_META[slug]; const isActive = selectedModel === slug; const color = modelColor(slug);
          return (
            <button key={slug} onClick={() => setSelectedModel(slug)} style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              padding: '6px 14px', borderRadius: '8px', cursor: 'pointer', fontFamily: 'inherit',
              border: `2px solid ${isActive ? color : '#e2e8f0'}`,
              background: isActive ? `${color}15` : 'white',
              color: isActive ? color : '#64748b',
              fontWeight: isActive ? 700 : 500, fontSize: '0.82rem', transition: 'all 0.15s ease',
            }}>
              {meta?.logo && <img src={meta.logo} alt="" style={{ width: 14, height: 14, objectFit: 'contain' }} />}
              {meta?.short ?? slug}
            </button>
          );
        })}
      </div>
      <Plot data={data} layout={{
        ...base,
        title: { text: `Multi-Metric Profile — ${modelLabel(selectedModel)}<br><sup>ASR, ORR and ERR across all mitigation stages</sup>`, x: 0.5, xanchor: 'center' },
        yaxis: { title: 'Rate (%)', range: [-3, 108], showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        xaxis: { title: 'Mitigation Strategy', showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        legend: { orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center' },
        margin: { t: 90, b: 100, l: 65, r: 20 }, showlegend: true,
      }} useResizeHandler style={{ width: '100%', height: '430px' }} config={{ responsive: true, displayModeBar: false }} />
      <MetricInfoPanel
        formula={'Three metrics on a single chart:\n  ASR (%) = attacks that succeeded (↓ lower=better)\n  ORR (%) = benign queries wrongly blocked (↓ lower=better)\n  ERR (%) = attacks caught early (↑ higher=better)'}
        unit="All three metrics in Percentage (0–100%)"
        description="The 'X pattern' — where ASR falls while ORR rises — is the classic security-usability tension. An ideal mitigation would show ASR falling AND ORR staying low AND ERR rising simultaneously. M2 typically shows dramatic ASR reduction but an equally dramatic ORR spike."
        ranges={null}
      />
    </>
  );
};

const ModelRankingLineChart = ({ allData, models }) => {
  const data = useMemo(() => {
    const rankMatrix = {};
    MITS.forEach((mit, mi) => {
      const scored = models
        .map(slug => ({ slug, val: allData[slug]?.mitigations?.[mit]?.attack_success_rate_pct ?? 999 }))
        .sort((a, b) => a.val - b.val);
      scored.forEach(({ slug }, ri) => {
        if (!rankMatrix[slug]) rankMatrix[slug] = [];
        rankMatrix[slug][mi] = ri + 1;
      });
    });

    return models.map(slug => {
      const color = modelColor(slug);
      return {
        name: modelLabel(slug),
        x: MIT_LABELS, y: rankMatrix[slug] || [],
        mode: 'lines+markers',
        line: { color, width: 2.5, shape: 'spline' },
        marker: { color, size: 10, symbol: 'circle', line: { color: 'white', width: 2 } },
        hovertemplate: `<b>${modelLabel(slug)}</b><br>%{x}<br>Rank: <b>#%{y}</b> (by ASR — lower is better)<extra></extra>`,
      };
    });
  }, [allData, models]);

  return (
    <>
      <Plot data={data} layout={{
        ...base,
        title: { text: 'Model Security Ranking by Mitigation Stage<br><sup>Rank #1 = lowest ASR (best). Crossing lines = rank reversal</sup>', x: 0.5, xanchor: 'center' },
        yaxis: {
          title: 'Security Rank (1 = best / lowest ASR)',
          autorange: 'reversed',
          tickvals: [1, 2, 3, 4, 5],
          ticktext: ['#1 Best', '#2', '#3', '#4', '#5 Worst'],
          showgrid: true, gridcolor: '#f1f5f9', automargin: true,
        },
        xaxis: { title: 'Mitigation Strategy', showgrid: true, gridcolor: '#f1f5f9', automargin: true },
        legend: { orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center' },
        margin: { t: 90, b: 100, l: 110, r: 20 }, showlegend: true,
      }} useResizeHandler style={{ width: '100%', height: '430px' }} config={{ responsive: true, displayModeBar: false }} />
      <ModelLegend models={models} />
      <MetricInfoPanel
        formula={'Rank = position of model when all 5 models are sorted by ASR (ascending)\nRank 1 = lowest ASR = most secure for that mitigation stage'}
        unit="Ordinal rank (1–5); not a continuous metric"
        description="Reveals which models are most responsive to each mitigation. A model that jumps from rank 5 to rank 1 specifically for M1 tells us M1 is particularly well-suited to that model's architecture. Crossing lines highlight these rank inversions."
        ranges={[
          { color: '#10b981', label: 'Best', range: 'Rank #1' },
          { color: '#f59e0b', label: 'Middle', range: 'Rank #2–3' },
          { color: '#ef4444', label: 'Weakest', range: 'Rank #4–5' },
        ]}
      />
    </>
  );
};

// ── Summary Table ─────────────────────────────────────────────────────────────
const SummaryTable = ({ allData, models }) => {
  const rows = useMemo(() => models.map(slug => {
    const mits = allData[slug]?.mitigations ?? {};
    const best = MITS.filter(m => mits[m]).reduce((prev, curr) => {
      const pA = mits[prev]?.attack_success_rate_pct ?? 999;
      const cA = mits[curr]?.attack_success_rate_pct ?? 999;
      return cA < pA ? curr : prev;
    }, 'none');
    return { slug, mits, best };
  }), [allData, models]);

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', fontFamily: 'Inter, sans-serif' }}>
        <thead>
          <tr style={{ background: 'linear-gradient(135deg, #4f46e5, #0891b2)', color: 'white' }}>
            <th style={{ padding: '12px 16px', textAlign: 'left', borderRadius: '8px 0 0 0' }}>Model</th>
            {MITS.map(m => (
              <th key={m} colSpan={2} style={{ padding: '12px 8px', textAlign: 'center', borderLeft: '1px solid rgba(255,255,255,0.2)' }}>
                {MIT_META[m].short}
              </th>
            ))}
            <th style={{ padding: '12px 10px', textAlign: 'center', borderLeft: '1px solid rgba(255,255,255,0.2)', borderRadius: '0 8px 0 0' }}>Best Mit.</th>
          </tr>
          <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
            <th style={{ padding: '8px 16px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}></th>
            {MITS.map(m => (
              <React.Fragment key={m}>
                <th style={{ padding: '8px 6px', textAlign: 'center', color: '#64748b', fontWeight: 600, borderLeft: '1px solid #e2e8f0', fontSize: '0.72rem' }}>ASR%↓</th>
                <th style={{ padding: '8px 6px', textAlign: 'center', color: '#64748b', fontWeight: 600, fontSize: '0.72rem' }}>ORR%↓</th>
              </React.Fragment>
            ))}
            <th style={{ padding: '8px 10px', textAlign: 'center', color: '#64748b', fontWeight: 600, borderLeft: '1px solid #e2e8f0', fontSize: '0.72rem' }}>Reliability↑</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ slug, mits, best }, ri) => {
            const meta = MODEL_META[slug];
            return (
              <tr key={slug} style={{ background: ri % 2 === 0 ? 'white' : '#f8fafc', borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '12px 16px', fontWeight: 600, color: '#1e293b', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {meta?.logo && <img src={meta.logo} alt="" style={{ width: 16, height: 16, objectFit: 'contain' }} />}
                  {meta?.label ?? slug}
                </td>
                {MITS.map(m => {
                  const d = mits[m];
                  const asr = d?.attack_success_rate_pct; const orr = d?.over_refusal_rate_pct;
                  const asrBg = asr == null ? '' : asr < 25 ? 'rgba(16,185,129,0.12)' : asr < 55 ? 'rgba(245,158,11,0.12)' : 'rgba(239,68,68,0.12)';
                  const orrBg = orr == null ? '' : orr < 20 ? 'rgba(16,185,129,0.12)' : orr < 60 ? 'rgba(245,158,11,0.12)' : 'rgba(239,68,68,0.12)';
                  return (
                    <React.Fragment key={m}>
                      <td style={{ padding: '12px 8px', textAlign: 'center', borderLeft: '1px solid #f1f5f9', background: asrBg, fontWeight: 600, color: '#1e293b' }}>
                        {asr != null ? `${asr}%` : '—'}
                      </td>
                      <td style={{ padding: '12px 8px', textAlign: 'center', background: orrBg, fontWeight: 500, color: '#334155' }}>
                        {orr != null ? `${orr}%` : '—'}
                      </td>
                    </React.Fragment>
                  );
                })}
                <td style={{ padding: '12px 10px', textAlign: 'center', borderLeft: '1px solid #f1f5f9' }}>
                  {(() => {
                    const d = mits[best];
                    if (!d) return '—';
                    const r = gmean(100 - d.attack_success_rate_pct, 100 - d.over_refusal_rate_pct);
                    return (
                      <span style={{ background: `${mitColor(best)}20`, color: mitColor(best), border: `1px solid ${mitColor(best)}50`, borderRadius: '999px', padding: '2px 10px', fontWeight: 700, fontSize: '0.8rem' }}>
                        {MIT_META[best]?.short} · {r}
                      </span>
                    );
                  })()}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

// ── Main Component ────────────────────────────────────────────────────────────
const SECTION_GROUPS = [
  {
    group: '📈 Line & Trend Charts',
    items: [
      { id: 'asr_trend',         label: '📈 ASR Trend' },
      { id: 'reliability_trend', label: '🏆 Reliability Trend' },
      { id: 'err_trend',         label: '⚡ ERR Trend' },
      { id: 'model_profile',     label: '🔍 Model Profile' },
      { id: 'ranking',           label: '🥇 Model Ranking' },
    ]
  },
  {
    group: '📊 Bar Charts',
    items: [
      { id: 'asr',        label: '🔴 ASR Bars' },
      { id: 'reduction',  label: '📉 ASR Reduction' },
      { id: 'orr',        label: '🛡️ ORR Bars' },
      { id: 'err',        label: '⚡ ERR Bars' },
      { id: 'reliability',label: '🏆 Reliability Bars' },
      { id: 'cld',        label: '📏 CLD Bars' },
    ]
  },
  {
    group: '🔬 Deep Analysis',
    items: [
      { id: 'scatter',  label: '⚖️ Security vs Usability' },
      { id: 'heatmap',  label: '🌡️ Heatmap' },
      { id: 'overview', label: '📋 Summary Table' },
    ]
  },
];

const SECTIONS = SECTION_GROUPS.flatMap(g => g.items);

const Results = () => {
  const [activeSection, setActiveSection] = useState('asr_trend');
  const [heatmapMit, setHeatmapMit] = useState('m1');
  const allData = getAllModelsComparisons();
  const models = Object.keys(allData).sort();

  if (models.length === 0) {
    return (
      <div className="warn-panel mt-1">
        <strong>⚠️ No Data Found</strong><br />
        No model result data is available in the comparison directory.
      </div>
    );
  }

  return (
    <div className="tab-pane">
      {/* Header */}
      <div className="glass-card mb-2" style={{ padding: '1.5rem 2rem', background: 'linear-gradient(135deg, rgba(79,70,229,0.06) 0%, rgba(8,145,178,0.06) 100%)' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h3 style={{ marginBottom: '0.4rem', fontSize: '1.3rem' }}>📊 Cross-Model Results Dashboard</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '620px' }}>
              Comprehensive comparison of all {models.length} evaluated LLMs across all mitigation strategies.
              Click any chart pill to explore. Each chart includes a formula & interpretation panel — click "Show formula" below the chart.
            </p>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
            {models.map(slug => {
              const meta = MODEL_META[slug];
              return (
                <div key={slug} style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'white', padding: '4px 12px', borderRadius: '999px', boxShadow: '0 1px 4px rgba(0,0,0,0.08)', border: '1px solid #e2e8f0', fontSize: '0.8rem', fontWeight: 600, color: '#1e293b' }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: modelColor(slug), flexShrink: 0 }} />
                  {meta?.logo && <img src={meta.logo} alt="" style={{ width: 14, height: 14, objectFit: 'contain' }} />}
                  {meta?.short ?? slug}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Section Nav — grouped */}
      <div style={{ marginBottom: '1.5rem', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {SECTION_GROUPS.map(grp => (
          <div key={grp.group}>
            <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '6px' }}>
              {grp.group}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {grp.items.map(s => (
                <button key={s.id} onClick={() => setActiveSection(s.id)} style={{
                  padding: '5px 14px', borderRadius: '999px', border: '1.5px solid',
                  borderColor: activeSection === s.id ? 'var(--primary)' : '#e2e8f0',
                  background: activeSection === s.id ? 'var(--primary)' : 'white',
                  color: activeSection === s.id ? 'white' : 'var(--text-muted)',
                  fontWeight: activeSection === s.id ? 700 : 500, fontSize: '0.82rem',
                  cursor: 'pointer', fontFamily: 'inherit', transition: 'all 0.15s ease',
                  boxShadow: activeSection === s.id ? '0 2px 8px rgba(79,70,229,0.25)' : 'none',
                }}>
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Chart sections */}
      {activeSection === 'asr_trend' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>📈 ASR Trend Across Mitigation Stages</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Each line = one model. A steeply falling line means the mitigation strongly reduces attacks for that model.</p>
          <AsrTrendLineChart allData={allData} models={models} />
        </div>
      )}

      {activeSection === 'reliability_trend' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>🏆 Reliability Score (0–100) Trend</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>√(Safety × Availability) per mitigation. A rising line = mitigation improves the security-usability balance.</p>
          <ReliabilityTrendLineChart allData={allData} models={models} />
        </div>
      )}

      {activeSection === 'err_trend' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>⚡ ERR Trend Across Mitigation Stages</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Escalation Resistance Rate (%) — rising line = mitigation catches more attacks before they fully escalate.</p>
          <ErrTrendLineChart allData={allData} models={models} />
        </div>
      )}

      {activeSection === 'model_profile' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>🔍 Multi-Metric Profile (Per Model)</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Select a model to see its ASR, ORR, and ERR across all mitigations — the complete performance fingerprint.</p>
          <ModelProfileLineChart allData={allData} models={models} />
        </div>
      )}

      {activeSection === 'ranking' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>🥇 Model Security Ranking by Mitigation Stage</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Rank #1 = lowest ASR (best). Crossing lines reveal which models benefit disproportionately from specific mitigations.</p>
          <ModelRankingLineChart allData={allData} models={models} />
        </div>
      )}

      {activeSection === 'asr' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>🔴 Attack Success Rate (%) by Model & Mitigation</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Lower is better. Shows how well each mitigation stops attacks from succeeding.</p>
          <AsrGroupedChart allData={allData} models={models} />
        </div>
      )}

      {activeSection === 'reduction' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>📉 ASR Reduction vs Baseline (percentage points)</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Higher is better. How many percentage points of ASR each mitigation eliminated compared to no-mitigation baseline.</p>
          <AsrReductionChart allData={allData} models={models} />
        </div>
      )}

      {activeSection === 'orr' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>🛡️ Over-Refusal Rate (%) — Usability Cost</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Lower is better. High ORR means the mitigation is over-aggressive and disrupts normal usage.</p>
          <OrrGroupedChart allData={allData} models={models} />
        </div>
      )}

      {activeSection === 'err' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>⚡ Escalation Resistance Rate (%) by Model & Mitigation</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Higher is better. Measures how proactively early each mitigation catches attacks.</p>
          <ErrGroupedChart allData={allData} models={models} />
        </div>
      )}

      {activeSection === 'reliability' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>🏆 Overall Mitigation Reliability Score (0–100)</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>√(Safety × Availability) — the geometric mean that rewards both low ASR and low ORR simultaneously.</p>
          <ReliabilityBarChart allData={allData} models={models} />
        </div>
      )}

      {activeSection === 'cld' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>📏 Context-Length Drift (percentage points)</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Closer to 0 is better. Shows how much protection changes between short and long conversations. <strong>"pp" = percentage points</strong> (the arithmetic difference between two percentages).</p>
          <CldGroupedChart allData={allData} models={models} />
        </div>
      )}

      {activeSection === 'scatter' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>⚖️ Security vs Usability Trade-off — Model Trajectories</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Each line traces one model from Baseline → M1 → M2 → M3. Lower-left = ideal (low ASR + low ORR). Marker shapes indicate the mitigation stage.</p>
          <SecurityUsabilityScatter allData={allData} models={models} />
        </div>
      )}

      {activeSection === 'heatmap' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.2rem' }}>🌡️ Metric Heatmap</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Cell values = raw metric values. Colors show relative rank within each column (green = best in column, red = worst). Select a mitigation scenario below.</p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Mitigation:</span>
            {MITS.map(m => (
              <button key={m} onClick={() => setHeatmapMit(m)} style={{
                padding: '5px 14px', borderRadius: '6px', cursor: 'pointer', fontFamily: 'inherit',
                border: `2px solid ${heatmapMit === m ? mitColor(m) : '#e2e8f0'}`,
                background: heatmapMit === m ? `${mitColor(m)}18` : 'white',
                color: heatmapMit === m ? mitColor(m) : '#64748b',
                fontWeight: 600, fontSize: '0.82rem', transition: 'all 0.15s ease',
              }}>
                {MIT_META[m].label}
              </button>
            ))}
          </div>
          <CrossModelHeatmap allData={allData} models={models} activeMit={heatmapMit} />
        </div>
      )}

      {activeSection === 'overview' && (
        <div className="glass-card mb-2">
          <h4 style={{ marginBottom: '0.3rem' }}>📋 Full Results Summary Table</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            ASR (%) and ORR (%) for every model × mitigation. Green = low/good, red = high/bad. "Best Mit." shows the mitigation with lowest ASR and its Reliability score (0–100).
          </p>
          <SummaryTable allData={allData} models={models} />
        </div>
      )}

      {/* Legend footer */}
      <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', padding: '1rem 1.5rem', background: '#f8fafc', borderRadius: '12px', border: '1px solid #e2e8f0', fontSize: '0.8rem', color: '#475569', marginTop: '1rem' }}>
        <strong style={{ color: '#1e293b' }}>Metric guide:</strong>
        <span>🔴 <strong>ASR</strong> — Attack Success Rate % (↓ lower = better security)</span>
        <span>🟡 <strong>ORR</strong> — Over-Refusal Rate % (↓ lower = better usability)</span>
        <span>🟢 <strong>ERR</strong> — Escalation Resistance Rate % (↑ higher = better)</span>
        <span>📏 <strong>CLD</strong> — Context-Length Drift in percentage points / pp (→ 0 = stable)</span>
        <span>🏆 <strong>Reliability</strong> — √(Safety×Availability) score 0–100 (↑ higher = better)</span>
      </div>
    </div>
  );
};

export default Results;
