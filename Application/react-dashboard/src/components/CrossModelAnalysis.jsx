import React, { useState, useMemo, useEffect } from 'react';
import Plotly from 'plotly.js-dist-min';
import PlotlyFactory from 'react-plotly.js/factory';
import { getAllModelsComparisons, loadMitigationResultsAcrossModels } from '../utils/dataStore';
import { CrossModelResponseCurveChart, CrossModelReliabilityChart } from './Charts';

import googleLogo from '../assets/google_logo.png';
import mistralLogo from '../assets/mistral_logo.png';
import openaiLogo from '../assets/openai_logo.png';
import qwenLogo from '../assets/qwen_logo.png';
import moonshotLogo from '../assets/moonshot_logo.png';

const createPlot = PlotlyFactory.default || PlotlyFactory;
const Plot = createPlot(Plotly);

const MITIGATIONS = [
  { id: 'none', label: 'Baseline (None)' },
  { id: 'm1', label: 'M1 — Prompt Hardening' },
  { id: 'm2', label: 'M2 — I/O Gate' },
  { id: 'm3', label: 'M3 — State Monitor' }
];

const baseLayout = {
  font: { family: "Inter, sans-serif", color: "#1e293b", size: 12 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  margin: { t: 90, b: 80, l: 80, r: 20 },
  legend: { orientation: 'h', y: -0.3, x: 0.5, xanchor: 'center' },
  hoverlabel: { namelength: -1 }
};

const CrossModelAnalysis = () => {
  const [activeMit, setActiveMit] = useState('none');
  const [crossModelResults, setCrossModelResults] = useState({});
  const [loadingResults, setLoadingResults] = useState(false);

  const allData = getAllModelsComparisons();
  const models = Object.keys(allData).sort(); // Sort alphabetically

  // Fetch all models' results.json traces when mitigation changes
  useEffect(() => {
    let active = true;
    setLoadingResults(true);
    loadMitigationResultsAcrossModels(activeMit).then(data => {
      if (active) {
        setCrossModelResults(data);
        setLoadingResults(false);
      }
    });
    return () => { active = false; };
  }, [activeMit]);

  // Format model names for display
  const formatModelName = (slug) => {
    if (slug === 'openai-gpt-oss-120b') return `<img src="${openaiLogo}" height="14" width="14" /> OpenAI GPT-OSS 120B`;
    if (slug.includes('gemma-3')) return `<img src="${googleLogo}" height="14" width="14" /> Google Gemma 3 27B`;
    if (slug.includes('mistral-small')) return `<img src="${mistralLogo}" height="14" width="14" /> Mistral Small`;
    if (slug.includes('qwen3')) return `<img src="${qwenLogo}" height="14" width="14" /> Alibaba Qwen3 80B`;
    if (slug.includes('kimi')) return `<img src="${moonshotLogo}" height="14" width="14" /> Moonshot Kimi K2`;
    return slug;
  };

  const chartData = useMemo(() => {
    // Array of objects containing metrics for the selected mitigation for each model
    const extracted = models.map(model => {
      const data = allData[model]?.mitigations?.[activeMit];
      return {
        modelName: formatModelName(model),
        asr: data?.attack_success_rate_pct ?? null,
        cld: data?.context_length_drift_pct ?? null,
        dl: data?.mean_ai_latency_turns ?? null,
        err: data?.err_overall ?? null,
        orr: data?.over_refusal_rate_pct ?? null,
      };
    });

    const validModels = extracted.filter(m => m.asr !== null);
    const xLabels = validModels.map(m => m.modelName);

    // Bar chart for ASR
    const asrBarData = [{
      x: xLabels,
      y: validModels.map(m => m.asr),
      type: 'bar',
      name: 'Attack Success Rate (%) (Lower=Better)',
      marker: { color: '#ef4444' }, // Red
      text: validModels.map(m => m.asr !== null ? `${m.asr}%` : 'N/A'),
      textposition: 'outside'
    }];

    // Bar charts for Auxiliary Metrics (CLD, Detection Latency, ERR)
    const errBarData = [{
      x: xLabels,
      y: validModels.map(m => m.err),
      type: 'bar',
      name: 'Escalation Resistance Rate (%) (Higher=Better)',
      marker: { color: '#10b981' }, // Green
      text: validModels.map(m => m.err !== null ? `${m.err}%` : 'N/A'),
      textposition: 'outside'
    }];

    const cldBarData = [{
      x: xLabels,
      y: validModels.map(m => m.cld),
      type: 'bar',
      name: 'Context-Length Drift (pp) (Closer to 0=Better)',
      marker: { color: '#f59e0b' }, // Orange
      text: validModels.map(m => m.cld !== null ? `${m.cld > 0 ? '+' : ''}${m.cld}` : 'N/A'),
      textposition: 'outside'
    }];
    
    const dlBarData = [{
      x: xLabels,
      y: validModels.map(m => m.dl),
      type: 'bar',
      name: 'Detection Latency (Turns) (Lower=Better)',
      marker: { color: '#3b82f6' }, // Blue
      text: validModels.map(m => m.dl !== null ? m.dl : 'N/A'),
      textposition: 'outside'
    }];

    return { 
      xLabels, 
      asrBarData, 
      errBarData, 
      cldBarData, 
      dlBarData, 
      modelCount: validModels.length,
      chartDataRaw: validModels
    };
  }, [allData, models, activeMit]);

  if (models.length === 0) {
    return <div className="warn-panel mt-1"><strong>⚠️ No Data Found</strong><br/>No multi-model result data is available.</div>;
  }

  return (
    <div className="tab-pane">
      <div className="glass-card mb-2" style={{ padding: '1rem 1.5rem' }}>
        <h3 className="mb-1">🌐 Cross-Model Mitigation Analysis</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1rem' }}>
          Evaluate how effective a specific mitigation strategy is across different model architectures and sizes. Select a mitigation strategy below to update the visualizations.
        </p>

        <div className="select-wrapper" style={{ margin: 0 }}>
          <div style={{ fontWeight: 600 }}>Active Mitigation Strategy:</div>
          <select value={activeMit} onChange={e => setActiveMit(e.target.value)}>
            {MITIGATIONS.map(m => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        </div>
      </div>

      {chartData.modelCount === 0 ? (
        <div className="warn-panel mt-1">
          <strong>⚠️ Incomplete Runs</strong><br/>
          None of the models have completed results for <b>{MITIGATIONS.find(m => m.id === activeMit)?.label}</b> yet. They may still be running in the harness.
        </div>
      ) : (
        <>
          <div className="grid-2 mb-2">
            <div className="glass-card">
              <Plot 
                data={chartData.asrBarData} 
                layout={{ 
                  ...baseLayout, 
                  title: 'Attack Success Rate (ASR) Across Models<br><sup>Lower is better. Shows baseline susceptibility or mitigation coverage gaps.</sup>',
                  yaxis: { title: 'Attack Success Rate (%)', range: [-5, 110], showgrid: true, gridcolor: '#f1f5f9', automargin: true },
                  xaxis: { title: 'Model', showgrid: false, automargin: true },
                  showlegend: true
                }} 
                useResizeHandler 
                style={{ width: '100%', height: '340px' }} 
                config={{ responsive: true, displayModeBar: false }} 
              />
            </div>

            <div className="glass-card">
              <Plot 
                data={chartData.errBarData} 
                layout={{ 
                  ...baseLayout, 
                  title: 'Escalation Resistance Rate (ERR)<br><sup>Higher is better. How well the mitigation catches adversarial build-up.</sup>',
                  yaxis: { title: 'ERR (%)', range: [0, 110], showgrid:true, gridcolor:'#f1f5f9', automargin: true },
                  xaxis: { title: 'Model', showgrid: false, automargin: true },
                  showlegend: true
                }} 
                useResizeHandler 
                style={{ width: '100%', height: '340px' }} 
                config={{ responsive: true, displayModeBar: false }} 
              />
            </div>
          </div>

          <div className="grid-2 mb-2">
            <div className="glass-card">
              <Plot 
                data={chartData.cldBarData} 
                layout={{ 
                  ...baseLayout, 
                  title: 'Context-Length Drift (CLD)<br><sup>Closer to 0 is better. Positive means mitigation fails on long contexts.</sup>',
                  yaxis: { title: 'Drift (percentage points)', showgrid:true, gridcolor:'#f1f5f9', automargin: true },
                  xaxis: { title: 'Model', showgrid: false, automargin: true },
                  showlegend: true
                }} 
                useResizeHandler 
                style={{ width: '100%', height: '340px' }} 
                config={{ responsive: true, displayModeBar: false }} 
              />
            </div>

            <div className="glass-card">
              <Plot 
                data={chartData.dlBarData} 
                layout={{ 
                  ...baseLayout, 
                  title: 'Detection Latency<br><sup>Lower is better. Average turns taken before the attack was blocked.</sup>',
                  yaxis: { title: 'Latency (Turns)', showgrid:true, gridcolor:'#f1f5f9', automargin: true },
                  xaxis: { title: 'Model', showgrid: false, automargin: true },
                  showlegend: true
                }} 
                useResizeHandler 
                style={{ width: '100%', height: '340px' }} 
                config={{ responsive: true, displayModeBar: false }} 
              />
            </div>
          </div>

          <div className="card mt-2">
            <CrossModelReliabilityChart chartData={chartData.chartDataRaw} />
          </div>



          <div className="card mt-2">
            {loadingResults ? (
              <div style={{ height: '380px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <p className="text-secondary">Loading multi-turn journey data...</p>
              </div>
            ) : (
              <>
                <CrossModelResponseCurveChart 
                   resultsMap={crossModelResults} 
                   models={models} 
                   formatModel={formatModelName} 
                />
                <div style={{ textAlign: 'center', fontSize: '12px', color: '#475569', marginTop: '-10px', marginBottom: '15px' }}>
                  <span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: '#94a3b8', marginRight: '6px', verticalAlign: 'middle', borderRadius: '2px' }}></span>
                  <span style={{ verticalAlign: 'middle' }}>Cumulative Detection Rate (%)</span>
                </div>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default CrossModelAnalysis;
