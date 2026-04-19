import React, { useMemo } from 'react';
import Plotly from 'plotly.js-dist-min';
import PlotlyFactory from 'react-plotly.js/factory';

const createPlot = PlotlyFactory.default || PlotlyFactory;
const Plot = createPlot(Plotly);

const COLORS = {
  none: "rgba(148, 163, 184, 1)",
  m1: "rgba(99, 102, 241, 1)",
  m2: "rgba(8, 145, 178, 1)",
  m3: "rgba(245, 158, 11, 1)",
};
const FILL_RGBA = {
  none: "rgba(148,163,184,0.12)",
  m1:   "rgba(99,102,241,0.12)",
  m2:   "rgba(8,145,178,0.12)",
  m3:   "rgba(245,158,11,0.12)",
};

const NAMES = {
  none: "Baseline",
  m1: "M1: Prompt Hardening",
  m2: "M2: I/O Gate",
  m3: "M3: State Monitor",
};

const baseLayout = {
  font: { family: "Inter, sans-serif", color: "#1e293b", size: 12 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  margin: { t: 55, b: 80, l: 60, r: 20 },
  legend: { orientation: 'h', y: -0.3, x: 0.5, xanchor: 'center' },
  hoverlabel: { namelength: -1 }
};

export const ResponseCurvesChart = ({ results, mits }) => {
  const data = useMemo(() => {
    let maxTurn = 17;
    // Find max turn
    mits.forEach(mit => {
      (results[mit] || []).forEach(r => {
        if (r.is_attack) {
          if (r.injection_turn) maxTurn = Math.max(maxTurn, r.injection_turn);
          if (r.caught_at_turn) maxTurn = Math.max(maxTurn, r.caught_at_turn);
        }
      });
    });

    return mits.map(mit => {
      const recs = (results[mit] || []).filter(r => r.is_attack);
      if (recs.length === 0) return null;
      
      const total = recs.length;
      const byTurn = {};
      recs.forEach(r => {
        if (r.caught_at_turn !== null && r.caught_at_turn !== undefined) {
          byTurn[r.caught_at_turn] = (byTurn[r.caught_at_turn] || 0) + 1;
        }
      });
      
      const turns = Array.from({length: maxTurn + 1}, (_, i) => i);
      let running = 0;
      const cumulative = turns.map(t => {
        running += (byTurn[t] || 0);
        return parseFloat((running / total * 100).toFixed(1));
      });

      return {
        x: turns,
        y: cumulative,
        name: NAMES[mit],
        mode: 'lines+markers',
        marker: { color: COLORS[mit], size: 5 },
        line: { color: COLORS[mit], width: 2.5 }
      };
    }).filter(Boolean);
  }, [results, mits]);

  return <Plot data={data} layout={{ ...baseLayout, title: 'Unified Mitigation Response Comparison<br><sup>Cumulative % of attacks caught by turn number — higher & earlier is better</sup>', yaxis: { title: 'Attacks Caught (%)', range: [0, 105], showgrid:true, gridcolor:'#f1f5f9' }, xaxis: { title: 'Turn Number', showgrid:true, gridcolor:'#f1f5f9' } }} useResizeHandler style={{ width: '100%', height: '380px' }} config={{ responsive: true, displayModeBar: false }} />;
};

export const CrossModelResponseCurveChart = ({ resultsMap, models, formatModel }) => {
  const data = useMemo(() => {
    let maxTurn = 17;
    const palette = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"];
    return models.map((model, idx) => {
      const results = resultsMap[model] || [];
      const recs = results.filter(r => r.is_attack);
      if (recs.length === 0) return null;

      recs.forEach(r => {
        if (r.injection_turn) maxTurn = Math.max(maxTurn, r.injection_turn);
        if (r.caught_at_turn) maxTurn = Math.max(maxTurn, r.caught_at_turn);
      });

      const total = recs.length;
      const byTurn = {};
      recs.forEach(r => {
        if (r.caught_at_turn !== null && r.caught_at_turn !== undefined) {
          byTurn[r.caught_at_turn] = (byTurn[r.caught_at_turn] || 0) + 1;
        }
      });
      
      const turns = Array.from({length: maxTurn + 1}, (_, i) => i);
      let running = 0;
      const cumulative = turns.map(t => {
        running += (byTurn[t] || 0);
        return parseFloat((running / total * 100).toFixed(1));
      });

      const color = palette[idx % palette.length];

      return {
        x: turns,
        y: cumulative,
        name: formatModel ? formatModel(model) : model,
        mode: 'lines+markers',
        marker: { color, size: 5 },
        line: { color, width: 2.5 }
      };
    }).filter(Boolean);
  }, [resultsMap, models, formatModel]);

  return <Plot data={data} layout={{ ...baseLayout, title: 'Mitigation Response Journey Across Models<br><sup>Cumulative % of attacks caught by turn number — higher & earlier is better</sup>', yaxis: { title: 'Attacks Caught (%)', range: [0, 105], showgrid:true, gridcolor:'#f1f5f9', automargin:true }, xaxis: { title: 'Turn Number', showgrid:true, gridcolor:'#f1f5f9', automargin:true }, showlegend: true, margin: { t: 90, b: 80, l: 60, r: 20 }, legend: { orientation: 'h', y: -0.3, x: 0.5, xanchor: 'center' } }} useResizeHandler style={{ width: '100%', height: '380px' }} config={{ responsive: true, displayModeBar: false }} />;
};
export const MitigationDeltaBarChart = ({ allData, models, formatModel }) => {
  const data = useMemo(() => {
    const mitigations = ['m1', 'm2', 'm3'];
    const xValues = models.map(m => formatModel ? formatModel(m) : m);
    
    return mitigations.map(mit => {
      const yValues = models.map(model => {
        const noneAsr = allData[model]?.mitigations?.['none']?.attack_success_rate_pct ?? null;
        const mitAsr = allData[model]?.mitigations?.[mit]?.attack_success_rate_pct ?? null;
        if (noneAsr === null || mitAsr === null) return null;
        return parseFloat((noneAsr - mitAsr).toFixed(1));
      });

      return {
        x: xValues,
        y: yValues,
        name: `${NAMES[mit] || mit} (ASR Reduction pp, Higher=Better)`,
        type: 'bar',
        marker: { color: COLORS[mit] }
      };
    });
  }, [allData, models, formatModel]);

  return <Plot data={data} layout={{ ...baseLayout, barmode: 'group', title: 'Mitigation Efficacy (ASR Reduction vs Baseline)<br><sup>Higher is better. Measures the absolute drop in ASR compared to \'None\'.</sup>', yaxis: { title: 'ASR Reduction (pp)', showgrid:true, gridcolor:'#f1f5f9', automargin:true }, xaxis: { title: 'Model', showgrid:false, automargin:true }, showlegend: true, margin: { t: 90, b: 80, l: 60, r: 20 }, legend: { orientation: 'h', y: -0.3, x: 0.5, xanchor: 'center' } }} useResizeHandler style={{ width: '100%', height: '340px' }} config={{ responsive: true, displayModeBar: false }} />;
};

export const CrossModelReliabilityChart = ({ chartData }) => {
  const data = useMemo(() => {
    // Normal order for left-to-right plotting
    const xLabels = chartData.map(d => d.modelName);
    
    const safetyScores = chartData.map(d => parseFloat((100 - (d.asr ?? 0)).toFixed(1)));
    const availScores = chartData.map(d => parseFloat((100 - (d.orr ?? 0)).toFixed(1)));
    const combined = safetyScores.map((s, i) => {
      const a = availScores[i];
      return (s > 0 && a > 0) ? parseFloat(Math.sqrt(s * a).toFixed(1)) : 0.0;
    });
    
    return [
      {
        name: "Safety Score (100−ASR)", x: xLabels, y: safetyScores, type: 'bar', marker: { color: "#22c55e" }, text: safetyScores.map(v => `${v}`), textposition: "outside"
      },
      {
        name: "Availability (100−ORR)", x: xLabels, y: availScores, type: 'bar', marker: { color: "#0891b2" }, text: availScores.map(v => `${v}`), textposition: "outside"
      },
      {
        name: "Overall Reliability (√Safety×Avail)", x: xLabels, y: combined, type: 'bar', marker: { color: "#6366f1" }, text: combined.map(v => `<b>${v}</b>`), textposition: "outside"
      }
    ];
  }, [chartData]);

  return <Plot data={data} layout={{ ...baseLayout, margin: { t: 90, b: 80, l: 60, r: 20 }, barmode: 'group', title: 'Overall Mitigation Reliability Across Models<br><sup>Geometric Mean balances Safety vs. Usability — higher is better</sup>', yaxis: { title: 'Score (0–100)', range: [0, 115], showgrid:true, gridcolor:'#f1f5f9', automargin:true }, xaxis: { title: 'Model', showgrid:false, automargin:true }, showlegend: true, legend: { orientation: 'h', y: -0.3, x: 0.5, xanchor: 'center' } }} useResizeHandler style={{ width: '100%', height: '420px' }} config={{ responsive: true, displayModeBar: false }} />;
};

export const RunHistoryChart = ({ results, mits }) => {
  const data = useMemo(() => {
    return mits.map(mit => {
      const attackRecs = (results[mit] || []).filter(r => r.is_attack);
      if (attackRecs.length === 0) return null;
      
      let totalSeen = 0;
      let totalFailed = 0;
      const runningX = [];
      const runningY = [];
      
      attackRecs.forEach(r => {
        totalSeen += 1;
        if (r.attack_succeeded) totalFailed += 1;
        runningX.push(totalSeen);
        runningY.push(parseFloat((totalFailed / totalSeen * 100).toFixed(1)));
      });

      return {
        x: runningX,
        y: runningY,
        name: NAMES[mit],
        mode: 'lines',
        fill: 'tozeroy',
        fillcolor: FILL_RGBA[mit],
        line: { color: COLORS[mit], width: 2 }
      };
    }).filter(Boolean);
  }, [results, mits]);
  
  return <Plot data={data} layout={{ ...baseLayout, title: 'Unified Run History Comparison<br><sup>Running average ASR as the evaluation progresses — lower convergence = better</sup>', yaxis: { title: 'Running Average ASR (%)', range: [0, 105], showgrid:true, gridcolor:'#f1f5f9' }, xaxis: { title: 'Total Progression (Attack Samples)', showgrid:true, gridcolor:'#f1f5f9' } }} useResizeHandler style={{ width: '100%', height: '380px' }} config={{ responsive: true, displayModeBar: false }} />;
};

export const EfficiencyChart = ({ comp, mits }) => {
  const data = useMemo(() => {
    const mitsData = comp?.mitigations || {};
    const validMits = mits.filter(m => mitsData[m]);
    const labels = validMits.map(m => NAMES[m]);
    const asrs = validMits.map(m => mitsData[m].attack_success_rate_pct);
    const lats = validMits.map(m => mitsData[m].mean_ai_latency_turns);
    
    return [
      {
        name: "ASR % (lower=better)",
        x: labels,
        y: asrs,
        type: 'bar',
        marker: { color: asrs.map(a => a > 30 ? "#ef4444" : a > 15 ? "#f97316" : "#22c55e") },
        text: asrs.map(v => `${v.toFixed(1)}%`),
        textposition: 'outside',
        yaxis: 'y1',
        offsetgroup: 1
      },
      {
        name: "AI Latency (turns, lower=better)",
        x: labels,
        y: lats,
        type: 'bar',
        marker: { color: "#93c5fd" },
        text: lats.map(v => v.toFixed(2)),
        textposition: 'outside',
        yaxis: 'y2',
        offsetgroup: 2
      }
    ];
  }, [comp, mits]);

  return <Plot data={data} layout={{ ...baseLayout, title: 'Efficiency Analysis: Security vs. Performance<br><sup>Lower ASR (red) and lower latency (blue) = better mitigation</sup>', barmode: 'group', xaxis: { title: 'Mitigation Strategy' }, yaxis: { title: 'ASR (%)', range: [0, 110], showgrid:true, gridcolor:'#f1f5f9' }, yaxis2: { title: "Latency (Turns)", overlaying: "y", side: "right", showgrid:false, range: [0, 1.5] }, margin: { t: 55, b: 30, l: 40, r: 40 } }} useResizeHandler style={{ width: '100%', height: '380px' }} config={{ responsive: true, displayModeBar: false }} />;
};

export const AsrByLengthChart = ({ metrics, mits }) => {
  const data = useMemo(() => {
    const buckets = ["short", "medium", "long"];
    const bucketLabels = ["Short (<=8 turns)", "Medium (9-14)", "Long (>14 turns)"];
    
    return mits.map((mit, i) => {
      const m = metrics[mit] || {};
      const asrMap = {};
      (m.asr_by_length_group || []).forEach(entry => {
        asrMap[entry.length_group] = entry.asr_pct;
      });
      
      const yVals = buckets.map(lg => asrMap[lg] || 0);
      return {
        name: NAMES[mit],
        x: bucketLabels,
        y: yVals,
        type: 'bar',
        marker: { color: COLORS[mit] },
        text: yVals.map(v => `${Math.round(v)}%`),
        textposition: "outside"
      };
    });
  }, [metrics, mits]);

  return <Plot data={data} layout={{ ...baseLayout, title: 'ASR by Mitigation & Conversation Length<br><sup>Lower is better — reveals if protection weakens in longer chats</sup>', barmode: 'group', yaxis: { title: 'ASR (%)', range: [0, 115], showgrid:true, gridcolor:'#f1f5f9' }, xaxis: { title: 'Conversation Length', showgrid:false } }} useResizeHandler style={{ width: '100%', height: '380px' }} config={{ responsive: true, displayModeBar: false }} />;
};

export const ReliabilityChart = ({ comp, mits }) => {
  const data = useMemo(() => {
    const mitsData = comp?.mitigations || {};
    const validMits = mits.filter(m => mitsData[m]).reverse();
    const labels = validMits.map(m => NAMES[m]);
    
    const safetyScores = validMits.map(m => parseFloat((100 - mitsData[m].attack_success_rate_pct).toFixed(1)));
    const availScores = validMits.map(m => parseFloat((100 - mitsData[m].over_refusal_rate_pct).toFixed(1)));
    const combined = safetyScores.map((s, i) => {
      const a = availScores[i];
      return (s > 0 && a > 0) ? parseFloat(Math.sqrt(s * a).toFixed(1)) : 0.0;
    });
    
    return [
      {
        name: "Safety Score (100−ASR)", y: labels, x: safetyScores, type: 'bar', orientation: 'h', marker: { color: "#22c55e" }, text: safetyScores.map(v => `${v}`), textposition: "auto"
      },
      {
        name: "Availability (100−ORR)", y: labels, x: availScores, type: 'bar', orientation: 'h', marker: { color: "#0891b2" }, text: availScores.map(v => `${v}`), textposition: "auto"
      },
      {
        name: "Overall Reliability (√Safety×Avail)", y: labels, x: combined, type: 'bar', orientation: 'h', marker: { color: "#6366f1" }, text: combined.map(v => `<b>${v}</b>`), textposition: "auto"
      }
    ];
  }, [comp, mits]);

  return <Plot data={data} layout={{ ...baseLayout, margin: { t: 55, b: 30, l: 150, r: 40 }, barmode: 'group', title: 'Overall Mitigation Reliability<br><sup>Geometric mean balances Safety vs. Usability — higher is better</sup>', xaxis: { title: 'Score (0–100)', range: [0, 115], showgrid:true, gridcolor:'#f1f5f9' }, yaxis: { title: 'Mitigation Strategy', showgrid:false, tickpad: 12 } }} useResizeHandler style={{ width: '100%', height: '380px' }} config={{ responsive: true, displayModeBar: false }} />;
};

export const HeatmapChart = ({ comp, mits }) => {
  const data = useMemo(() => {
    const mitsData = comp?.mitigations || {};
    const validMits = mits.filter(m => mitsData[m]);
    const metricsNames = ["ASR", "DL", "ORR", "ERR", "Trust"];
    
    const rows = [];
    const annot = [];
    
    validMits.forEach(mit => {
      const d = mitsData[mit];
      const safety = 100 - d.attack_success_rate_pct;
      const avail = 100 - d.over_refusal_rate_pct;
      const combined = (safety > 0 && avail > 0) ? Math.sqrt(safety * avail) : 0;
      
      rows.push([
        d.attack_success_rate_pct, d.mean_ai_latency_turns, d.over_refusal_rate_pct, d.err_overall, combined
      ]);
      annot.push([
        `${d.attack_success_rate_pct.toFixed(1)}%`,
        `${d.mean_ai_latency_turns.toFixed(2)}`,
        `${d.over_refusal_rate_pct.toFixed(1)}%`,
        `${d.err_overall.toFixed(1)}%`,
        `${combined.toFixed(1)}`
      ]);
    });
    
    // Normalize logic
    const norm = [];
    const lowerBetter = [true, true, true, false, false];
    const n_mits = rows.length;
    
    for (let r = 0; r < n_mits; r++) { norm.push([0,0,0,0,0]); }
    
    for (let c = 0; c < 5; c++) {
      let colVals = rows.map(r => r[c]);
      let mn = Math.min(...colVals);
      let mx = Math.max(...colVals);
      for (let r = 0; r < n_mits; r++) {
        if (mx === mn) {
          norm[r][c] = 0.5;
        } else {
          let raw = (rows[r][c] - mn) / (mx - mn);
          norm[r][c] = lowerBetter[c] ? (1 - raw) : raw;
        }
      }
    }
    
    return [{
      z: norm,
      x: metricsNames,
      y: validMits.map(m => NAMES[m]),
      type: 'heatmap',
      colorscale: [[0, "#fca5a5"], [0.5, "#fde68a"], [1, "#86efac"]],
      zmin: 0, zmax: 1,
      showscale: true,
      text: annot,
      texttemplate: "%{text}",
      textfont: {size: 12, color: "#1e293b", family: "Inter"},
      hovertemplate: "<b>%{y}</b><br>%{x}: %{text}<extra></extra>"
    }];
  }, [comp, mits]);

  return <Plot data={data} layout={{ ...baseLayout, margin: { t: 55, b: 30, l: 140, r: 80 }, title: 'Mitigation Comparison Heatmap<br><sup>Green = stronger performance on that metric</sup>', xaxis: { title: 'Metric', side: "bottom" }, yaxis: { title: 'Mitigation Strategy' } }} useResizeHandler style={{ width: '100%', height: '360px' }} config={{ responsive: true, displayModeBar: false }} />;
};

export const DatasetPieChart = ({ dataObj }) => {
  return <Plot data={[{ labels: Object.keys(dataObj), values: Object.values(dataObj), type: 'pie', hole: 0.52, marker: { colors: ["#0284c7","#059669","#d97706","#7c3aed","#e11d48","#4f46e5"] }, textinfo: 'label+percent' }]} layout={{ ...baseLayout, title: 'Topic Distribution', showlegend: false }} useResizeHandler style={{ width: '100%', height: '300px' }} config={{ responsive: true, displayModeBar: false }} />;
};

export const DatasetBarChart = ({ lenData }) => {
  const data = [{
    x: Object.keys(lenData),
    y: Object.values(lenData),
    type: 'bar',
    marker: { color: ["#4f46e5", "#0891b2", "#d97706"] },
    text: Object.values(lenData).map(v => String(v)),
    textposition: 'outside'
  }];
  return <Plot data={data} layout={{ ...baseLayout, title: 'Conversations by Length Group', yaxis: { title: 'Count', showgrid:true, gridcolor:'#f1f5f9', range: [0, Math.max(...Object.values(lenData)) * 1.25] }, xaxis: { title: 'Length Group', showgrid:false } }} useResizeHandler style={{ width: '100%', height: '300px' }} config={{ responsive: true, displayModeBar: false }} />;
};
