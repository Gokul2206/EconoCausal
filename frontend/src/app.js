import React, { useState, useMemo } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import './app.css';
import PrescriptionTable from './PrescriptionTable';

const Plot = createPlotlyComponent(Plotly);

const causalModel = {
  treatment: 'discount_amount',
  outcome: 'purchase_made',
  confounders: ['income', 'prior_orders', 'tenure_days', 'season', 'engagement_proxy'],
};

const REQUIRED_COLUMNS = ['customer_id', 'discount_amount', 'purchase_made', 'income', 'prior_orders', 'tenure_days', 'season'];
const SCORE_COLUMNS = ['customer_id', 'ite_10', 'ite_20', 'ite_diff', 'discount_amount', 'purchase_made'];
const N_BINS = 20;

function parseCsv(text) {
  const lines = text.split('\n').filter((l) => l.trim());
  const headers = lines[0].split(',').map((h) => h.trim());
  const rows = lines.slice(1).map((line) => {
    const cells = line.split(',');
    const obj = {};
    headers.forEach((h, i) => { obj[h] = cells[i]; });
    return obj;
  });
  return { headers, rows };
}

function targetingCurve(rows, scoreKey, level) {
  const sub = rows
    .filter((r) => r.discount_amount === 0 || r.discount_amount === level)
    .sort((a, b) => b[scoreKey] - a[scoreKey]);
  const n = sub.length;
  const ks = Array.from({ length: N_BINS }, (_, i) =>
    Math.floor(n / N_BINS) * (i + 1) + (i === N_BINS - 1 ? n - Math.floor(n / N_BINS) * N_BINS : 0));

  let cumT = 0, cumC = 0, cumYT = 0, cumYC = 0, idx = 0;
  const values = [];
  ks.forEach((k) => {
    while (idx < k) {
      const r = sub[idx];
      if (r.discount_amount === level) { cumT += 1; cumYT += r.purchase_made; }
      else { cumC += 1; cumYC += r.purchase_made; }
      idx += 1;
    }
    const rateT = cumT ? cumYT / cumT : 0;
    const rateC = cumC ? cumYC / cumC : 0;
    values.push((rateT - rateC) * k);
  });
  const total = values[values.length - 1];
  const x = ks.map((k) => (k / n) * 100);
  const random = x.map((xi) => (total * xi) / 100);
  return { x, values, random };
}

function upliftByDecile(rows, scoreKey, level) {
  const sub = rows
    .filter((r) => r.discount_amount === 0 || r.discount_amount === level)
    .sort((a, b) => b[scoreKey] - a[scoreKey]);
  const n = sub.length;
  const deciles = [];
  for (let d = 0; d < 10; d++) {
    const start = Math.floor((n * d) / 10);
    const end = Math.floor((n * (d + 1)) / 10);
    let tN = 0, tY = 0, cN = 0, cY = 0;
    for (let i = start; i < end; i++) {
      const r = sub[i];
      if (r.discount_amount === level) { tN += 1; tY += r.purchase_made; }
      else { cN += 1; cY += r.purchase_made; }
    }
    const rateT = tN ? tY / tN : 0;
    const rateC = cN ? cY / cN : 0;
    deciles.push({ label: `D${d + 1}`, uplift: (rateT - rateC) * 100 });
  }
  return deciles;
}

function App() {
  const [file, setFile] = useState(null);
  const [budget, setBudget] = useState(50000);
  const [budgetError, setBudgetError] = useState('');
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const [driftReport, setDriftReport] = useState(null);
  const [driftError, setDriftError] = useState('');

  const [scoreRows, setScoreRows] = useState(null);
  const [scoresError, setScoresError] = useState('');
  const [level, setLevel] = useState('ite_20');
  const [minEffect, setMinEffect] = useState(0);

  const lvlAmount = level === 'ite_10' ? 10 : 20;

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setDriftError('');
    setDriftReport(null);

    const reader = new FileReader();
    reader.onload = async (event) => {
      const { headers, rows } = parseCsv(event.target.result);
      const missing = REQUIRED_COLUMNS.filter((c) => !headers.includes(c));
      if (missing.length > 0) {
        setData(null);
        setError(`Missing columns: ${missing.join(', ')}`);
        setLoading(false);
        return;
      } 
      setError('');
      setData({ filename: file.name, rows: rows.length, columns: headers.length });

      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("http://localhost:8000/api/detect_drift", {
          method: "POST",
          body: formData,
        });
        if (!response.ok) throw new Error("API Offline");
        const driftData = await response.json();
        setDriftReport(driftData);
      } catch (err) {
        setDriftError(err.message);
      } finally {
        setLoading(false);
      }
    };
    reader.readAsText(file);
  };

  const handleScores = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const { headers, rows } = parseCsv(event.target.result);
      const missing = SCORE_COLUMNS.filter((c) => !headers.includes(c));
      if (missing.length > 0) {
        setScoreRows(null);
        setScoresError(`Missing: ${missing.join(', ')}`);
        return;
      }
      setScoresError('');
      setScoreRows(rows.map((r) => ({
        ite_10: parseFloat(r.ite_10),
        ite_20: parseFloat(r.ite_20),
        discount_amount: Number(r.discount_amount),
        purchase_made: Number(r.purchase_made),
      })));
    };
    reader.readAsText(f);
  };

  const filtered = useMemo(() => {
    if (!scoreRows) return [];
    return scoreRows.filter((r) => r[level] >= minEffect);
  }, [scoreRows, level, minEffect]);

  const summary = useMemo(() => {
    if (filtered.length === 0) return null;
    const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    return {
      count: filtered.length,
      meanEffect: mean(filtered.map((r) => r[level])),
      meanOutcome: mean(filtered.map((r) => r.purchase_made)),
    };
  }, [filtered, level]);

  const roiText = useMemo(() => {
    if (!scoreRows || filtered.length === 0) return null;
    const aiSales = filtered.reduce((sum, r) => sum + r[level], 0);
    const totalMeanEffect = scoreRows.reduce((sum, r) => sum + r[level], 0) / scoreRows.length;
    const randomSales = filtered.length * totalMeanEffect;
    const moneySaved = (aiSales - randomSales) * 50; 
    if (moneySaved <= 0) return "Adjust sliders to find a profitable strategy.";
    return `💰 Strategy saves $${Math.round(moneySaved).toLocaleString('en-US')}`;
  }, [scoreRows, filtered, level]);

  const histogram = useMemo(() => {
    if (!scoreRows) return [];
    const values = scoreRows.map((r) => r[level]);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const width = (max - min) / N_BINS || 1;
    const counts = Array(N_BINS).fill(0);
    values.forEach((v) => {
      const i = Math.min(N_BINS - 1, Math.floor((v - min) / width));
      counts[i] += 1;
    });
    return counts.map((c, i) => ({ start: min + i * width, count: c }));
  }, [scoreRows, level]);

  const maxCount = Math.max(1, ...histogram.map((b) => b.count));

  const curve = useMemo(() => {
    if (!scoreRows) return null;
    return targetingCurve(scoreRows, level, lvlAmount);
  }, [scoreRows, level, lvlAmount]);

  const deciles = useMemo(() => {
    if (!scoreRows) return null;
    return upliftByDecile(scoreRows, level, lvlAmount);
  }, [scoreRows, level, lvlAmount]);

  return (
    <div className="app-wrapper">
      
      {/* LEFT SIDEBAR (Fixed Toolbar) */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>🎯 EconoCausal</h1>
          <p>Causal AI Engine</p>
        </div>

        <div className="sidebar-section">
          <h3>1. System Data</h3>
          <form onSubmit={handleUpload}>
            <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} required />
            <button type="submit" disabled={!file || loading}>
              {loading ? 'Scanning API...' : 'Upload Validation Data'}
            </button>
          </form>
          {error && <div className="error-box">{error}</div>}
          {driftError && <div className="error-box">{driftError}</div>}
          
          {driftReport && (
            <div className={`status-box ${driftReport.overall_drift_detected ? 'drift-warn' : 'drift-safe'}`}>
              <strong>{driftReport.overall_drift_detected ? '⚠️ Drift Detected' : '✅ Data Validated'}</strong>
              <p>{driftReport.warning_message}</p>
            </div>
          )}
        </div>

        <div className="sidebar-section">
          <h3>2. Budget & ITE Scores</h3>
          <label>Campaign Budget ($)</label>
          <input type="number" value={budget}
            onChange={(e) => setBudget(Number(e.target.value))}
            min="1000" step="1000" />
          
          <label style={{marginTop: '15px'}}>Load Scores (ite_scores.csv)</label>
          <input type="file" accept=".csv" onChange={handleScores} />
          {scoresError && <div className="error-box">{scoresError}</div>}
        </div>

        {scoreRows && (
          <div className="sidebar-section highlight-section">
            <h3>3. Strategy Controls</h3>
            <label>Discount Tier</label>
            <select value={level} onChange={(e) => setLevel(e.target.value)}>
              <option value="ite_10">$10 Promotion</option>
              <option value="ite_20">$20 Promotion</option>
            </select>
            
            <label style={{marginTop: '15px'}}>Targeting Threshold: {minEffect.toFixed(2)}</label>
            <input type="range" min="-0.1" max="0.2" step="0.01"
              value={minEffect} onChange={(e) => setMinEffect(Number(e.target.value))} />
          </div>
        )}

        <div className="sidebar-section dag-mini">
          <h3>Causal Graph Info</h3>
          <code>Treatment: {causalModel.treatment}</code>
          <code>Outcome: {causalModel.outcome}</code>
          <p>Confounders: {causalModel.confounders.join(', ')}</p>
        </div>
      </aside>

      {/* RIGHT MAIN CONTENT (Scrolls independently) */}
      <main className="main-content">
        {!scoreRows ? (
          <div className="empty-state">
            <h2>Welcome to EconoCausal</h2>
            <p>Please upload your campaign data and ITE scores in the sidebar to begin analysis.</p>
          </div>
        ) : (
          <>
            {/* Top Stats Bar */}
            <div className="top-stats-bar">
              <div className="stat-card">
                <h4>Audience Size</h4>
                <div className="value">{summary?.count.toLocaleString('en-US') || 0}</div>
              </div>
              <div className="stat-card">
                <h4>Mean Effect</h4>
                <div className="value">{summary ? summary.meanEffect.toFixed(4) : '0.00'}</div>
              </div>
              <div className="stat-card highlight-stat">
                <h4>ROI Summary</h4>
                <div className="value text-green">{roiText}</div>
              </div>
            </div>

            {/* Dashboard Grid */}
            <div className="dashboard-grid">
              
              {/* Histogram Full Width */}
              <div className="chart-panel full-width">
                <h3 className="panel-title">ITE Distribution</h3>
                <div className="histogram">
                  {histogram.map((b, i) => (
                    <div key={i} className="bar"
                      title={`${b.start.toFixed(3)}: ${b.count}`}
                      style={{ height: `${(b.count / maxCount) * 100}%` }} />
                  ))}
                </div>
              </div>

              {/* Side by Side Plots */}
              <div className="chart-panel">
                <Plot
                  data={[
                    { x: curve?.x, y: curve?.values, type: 'scatter', mode: 'lines+markers', name: 'AI Strategy', marker: {color: '#4f46e5'} },
                    { x: curve?.x, y: curve?.random, type: 'scatter', mode: 'lines', line: { dash: 'dash', color: '#94a3b8' }, name: 'Random' },
                  ]}
                  layout={{ title: 'Targeting Curve', margin: { l: 40, r: 20, t: 40, b: 40 }, autosize: true, height: 300 }}
                  useResizeHandler style={{ width: '100%' }} config={{ displayModeBar: false }}
                />
              </div>

              <div className="chart-panel">
                <Plot
                  data={[
                    { x: deciles?.map(d=>d.label), y: deciles?.map(d=>d.uplift), type: 'bar', marker: { color: deciles?.map(d => d.uplift >= 0 ? '#10b981' : '#ef4444') } },
                  ]}
                  layout={{ title: 'Uplift by Decile', margin: { l: 40, r: 20, t: 40, b: 40 }, autosize: true, height: 300 }}
                  useResizeHandler style={{ width: '100%' }} config={{ displayModeBar: false }}
                />
              </div>

            </div>

            {/* Prescription Table */}
            <div className="table-panel">
              <PrescriptionTable />
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;