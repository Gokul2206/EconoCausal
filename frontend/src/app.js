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
  const [activeTab, setActiveTab] = useState('dashboard');

  const [driftReport, setDriftReport] = useState(null);
  const [driftError, setDriftError] = useState('');

  const [scoreRows, setScoreRows] = useState(null);
  const [scoresFile, setScoresFile] = useState(null);
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
        const response = await fetch("http://localhost:8000/api/detect_drift", { method: "POST", body: formData });
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

  const handleScores = async () => {
    setLoading(true);
    setScoresError('');
    
    try {
      // Ask Python for the data instead of uploading a file!
      const response = await fetch("http://localhost:8000/api/scores");
      
      if (!response.ok) {
        throw new Error("Failed to fetch scores from Python API.");
      }
      
      const data = await response.json();
      
      // Load the data into the dashboard
      setScoreRows(data.map((r) => ({
        ite_10: parseFloat(r.ite_10),
        ite_20: parseFloat(r.ite_20),
        discount_amount: Number(r.discount_amount),
        purchase_made: Number(r.purchase_made),
      })));
      
    } catch (err) {
      setScoresError(err.message);
    } finally {
      setLoading(false);
    }
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
    return `+$${Math.round(moneySaved).toLocaleString('en-US')}`;
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
      {/* INTERACTIVE TOP MENU BAR */}
      <nav className="top-navbar">
        <div className="brand">
          <span className="logo-icon">💠</span> EconoCausal
        </div>
        <div className="nav-menu">
          <button className={activeTab === 'dashboard' ? 'active' : ''} onClick={() => setActiveTab('dashboard')}>Overview</button>
          <button className={activeTab === 'model' ? 'active' : ''} onClick={() => setActiveTab('model')}>Causal Model</button>
          <button className={activeTab === 'settings' ? 'active' : ''} onClick={() => setActiveTab('settings')}>Settings</button>
        </div>
        <div className="user-profile">
          <div className="avatar">AI</div>
        </div>
      </nav>

      <main className="main-content">
        
        {/* INTERACTIVE CONTROL STRIP */}
        <section className="control-strip">
          
          <div className="control-card">
            <h3>1. Data Drift Validation</h3>
            <p className="subtitle">Connect campaign data to API</p>
            <form onSubmit={handleUpload} className="upload-form">
              <label className="custom-file-upload">
                <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} required />
                {file ? file.name : 'Choose CSV File'}
              </label>
              <button type="submit" disabled={!file || loading} className="primary-btn">
                {loading ? 'Scanning API...' : 'Validate'}
              </button>
            </form>
            {driftReport && (
              <div className={`status-badge ${driftReport.overall_drift_detected ? 'warn' : 'success'}`}>
                {driftReport.overall_drift_detected ? '⚠️ Model Drift Detected' : '✅ Data Safe'}
              </div>
            )}
            {driftError && <div className="error-text">{driftError}</div>}
          </div>

          <div className="control-card">
            <h3>2. Load Intelligence</h3>
            <p className="subtitle">Upload ITE Predictions</p>
            <button 
              onClick={handleScores} 
              className="primary-btn" 
              style={{ width: '100%', marginTop: '15px' }}
              disabled={loading}
            >
              {loading ? 'Fetching from API...' : 'Load Causal Intelligence'}
            </button>
            {scoresError && <div className="error-text">{scoresError}</div>}
          </div>

          <div className="control-card">
            <h3>3. Strategic Tuning</h3>
            <p className="subtitle">Set boundaries</p>
            <div className="tuning-controls">
              <div className="input-group">
                <label>Budget ($)</label>
                <input type="number" value={budget} onChange={(e) => setBudget(Number(e.target.value))} min="1000" step="1000" />
              </div>
              <div className="input-group">
                <label>Promotion Tier</label>
                <select value={level} onChange={(e) => setLevel(e.target.value)}>
                  <option value="ite_10">$10 Off</option>
                  <option value="ite_20">$20 Off</option>
                </select>
              </div>
            </div>
          </div>

        </section>

        {/* ANALYTICS BOARD */}
        {!scoreRows ? (
          <div className="empty-state">
            <div className="empty-icon">📊</div>
            <h2>Awaiting Data</h2>
            <p>Upload your campaign data and ITE scores in the control strip above to begin.</p>
          </div>
        ) : (
          <div className="analytics-board">
            
            {/* KPI ROW */}
            <div className="kpi-row">
              <div className="kpi-card">
                <h4>Eligible Audience</h4>
                <div className="kpi-value">{summary?.count.toLocaleString('en-US') || 0}</div>
              </div>
              <div className="kpi-card">
                <h4>Mean Causal Lift</h4>
                <div className="kpi-value">{(summary ? summary.meanEffect * 100 : 0).toFixed(2)}%</div>
              </div>
              <div className="kpi-card highlight-kpi">
                <h4>Value Generated vs Random</h4>
                <div className="kpi-value profit-text">{roiText}</div>
              </div>
            </div>

            {/* DYNAMIC SLIDER & HISTOGRAM */}
            <div className="chart-card full-width">
              <div className="slider-header">
                <h3>Targeting Threshold & Distribution</h3>
                <div className="slider-value">Min Effect: {minEffect.toFixed(2)}</div>
              </div>
              <input type="range" className="modern-slider" min="-0.1" max="0.2" step="0.01" value={minEffect} onChange={(e) => setMinEffect(Number(e.target.value))} />
              
              <div className="histogram">
                {histogram.map((b, i) => {
                  const isTargeted = b.start >= minEffect;
                  return (
                    <div key={i} className={`bar ${isTargeted ? 'bar-active' : 'bar-inactive'}`}
                      title={`${b.start.toFixed(3)}: ${b.count}`}
                      style={{ height: `${(b.count / maxCount) * 100}%` }} />
                  );
                })}
              </div>
            </div>

            {/* CHARTS GRID */}
            <div className="charts-grid">
              <div className="chart-card">
                <Plot
                  data={[
                    { x: curve?.x, y: curve?.values, type: 'scatter', mode: 'lines', name: 'AI Strategy', line: { color: '#6366f1', width: 3 }, fill: 'tozeroy', fillcolor: 'rgba(99, 102, 241, 0.1)' },
                    { x: curve?.x, y: curve?.random, type: 'scatter', mode: 'lines', line: { dash: 'dash', color: '#94a3b8', width: 2 }, name: 'Random' },
                  ]}
                  layout={{ title: { text: 'Cumulative Incremental Revenue', font: { family: 'Inter', size: 16 } }, margin: { l: 50, r: 20, t: 50, b: 40 }, autosize: true, height: 320, plot_bgcolor: 'transparent', paper_bgcolor: 'transparent' }}
                  useResizeHandler style={{ width: '100%' }} config={{ displayModeBar: false }}
                />
              </div>

              <div className="chart-card">
                <Plot
                  data={[
                    { x: deciles?.map(d=>d.label), y: deciles?.map(d=>d.uplift), type: 'bar', marker: { color: deciles?.map(d => d.uplift >= 0 ? '#10b981' : '#f43f5e'), borderRadius: 4 } },
                  ]}
                  layout={{ title: { text: 'Observed Uplift by Decile', font: { family: 'Inter', size: 16 } }, margin: { l: 50, r: 20, t: 50, b: 40 }, autosize: true, height: 320, plot_bgcolor: 'transparent', paper_bgcolor: 'transparent' }}
                  useResizeHandler style={{ width: '100%' }} config={{ displayModeBar: false }}
                />
              </div>
            </div>

            <div className="chart-card full-width" style={{marginTop: '1.5rem'}}>
               <h3 style={{marginBottom: '1rem', color: '#1e293b'}}>Targeting Prescription</h3>
               <PrescriptionTable />
            </div>

          </div>
        )}
      </main>
    </div>
  );
}

export default App;