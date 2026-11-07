import React, { useState, useMemo } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import './app.css';
import PrescriptionTable from './PrescriptionTable';

const Plot = createPlotlyComponent(Plotly);
const N_BINS = 20;

// --- Helper Math Functions ---
function targetingCurve(rows, scoreKey, level) {
  const sub = rows.filter((r) => r.discount_amount === 0 || r.discount_amount === level).sort((a, b) => b[scoreKey] - a[scoreKey]);
  const n = sub.length;
  const ks = Array.from({ length: N_BINS }, (_, i) => Math.floor(n / N_BINS) * (i + 1) + (i === N_BINS - 1 ? n - Math.floor(n / N_BINS) * N_BINS : 0));
  let cumT = 0, cumC = 0, cumYT = 0, cumYC = 0, idx = 0;
  const values = [];
  ks.forEach((k) => {
    while (idx < k) {
      const r = sub[idx];
      if (r.discount_amount === level) { cumT += 1; cumYT += r.purchase_made; } else { cumC += 1; cumYC += r.purchase_made; }
      idx += 1;
    }
    values.push(((cumT ? cumYT / cumT : 0) - (cumC ? cumYC / cumC : 0)) * k);
  });
  const total = values[values.length - 1];
  return { x: ks.map((k) => (k / n) * 100), values, random: ks.map((k) => (total * (k / n * 100)) / 100) };
}

function upliftByDecile(rows, scoreKey, level) {
  const sub = rows.filter((r) => r.discount_amount === 0 || r.discount_amount === level).sort((a, b) => b[scoreKey] - a[scoreKey]);
  const n = sub.length;
  const deciles = [];
  for (let d = 0; d < 10; d++) {
    const start = Math.floor((n * d) / 10), end = Math.floor((n * (d + 1)) / 10);
    let tN = 0, tY = 0, cN = 0, cY = 0;
    for (let i = start; i < end; i++) {
      const r = sub[i];
      if (r.discount_amount === level) { tN += 1; tY += r.purchase_made; } else { cN += 1; cY += r.purchase_made; }
    }
    deciles.push({ label: `Top ${ (d + 1) * 10 }%`, uplift: ((tN ? tY / tN : 0) - (cN ? cY / cN : 0)) * 100 });
  }
  return deciles;
}

function App() {
  const [activeTab, setActiveTab] = useState('summary');
  const [file, setFile] = useState(null);
  
  // Business Controls
  const [budget, setBudget] = useState(1000);
  const [level, setLevel] = useState('ite_20');
  const [minEffect, setMinEffect] = useState(0);

  // System States
  const [loading, setLoading] = useState(false);
  const [loadingScores, setLoadingScores] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  
  // Data States
  const [driftReport, setDriftReport] = useState(null);
  const [driftError, setDriftError] = useState('');
  const [scoreRows, setScoreRows] = useState(null);
  const [scoresError, setScoresError] = useState('');
  
  // NEW: Week 3 Optimizer State
  const [optData, setOptData] = useState(null);

  const lvlAmount = level === 'ite_10' ? 10 : 20;

  // --- API Integrations ---
  const handleDriftCheck = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true); setDriftError(''); setDriftReport(null);
    const formData = new FormData(); formData.append("file", file);
    try {
      const response = await fetch("http://localhost:8000/api/detect_drift", { method: "POST", body: formData });
      if (!response.ok) throw new Error("Could not connect to AI Engine.");
      setDriftReport(await response.json());
    } catch (err) { setDriftError(err.message); } finally { setLoading(false); }
  };

  const loadAIStrategy = async () => {
    setLoadingScores(true); setScoresError('');
    try {
      const response = await fetch("http://localhost:8000/api/scores");
      if (!response.ok) throw new Error("Failed to load strategy from API.");
      const data = await response.json();
      setScoreRows(data.map((r) => ({
        ite_10: parseFloat(r.ite_10), ite_20: parseFloat(r.ite_20), discount_amount: Number(r.discount_amount), purchase_made: Number(r.purchase_made),
      })));
    } catch (err) { setScoresError(err.message); } finally { setLoadingScores(false); }
  };

  // NEW: Call the SciPy LP Optimizer
  const runOptimization = async () => {
    setOptimizing(true);
    try {
      const response = await fetch(`http://localhost:8000/api/optimize?budget=${budget}`);
      if (!response.ok) throw new Error("Optimizer failed.");
      setOptData(await response.json());
    } catch (err) { console.error(err); } finally { setOptimizing(false); }
  };

  // --- Business Logic & Calculations ---
  const filtered = useMemo(() => scoreRows ? scoreRows.filter((r) => r[level] >= minEffect) : [], [scoreRows, level, minEffect]);
  const summary = useMemo(() => {
    if (filtered.length === 0) return null;
    return { count: filtered.length, meanEffect: filtered.reduce((a, b) => a + b[level], 0) / filtered.length };
  }, [filtered, level]);

  const roiMetrics = useMemo(() => {
    if (!scoreRows || filtered.length === 0) return { saved: 0, text: "Adjust targeting to find profit." };
    const aiSales = filtered.reduce((sum, r) => sum + r[level], 0);
    const totalMeanEffect = scoreRows.reduce((sum, r) => sum + r[level], 0) / scoreRows.length;
    const randomSales = filtered.length * totalMeanEffect;
    const moneySaved = (aiSales - randomSales) * 50; 
    if (moneySaved <= 0) return { saved: 0, text: "Strategy currently yields negative return." };
    return { saved: Math.round(moneySaved), text: `AI Targeting is projected to generate $${Math.round(moneySaved).toLocaleString()} in incremental profit compared to random promotions.` };
  }, [scoreRows, filtered, level]);

  const histogram = useMemo(() => {
    if (!scoreRows) return [];
    const values = scoreRows.map((r) => r[level]);
    const min = Math.min(...values), max = Math.max(...values);
    const width = (max - min) / N_BINS || 1;
    const counts = Array(N_BINS).fill(0);
    values.forEach((v) => { counts[Math.min(N_BINS - 1, Math.floor((v - min) / width))] += 1; });
    return counts.map((c, i) => ({ start: min + i * width, count: c }));
  }, [scoreRows, level]);
  const maxCount = Math.max(1, ...histogram.map((b) => b.count));

  const curve = useMemo(() => scoreRows ? targetingCurve(scoreRows, level, lvlAmount) : null, [scoreRows, level, lvlAmount]);
  const deciles = useMemo(() => scoreRows ? upliftByDecile(scoreRows, level, lvlAmount) : null, [scoreRows, level, lvlAmount]);

  return (
    <div className="executive-layout">
      
      <aside className="side-nav">
        <div className="brand">
          <div className="logo-box">E</div>
          <div className="brand-text"><h2>EconoCausal</h2><span>Executive Console</span></div>
        </div>
        
        <nav className="nav-links">
          <button className={`nav-item ${activeTab === 'summary' ? 'active' : ''}`} onClick={() => setActiveTab('summary')}><span className="icon">📊</span> Executive Summary</button>
          <button className={`nav-item ${activeTab === 'strategy' ? 'active' : ''}`} onClick={() => setActiveTab('strategy')}><span className="icon">🎯</span> Strategy Simulator</button>
          <button className={`nav-item ${activeTab === 'data' ? 'active' : ''}`} onClick={() => setActiveTab('data')}><span className="icon">⚙️</span> Data Operations</button>
        </nav>
      </aside>

      <main className="content-area">
        
        {/* --- VIEW 1: EXECUTIVE SUMMARY --- */}
        {activeTab === 'summary' && (
          <div className="tab-pane animate-fade">
            <header className="page-header">
              <h1>Executive Summary</h1>
              <p>High-level overview of campaign performance and AI-driven recommendations.</p>
            </header>

            {!scoreRows ? (
              <div className="empty-state"><p>No strategy loaded. Please initialize the AI Engine in Data Operations.</p><button className="btn-primary mt-3" onClick={() => setActiveTab('data')}>Go to Data Operations</button></div>
            ) : (
              <>
                <div className="kpi-grid">
                  <div className="kpi-card">
                    <h4>Optimal Audience Size</h4>
                    <div className="kpi-value">{optData ? optData.summary.customers_targeted.toLocaleString() : summary?.count.toLocaleString()}</div>
                    <div className="kpi-label">Customers receiving promotion</div>
                  </div>
                  <div className="kpi-card">
                    <h4>Budget Utilized</h4>
                    <div className="kpi-value text-success">${optData ? optData.summary.budget_utilized.toLocaleString() : "---"}</div>
                    <div className="kpi-label">Capital allocated efficiently</div>
                  </div>
                  <div className="kpi-card">
                    <h4>AI Revenue Lift</h4>
                    <div className="kpi-value">+{optData ? (optData.summary.projected_revenue_lift * 100).toFixed(1) : (summary?.meanEffect * 100).toFixed(1)}%</div>
                    <div className="kpi-label">Expected boost in purchase rate</div>
                  </div>
                </div>

                <div className="insight-box">
                  <h3>AI Recommendation</h3>
                  <p>{optData ? `The SciPy Optimization Solver successfully allocated $${optData.summary.budget_utilized} to target ${optData.summary.customers_targeted} highly-persuadable customers.` : roiMetrics.text}</p>
                </div>

                <div className="content-card mt-4">
                  <h3>Actionable Customer Segments (Final Prescription)</h3>
                  <p className="subtext">The list of customers chosen by the SciPy Optimization algorithm.</p>
                  <PrescriptionTable data={optData?.prescriptions} />
                </div>
              </>
            )}
          </div>
        )}

        {/* --- VIEW 2: STRATEGY SIMULATOR --- */}
        {activeTab === 'strategy' && (
          <div className="tab-pane animate-fade">
            <header className="page-header">
              <h1>Strategy Simulator</h1>
              <p>Adjust business boundaries to optimize profit margins and audience targeting.</p>
            </header>

            {!scoreRows ? (
              <div className="empty-state"><p>Strategy data required. Please pull data from the API first.</p><button className="btn-primary mt-3" onClick={() => setActiveTab('data')}>Go to Data Operations</button></div>
            ) : (
              <div className="strategy-layout">
                {/* Left Panel: Controls */}
                <div className="control-panel">
                  
                  {/* NEW OPTIMIZER CONTROLS */}
                  <div className="form-group mb-4" style={{ padding: '1rem', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '8px' }}>
                    <label style={{color: '#166534'}}>Strict Budget Constraint ($)</label>
                    <input type="number" value={budget} onChange={(e) => setBudget(Number(e.target.value))} min="10" step="100" style={{ width: '100%', padding: '0.5rem', marginBottom: '10px' }} />
                    <button className="btn-primary w-100" onClick={runOptimization} disabled={optimizing} style={{ background: '#166534' }}>
                      {optimizing ? 'Calculating LP Model...' : 'Run SciPy Optimizer'}
                    </button>
                    {optData && <p style={{fontSize: '0.8rem', color: '#15803d', marginTop: '10px'}}>✅ Found perfect allocation!</p>}
                  </div>

                  <hr style={{border: 0, borderTop: '1px solid #e2e8f0', margin: '1.5rem 0'}} />

                  <div className="form-group">
                    <label>Exploratory Tier</label>
                    <select value={level} onChange={(e) => setLevel(e.target.value)}>
                      <option value="ite_10">Standard Offer ($10 Off)</option>
                      <option value="ite_20">Aggressive Offer ($20 Off)</option>
                    </select>
                  </div>
                  
                  <div className="form-group mt-4">
                    <label>Targeting Threshold (Minimum Expected Lift)</label>
                    <div className="threshold-readout">Targeting customers with &gt; {(minEffect * 100).toFixed(1)}% lift</div>
                    <input type="range" className="range-slider" min="-0.1" max="0.2" step="0.01" value={minEffect} onChange={(e) => setMinEffect(Number(e.target.value))} />
                  </div>

                  <div className="audience-viz mt-4">
                    <label>Customer Responsiveness Distribution</label>
                    <div className="histogram">
                      {histogram.map((b, i) => (
                        <div key={i} className={`bar ${b.start >= minEffect ? 'active-bar' : 'inactive-bar'}`}
                          title={`Lift: ${(b.start*100).toFixed(1)}% | Customers: ${b.count}`}
                          style={{ height: `${(b.count / maxCount) * 100}%` }} />
                      ))}
                    </div>
                  </div>
                </div>

                {/* Right Panel: Charts */}
                <div className="charts-panel">
                  <div className="content-card mb-4">
                    <Plot
                      data={[{ x: deciles?.map(d=>d.label), y: deciles?.map(d=>d.uplift), type: 'bar', marker: { color: deciles?.map(d => d.uplift >= 0 ? '#0284c7' : '#e11d48'), borderRadius: 2 } }]}
                      layout={{ title: { text: 'Revenue Lift by Customer Segment', font: { family: 'Inter', size: 15 } }, margin: { l: 40, r: 20, t: 40, b: 30 }, height: 260 }}
                      useResizeHandler style={{ width: '100%' }} config={{ displayModeBar: false }}
                    />
                  </div>
                  <div className="content-card">
                    <Plot
                      data={[
                        { x: curve?.x, y: curve?.values, type: 'scatter', mode: 'lines', name: 'AI Strategy', line: { color: '#0284c7', width: 3 }, fill: 'tozeroy', fillcolor: 'rgba(2, 132, 199, 0.1)' },
                        { x: curve?.x, y: curve?.random, type: 'scatter', mode: 'lines', line: { dash: 'dash', color: '#94a3b8', width: 2 }, name: 'Random Targeting' },
                      ]}
                      layout={{ title: { text: 'Cumulative Campaign Performance', font: { family: 'Inter', size: 15 } }, margin: { l: 40, r: 20, t: 40, b: 30 }, height: 260 }}
                      useResizeHandler style={{ width: '100%' }} config={{ displayModeBar: false }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* --- VIEW 3: DATA OPERATIONS --- */}
        {activeTab === 'data' && (
          <div className="tab-pane animate-fade">
            <header className="page-header">
              <h1>Data Operations</h1>
              <p>System health, API connections, and data validation.</p>
            </header>

            <div className="operations-grid">
              <div className="content-card">
                <h3>1. Pull Causal Strategy</h3>
                <p className="subtext">Fetch the latest generated intelligence from the backend ML engine.</p>
                <button className="btn-primary w-100" onClick={loadAIStrategy} disabled={loadingScores}>
                  {loadingScores ? 'Connecting to API...' : (scoreRows ? 'Refresh Strategy Data' : 'Pull Data from API')}
                </button>
                {scoreRows && !scoresError && <div className="alert-success mt-3">✅ Strategy loaded successfully. View Executive Summary.</div>}
              </div>

              <div className="content-card">
                <h3>2. Quality Assurance (Drift Check)</h3>
                <p className="subtext">Upload recent campaign data to verify customer behavior hasn't changed.</p>
                <form onSubmit={handleDriftCheck}>
                  <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} className="file-input" required />
                  <button type="submit" className="btn-secondary w-100 mt-2" disabled={!file || loading}>{loading ? 'Running AI Diagnostics...' : 'Run Diagnostics'}</button>
                </form>
                {driftReport && (
                  <div className={`alert-box mt-3 ${driftReport.overall_drift_detected ? 'alert-warning' : 'alert-success'}`}>
                    <strong>{driftReport.overall_drift_detected ? '⚠️ Attention Required' : '✅ Data is Stable'}</strong>
                    <p style={{margin: '5px 0 0 0', fontSize: '0.9rem'}}>{driftReport.warning_message}</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;