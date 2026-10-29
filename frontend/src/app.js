import React, { useState, useMemo } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import './app.css';

// Import the new Prescription Table component
import PrescriptionTable from './PrescriptionTable';

const Plot = createPlotlyComponent(Plotly);

const causalModel = {
  treatment: 'discount_amount',
  outcome: 'purchase_made',
  confounders: ['income', 'prior_orders', 'tenure_days', 'season', 'engagement_proxy'],
};

const REQUIRED_COLUMNS = [
  'customer_id', 'discount_amount', 'purchase_made',
  'income', 'prior_orders', 'tenure_days', 'season',
];

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

// Targeting curve: $0 customers vs customers who got `level`, ranked by score.
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

// Observed repeat-rate gain within each decile of predicted effect (unadjusted)
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

  const [scoreRows, setScoreRows] = useState(null);
  const [scoresError, setScoresError] = useState('');
  const [level, setLevel] = useState('ite_20');
  const [minEffect, setMinEffect] = useState(0);

  const lvlAmount = level === 'ite_10' ? 10 : 20;

  const handleUpload = (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    const reader = new FileReader();
    reader.onload = (event) => {
      const { headers, rows } = parseCsv(event.target.result);
      const missing = REQUIRED_COLUMNS.filter((c) => !headers.includes(c));
      if (missing.length > 0) {
        setData(null);
        setError(`Missing required columns: ${missing.join(', ')}`);
      } else {
        setError('');
        setData({ filename: file.name, rows: rows.length, columns: headers.length });
      }
      setLoading(false);
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
        setScoresError(`Scores file missing: ${missing.join(', ')}`);
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
    <div className="App">
      <header className="header">
        <h1>🎯 EconoCausal</h1>
        <p>Dynamic Pricing via Double Machine Learning</p>
      </header>

      <main className="main">
        <div className="container">
          <div className="card">
            <h2>📊 Upload Campaign Data</h2>
            <form onSubmit={handleUpload}>
              <div className="form-group">
                <input type="file" accept=".csv"
                  onChange={(e) => setFile(e.target.files[0])} required />
              </div>
              <button type="submit" disabled={!file || loading}>
                {loading ? 'Processing...' : 'Upload CSV'}
              </button>
            </form>
            <p className="hint">Select mock_customers.csv from data/mock/</p>
            {error && <p style={{ color: '#c62828', marginTop: '10px' }}>{error}</p>}
          </div>

          <div className="card">
            <h2>💰 Budget Constraint</h2>
            <div className="form-group">
              <label>Marketing Budget ($):</label>
              <input type="number" value={budget}
                onChange={(e) => {
                  const value = Number(e.target.value);
                  setBudget(value);
                  setBudgetError(value <= 0 ? 'Budget must be greater than 0' : '');
                }}
                min="1000" step="1000" />
            </div>
            {budgetError && <p style={{ color: '#c62828', marginTop: '10px' }}>{budgetError}</p>}
            <div className="budget-display">${budget.toLocaleString('en-US')}</div>
          </div>
        </div>

        <div className="dag-panel">
          <h2>📐 Causal Graph (DAG)</h2>
          <div className="dag-node dag-treatment">
            <span className="dag-label">Treatment</span>
            <code>{causalModel.treatment}</code>
          </div>
          <div className="dag-node dag-outcome">
            <span className="dag-label">Outcome</span>
            <code>{causalModel.outcome}</code>
          </div>
          <div className="dag-node dag-confounders">
            <span className="dag-label">Confounders</span>
            {causalModel.confounders.map((c) => <code key={c}>{c}</code>)}
          </div>
          <p className="dag-note">Reasons are in docs/causal_assumptions.md.</p>
        </div>

        {data && (
          <div className="status">
            <h3>✓ Configuration Ready</h3>
            <p><strong>File:</strong> {data.filename}</p>
            <p><strong>Customers:</strong> {data.rows.toLocaleString('en-US')}</p>
            <p><strong>Budget:</strong> ${budget.toLocaleString('en-US')}</p>
          </div>
        )}

        <div className="card scores-card">
          <h2>📈 Individual Treatment Effects</h2>
          <p className="hint">Load data/processed/ite_scores_crossfit.csv</p>
          <input type="file" accept=".csv" onChange={handleScores} />
          {scoresError && <p style={{ color: '#c62828', marginTop: '10px' }}>{scoresError}</p>}

          {scoreRows && (
            <>
              <div className="filters">
                <label>
                  Discount level:&nbsp;
                  <select value={level} onChange={(e) => setLevel(e.target.value)}>
                    <option value="ite_10">$10 vs $0</option>
                    <option value="ite_20">$20 vs $0</option>
                  </select>
                </label>
                <label>
                  Minimum effect: {minEffect.toFixed(2)}&nbsp;
                  <input type="range" min="-0.1" max="0.2" step="0.01"
                    value={minEffect} onChange={(e) => setMinEffect(Number(e.target.value))} />
                </label>
              </div>

              {summary && (
                <p className="summary-line">
                  <strong>Customers matching:</strong> {summary.count.toLocaleString('en-US')} &nbsp;·&nbsp;
                  <strong>Mean effect:</strong> {summary.meanEffect.toFixed(4)} &nbsp;·&nbsp;
                  <strong>Repeat rate:</strong> {(summary.meanOutcome * 100).toFixed(1)}%
                </p>
              )}

              <div className="scores-row">
                <div>
                  <div className="histogram">
                    {histogram.map((b, i) => (
                      <div key={i} className="bar"
                        title={`${b.start.toFixed(3)}: ${b.count}`}
                        style={{ height: `${(b.count / maxCount) * 140}px` }} />
                    ))}
                  </div>
                  <p className="hint">Distribution of estimated effects.</p>
                </div>

                {curve && (
                  <div className="plot-wrap">
                    <Plot
                      data={[
                        { x: curve.x, y: curve.values, type: 'scatter', mode: 'lines+markers',
                          name: `Ranked by ${level === 'ite_10' ? '$10' : '$20'} effect` },
                        { x: curve.x, y: curve.random, type: 'scatter', mode: 'lines',
                          line: { dash: 'dash' }, name: 'Random rollout' },
                      ]}
                      layout={{
                        title: { text: 'Targeting curve (out-of-fold, unadjusted)', font: { size: 15 } },
                        xaxis: { title: 'Percent targeted (ranked)' },
                        yaxis: { title: 'Cumulative incremental repeats' },
                        template: 'plotly_white',
                        margin: { l: 60, r: 20, t: 50, b: 50 },
                        legend: { orientation: 'h', y: -0.25 },
                        autosize: true,
                        height: 340,
                      }}
                      useResizeHandler
                      style={{ width: '100%' }}
                      config={{ displayModeBar: false, responsive: true }}
                    />
                  </div>
                )}

                {deciles && (
                  <div className="plot-wrap">
                    <Plot
                      data={[
                        { x: deciles.map((d) => d.label), y: deciles.map((d) => d.uplift),
                          type: 'bar', name: 'Observed uplift',
                          marker: { color: deciles.map((d) => (d.uplift >= 0 ? '#1f77b4' : '#d62728')) } },
                      ]}
                      layout={{
                        title: { text: 'Uplift by decile, (unadjusted)', font: { size: 15 } },
                        xaxis: { title: 'Decile (D1 = highest predicted effect)' },
                        yaxis: { title: 'Repeat-rate gain (percentage points)' },
                        template: 'plotly_white',
                        margin: { l: 60, r: 20, t: 50, b: 50 },
                        height: 340,
                        autosize: true,
                      }}
                      useResizeHandler
                      style={{ width: '100%' }}
                      config={{ displayModeBar: false, responsive: true }}
                    />
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* The new Prescription Table rendered underneath everything else */}
        <div className="card prescription-card" style={{ marginTop: '2rem' }}>
          <PrescriptionTable />
        </div>

      </main>
    </div>
  );
}

export default App;