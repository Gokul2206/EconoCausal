import React, { useState } from 'react';
import './App.css';

// Causal model from scripts/causal_analysis.py
const causalModel = {
  treatment: 'discount_given',
  outcome: 'purchase_made',
  confounders: ['income_proxy', 'tenure_days', 'favorite_season'],
};

function App() {
  const [file, setFile] = useState(null);
  const [budget, setBudget] = useState(50000);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    const reader = new FileReader();

    reader.onload = (event) => {
      try {
        const csv = event.target.result;
        const lines = csv.split('\n');
        const headers = lines[0].split(',');
        const dataRows = lines.slice(1).filter((line) => line.trim()).length;

        setData({
          filename: file.name,
          rows: dataRows,
          columns: headers.length,
        });
      } catch (err) {
        alert('Error reading file');
      }
      setLoading(false);
    };

    reader.readAsText(file);
  };

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
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setFile(e.target.files[0])}
                  required
                />
              </div>
              <button type="submit" disabled={!file || loading}>
                {loading ? 'Processing...' : 'Upload CSV'}
              </button>
            </form>
            <p className="hint">Select customer_data.csv from data/processed/</p>
          </div>

          <div className="card">
            <h2>💰 Budget Constraint</h2>
            <div className="form-group">
              <label>Marketing Budget ($):</label>
              <input
                type="number"
                value={budget}
                onChange={(e) => setBudget(Number(e.target.value))}
                min="1000"
                step="1000"
              />
            </div>
            <div className="budget-display">${budget.toLocaleString()}</div>
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
            {causalModel.confounders.map((c) => (
              <code key={c}>{c}</code>
            ))}
          </div>

          <p className="dag-note">
            Each confounder points to both <code>{causalModel.treatment}</code> and{' '}
            <code>{causalModel.outcome}</code>. Adjusting for them closes the backdoor paths.
          </p>
          <p className="dag-status">✓ DoWhy backdoor adjustment set (see causal_analysis.py output)</p>
        </div>

        {data && (
          <div className="status">
            <h3>✓ Configuration Ready</h3>
            <p><strong>File:</strong> {data.filename}</p>
            <p><strong>Customers:</strong> {data.rows.toLocaleString()}</p>
            <p><strong>Budget:</strong> ${budget.toLocaleString()}</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;