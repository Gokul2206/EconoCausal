import React, { useState } from 'react';
import './app.css';

// Causal model from scripts/causal_analysis.py (mock data)
const causalModel = {
  treatment: 'discount_given',
  outcome: 'purchase_made',
  confounders: ['income_proxy', 'prior_orders', 'days_since_prev',
                'prior_spend', 'order_season'],
};

const REQUIRED_COLUMNS = [
  'discount_given',
  'purchase_made',
  'income_proxy',
  'prior_orders',
  'days_since_prev',
  'prior_spend',
  'order_season',
];

function App() {
  const [file, setFile] = useState(null);
  const [budget, setBudget] = useState(50000);
  const [budgetError, setBudgetError] = useState('');
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleUpload = (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    const reader = new FileReader();

    reader.onload = (event) => {
      const csv = event.target.result;
      const lines = csv.split('\n').filter((line) => line.trim());
      const headers = lines[0].split(',').map((h) => h.trim());
      const missing = REQUIRED_COLUMNS.filter((c) => !headers.includes(c));

      if (missing.length > 0) {
        setData(null);
        setError(`Missing required columns: ${missing.join(', ')}`);
      } else {
        setError('');
        setData({
          filename: file.name,
          rows: lines.length - 1,
          columns: headers.length,
        });
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
            <p className="hint">Select mock_customers.csv from data/mock/</p>
            {error && <p style={{ color: '#c62828', marginTop: '10px' }}>{error}</p>}
          </div>

          <div className="card">
            <h2>💰 Budget Constraint</h2>
            <div className="form-group">
              <label>Marketing Budget ($):</label>
              <input
                type="number"
                value={budget}
                onChange={(e) => {
                  const value = Number(e.target.value);
                  setBudget(value);
                  setBudgetError(value <= 0 ? 'Budget must be greater than 0' : '');
                }}
                min="1000"
                step="1000"
              />
            </div>
            {budgetError && <p style={{ color: '#c62828', marginTop: '10px' }}>{budgetError}</p>}
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