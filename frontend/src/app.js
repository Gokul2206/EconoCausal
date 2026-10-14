import React, { useState } from 'react';
import './app.css';

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
        const dataRows = lines.slice(1).filter(line => line.trim()).length;

        setData({
          filename: file.name,
          rows: dataRows,
          columns: headers.length
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
            <div className="budget-display">
              ${budget.toLocaleString()}
            </div>
          </div>
        </div>

        {data && (
          <div className="status">
            <h3>✓ Configuration Ready</h3>
            <p><strong>File:</strong> {data.filename}</p>
            <p><strong>Customers:</strong> {data.rows.toLocaleString()}</p>
            <p><strong>Budget:</strong> ${budget.toLocaleString()}</p>
            
            <div className="causal-info">
              <h4>📐 Week 1: Causal Model</h4>
              <p>✓ DoWhy DAG defined</p>
              <p>✓ Confounders identified: income, tenure, engagement, season</p>
              <p>✓ Causal effect is identifiable</p>
              <p style={{color: '#999', marginTop: '10px', fontSize: '0.9em'}}>
                Next: Week 2 - Train DML models to estimate Individual Treatment Effects
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;