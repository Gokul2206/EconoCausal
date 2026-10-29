import React, { useState } from 'react';
import Papa from 'papaparse';
import { Search, TrendingUp, Upload, ArrowUpDown } from 'lucide-react';

const PrescriptionTable = () => {
  const [allocations, setAllocations] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: 'predicted_effect', direction: 'desc' });

  const AOV = 100; // Match the Python AOV assumption

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setLoading(true);
    setError('');

    const reader = new FileReader();
    reader.onload = (event) => {
      Papa.parse(event.target.result, {
        header: true,
        dynamicTyping: true,
        complete: (results) => {
          const requiredCols = ['customer_id', 'discount', 'cost', 'predicted_effect'];
          const headers = results.meta.fields || [];
          const missing = requiredCols.filter(col => !headers.includes(col));
          
          if (missing.length > 0) {
            setError(`Missing columns: ${missing.join(', ')}`);
            setAllocations([]);
          } else {
            setAllocations(results.data.filter(row => row.customer_id));
          }
          setLoading(false);
        },
        error: (err) => {
          setError(`Error parsing CSV: ${err.message}`);
          setLoading(false);
        }
      });
    };
    reader.readAsText(file);
  };

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const totalSpend = allocations.reduce((sum, row) => sum + (Number(row.cost) || 0), 0);
  const totalRevenue = allocations.reduce((sum, row) => sum + ((Number(row.predicted_effect) || 0) * AOV), 0);
  const roi = totalSpend > 0 ? ((totalRevenue - totalSpend) / totalSpend) * 100 : 0;
  
  const sortedData = [...allocations].sort((a, b) => {
    if (a[sortConfig.key] < b[sortConfig.key]) return sortConfig.direction === 'asc' ? -1 : 1;
    if (a[sortConfig.key] > b[sortConfig.key]) return sortConfig.direction === 'asc' ? 1 : -1;
    return 0;
  });

  const filteredData = sortedData.filter(row => 
    row.customer_id?.toString().toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="prescription-container">
      <div className="prescription-header">
        <div>
          <h2>Prescription Engine</h2>
          <p className="hint">Mathematically optimal discount allocations maximizing predicted revenue.</p>
        </div>
        
        <div className="file-upload-box">
          <label className="file-upload-label">
            <Upload size={18} />
            <span>Upload allocation.csv</span>
            <input 
              type="file" 
              accept=".csv" 
              onChange={handleFileUpload} 
              style={{ display: 'none' }} 
            />
          </label>
          {error && <p style={{ color: '#d32f2f', fontSize: '0.8rem', marginTop: '5px' }}>{error}</p>}
        </div>
      </div>

      <div className="kpi-grid">
        <div className="kpi-card">
          <p className="kpi-title">Total Budget Allocated</p>
          <p className="kpi-value">${totalSpend.toLocaleString()}</p>
        </div>
        <div className="kpi-card">
          <p className="kpi-title">Customers Targeted</p>
          <p className="kpi-value">{allocations.length.toLocaleString()}</p>
        </div>
        <div className="kpi-card">
          <p className="kpi-title">Predicted Revenue (ROI)</p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <p className="kpi-value green">${totalRevenue.toLocaleString(undefined, {maximumFractionDigits: 0})}</p>
            <span style={{ color: '#2e7d32', fontWeight: '600', fontSize: '1rem' }}>(+{roi.toFixed(1)}%)</span>
            <TrendingUp size={24} color="#2e7d32" />
          </div>
        </div>
      </div>

      <div className="table-card">
        <div className="table-toolbar">
          <div className="search-wrapper">
            <Search size={18} className="search-icon" />
            <input
              type="text"
              placeholder="Search Customer ID..."
              className="search-input"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              disabled={allocations.length === 0}
            />
          </div>
        </div>

        <div className="table-responsive">
          <table className="styled-table">
            <thead>
              <tr>
                <th onClick={() => handleSort('customer_id')} style={{ cursor: 'pointer' }}>
                  Customer ID <ArrowUpDown size={14} style={{ display: 'inline', marginLeft: '4px' }}/>
                </th>
                <th onClick={() => handleSort('discount')} style={{ cursor: 'pointer' }}>
                  Allocated Discount <ArrowUpDown size={14} style={{ display: 'inline', marginLeft: '4px' }}/>
                </th>
                <th onClick={() => handleSort('cost')} style={{ cursor: 'pointer' }}>
                  Cost <ArrowUpDown size={14} style={{ display: 'inline', marginLeft: '4px' }}/>
                </th>
                <th onClick={() => handleSort('predicted_effect')} style={{ cursor: 'pointer' }}>
                  Predicted Revenue <ArrowUpDown size={14} style={{ display: 'inline', marginLeft: '4px' }}/>
                </th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan="4" className="text-center">Loading optimization results...</td></tr>
              ) : allocations.length === 0 ? (
                <tr><td colSpan="4" className="text-center">Upload an allocation.csv file to view prescription.</td></tr>
              ) : filteredData.length === 0 ? (
                <tr><td colSpan="4" className="text-center">No customers found matching that ID.</td></tr>
              ) : (
                filteredData.slice(0, 50).map((row, idx) => (
                  <tr key={idx}>
                    <td style={{ fontFamily: 'monospace' }}>{row.customer_id}</td>
                    <td>
                      <span className="badge-green">${row.discount} OFF</span>
                    </td>
                    <td>${row.cost}</td>
                    <td style={{ fontWeight: '500' }}>
                      ${(Number(row.predicted_effect) * AOV).toFixed(2)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          
          {!loading && filteredData.length > 50 && (
            <div className="table-footer">
              Showing first 50 rows of {filteredData.length.toLocaleString()} matching results.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PrescriptionTable;