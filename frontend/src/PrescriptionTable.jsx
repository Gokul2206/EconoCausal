import React from 'react';

export default function PrescriptionTable({ data }) {
  if (!data || data.length === 0) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', background: '#f8fafc', borderRadius: '8px', border: '1px dashed #cbd5e1' }}>
        <p style={{ color: '#64748b', fontSize: '0.95rem' }}>Awaiting optimization instructions.</p>
        <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '0.5rem' }}>Run the SciPy Optimizer in the Strategy Simulator tab to generate the final rollout matrix.</p>
      </div>
    );
  }

  return (
    <div style={{ maxHeight: '400px', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem', textAlign: 'left' }}>
        <thead style={{ position: 'sticky', top: 0, backgroundColor: '#f8fafc', zIndex: 1 }}>
          <tr>
            <th style={{ padding: '0.85rem 1rem', borderBottom: '1px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>Customer ID</th>
            <th style={{ padding: '0.85rem 1rem', borderBottom: '1px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>AI Prescription</th>
            <th style={{ padding: '0.85rem 1rem', borderBottom: '1px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>Expected Conversion Lift</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
              {/* FIX APPLIED HERE: Added String() to prevent the substring crash */}
              <td style={{ padding: '0.85rem 1rem', fontFamily: 'monospace', color: '#64748b' }}>{String(row.customer_id).substring(0, 8)}...</td>
              <td style={{ padding: '0.85rem 1rem' }}>
                <span style={{ 
                  padding: '0.25rem 0.75rem', 
                  borderRadius: '99px', 
                  fontWeight: 600,
                  fontSize: '0.8rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  backgroundColor: row.prescription === 20 ? '#fee2e2' : '#e0e7ff',
                  color: row.prescription === 20 ? '#b91c1c' : '#4338ca' 
                }}>
                  ${row.prescription} Discount
                </span>
              </td>
              <td style={{ padding: '0.85rem 1rem', fontWeight: 600, color: '#059669' }}>
                +{(row.prescription === 10 ? row.ite_10 * 100 : row.ite_20 * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}