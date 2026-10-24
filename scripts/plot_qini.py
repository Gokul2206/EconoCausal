import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import trapezoid

SCORES_PATH = 'data/processed/ite_scores_crossfit.csv'
OUT_PATH = 'data/processed/qini_curve.html'
N_BINS = 20
N_BOOT = 200
RNG = np.random.default_rng(42)


def uplift_curve(df, score_col, level):
    # Only $0 customers and customers who got this level
    sub = df[df['discount_amount'].isin([0, level])]
    treated = (sub['discount_amount'] == level).values.astype(int)
    y = sub['purchase_made'].values.astype(int)
    s = sub[score_col].values

    order = np.argsort(-s)
    treated, y = treated[order], y[order]
    n = len(sub)

    cum_t = np.cumsum(treated)
    cum_c = np.cumsum(1 - treated)
    cum_yt = np.cumsum(y * treated)
    cum_yc = np.cumsum(y * (1 - treated))

    ks = np.linspace(n // N_BINS, n, N_BINS, dtype=int)
    values = np.array([
        (cum_yt[k - 1] / cum_t[k - 1] - cum_yc[k - 1] / cum_c[k - 1]) * k
        for k in ks
    ])
    random_line = values[-1] * ks / n
    return ks / n * 100, values, random_line


def qini_coefficient(values, random_line):
    return trapezoid(values - random_line, dx=1.0 / N_BINS)


def bootstrap_band(df, score_col, level):
    curves, qinis = [], []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(df), len(df))
        _, v, r = uplift_curve(df.iloc[idx], score_col, level)
        curves.append(v)
        qinis.append(qini_coefficient(v, r))
    curves = np.array(curves)
    qinis = np.array(qinis)
    return (np.percentile(curves, 5, axis=0), np.percentile(curves, 95, axis=0),
            np.percentile(qinis, 5), np.percentile(qinis, 95))


def main():
    df = pd.read_csv(SCORES_PATH)

    fig = go.Figure()
    colors = {10: '#1f77b4', 20: '#d62728'}
    rows = []

    for level in [10, 20]:
        col = f'ite_{level}'
        x, v, rand = uplift_curve(df, col, level)
        lo, hi, q_lo, q_hi = bootstrap_band(df, col, level)
        q = qini_coefficient(v, rand)

        rows.append({
            'comparison': f'${level} vs $0',
            'customers': int((df['discount_amount'].isin([0, level])).sum()),
            'qini_coefficient': round(q, 2),
            'ci_90_low': round(q_lo, 2),
            'ci_90_high': round(q_hi, 2),
            'beats_random': q_lo > 0,
        })

        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[hi, lo[::-1]],
                                 fill='toself', line=dict(width=0),
                                 fillcolor=colors[level], opacity=0.15,
                                 name=f'${level} 90% band', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x, y=v, mode='lines+markers',
                                 line=dict(color=colors[level]),
                                 name=f'Ranked by ${level} effect (${level} vs $0)'))
        fig.add_trace(go.Scatter(x=x, y=rand, mode='lines',
                                 line=dict(dash='dash', color=colors[level]),
                                 name=f'Random (${level} vs $0)'))

    fig.update_layout(
        title='Targeting curves, out-of-fold on all customers, with 90% bootstrap bands',
        xaxis_title='Percent of customers targeted (ranked)',
        yaxis_title='Cumulative incremental repeat purchases (observed)',
        template='plotly_white',
    )

    os.makedirs('data/processed', exist_ok=True)
    fig.write_html(OUT_PATH)

    table = pd.DataFrame(rows)
    table.to_csv('data/processed/qini_coefficients.csv', index=False)
    print(f"Saved {OUT_PATH}")
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()