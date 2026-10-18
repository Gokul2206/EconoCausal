Here's the README without the week references:

# EconoCausal

Dynamic pricing via Double Machine Learning. This project estimates how a discount causally affects repeat purchases, so a marketing budget can target customers whose behavior actually changes with a discount.

## Current Status

- **Data processing:** `scripts/process_data.py` merges the Brazilian E-Commerce (Olist) dataset and builds one row per customer (95,146 customers, 13 columns).
- **Causal graph:** `scripts/causal_analysis.py` defines the DAG in DoWhy and prints the backdoor adjustment set (`income_proxy`, `tenure_days`, `favorite_season`).
- **Dashboard:** a React app with a DAG panel, a CSV upload, and a budget input.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place the Olist CSV files in `data/raw/`.

## Run

```bash
python scripts/process_data.py
python scripts/causal_analysis.py

cd frontend
npm install
npm start
```

## Assumptions and Limitations

- **Treatment:** the discount is inferred as a price below 85% of the category median. There are no recorded discount records.
- **Outcome:** repeat purchase means 2 or more orders.
- **Confounders:** `income_proxy` is a hand-made state lookup. `favorite_season` and `tenure_days` may be affected by the discount, so they are not strictly pre-treatment.
- **Identification:** DoWhy shows the effect is identifiable if the DAG is correct. The DAG is an assumption, not a tested result. Refutation tests are planned as a later step.

## Next Steps

Train the outcome and propensity models with Double ML, estimate individual treatment effects, and plot uplift curves with Plotly.