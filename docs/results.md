# RESULTS: Weeks 1 and 2 (mock dataset, frozen file)

Dataset: data/mock/mock_customers.csv, 100,000 customers.
Treatment: discount_amount ($0, $10, $20).
Outcome: purchase_made (repeat purchase within 90 days).

## 1. Causal graph and adjustment set (Week 1)

DAG in DoWhy (scripts/causal_analysis.py). Backdoor adjustment set: income, prior_orders, tenure_days, season, engagement_proxy.
Outcome model additionally uses device_mobile (outcome-only driver, not a confounder).
Assumptions and limits: docs/causal_assumptions.md.

## 2. Headline estimate: DML effect vs $0 (scripts/estimate_effects.py)

Treatment model features: income, prior_orders, tenure_days, season, engagement_proxy (5 features).
Outcome model features: the same five plus device_mobile (6 features).
Confidence intervals are for the DML estimate on the 20% test split only.
Saved output: data/processed/headline_estimates.csv.

| Comparison | Naive gap | DML effect | 95% CI (test split) | Realized treated mean | DML minus realized |
|---|---|---|---|---|---|
| $10 vs $0 | 0.0787 | 0.0487 | [0.0300, 0.0674] | 0.0370 | +0.012 |
| $20 vs $0 | 0.0794 | 0.0442 | [0.0262, 0.0622] | 0.0322 | +0.012 |

Realized treated mean is the planted average effect on treated test customers. It is used for validation only, never as a model input.

## 3. Refutations and sensitivity (scripts/refute_model.py)

Label: DoWhy audit estimate. It uses LightGBM nuisances set inside DoWhy, so its point estimates differ from the headline. Reviewers should compare the audit with the headline as a robustness check, not as a second result.

| Comparison | Estimate | Random common cause | Placebo | Data subset | Sensitivity 0.05 | Sensitivity 0.1 |
|---|---|---|---|---|---|---|
| $10 vs $0 | 0.0536 | 0.0537 | -0.0001 | 0.0537 | 0.0298 | -0.0097 |
| $20 vs $0 | 0.0467 | 0.0467 | -0.0003 | 0.0468 | 0.0224 | -0.0106 |

Reading: the three refutations pass. Under the sensitivity sweep, the estimate falls to about the realized level at strength 0.05 and reaches zero at about strength 0.1.

## 4. Ranking evidence (scripts/plot_qini.py, out-of-fold forest scores)

| Comparison | Customers | Qini | 90% interval | Beats random |
|---|---|---|---|---|
| $10 vs $0 | 79,960 | 455 | [329, 587] | Yes |
| $20 vs $0 | 80,040 | 848 | [739, 980] | Yes |

Qini values are counts and depend on sample size. Compare levels within this run only.

## 5. Bias reduction from the proxy

Without engagement_proxy, the headline DML minus realized gap was about +0.024 for both levels. With engagement_proxy, it is +0.012 (table 2). This comparison is from the earlier run on the same design without the proxy.

## 6. Summary claim

The headline estimate adjusts for the four observed confounders and the measured proxy engagement_proxy. Its residual bias against the planted mean is about 0.012 for both levels, down from about 0.024 without the proxy. The sensitivity sweep shows the estimate reaches zero at a hidden confounder of strength about 0.1. This is a measured reduction of hidden confounding with a stated bound. It is not a proof that the effect is isolated from hidden confounders.

## 7. Week 3

Optimizer: scripts/optimize_budget.py (SciPy milp). At a $50,000 budget, the optimized allocation predicts about 600 extra repeats against about 290 for either baseline. Revenue figures use an assumed average order value of $100, which is not in the data. Outputs: data/processed/allocation.csv and data/processed/optimization_metrics.json.