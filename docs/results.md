RESULTS (Weeks 1 and 2, mock dataset, frozen file)

Dataset: data/mock/mock_customers.csv, 100,000 customers.
Treatment: discount_amount ($0, $10, $20). Outcome: purchase_made (repeat within 90 days).

HEADLINE: DML EFFECT VS $0 (scripts/estimate_effects.py)

Adjustment set: income, prior_orders, tenure_days, season, engagement_proxy (treatment model);
the same plus device_mobile (outcome model).

| Comparison | Naive gap | DML effect | 95% CI | Realized treated mean | DML minus realized |
|---|---|---|---|---|---|
| $10 vs $0 | 0.0787 | 0.0487 | [0.0300, 0.0674] | 0.0370 | +0.012 |
| $20 vs $0 | 0.0794 | 0.0442 | [0.0262, 0.0622] | 0.0322 | +0.012 |

Realized treated mean is the planted average effect on treated test customers (validation only).

AUDIT: DOWHY REFUTATIONS AND SENSITIVITY (scripts/refute_model.py)

Label: DoWhy audit estimate. Uses LightGBM nuisances set inside DoWhy, so the point
estimates differ slightly from the headline.

| Comparison | Estimate | Random common cause | Placebo | Data subset | Sensitivity 0.05 | Sensitivity 0.1 |
|---|---|---|---|---|---|---|
| $10 vs $0 | 0.0536 | 0.0537 | -0.0001 | 0.0537 | 0.0298 | -0.0097 |
| $20 vs $0 | 0.0467 | 0.0467 | -0.0003 | 0.0468 | 0.0224 | -0.0106 |

RANKING EVIDENCE (scripts/plot_qini.py, out-of-fold forest scores)

| Comparison | Customers | Qini | 90% interval | Beats random |
|---|---|---|---|---|
| $10 vs $0 | 79,960 | 455 | [329, 587] | Yes |
| $20 vs $0 | 80,040 | 848 | [739, 980] | Yes |

Qini values are counts and depend on sample size; compare levels within this run only.

SUMMARY CLAIM

The headline estimate is adjusted for the four observed confounders and the measured
proxy engagement_proxy. Its residual bias against the planted mean is about 0.012 for
both levels. The sensitivity sweep shows the conclusion is fragile to a hidden
confounder of strength about 0.1. The result is a measured reduction of hidden
confounding with a stated bound. It is not a proof that the effect is isolated from
hidden confounders.