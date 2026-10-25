import warnings
import pandas as pd
from dowhy import CausalModel
from lightgbm import LGBMRegressor, LGBMClassifier

warnings.filterwarnings('ignore')

DATA_PATH = 'data/mock/mock_customers.csv'
CONFOUNDERS = ['income', 'prior_orders', 'tenure_days', 'season', 'engagement_proxy']

GRAPH = """
digraph {
income -> treat;
income -> purchase_made;
prior_orders -> treat;
prior_orders -> purchase_made;
tenure_days -> treat;
tenure_days -> purchase_made;
season -> treat;
season -> purchase_made;
engagement_proxy -> treat;
engagement_proxy -> purchase_made;
treat -> purchase_made;
}
"""

REFUTERS = [
    ('random_common_cause', {}),
    ('placebo_treatment_refuter', {'placebo_type': 'permute'}),
    ('data_subset_refuter', {'subset_fraction': 0.8}),
]

SENSITIVITY_STRENGTHS = [0.05, 0.1, 0.2, 0.3]


def lgbm_reg():
    return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                         min_child_samples=50, random_state=42, verbose=-1)


def lgbm_clf():
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                          min_child_samples=50, random_state=42, verbose=-1)


def find_inner_estimator(estimate):
    for name in ['_econml_estimator', 'econml_estimator', 'estimator', '_estimator']:
        inner = getattr(estimate.estimator, name, None)
        if inner is not None and inner is not estimate.estimator:
            return inner
    return None


def refute_comparison(df, level):
    sub = df[df['discount_amount'].isin([0, level])].copy()
    sub['treat'] = (sub['discount_amount'] == level).astype(int)
    sub['season'] = sub['season'].astype(int)

    model = CausalModel(
        data=sub,
        treatment='treat',
        outcome='purchase_made',
        common_causes=CONFOUNDERS,
        graph=GRAPH,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)

    estimate = model.estimate_effect(
        estimand,
        method_name='backdoor.econml.dml.LinearDML',
        method_params={
            'init_params': {
                'model_y': lgbm_reg(),
                'model_t': lgbm_clf(),
                'discrete_treatment': True,
                'cv': 5,
                'random_state': 42,
            },
            'fit_params': {},
        },
    )

    inner = find_inner_estimator(estimate)
    inner_name = type(inner).__name__ if inner is not None else 'not found'
    print(f"\n=== ${level} vs $0 ===")
    print(f"Estimator wrapper: {type(estimate.estimator).__name__}")
    print(f"Wrapped EconML class: {inner_name}")
    if inner is None or 'DML' not in inner_name:
        raise RuntimeError(f"Expected a DML class inside the wrapper, got: {inner_name}")

    print(f"Customers: {len(sub):,}  (treated: {sub['treat'].sum():,})")
    print(f"Estimate: {estimate.value:.4f}")

    for method, kwargs in REFUTERS:
        r = model.refute_estimate(estimand, estimate, method_name=method, **kwargs)
        print(f"  {method}: original {r.estimated_effect:.4f}   new {r.new_effect:.4f}")

    for strength in SENSITIVITY_STRENGTHS:
        sens = model.refute_estimate(
            estimand, estimate,
            method_name='add_unobserved_common_cause',
            confounders_effect_on_treatment='binary_flip',
            confounders_effect_on_outcome='linear',
            effect_strength_on_treatment=strength,
            effect_strength_on_outcome=strength,
        )
        print(f"  sensitivity (strength {strength}): original {sens.estimated_effect:.4f}   new {sens.new_effect:.4f}")


def main():
    df = pd.read_csv(DATA_PATH)
    for level in [10, 20]:
        refute_comparison(df, level)


if __name__ == '__main__':
    main()