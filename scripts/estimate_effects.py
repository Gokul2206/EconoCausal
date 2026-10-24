import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import train_test_split, KFold
from econml.dml import LinearDML

from outcome_model import fit_outcome_model
from treatment_model import fit_treatment_model

DATA_PATH = 'data/mock/mock_customers.csv'
CONFOUNDERS = ['income', 'prior_orders', 'tenure_days', 'season']
RANDOM_STATE = 42


class ProbaRegressor(BaseEstimator, RegressorMixin):
    """Wraps a fitted classifier so LinearDML can use it as the outcome model.
    LinearDML expects a regressor interface; this returns P(purchase_made = 1)."""

    def __init__(self, classifier=None):
        self.classifier = classifier

    def fit(self, X, y):
        self.model_ = clone(self.classifier).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict_proba(X)[:, 1]


def design(df):
    # Same design matrix as the model scripts: season one-hot encoded
    return pd.get_dummies(df[CONFOUNDERS], columns=['season'],
                          drop_first=True).astype(float).values


def main():
    df = pd.read_csv(DATA_PATH)
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE,
        stratify=df['discount_amount'])

    # Chosen nuisance models (selected on the training split only)
    outcome_fit, out_summary = fit_outcome_model(df_train, df_test)
    treat_fit, trt_summary = fit_treatment_model(df_train, df_test)

    X_tr, X_te = design(df_train), design(df_test)
    Y_tr = df_train['purchase_made'].values
    T_tr = df_train['discount_amount'].values

    # Confounders are both the nuisance inputs and the effect modifiers,
    # so the estimate is the average effect among the customers who got each level.
    est = LinearDML(
        model_y=ProbaRegressor(outcome_fit),
        model_t=clone(treat_fit),
        discrete_treatment=True,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
    )
    est.fit(Y_tr, T_tr, X=X_tr)

    rows = []
    for lvl in [10, 20]:
        mask = df_test['discount_amount'].values == lvl
        X_lvl = X_te[mask]

        dml = est.effect(X_lvl, T0=0, T1=lvl).mean()
        lb, ub = est.effect_interval(X_lvl, T0=0, T1=lvl, alpha=0.05)

        Y_te = df_test['purchase_made'].values
        T_te = df_test['discount_amount'].values
        naive = Y_te[T_te == lvl].mean() - Y_te[T_te == 0].mean()
        realized = df_test.loc[mask, 'true_ite'].mean()

        rows.append({
            'comparison': f'${lvl} vs $0',
            'naive_gap': round(naive, 4),
            'dml_effect': round(dml, 4),
            'dml_ci_low': round(lb.mean(), 4),
            'dml_ci_high': round(ub.mean(), 4),
            'realized_treated_mean': round(realized, 4),
            'n_treated_test': int(mask.sum()),
        })

    results = pd.DataFrame(rows)
    print("Outcome model:", out_summary['chosen'], "| Treatment model:", trt_summary['chosen'])
    print(results.to_string(index=False))

    # Save the fitted nuisance choices for the record
    pd.DataFrame([out_summary, trt_summary]).to_csv(
        'data/processed/nuisance_summary.csv', index=False)


if __name__ == '__main__':
    main()