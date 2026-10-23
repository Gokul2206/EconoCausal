import numpy as np
import os
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import log_loss, roc_auc_score
from econml.dml import LinearDML, CausalForestDML

# Features chosen by backward selection, one list per model
OUTCOME_FEATURES = ['income_proxy', 'has_spend', 'n_items', 'avg_weight']
PROPENSITY_FEATURES = ['income_proxy', 'has_spend']
ALL_FEATURES = sorted(set(OUTCOME_FEATURES) | set(PROPENSITY_FEATURES))


def build_logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


# ---------- Model 1: outcome model ----------

class OutcomeModel(BaseEstimator, RegressorMixin):
    def __init__(self, all_features=None):
        self.all_features = all_features

    def _cols(self):
        return [self.all_features.index(c) for c in OUTCOME_FEATURES]

    def fit(self, X, y):
        self.model_ = build_logistic().fit(X[:, self._cols()], y)
        return self

    def predict(self, X):
        return self.model_.predict_proba(X[:, self._cols()])[:, 1]


# ---------- Model 2: propensity model ----------

class PropensityModel(BaseEstimator, ClassifierMixin):
    def __init__(self, all_features=None):
        self.all_features = all_features

    def _cols(self):
        return [self.all_features.index(c) for c in PROPENSITY_FEATURES]

    def fit(self, X, y):
        self.model_ = build_logistic().fit(X[:, self._cols()], y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X[:, self._cols()])

    def predict(self, X):
        return self.model_.predict(X[:, self._cols()])


def load_data():
    df = pd.read_csv('data/mock/mock_customers.csv')
    df['has_prior'] = (df['prior_orders'] > 0).astype(int)
    df['has_spend'] = (df['prior_spend'] > 0).astype(int)
    return df


def check_models(X, Y, T, groups, cv):
    y_hat = cross_val_predict(
        build_logistic(), X[:, [ALL_FEATURES.index(c) for c in OUTCOME_FEATURES]],
        Y, cv=cv, groups=groups, method='predict_proba')[:, 1]
    t_hat = cross_val_predict(
        build_logistic(), X[:, [ALL_FEATURES.index(c) for c in PROPENSITY_FEATURES]],
        T, cv=cv, groups=groups, method='predict_proba')[:, 1]

    print("Model 1: outcome (purchase_made)")
    print(f"  features: {OUTCOME_FEATURES}")
    print(f"  log loss: {log_loss(Y, y_hat):.5f}   AUC: {roc_auc_score(Y, y_hat):.3f}")
    print("Model 2: propensity (discount_given)")
    print(f"  features: {PROPENSITY_FEATURES}")
    print(f"  log loss: {log_loss(T, t_hat):.5f}   AUC: {roc_auc_score(T, t_hat):.3f}")
    print(f"  range:    {t_hat.min():.3f} to {t_hat.max():.3f}")


def estimate_ate(X, Y, T, groups, cv):
    est = LinearDML(
        model_y=OutcomeModel(ALL_FEATURES),
        model_t=PropensityModel(ALL_FEATURES),
        discrete_treatment=True,
        cv=cv,
        random_state=42,
    )
    est.fit(Y, T, X=X, groups=groups)
    return est


def estimate_ite(X, Y, T, groups, cv):
    forest = CausalForestDML(
        model_y=OutcomeModel(ALL_FEATURES),
        model_t=PropensityModel(ALL_FEATURES),
        discrete_treatment=True,
        cv=cv,
        n_estimators=500,
        min_samples_leaf=50,
        random_state=42,
    )
    forest.fit(Y, T, X=X, groups=groups)
    return forest


def main():
    df = load_data()
    X = df[ALL_FEATURES].values
    Y = df['purchase_made'].values
    T = df['discount_given'].values
    groups = df['customer_id'].values
    cv = GroupKFold(n_splits=5)

    check_models(X, Y, T, groups, cv)

    # Average effect
    est = estimate_ate(X, Y, T, groups, cv)
    Xt = X[T == 1]
    effects = est.effect(Xt)
    lb, ub = est.effect_interval(Xt, alpha=0.05)

    naive = Y[T == 1].mean() - Y[T == 0].mean()
    planted = df.loc[T == 1, 'true_ite'].mean()

    print("\nDML average effect")
    print(f"  naive gap:            {naive:.4f}")
    print(f"  DML effect (treated): {effects.mean():.4f}  "
          f"95% CI: [{lb.mean():.4f}, {ub.mean():.4f}]")
    print(f"  planted (treated):    {planted:.4f}")

    # Per-customer effect, compared on treated customers only
    forest = estimate_ite(X, Y, T, groups, cv)
    ite_hat = forest.effect(X).flatten()
    ite_hat_t = ite_hat[T == 1]
    planted_t = df.loc[T == 1, 'true_ite'].values

    print("\nPer-customer effect (treated customers)")
    print(f"  estimated mean: {ite_hat_t.mean():.4f}   std: {ite_hat_t.std():.4f}")
    print(f"  planted mean:   {planted_t.mean():.4f}   std: {planted_t.std():.4f}")
    print(f"  correlation:    {np.corrcoef(ite_hat_t, planted_t)[0, 1]:.3f}")

    # Save per-customer scores for the optimizer and dashboard
    os.makedirs('data/processed', exist_ok=True)
    scores = df[['customer_id', 'income_proxy', 'has_spend',
                 'n_items', 'avg_weight']].copy()
    scores['ite_hat'] = ite_hat
    scores.to_csv('data/processed/ite_scores.csv', index=False)
    print(f"\nSaved data/processed/ite_scores.csv: {len(scores):,} rows")


if __name__ == '__main__':
    main()