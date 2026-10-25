import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.model_selection import train_test_split, KFold
from econml.dml import LinearDML

from outcome_model import fit_outcome_model, OUTCOME_FEATURES
from treatment_model import fit_treatment_model, TREATMENT_FEATURES

DATA_PATH = 'data/mock/mock_customers.csv'
RANDOM_STATE = 42


def design(df):
    # One matrix with every column either model needs
    return pd.get_dummies(df[OUTCOME_FEATURES], columns=['season'],
                          drop_first=True).astype(float)


class OutcomeWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, clf=None, cols=None):
        self.clf = clf
        self.cols = cols

    def fit(self, X, y):
        self.model_ = clone(self.clf).fit(X[:, self.cols], y)
        return self

    def predict(self, X):
        return self.model_.predict_proba(X[:, self.cols])[:, 1]


class TreatmentWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, clf=None, cols=None):
        self.clf = clf
        self.cols = cols

    def fit(self, X, y):
        self.model_ = clone(self.clf).fit(X[:, self.cols], y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X[:, self.cols])

    def predict_proba(self, X):
        return self.model_.predict_proba(X[:, self.cols])


def main():
    df = pd.read_csv(DATA_PATH)
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE,
        stratify=df['discount_amount'])

    outcome_fit, out_summary = fit_outcome_model(df_train, df_test)
    treat_fit, trt_summary = fit_treatment_model(df_train, df_test)

    X_all = design(df)
    cols = list(X_all.columns)
    out_cols = [i for i, c in enumerate(cols)
                if c in OUTCOME_FEATURES or c.startswith('season_')]
    trt_cols = [i for i, c in enumerate(cols)
                if c in TREATMENT_FEATURES or c.startswith('season_')]

    X_tr = X_all.loc[df_train.index].values
    X_te = X_all.loc[df_test.index].values
    Y_tr = df_train['purchase_made'].values
    T_tr = df_train['discount_amount'].values

    est = LinearDML(
        model_y=OutcomeWrapper(outcome_fit, out_cols),
        model_t=TreatmentWrapper(treat_fit, trt_cols),
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

    print("Outcome features:", OUTCOME_FEATURES)
    print("Treatment features:", TREATMENT_FEATURES)
    print("Outcome model:", out_summary['chosen'], "| Treatment model:", trt_summary['chosen'])
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()