import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold
from econml.dml import CausalForestDML

from outcome_model import fit_outcome_model
from treatment_model import fit_treatment_model
from estimate_effects import ProbaRegressor, design

DATA_PATH = 'data/mock/mock_customers.csv'
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_PATH)
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE,
        stratify=df['discount_amount'])

    outcome_fit, _ = fit_outcome_model(df_train, df_test)
    treat_fit, _ = fit_treatment_model(df_train, df_test)

    X_tr, X_te = design(df_train), design(df_test)
    Y_tr = df_train['purchase_made'].values
    T_tr = df_train['discount_amount'].values
    Y_te = df_test['purchase_made'].values
    T_te = df_test['discount_amount'].values

    forest = CausalForestDML(
        model_y=ProbaRegressor(outcome_fit),
        model_t=clone(treat_fit),
        discrete_treatment=True,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_estimators=500,
        min_samples_leaf=50,
        random_state=RANDOM_STATE,
    )
    forest.fit(Y_tr, T_tr, X=X_tr)

    rows = []
    for lvl in [10, 20]:
        # Per-customer effect of this level vs $0, on the test split
        ite = forest.effect(X_te, T0=0, T1=lvl)
        mask = T_te == lvl
        realized = df_test.loc[mask, 'true_ite'].values
        est_on_treated = ite[mask]

        rows.append({
            'comparison': f'${lvl} vs $0',
            'est_mean': round(est_on_treated.mean(), 4),
            'est_std': round(est_on_treated.std(), 4),
            'realized_mean': round(realized.mean(), 4),
            'realized_std': round(realized.std(), 4),
            'correlation': round(np.corrcoef(est_on_treated, realized)[0, 1], 3),
            'n_treated_test': int(mask.sum()),
        })

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()