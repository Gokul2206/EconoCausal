import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from econml.dml import CausalForestDML

from outcome_model import fit_outcome_model
from treatment_model import fit_treatment_model
from estimate_effects import ProbaRegressor, design

DATA_PATH = 'data/mock/mock_customers.csv'
OUT_PATH = 'data/processed/ite_scores_crossfit.csv'
RANDOM_STATE = 42
N_FOLDS = 5


def main():
    df = pd.read_csv(DATA_PATH)
    X = design(df)
    Y = df['purchase_made'].values
    T = df['discount_amount'].values

    # Nuisance models chosen on a fixed 80% split, as before
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE,
        stratify=df['discount_amount'])
    outcome_fit, _ = fit_outcome_model(df_train, df_test)
    treat_fit, _ = fit_treatment_model(df_train, df_test)

    ite_10 = np.zeros(len(df))
    ite_20 = np.zeros(len(df))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        forest = CausalForestDML(
            model_y=ProbaRegressor(clone(outcome_fit)),
            model_t=clone(treat_fit),
            discrete_treatment=True,
            cv=5,
            n_estimators=500,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
        )
        forest.fit(Y[tr], T[tr], X=X[tr])
        ite_10[te] = forest.effect(X[te], T0=0, T1=10)
        ite_20[te] = forest.effect(X[te], T0=0, T1=20)
        print(f"Fold {fold} done")

    scores = pd.DataFrame({
        'customer_id': df['customer_id'].values,
        'ite_10': ite_10,
        'ite_20': ite_20,
        'ite_diff': ite_20 - ite_10,
        'discount_amount': df['discount_amount'].values,
        'purchase_made': Y,
    })

    os.makedirs('data/processed', exist_ok=True)
    scores.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}: {len(scores):,} rows")


if __name__ == '__main__':
    main()