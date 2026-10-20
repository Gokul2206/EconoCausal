import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, LGBMRegressor
from econml.dml import LinearDML

df = pd.read_csv('data/processed/customer_data.csv')

season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
df['season_encoded'] = df['favorite_season'].map(season_map)

X = df[['income_proxy', 'tenure_days', 'season_encoded']].values
T = df['discount_given'].values
Y = df['purchase_made'].values

cv = KFold(n_splits=5, shuffle=True, random_state=42)

def make_classifier():
    return LGBMClassifier(n_estimators=200, learning_rate=0.05,
                          num_leaves=31, random_state=42, verbose=-1)

def make_regressor():
    return LGBMRegressor(n_estimators=200, learning_rate=0.05,
                         num_leaves=31, random_state=42, verbose=-1)

# Baseline models: out-of-fold predictions and AUC
Y_hat = cross_val_predict(make_classifier(), X, Y, cv=cv, method='predict_proba')[:, 1]
T_hat = cross_val_predict(make_classifier(), X, T, cv=cv, method='predict_proba')[:, 1]

print(f"Outcome AUC: {roc_auc_score(Y, Y_hat):.3f}")
print(f"Propensity AUC: {roc_auc_score(T, T_hat):.3f}")
print(f"Propensity range: {T_hat.min():.3f} to {T_hat.max():.3f}")

# Average effect of discount on repeat purchase
est = LinearDML(
    model_y=make_regressor(),
    model_t=make_classifier(),
    discrete_treatment=True,
    cv=cv,
    random_state=42,
)
est.fit(Y, T, X=X)

ate = est.ate(X)
lb, ub = est.ate_interval(X, alpha=0.05)

print(f"ATE: {ate:.4f}")
print(f"95% CI: [{lb:.4f}, {ub:.4f}]")