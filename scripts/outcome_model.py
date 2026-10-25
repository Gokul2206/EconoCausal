import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

# Outcome model inputs: every pre-offer observed column.
# device_mobile is an outcome-only driver (data check: AUC vs outcome 0.528,
# vs discount 0.500). Adding it only reduces residual noise in the outcome
# nuisance; it does not change confounding.
OUTCOME_FEATURES = ['income', 'prior_orders', 'tenure_days', 'season', 'device_mobile', 'engagement_proxy']
RANDOM_STATE = 42
TARGET = 'purchase_made'

LGBM_GRID = {
    'num_leaves': [4, 8, 16],
    'min_child_samples': [100, 300],
    'learning_rate': [0.05],
    'n_estimators': [200, 400],
    'reg_lambda': [1.0, 10.0],
}

RF_GRID = {
    'n_estimators': [200, 400],
    'min_samples_leaf': [50, 200],
    'max_depth': [4, 8],
}


def build_X(df):
    return pd.get_dummies(df[OUTCOME_FEATURES], columns=['season'],
                          drop_first=True).astype(float)


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def fit_outcome_model(df_train, df_test):
    """Model 1: predicts purchase_made from all pre-offer observed columns.
    Compares logistic, tuned LightGBM, and tuned Random Forest on inner-CV
    log loss (training split only). The test split is used only to report."""
    X_tr = build_X(df_train)
    X_te = build_X(df_test)
    y_tr = df_train[TARGET].values
    y_te = df_test[TARGET].values

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    logit_cv = -cross_val_score(logistic(), X_tr, y_tr, cv=inner_cv,
                                scoring='neg_log_loss').mean()

    lgbm_search = GridSearchCV(
        LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       random_state=RANDOM_STATE, verbose=-1),
        LGBM_GRID, cv=inner_cv, scoring='neg_log_loss', n_jobs=-1).fit(X_tr, y_tr)
    lgbm_cv = -lgbm_search.best_score_

    rf_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        RF_GRID, cv=inner_cv, scoring='neg_log_loss', n_jobs=-1).fit(X_tr, y_tr)
    rf_cv = -rf_search.best_score_

    candidates = {
        'Logistic': (logit_cv, lambda: logistic().fit(X_tr, y_tr)),
        'LightGBM': (lgbm_cv, lambda: lgbm_search.best_estimator_),
        'RandomForest': (rf_cv, lambda: rf_search.best_estimator_),
    }
    chosen = min(candidates, key=lambda k: candidates[k][0])
    model = candidates[chosen][1]()

    summary = {
        'model': 'outcome (purchase_made)',
        'features': OUTCOME_FEATURES,
        'logistic_cv_logloss': round(logit_cv, 5),
        'lgbm_cv_logloss': round(lgbm_cv, 5),
        'rf_cv_logloss': round(rf_cv, 5),
        'chosen': chosen,
        'test_logloss': round(log_loss(y_te, model.predict_proba(X_te)), 5),
    }
    return model, summary


if __name__ == '__main__':
    df = pd.read_csv('data/mock/mock_customers.csv')
    df_train, df_test = train_test_split(df, test_size=0.2,
                                         random_state=RANDOM_STATE,
                                         stratify=df['discount_amount'])
    _, summary = fit_outcome_model(df_train, df_test)
    print(pd.Series(summary).to_string())