import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

CONFOUNDERS = ['income', 'prior_orders', 'tenure_days', 'season']
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


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def fit_outcome_model(df_train, df_test):
    """Model 1: predicts purchase_made from the four confounders.
    Compares logistic, tuned LightGBM, and tuned Random Forest on inner-CV
    log loss (training split only). The test split is used only to report."""
    X_tr = pd.get_dummies(df_train[CONFOUNDERS], columns=['season'], drop_first=True).astype(float)
    X_te = pd.get_dummies(df_test[CONFOUNDERS], columns=['season'], drop_first=True).astype(float)
    y_tr = df_train[TARGET].values
    y_te = df_test[TARGET].values

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Candidate 1: logistic regression
    logit_cv = -cross_val_score(logistic(), X_tr, y_tr, cv=inner_cv,
                                scoring='neg_log_loss').mean()

    # Candidate 2: tuned LightGBM
    lgbm_search = GridSearchCV(
        LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       random_state=RANDOM_STATE, verbose=-1),
        LGBM_GRID, cv=inner_cv, scoring='neg_log_loss', n_jobs=-1).fit(X_tr, y_tr)
    lgbm_cv = -lgbm_search.best_score_

    # Candidate 3: tuned Random Forest
    rf_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        RF_GRID, cv=inner_cv, scoring='neg_log_loss', n_jobs=-1).fit(X_tr, y_tr)
    rf_cv = -rf_search.best_score_

    # Choose the lowest inner-CV log loss
    candidates = {
        'Logistic': (logit_cv, lambda: logistic().fit(X_tr, y_tr)),
        'LightGBM': (lgbm_cv, lambda: lgbm_search.best_estimator_),
        'RandomForest': (rf_cv, lambda: rf_search.best_estimator_),
    }
    chosen = min(candidates, key=lambda k: candidates[k][0])
    model = candidates[chosen][1]()

    summary = {
        'model': 'outcome (purchase_made)',
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