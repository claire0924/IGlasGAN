# -*- coding: utf-8 -*-
"""
Trains XGBoost regressors on training datasets to predict glass properties and saves the fitted model.

"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import joblib

df = pd.read_csv('property_CTE.csv')
df = df.set_index('GlasNo')

y = df.iloc[:, -1]
X = df.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def float_range(start, end, step):
    while start < end:
        yield round(start, 2)
        start += step


space = {
    'n_estimators': hp.choice('n_estimators', [i for i in range(100, 2000, 1)]),
    'max_depth': hp.choice('max_depth', [i for i in range(1, 20, 1)]),
    'learning_rate': hp.choice('learning_rate', [i for i in list(float_range(0.01, 0.5, 0.01))]),
    'gamma': hp.choice('gamma', [i for i in list(float_range(0, 0.5, 0.01))]),
    'min_child_weight': hp.choice('min_child_weight', [i for i in range(1, 10, 1)]),
    'subsample': hp.choice('subsample', [i for i in list(float_range(0.1, 1, 0.01))]),
    'colsample_bytree': hp.choice('colsample_bytree', [i for i in list(float_range(0.1, 1, 0.01))]),
}
space_ = {
    'max_depth': [i for i in range(1, 20, 1)],
    'learning_rate': [i for i in list(float_range(0.01, 0.5, 0.01))],
    'n_estimators': [i for i in range(100, 2000, 1)],
    'gamma': [i for i in list(float_range(0, 0.5, 0.01))],
    'min_child_weight': [i for i in range(1, 10, 1)],
    'subsample': [i for i in list(float_range(0.1, 1, 0.01))],
    'colsample_bytree': [i for i in list(float_range(0.1, 1, 0.01))]
}


def objective(params):
    model = xgb.XGBRegressor(**params, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error').mean()
    return -score


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=500,
            trials=trials)

best_params = {key: val[best[key]] for key, val in space_.items()}
print("Best parameters!!!: ", best_params)

best_xgb = xgb.XGBRegressor(**best_params, random_state=42)
best_xgb.fit(X_train, y_train)

y_train_pred = best_xgb.predict(X_train)
y_test_pred = best_xgb.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f'train: r2={train_r2}')
print(f'train: mae={train_mae}')
print(f'train: mape={train_mape}')
print(f'train: rmse={train_rmse}')

print(f'test: r2={test_r2}')
print(f'test: mae={test_mae}')
print(f'test: mape={test_mape}')
print(f'test: rmse={test_rmse}')

joblib.dump(best_xgb, 'xgb_CTE.pkl')
