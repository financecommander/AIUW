import pytest
import numpy as np
from src.models.ensemble import EnsembleModel

def test_ensemble_predict():
    xgb_params = {'max_depth': 3, 'n_estimators': 10}
    lgb_params = {'max_depth': 3, 'n_estimators': 10}
    model = EnsembleModel(xgb_params, lgb_params)
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 10
    assert all(p in [0, 1] for p in preds)

def test_ensemble_predict_proba():
    xgb_params = {'max_depth': 3, 'n_estimators': 10}
    lgb_params = {'max_depth': 3, 'n_estimators': 10}
    model = EnsembleModel(xgb_params, lgb_params)
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == 10
    assert all(0 <= p <= 1 for p in probs)
