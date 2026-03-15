import pytest
import numpy as np
from src.models.ensemble import EnsembleModel

def test_ensemble_model():
    config = {
        'xgboost': {'n_estimators': 10, 'max_depth': 3},
        'lightgbm': {'n_estimators': 10, 'max_depth': 3},
        'champion_weight': 0.7
    }
    model = EnsembleModel(config)
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    model.fit(X, y)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    assert len(preds) == 10
    assert len(probs) == 10
    assert all(0 <= p <= 1 for p in probs)
