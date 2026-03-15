from typing import List
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class Ensemble(BaseEstimator):
    def __init__(self, xgboost_params, lightgbm_params):
        self.xgboost = XGBClassifier(**xgboost_params)
        self.lightgbm = LGBMClassifier(**lightgbm_params)

    def fit(self, X, y):
        self.xgboost.fit(X, y)
        self.lightgbm.fit(X, y)
        return self

    def predict(self, X):
        return (self.xgboost.predict(X) + self.lightgbm.predict(X)) / 2
