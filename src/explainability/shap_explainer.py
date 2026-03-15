import shap
import xgboost as xgb
import numpy as np
from typing import Dict, Any, List

class SHAPExplainer:
    def __init__(self, model: xgb.XGBClassifier, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain_instance(self, X: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP values for a single instance."""
        shap_values = self.explainer.shap_values(X)
        return {
            'shap_values': shap_values.tolist(),
            'base_value': float(self.explainer.expected_value),
            'features': self.feature_names
        }

    def explain_batch(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """Generate SHAP values for a batch of instances."""
        shap_values = self.explainer.shap_values(X)
        return [
            {
                'shap_values': sv.tolist(),
                'base_value': float(self.explainer.expected_value),
                'features': self.feature_names
            }
            for sv in shap_values
        ]
