import shap
import numpy as np
from typing import Dict, Any, List

class SHAPExplainer:
    def __init__(self, model: Any, background_data: np.ndarray):
        self.model = model
        self.explainer = shap.TreeExplainer(model, background_data)

    def get_feature_attributions(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for input data."""
        return self.explainer.shap_values(X)

    def get_top_features(self, shap_values: np.ndarray, feature_names: List[str], n_top: int = 5) -> List[Dict[str, Any]]:
        """Extract top contributing features for adverse action."""
        results = []
        for i in range(len(shap_values)):
            feature_contribs = sorted(
                [(name, val) for name, val in zip(feature_names, shap_values[i])],
                key=lambda x: abs(x[1]), reverse=True
            )[:n_top]
            results.append({
                'sample_idx': i,
                'top_features': [{'name': name, 'contribution': val} for name, val in feature_contribs]
            })
        return results
