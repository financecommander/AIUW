import shap
import numpy as np
from typing import Any, List, Dict

class SHAPExplainer:
    def __init__(self, model: Any, background_data: np.ndarray):
        self.model = model
        self.explainer = shap.TreeExplainer(model, background_data)

    def explain_instance(self, X: np.ndarray) -> np.ndarray:
        return self.explainer.shap_values(X)

    def get_feature_attributions(self, X: np.ndarray, feature_names: List[str]) -> List[Dict[str, float]]:
        shap_values = self.explain_instance(X)
        attributions = []
        for i in range(len(X)):
            instance_attr = {feature_names[j]: shap_values[i][j] for j in range(len(feature_names))}
            attributions.append(instance_attr)
        return attributions
