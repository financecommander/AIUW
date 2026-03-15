from typing import Dict, List, Any
from src.models.ensemble import EnsembleModel
from src.data.preprocess import preprocess_data

class InferencePipeline:
    def __init__(self, model: EnsembleModel):
        self.model = model

    def run(self, raw_data: List[Dict[str, Any]]) -> List[float]:
        processed_data = preprocess_data(raw_data)
        return self.model.predict_proba(processed_data)

    def classify(self, raw_data: List[Dict[str, Any]]) -> List[int]:
        processed_data = preprocess_data(raw_data)
        return self.model.predict(processed_data)
