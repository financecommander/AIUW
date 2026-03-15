from typing import Dict, List, Any

def preprocess_data(raw_data: List[Dict[str, Any]]) -> List[List[float]]:
    feature_order = ['age', 'income', 'credit_score', 'debt_ratio']
    processed = []
    for record in raw_data:
        features = [float(record.get(f, 0.0)) for f in feature_order]
        processed.append(features)
    return processed
