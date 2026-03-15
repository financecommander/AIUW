import pytest
from src.models.ensemble import EnsembleModel
from src.inference.pipeline import InferencePipeline

def test_pipeline_run():
    model = EnsembleModel()
    pipeline = InferencePipeline(model)
    sample_data = [{'age': 30, 'income': 50000, 'credit_score': 700, 'debt_ratio': 0.3}]
    with pytest.raises(ValueError):  # Model not fitted
        pipeline.run(sample_data)
