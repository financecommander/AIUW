from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Any
from src.models.ensemble import EnsembleModel

app = FastAPI(title="AI Underwriting Engine")

# Placeholder for model loading
model = None

class ScoreRequest(BaseModel):
    features: List[float]

class BatchScoreRequest(BaseModel):
    features: List[List[float]]

@app.on_event("startup")
async def startup_event():
    global model
    # TODO: Load model from config or checkpoint
    model_config = {'xgboost': {'n_estimators': 100}, 'lightgbm': {'n_estimators': 100}}
    model = EnsembleModel(model_config)

@app.post("/score")
async def score(request: ScoreRequest) -> Dict[str, Any]:
    try:
        features = np.array(request.features).reshape(1, -1)
        prob = model.predict_proba(features)[0]
        return {'score': float(prob), 'decision': int(prob >= 0.5)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_score")
async def batch_score(request: BatchScoreRequest) -> List[Dict[str, Any]]:
    try:
        features = np.array(request.features)
        probs = model.predict_proba(features)
        return [{'score': float(p), 'decision': int(p >= 0.5)} for p in probs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health() -> Dict[str, str]:
    return {'status': 'healthy'}
