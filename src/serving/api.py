from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Dict
from src.models.ensemble import EnsembleModel

app = FastAPI(title='AI Underwriting Engine')

# Placeholder for model loading
model = None  # TODO: Load model on startup

class ScoreRequest(BaseModel):
    features: List[float]

class BatchScoreRequest(BaseModel):
    features: List[List[float]]

@app.on_event('startup')
async def startup_event():
    global model
    model = EnsembleModel({})  # TODO: Load config and weights

@app.post('/score')
async def score(request: ScoreRequest):
    try:
        X = np.array([request.features])
        proba = model.predict_proba(X)[0]
        return {'probability': proba.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/batch_score')
async def batch_score(request: BatchScoreRequest):
    try:
        X = np.array(request.features)
        proba = model.predict_proba(X)
        return {'probabilities': proba.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
async def health():
    return {'status': 'healthy'}
