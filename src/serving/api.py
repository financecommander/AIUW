from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Dict
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
    # TODO: Load model from checkpoint or Triton server
    model = EnsembleModel({}, {})

@app.post("/score")
async def score(request: ScoreRequest) -> Dict[str, float]:
    try:
        features = np.array(request.features).reshape(1, -1)
        prob = model.predict_proba(features)[0]
        return {"score": float(prob)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

@app.post("/batch_score")
async def batch_score(request: BatchScoreRequest) -> Dict[str, List[float]]:
    try:
        features = np.array(request.features)
        probs = model.predict_proba(features).tolist()
        return {"scores": probs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring error: {str(e)}")

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}
