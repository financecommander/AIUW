from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Dict

app = FastAPI(title="AI Underwriting Engine")

# Placeholder for model loading
model = None  # TODO: Load ensemble model

class ScoreRequest(BaseModel):
    features: List[float]

class BatchScoreRequest(BaseModel):
    features: List[List[float]]

@app.on_event("startup")
async def startup_event():
    global model
    # TODO: Initialize model from checkpoint
    pass

@app.post("/score")
async def score(request: ScoreRequest) -> Dict[str, float]:
    try:
        input_data = np.array(request.features).reshape(1, -1)
        score = model.predict_proba(input_data)[0]
        return {"score": float(score)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

@app.post("/batch_score")
async def batch_score(request: BatchScoreRequest) -> Dict[str, List[float]]:
    try:
        input_data = np.array(request.features)
        scores = model.predict_proba(input_data).tolist()
        return {"scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring error: {str(e)}")

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}
