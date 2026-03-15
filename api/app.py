from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.models.ensemble import EnsembleModel
from src.inference.pipeline import InferencePipeline
from src.compliance.proxy_detection import detect_proxies

app = FastAPI(title="AI Underwriting Engine")
model = EnsembleModel()
pipeline = InferencePipeline(model)

class UnderwritingRequest(BaseModel):
    data: List[Dict[str, Any]]

@app.post("/underwrite")
async def underwrite(request: UnderwritingRequest):
    try:
        proxies = detect_proxies(request.data)
        if proxies:
            return {"status": "rejected", "reason": "proxy detected", "details": proxies}
        scores = pipeline.run(request.data)
        decisions = pipeline.classify(request.data)
        return {"status": "success", "scores": scores, "decisions": decisions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
