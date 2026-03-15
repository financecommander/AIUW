from fastapi import FastAPI
from pydantic import BaseModel

class ScoreRequest(BaseModel):
    loan_data: List[float]

class ScoreResponse(BaseModel):
    score: float

app = FastAPI()

@app.post("/score")
async def score(score_request: ScoreRequest):
    # TODO: Implement scoring endpoint
    pass

@app.post("/batch_score")
async def batch_score(score_requests: List[ScoreRequest]):
    # TODO: Implement batch scoring endpoint
    pass

@app.get("/health")
async def health_check):
    # TODO: Implement health check endpoint
    pass
