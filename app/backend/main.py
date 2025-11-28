from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.backend.model import ai_engine

# --- Lifespan Manager ---
# This runs BEFORE the server accepts requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model into GPU memory
    ai_engine.load_model()
    yield
    print("Shutting down model...")

# --- App Definition ---
app = FastAPI(title="Medical Fact-Checker API", lifespan=lifespan)

# --- Data Models ---
class ClaimRequest(BaseModel):
    text: str

class ClaimResponse(BaseModel):
    verdict: str
    explanation: str
    confidence_score: float = 0.0 # Placeholder for future work

# --- Routes ---
@app.get("/")
def health_check():
    return {"status": "active", "model": "Llama-3.2-3B-FineTuned"}

@app.post("/predict", response_model=ClaimResponse)
def check_claim(request: ClaimRequest):
    try:
        result = ai_engine.predict(request.text)
        return ClaimResponse(
            verdict=result["verdict"],
            explanation=result["explanation"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))