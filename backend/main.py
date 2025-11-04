from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import torch
import logging
import httpx  # For making API calls
import os     # To get API keys
from typing import List, Optional
from pathlib import Path

# --- START DEBUGGING BLOCK ---
print("--- STARTING SERVER AND DEBUGGING .ENV ---")
BASE_DIR = Path(__file__).resolve().parent
print(f"[DEBUG] Base directory is: {BASE_DIR}")

DOTENV_PATH = BASE_DIR / ".env"
print(f"[DEBUG] Looking for .env file at: {DOTENV_PATH}")

# Load environment variables
load_dotenv(dotenv_path=DOTENV_PATH)

if os.path.exists(DOTENV_PATH):
    print("[DEBUG] .env file FOUND.")
else:
    print("[DEBUG] *** .env file NOT FOUND at this path! ***")

# --- END DEBUGGING BLOCK ---

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Google Search API Config ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")

# --- MORE DEBUGGING ---
print(f"[DEBUG] GOOGLE_API_KEY value is: {GOOGLE_API_KEY}")
print(f"[DEBUG] GOOGLE_CSE_ID value is: {GOOGLE_CSE_ID}")
print("------------------------------------------")
# --- END DEBUGGING ---

SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    logger.warning("Google API Key or CSE ID not found. Fact-checking will not work.")
    GOOGLE_API_KEY = None  # Ensure it's None if missing

app = FastAPI(title="Fact-Checking API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
logger.info("Loading RoBERTa model...")
try:
    classifier = pipeline(
        "zero-shot-classification",
        model="roberta-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    classifier = None

# --- New Pydantic Models ---

class TextInput(BaseModel):
    text: str

class EvidenceItem(BaseModel):
    source_title: str
    url: str
    snippet: str

class PredictionOutput(BaseModel):
    main_claim: str
    # Key metrics from AI
    truth_probability: float
    label: str
    # Supporting evidence from Google
    evidence: List[EvidenceItem]

@app.get("/")
def read_root():
    return {
        "message": "Fact-Checking API",
        "status": "running",
        "model": "roberta-large-mnli (for NLI)"
    }

async def search_google(query: str) -> List[dict]:
    """Helper function to call Google Search API"""
    if not GOOGLE_API_KEY:
        # We don't raise an error, just return no results
        # The AI prediction can still work
        logger.warning("Google Search is not configured. Skipping evidence search.")
        return []
        
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": 3  # Get top 3 results
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(SEARCH_URL, params=params)
            response.raise_for_status()
            return response.json().get("items", [])
        except httpx.HTTPStatusError as e:
            logger.error(f"Google Search API error: {e}")
            return [] # Don't fail the whole request
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return [] # Don't fail the whole request

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: TextInput):
    """
    1. Analyzes text with RoBERTa for a true/false prediction.
    2. Searches Google for supporting evidence.
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not input_data.text or len(input_data.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    claim = input_data.text
    
    try:
        # --- Step 1: Get AI Prediction (Your "Key Metrics") ---
        ai_result = classifier(
            claim,
            candidate_labels=["truthful news", "fake news"],
            hypothesis_template="This text is {}."
        )
        
        # Extract scores
        labels = ai_result['labels']
        scores = ai_result['scores']
        
        # Find the score for "truthful news"
        truth_index = labels.index("truthful news")
        truth_probability = scores[truth_index]
        
        # Determine label
        label = "true" if truth_probability > 0.5 else "false"

        # --- Step 2: Search for Supporting Evidence ---
        search_results = await search_google(claim)
        
        evidence_list = []
        for item in search_results:
            snippet = item.get("snippet", "")
            if not snippet:
                continue
            
            evidence_list.append(EvidenceItem(
                source_title=item.get("title", "No Title"),
                url=item.get("link", "#"),
                snippet=snippet.replace("\n", " "),
            ))
        
        # --- Step 3: Combine and Return ---
        return PredictionOutput(
            main_claim=claim,
            truth_probability=round(truth_probability, 4),
            label=label,
            evidence=evidence_list
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "search_api_configured": GOOGLE_API_KEY is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)