from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import torch
import logging
import requests
from bs4 import BeautifulSoup
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detection API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
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


# ----- Input & Output Schemas -----
class TextInput(BaseModel):
    input_text: str

    def get_content(self) -> str:
        text = self.input_text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        # If it looks like a URL, fetch content
        if re.match(r'^https?://', text):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(text, headers=headers, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()

                content = soup.get_text(separator=" ", strip=True)
                return content[:5000]
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not fetch content from URL: {e}")

        # Otherwise, treat as plain text
        return text


class PredictionOutput(BaseModel):
    truth_probability: float
    label: str


# ----- Routes -----
@app.get("/")
def root():
    return {"message": "Fake News Detection API", "status": "running"}


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: TextInput):
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        content = input_data.get_content()

        result = classifier(
            content,
            candidate_labels=["truthful news", "fake news"],
            hypothesis_template="This text is {}."
        )

        labels = result["labels"]
        scores = result["scores"]
        truth_index = labels.index("truthful news")
        truth_probability = scores[truth_index]
        label = "true" if truth_probability > 0.5 else "false"

        return PredictionOutput(
            truth_probability=round(truth_probability, 4),
            label=label
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": classifier is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
