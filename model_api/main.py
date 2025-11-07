import torch
from transformers import pipeline, AutoTokenizer
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import logging

# --- 1. Load Model and Tokenizer (from your evaluate_finetuned.py) ---
# This code runs ONCE when the API starts
# ---------------------------------------------------------------------
HUB_MODEL_NAME = "sereotubu/biobert-finetune-v2"  # From your script
DEVICE = 0 if torch.cuda.is_available() else -1

logging.info(f"Loading tokenizer from {HUB_MODEL_NAME}...")
# Force truncation to 512, just like in your script
tokenizer = AutoTokenizer.from_pretrained(
    HUB_MODEL_NAME,
    model_max_length=512
)

logging.info("Loading text-classification pipeline...")
# Load the classifier pipeline using the specific model and tokenizer
classifier = pipeline(
    "text-classification",
    model=HUB_MODEL_NAME,
    tokenizer=tokenizer,
    device=DEVICE,
    truncation=True  # Ensure pipeline truncates
)
logging.info(f"Model ({HUB_MODEL_NAME}) pipeline loaded successfully!")
# ---------------------------------------------------------------------


# --- 2. Define the API Application ---
app = FastAPI(
    title="Medical Fake News Detection API",
    description="API for the ELEC490 Group 16 AI Fake News Detection project."
)

# --- 3. Define the Input Data Structure ---
# This tells FastAPI what kind of JSON to expect
class PredictionRequest(BaseModel):
    title: str | None = None  # Title is optional
    text: str

# --- 4. Create the Prediction Endpoint ---
@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Receives text and (optionally) a title, combines them,
    and returns a 'True' or 'False' prediction.
    """
    
    # --- 5. Replicate Preprocessing Logic (from your evaluate_finetuned.py) ---
    # Combine title and text exactly as you did in your evaluation script
    title = request.title
    text = request.text
    
    if title and isinstance(title, str) and len(title.strip()) > 0:
        input_text = title + " </s></s> " + text
    else:
        input_text = text
    
    # --- 6. Get Prediction ---
    logging.info(f"Running prediction on: {input_text[:100]}...") # Log snippet
    
    # The classifier returns a list, even for one item.
    # We just want the first (and only) result.
    prediction_result = classifier(input_text)[0] 
    
    logging.info(f"Prediction result: {prediction_result}")
    
    # Return the JSON response
    return {
        "input_text_processed": input_text,
        "prediction": prediction_result['label'],
        "confidence": prediction_result['score']
    }

# --- 7. (Optional) Health Check Endpoint ---
@app.get("/")
def health_check():
    return {"status": "API is running", "model_name": HUB_MODEL_NAME}

# --- 8. (Optional) Make the script runnable for local testing ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)