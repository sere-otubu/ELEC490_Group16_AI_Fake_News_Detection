import sys
from unittest.mock import MagicMock

# --- Mock heavy ML libraries before import ---
# This allows tests to run in GitHub Actions without installing 5GB of PyTorch
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()

from fastapi.testclient import TestClient
from unittest.mock import patch
from app.backend.main import app

# Create a test client
client = TestClient(app)

# 1. Test the Health Check (No GPU needed)
def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "active", "model": "Llama-3.2-3B-FineTuned"}

# 2. Test Prediction Logic (With MOCKED Model)
# Fake the AI engine so we don't need a GPU to run this test
@patch("app.backend.main.ai_engine") 
def test_predict_misinformation(mock_engine):
    # Setup the fake return value
    mock_engine.predict.return_value = {
        "claim": "Fake claim",
        "verdict": "False",
        "explanation": "MISINFORMATION DETECTED. This is false."
    }

    # Make the request
    payload = {"text": "Drinking bleach cures COVID."}
    response = client.post("/predict", json=payload)

    # Verify the API handles it correctly
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "False"
    assert "MISINFORMATION DETECTED" in data["explanation"]

# 3. Test Error Handling
def test_predict_empty_input():
    # Sending invalid JSON (missing 'text' field)
    response = client.post("/predict", json={})
    assert response.status_code == 422 # Validation Error