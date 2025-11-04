import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the 'app' object from your main.py file
from main import app

# Create a TestClient instance
client = TestClient(app)

# --- Test 1: Basic Endpoint Checks ---

def test_read_root():
    """Tests the root endpoint /"""
    response = client.get("/")
    assert response.status_code == 200
    # Check if the JSON response matches what's in main.py
    assert response.json() == {
        "message": "Fake News Detection API",
        "status": "running",
        "model": "roberta-large-mnli"
    }

def test_health_check():
    """Tests the /health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
# --- Test 2: Input Validation Tests (No Model Needed) ---

def test_predict_empty_text():
    """Tests the 400 error for empty input"""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "Text input cannot be empty"}

def test_predict_whitespace_text():
    """Tests the 400 error for input with only spaces"""
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
    assert response.json() == {"detail": "Text input cannot be empty"}

# --- Test 3: Mocked Model Prediction ---

# This is the most important part.
# We "patch" (temporarily replace) the 'classifier' object in the 'main' module.
# This avoids loading the 1.4GB RoBERTa model, making the test instant.

@patch('main.classifier', MagicMock(return_value={
    'labels': ['truthful news', 'fake news'],
    'scores': [0.98, 0.02]
}))
def test_predict_likely_true():
    """Tests a 'true' prediction using a mocked model"""
    response = client.post("/predict", json={"text": "This is a test text."})
    
    assert response.status_code == 200
    # Check if the logic in your /predict endpoint correctly processes the mock output
    assert response.json() == {
        "truth_probability": 0.98,
        "label": "true"
    }

@patch('main.classifier', MagicMock(return_value={
    'labels': ['truthful news', 'fake news'],
    'scores': [0.1, 0.9]
}))
def test_predict_likely_false():
    """Tests a 'false' prediction using a mocked model"""
    response = client.post("/predict", json={"text": "This is another test."})
    
    assert response.status_code == 200
    # Check if the logic in your /predict endpoint correctly processes the mock output
    assert response.json() == {
        "truth_probability": 0.1,
        "label": "false"
    }