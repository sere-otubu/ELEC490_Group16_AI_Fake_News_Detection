import torch
import uvicorn
import nltk
from collections import Counter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymed import PubMed
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware

# ---
# 1. LOAD MODELS & CLIENTS AT STARTUP
# ---
print("Loading NLI model and clients... This runs once at startup.")

try:
    NLI_MODEL_NAME = "pritamdeka/PubMedBERT-MNLI-MedNLI"
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    id2label = nli_model.config.id2label
    
    pubmed = PubMed(tool="ELEC498-Group16-Capstone", email="21eo4@queensu.ca")

    print("NLI model and PubMed client loaded.")

except Exception as e:
    print(f"Error loading models or clients: {e}")
    raise RuntimeError("Failed to load critical model. Exiting.") from e

# ---
# 2. DEFINE HELPER FUNCTIONS (Your Core Logic)
# ---
def classify_claim(evidence, claim):
    """Classifies the relationship between a piece of evidence and a claim."""
    try:
        inputs = nli_tokenizer(evidence, claim, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = nli_model(**inputs)
            logits = outputs.logits
        
        probabilities = torch.softmax(logits, dim=1)
        prediction_id = torch.argmax(probabilities, dim=1).item()
        
        predicted_label = id2label[prediction_id].upper()
        confidence_score = probabilities[0][prediction_id].item()
        
        return {"label": predicted_label, "score": confidence_score}
    except Exception as e:
        print(f"Error during classification: {e}")
        return {"label": "ERROR", "score": 0}

def get_evidence_from_pubmed(claim, max_results=5):
    """
    Searches PubMed for the claim. We'll rely on the NLI model
    to find the correct sentences in the abstracts.
    """
    search_query = claim
    print(f"Search Query: '{search_query}'")
    
    try:
        search_results = pubmed.query(search_query, max_results=max_results)
        abstracts = [article.abstract for article in search_results if article.abstract]
        print(f"Found {len(abstracts)} relevant abstracts.")
        return abstracts
    except Exception as e:
        print(f"Error querying PubMed: {e}")
        return []

# ---
# 3. CREATE THE FASTAPI APP
# ---
app = FastAPI()
# ... (CORS middleware code stays exactly the same) ...
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ... (Pydantic models ClaimRequest/VerifyResponse stay the same) ...
class ClaimRequest(BaseModel):
    claim: str

class VerifyResponse(BaseModel):
    verdict: str
    score: float
    details: str
    evidence: list[dict]

# ---
# 4. DEFINE THE API ENDPOINT (NEW SENTENCE-LEVEL LOGIC)
# ---
@app.post("/verify-claim", response_model=VerifyResponse)
async def verify_claim_endpoint(request: ClaimRequest):
    """
    Receives a claim, retrieves evidence, splits it into sentences,
    verifies each sentence, and returns a weighted verdict.
    """
    claim = request.claim
    if not claim.strip():
        raise HTTPException(status_code=400, detail="Claim text cannot be empty.")
    
    # 1. Retrieve Evidence (Abstracts)
    evidence_list = get_evidence_from_pubmed(claim)
    
    if not evidence_list:
        return VerifyResponse(
            verdict="Uncertain", 
            score=0.0, 
            details="Could not find sufficient evidence on PubMed.",
            evidence=[]
        )
    
    # 2. Verify against each SENTENCE
    verdict_counts = Counter()
    evidence_results = []
    total_score = 0.0  # Our net score
    
    top_evidence = {"label": "NEUTRAL", "score": 0, "abstract": ""}

    for abstract in evidence_list:
        # --- THIS IS THE NEW LOGIC ---
        # Split the abstract into individual sentences
        sentences = nltk.sent_tokenize(abstract)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Classify the *sentence*, not the abstract
            result = classify_claim(sentence, claim)
            
            if result["label"] == "ERROR":
                continue
                
            # Tally the net score
            if result["label"] == "ENTAILMENT":
                total_score += result["score"]
            elif result["label"] == "CONTRADICTION":
                total_score -= result["score"]
                
            verdict_counts[result["label"]] += 1

            # Store the *best* supporting/contradicting sentence
            if abs(result["score"]) > abs(top_evidence["score"]):
                top_evidence = {
                    "label": result["label"],
                    "score": result["score"],
                    "abstract": sentence # Show the sentence
                }

    if not verdict_counts:
        return VerifyResponse(
            verdict="Uncertain", 
            score=0.0, 
            details="Found evidence, but an error occurred during classification.",
            evidence=[]
        )
    
    # 3. Aggregate results
    
    # --- We can make the thresholds stronger now ---
    # We need a *much* stronger signal from ~75 sentences
    TRUE_THRESHOLD = 3.0  # e.g., 3-4 strong 'True' sentences
    FALSE_THRESHOLD = -3.0 # e.g., 3-4 strong 'False' sentences
    
    avg_confidence = total_score / (verdict_counts["ENTAILMENT"] + verdict_counts["CONTRADICTION"] + 1)
    details_string = f"Based on {len(evidence_list)} abstracts ({sum(verdict_counts.values())} sentences). Tally: {dict(verdict_counts)}. Net Score: {total_score:.2f}"

    if total_score > TRUE_THRESHOLD:
        final_verdict = "True"
    elif total_score < FALSE_THRESHOLD:
        final_verdict = "False"
    else:
        final_verdict = "Uncertain" 

    return VerifyResponse(
        verdict=final_verdict, 
        score=abs(avg_confidence), 
        details=details_string,
        evidence=[top_evidence] # Send back the single best sentence
    )

# ---
# 5. RUN THE SERVER
# ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)