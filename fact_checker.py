import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pymed import PubMed
from collections import Counter
import textwrap

# 1. THE NLI CLASSIFIER (THE "VERIFIER")
# Load a model that is already fine-tuned for medical NLI.
# This one is trained on both MNLI and MedNLI, making it robust.
NLI_MODEL_NAME = "pritamdeka/PubMedBERT-MNLI-MedNLI"
print(f"Loading NLI model {NLI_MODEL_NAME}... (This may take a moment)")

# Load the tokenizer and model from Hugging Face
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)

# Get the human-readable labels from the model's config
# This is usually: { 0: "entailment", 1: "neutral", 2: "contradiction" }
id2label = nli_model.config.id2label

print("NLI model loaded.\n")

def classify_claim(evidence, claim):
    """
    Classifies the relationship between a piece of evidence and a claim.
    Returns the label (e.g., "Contradiction") and the confidence score.
    """
    # Tokenize the evidence (premise) and claim (hypothesis)
    inputs = nli_tokenizer(evidence, claim, return_tensors="pt", truncation=True, max_length=512)
    
    # Get model's prediction
    with torch.no_grad():
        outputs = nli_model(**inputs)
        logits = outputs.logits
        
    # Get probabilities and the "winning" class
    probabilities = torch.softmax(logits, dim=1)
    prediction_id = torch.argmax(probabilities, dim=1).item()
    
    # Get the label name (e.g., "contradiction") and its score
    predicted_label = id2label[prediction_id]
    confidence_score = probabilities[0][prediction_id].item()
    
    return {
        "label": predicted_label.upper(),
        "score": confidence_score
    }

# 2. THE EVIDENCE RETRIEVER (THE "SEARCHER")
# Initialize the PubMed API client
pubmed = PubMed(tool="MyMedicalFactChecker", email="your-email@example.com")

def get_evidence_from_pubmed(query, max_results=3):
    """
    Searches PubMed for a query and returns the abstracts.
    """
    print(f"Searching PubMed for: '{query}'")
    
    # Run the query
    search_results = pubmed.query(query, max_results=max_results)
    
    # Collect the abstracts
    abstracts = []
    for article in search_results:
        # Sometimes abstracts are None, so we check
        if article.abstract:
            abstracts.append(article.abstract)
            
    print(f"Found {len(abstracts)} relevant abstracts.\n")
    return abstracts

# 3. THE MAIN PIPELINE
if __name__ == "__main__":
    
    # Example 1 (Should be a clear contradiction)
    # claim_to_check = "Ibuprofen is a safe medication for pregnant women in their third trimester."
    
    # Example 2 (Should find supporting evidence)
    claim_to_check = "Vaccines are effective in preventing infectious diseases."

    # Example 3 (Should be a contradiction)
    # claim_to_check = "The MMR vaccine causes autism."

    print("="*50)
    print(f"🩺 Medical Claim Verifier 🩺")
    print("="*50)
    print(f"CLAIM: \"{claim_to_check}\"\n")
    
    # Phase 2: Retrieve Evidence
    # We use the claim itself as the search query
    evidence_list = get_evidence_from_pubmed(claim_to_check)
    
    if not evidence_list:
        print("Could not find any evidence on PubMed for this claim.")
    else:
        # Phase 3: Verify against each piece of evidence
        print("--- Verifying Claim Against Evidence ---")
        
        all_verdicts = []
        
        for i, evidence_abstract in enumerate(evidence_list):
            print(f"\n[Evidence {i+1}]")
            
            # For readability, just print the first few lines of the abstract
            short_evidence = textwrap.shorten(evidence_abstract, width=150, placeholder="...")
            print(f"ABSTRACT: {short_evidence}")
            
            # Run the NLI model
            result = classify_claim(evidence_abstract, claim_to_check)
            all_verdicts.append(result["label"])
            
            print(f"VERDICT: {result['label']} (Score: {result['score']:.2f})")

        # 4. FINAL AGGREGATION (THE VERDICT)
        print("\n" + "="*50)
        print("📈 Final Tally 📈")
        
        if all_verdicts:
            verdict_counts = Counter(all_verdicts)
            print("Based on the retrieved abstracts, the verdicts are:")
            for label, count in verdict_counts.items():
                print(f"- {label}: {count} time(s)")
            
            # Simple logic for a final answer
            if "CONTRADICTION" in verdict_counts and "ENTAILMENT" not in verdict_counts:
                print("\nFINAL ASSESSMENT: 🚫 Likely False")
            elif "ENTAILMENT" in verdict_counts and "CONTRADICTION" not in verdict_counts:
                print("\nFINAL ASSESSMENT: ✅ Likely True")
            else:
                print("\nFINAL ASSESSMENT:  inconclusive or disputed evidence.")
        else:
            print("No verdicts to tally.")
            
        print("="*50)