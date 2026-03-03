"""
Emerging Threat (Temporal Shift) Test

This script evaluates how MedCheck RAG handles completely brand new, fabricated medical
misinformation compared to Vanilla LLMs. Because Vanilla LLMs have a training cutoff,
they cannot definitively "know" whether a brand new claim is true or false unless they guess.
By injecting synthetic "CDC/WHO Advisories" into the MedCheck vector database, we
demonstrate how RAG instantly adapts to new information and correctly debunks novel threats.

Queries:
1. Vanilla models (via OpenRouter)
2. MedCheck RAG API (live Render instance)

Usage: python tests/differentiators/test_emerging_threats.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

CLAIMS_FILE = Path(__file__).parent.parent / "quality" / "test_emerging_threats.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"
TIMEOUT = 60
REQUEST_DELAY = 1.0  # seconds between tests to avoid rate limits

LIVE_API_URL = "https://capstone-backend-77s6.onrender.com"

VANILLA_MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-haiku",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat"
]

def query_vanilla_llm(claim: str, api_key: str, client: httpx.Client, model: str) -> str:
    """Send claim to OpenRouter and explicitly demand a True/False answer."""
    prompt = (
        "You are a medical fact-checker. Evaluate the following novel medical claim and answer ONLY with 'True' or 'False'. "
        "Do not hedge, provide context, or refuse to answer.\n\n"
        f"Claim: {claim}\n\n"
        "Answer:"
    )

    try:
        r = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
            timeout=TIMEOUT
        )
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"].strip().lower()
            if "true" in content:
                return "True"
            elif "false" in content:
                return "False"
    except Exception as e:
        pass
    
    return "Error"

def query_rag_api(claim: str, api_key: str, client: httpx.Client) -> str:
    """Send claim to the Live MedCheck RAG API."""
    try:
        r = client.post(
            f"{LIVE_API_URL}/rag/query",
            json={"query": claim, "top_k": 3},
            headers={"X-OpenRouter-API-Key": api_key},
            timeout=TIMEOUT
        )
        if r.status_code == 200:
            data = r.json()
            chat_response = data.get("chat_response", "").upper()
            
            # The RAG system uses specific verdict tags
            if "[ACCURATE]" in chat_response:
                return "True"
            elif "[INACCURATE]" in chat_response or "[MISLEADING]" in chat_response or "[FALSE]" in chat_response:
                return "False"
            elif "[PARTIALLY ACCURATE]" in chat_response:
                return "True" 
    except Exception as e:
        pass
    
    return "Error"

def run_emerging_threats_test(limit: int | None = None) -> None:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
    
    api_key = os.environ.get("APP_OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or APP_OPENROUTER_API_KEY env var")
        sys.exit(1)

    if not CLAIMS_FILE.exists():
        print(f"ERROR: Could not find claims file at {CLAIMS_FILE}.")
        sys.exit(1)

    with open(CLAIMS_FILE) as f:
        test_claims = json.load(f)

    if limit:
        test_claims = test_claims[:limit]

    print(f"\n{'='*70}")
    print(f"  Emerging Threat (Temporal Shift) Test")
    print(f"  Live Target: {LIVE_API_URL}")
    print(f"  Claims: {len(test_claims)}")
    print(f"{'='*70}\n")

    client = httpx.Client(timeout=TIMEOUT)
    
    # Tracking metrics
    metrics = {
        "MedCheck-RAG": {"correct": 0, "errors": 0},
    }
    for model in VANILLA_MODELS:
        metrics[model] = {"correct": 0, "errors": 0}
        
    results_details = []

    for i, claim_obj in enumerate(test_claims, 1):
        claim_text = claim_obj["claim"]
        expected = claim_obj["expected_label"]
        print(f"\n[{i:2d}/{len(test_claims)}] Claim: {claim_text[:60]}... (Expected: {expected})")

        claim_results = {
            "claim": claim_text,
            "expected": expected,
            "topic": claim_obj.get("topic"),
            "source_file": claim_obj.get("source_file"),
            "predictions": {}
        }
        
        # 1. Test Vanilla Models
        for model in VANILLA_MODELS:
            pred = query_vanilla_llm(claim_text, api_key, client, model)
            claim_results["predictions"][model] = pred
            
            if pred == expected:
                metrics[model]["correct"] += 1
                status = "✅"
            elif pred == "Error":
                metrics[model]["errors"] += 1
                status = "⚠️ "
            else:
                status = "❌"
            print(f"  - {model[:20]:<20}: {pred:<6} {status}")
            time.sleep(REQUEST_DELAY)

        # 2. Test RAG API
        pred_rag = query_rag_api(claim_text, api_key, client)
        claim_results["predictions"]["MedCheck-RAG"] = pred_rag
        
        if pred_rag == expected:
            metrics["MedCheck-RAG"]["correct"] += 1
            status = "✅"
        elif pred_rag == "Error" or pred_rag == "":
            metrics["MedCheck-RAG"]["errors"] += 1
            status = "⚠️ "
        else:
            status = "❌"
        print(f"  - {'MedCheck-RAG':<20}: {pred_rag:<6} {status}")
        
        results_details.append(claim_results)
        time.sleep(REQUEST_DELAY)

    client.close()

    total = len(test_claims)
    
    print(f"\n{'='*70}")
    print(f"  ACCURACY RESULTS (Successfully calling out the fake claim)")
    print(f"{'='*70}")
    
    # Calculate and print final metrics
    for system, stats in metrics.items():
        correct = stats["correct"]
        acc = (correct / total) * 100
        stats["accuracy_pct"] = acc
        print(f"  {system:<25}: {acc:5.1f}% ({correct}/{total}) | Errors: {stats['errors']}")
    print(f"{'='*70}\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = run_dir / "9_emerging_threats_results.json"
    
    summary = {
        "test": "Emerging Threat (Temporal Shift) Test",
        "total_claims": total,
        "metrics": metrics,
        "details": results_details
    }
    
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Saved detailed results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emerging Threat (Temporal Shift) Test")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of claims to test")
    args = parser.parse_args()
    
    run_emerging_threats_test(args.limit)
