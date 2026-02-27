"""
RAG vs Vanilla LLM Comparison

Proves that the RAG pipeline adds measurable value over a raw LLM.
Sends the same claims to:
  (a) MedCheck RAG system (with retrieved medical literature)
  (b) Raw GPT-4o-mini via OpenRouter (no context documents)

Compares verdict accuracy between the two approaches.

Requires: running backend + OPENROUTER_API_KEY in environment.
Usage: python tests/differentiators/test_rag_vs_vanilla.py [--url URL]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

CLAIMS_FILE = Path(__file__).parent.parent / "quality" / "test_claims.json"
DEFAULT_URL = "http://localhost:8000"
TIMEOUT = 60

VALID_VERDICTS = [
    "PARTIALLY ACCURATE", "INACCURATE", "ACCURATE",
    "MISLEADING", "UNVERIFIABLE", "OUTDATED",
    "OPINION", "INCONCLUSIVE", "IRRELEVANT",
]


def extract_verdict(text: str) -> str | None:
    for line in text.split("\n"):
        if "verdict" in line.lower():
            for v in VALID_VERDICTS:
                if v.lower() in line.lower():
                    return v
    return None


def query_vanilla_llm(claim: str, api_key: str) -> str | None:
    """Send claim directly to GPT-4o-mini via OpenRouter (no RAG context)."""
    prompt = (
        "You are a medical fact-checker. Evaluate this health claim and respond with:\n"
        "**Verdict**: <ONE of: [ACCURATE] / [INACCURATE] / [PARTIALLY ACCURATE] / "
        "[MISLEADING] / [UNVERIFIABLE] / [OUTDATED] / [OPINION] / [INCONCLUSIVE]>\n\n"
        f"Claim: {claim}"
    )

    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        timeout=TIMEOUT,
    )
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    return None


def run_comparison(base_url: str) -> dict:
    api_key = os.environ.get("APP_OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or APP_OPENROUTER_API_KEY env var")
        sys.exit(1)

    with open(CLAIMS_FILE) as f:
        claims = json.load(f)

    # Use a subset for cost efficiency
    test_claims = claims[:15]

    print(f"\n{'='*70}")
    print(f"  RAG vs Vanilla LLM Comparison")
    print(f"  RAG Backend: {base_url}")
    print(f"  Vanilla LLM: openai/gpt-4o-mini (OpenRouter)")
    print(f"  Claims: {len(test_claims)}")
    print(f"{'='*70}\n")

    rag_correct = 0
    vanilla_correct = 0
    client = httpx.Client(timeout=TIMEOUT)

    for i, claim in enumerate(test_claims, 1):
        expected = claim["expected_verdict"]
        print(f"[{i:2d}/{len(test_claims)}] {claim['claim'][:55]}...")

        # (a) RAG system
        try:
            r = client.post(f"{base_url}/rag/query", json={"query": claim["claim"], "top_k": 3})
            rag_verdict = extract_verdict(r.json()["chat_response"]) if r.status_code == 200 else None
        except Exception:
            rag_verdict = None

        # (b) Vanilla LLM
        try:
            vanilla_response = query_vanilla_llm(claim["claim"], api_key)
            vanilla_verdict = extract_verdict(vanilla_response) if vanilla_response else None
        except Exception:
            vanilla_verdict = None

        rag_match = rag_verdict and expected.lower() in rag_verdict.lower()
        vanilla_match = vanilla_verdict and expected.lower() in vanilla_verdict.lower()

        if rag_match:
            rag_correct += 1
        if vanilla_match:
            vanilla_correct += 1

        rag_icon = "✅" if rag_match else "❌"
        van_icon = "✅" if vanilla_match else "❌"
        print(f"         RAG: {rag_icon} [{rag_verdict}]  |  Vanilla: {van_icon} [{vanilla_verdict}]")

    client.close()

    rag_acc = rag_correct / len(test_claims) * 100
    van_acc = vanilla_correct / len(test_claims) * 100
    delta = rag_acc - van_acc

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  RAG Accuracy:     {rag_correct}/{len(test_claims)} ({rag_acc:.1f}%)")
    print(f"  Vanilla Accuracy: {vanilla_correct}/{len(test_claims)} ({van_acc:.1f}%)")
    print(f"  Delta (RAG - Vanilla): {delta:+.1f}%  {'✅ PASS' if delta >= 10 else '⚠️ UNDER TARGET'} (target: ≥+10%)")
    print(f"{'='*70}\n")

    return {"rag_accuracy": rag_acc, "vanilla_accuracy": van_acc, "delta": delta}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG vs Vanilla LLM Comparison")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend base URL")
    args = parser.parse_args()
    run_comparison(args.url)
