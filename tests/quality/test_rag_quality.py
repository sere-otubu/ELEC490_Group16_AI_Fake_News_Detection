"""
RAG Response Quality Evaluation

Runs curated medical claims through the RAG system and evaluates:
1. Verdict accuracy (does it match expected verdict?)
2. Citation presence (are source documents returned?)
3. Confidence score presence

Requires a running backend instance.
Usage: python tests/quality/test_rag_quality.py [--url URL]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

# ── Configuration ────────────────────────────────────────────────────

DEFAULT_URL = "http://localhost:8000"
CLAIMS_FILE = Path(__file__).parent / "test_claims.json"
TIMEOUT = 60  # seconds per request (LLM can be slow)

# All valid verdict keywords the system can return
VALID_VERDICTS = [
    "PARTIALLY ACCURATE", "INACCURATE", "ACCURATE",
    "MISLEADING", "UNVERIFIABLE", "OUTDATED",
    "OPINION", "INCONCLUSIVE", "IRRELEVANT",
]


def extract_verdict(chat_response: str) -> str | None:
    """Parse the verdict from the LLM's structured response."""
    for line in chat_response.split("\n"):
        if "verdict" in line.lower():
            for v in VALID_VERDICTS:
                if v.lower() in line.lower():
                    return v
    return None


def extract_confidence(chat_response: str) -> float | None:
    """Parse the confidence score from the LLM's response."""
    import re
    for line in chat_response.split("\n"):
        if "confidence" in line.lower():
            match = re.search(r"(\d+\.\d+)", line)
            if match:
                return float(match.group(1))
    return None


def run_quality_test(base_url: str) -> dict:
    """Run all claims through the RAG system and evaluate results."""
    with open(CLAIMS_FILE) as f:
        claims = json.load(f)

    print(f"\n{'='*70}")
    print(f"  MedCheck AI — RAG Response Quality Test")
    print(f"  Target: {base_url}")
    print(f"  Claims: {len(claims)}")
    print(f"{'='*70}\n")

    results = {
        "total": len(claims),
        "correct_verdict": 0,
        "has_sources": 0,
        "has_confidence": 0,
        "errors": 0,
        "details": [],
    }

    client = httpx.Client(timeout=TIMEOUT)

    for i, claim in enumerate(claims, 1):
        print(f"[{i:2d}/{len(claims)}] {claim['claim'][:60]}...", end=" ", flush=True)

        start = time.perf_counter()
        try:
            r = client.post(
                f"{base_url}/rag/query",
                json={"query": claim["claim"], "top_k": 3},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            if r.status_code != 200:
                print(f"❌ HTTP {r.status_code}")
                results["errors"] += 1
                continue

            data = r.json()
            returned_verdict = extract_verdict(data["chat_response"])
            confidence = extract_confidence(data["chat_response"])
            has_sources = len(data.get("source_documents", [])) > 0

            # Check verdict match (allow flexibility for similar verdicts)
            expected = claim["expected_verdict"]
            verdict_correct = False
            if returned_verdict and expected.lower() in returned_verdict.lower():
                verdict_correct = True
            elif returned_verdict and returned_verdict.lower() in expected.lower():
                verdict_correct = True

            if verdict_correct:
                results["correct_verdict"] += 1
                print(f"✅ [{returned_verdict}] ({elapsed_ms:.0f}ms)")
            else:
                print(f"❌ Expected [{expected}], Got [{returned_verdict}] ({elapsed_ms:.0f}ms)")

            if has_sources:
                results["has_sources"] += 1
            if confidence is not None:
                results["has_confidence"] += 1

            results["details"].append({
                "claim": claim["claim"],
                "expected": expected,
                "returned": returned_verdict,
                "correct": verdict_correct,
                "confidence": confidence,
                "num_sources": len(data.get("source_documents", [])),
                "elapsed_ms": round(elapsed_ms),
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            results["errors"] += 1

    client.close()

    # ── Summary ──────────────────────────────────────────────────────
    accuracy = results["correct_verdict"] / results["total"] * 100 if results["total"] else 0
    source_rate = results["has_sources"] / results["total"] * 100 if results["total"] else 0

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Verdict Accuracy:  {results['correct_verdict']}/{results['total']} ({accuracy:.1f}%)  {'✅ PASS' if accuracy >= 80 else '❌ FAIL'} (target: ≥80%)")
    print(f"  Source Citation:   {results['has_sources']}/{results['total']} ({source_rate:.1f}%)")
    print(f"  Errors:            {results['errors']}")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedCheck RAG Quality Test")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend base URL")
    args = parser.parse_args()

    results = run_quality_test(args.url)
    sys.exit(0 if results["correct_verdict"] / results["total"] >= 0.8 else 1)
