"""
Consistency / Reproducibility Test

Measures how deterministic the system's verdicts are by submitting
the same claim 10 times and checking if the verdict is consistent.

Requires a running backend instance.
Usage: python tests/differentiators/test_consistency.py [--url URL]
"""

import argparse
import re
import sys
from collections import Counter

import httpx

DEFAULT_URL = "http://localhost:8000"
TIMEOUT = 60
RUNS_PER_CLAIM = 10

VALID_VERDICTS = [
    "PARTIALLY ACCURATE", "INACCURATE", "ACCURATE",
    "MISLEADING", "UNVERIFIABLE", "OUTDATED",
    "OPINION", "INCONCLUSIVE", "IRRELEVANT",
]

# Claims to test consistency on
TEST_CLAIMS = [
    "Vaccines cause autism",
    "Regular exercise reduces the risk of heart disease",
    "Drinking bleach can cure diseases",
    "Hydroxychloroquine is a proven cure for COVID-19",
    "Hand washing with soap reduces transmission of respiratory infections",
    "5G towers spread COVID-19",
    "Smoking increases the risk of lung cancer",
    "Essential oils can replace chemotherapy for cancer treatment",
    "Wearing a mask reduces the spread of airborne respiratory viruses",
    "Antibiotics can cure viral infections like the common cold",
]


def extract_verdict(text: str) -> str | None:
    for line in text.split("\n"):
        if "verdict" in line.lower():
            for v in VALID_VERDICTS:
                if v.lower() in line.lower():
                    return v
    return None


def run_consistency_test(base_url: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  Consistency / Reproducibility Test")
    print(f"  Target: {base_url}")
    print(f"  Claims: {len(TEST_CLAIMS)} × {RUNS_PER_CLAIM} runs each")
    print(f"{'='*70}\n")

    client = httpx.Client(timeout=TIMEOUT)
    results = []

    for claim in TEST_CLAIMS:
        print(f"  Claim: \"{claim[:55]}...\"")
        verdicts = []

        for run in range(RUNS_PER_CLAIM):
            try:
                r = client.post(
                    f"{base_url}/rag/query",
                    json={"query": claim, "top_k": 3},
                )
                if r.status_code == 200:
                    verdict = extract_verdict(r.json()["chat_response"])
                    verdicts.append(verdict or "UNKNOWN")
                else:
                    verdicts.append("ERROR")
            except Exception:
                verdicts.append("ERROR")

            print(f"    Run {run+1:2d}: {verdicts[-1]}")

        # Calculate consistency
        counter = Counter(verdicts)
        majority_verdict, majority_count = counter.most_common(1)[0]
        consistency = majority_count / len(verdicts) * 100

        status = "✅" if consistency >= 80 else "❌"
        print(f"    → Majority: [{majority_verdict}] {majority_count}/{len(verdicts)} "
              f"({consistency:.0f}%) {status}\n")

        results.append({
            "claim": claim,
            "verdicts": verdicts,
            "majority_verdict": majority_verdict,
            "consistency_pct": consistency,
            "pass": consistency >= 80,
        })

    client.close()

    # Summary
    passing = sum(1 for r in results if r["pass"])
    avg_consistency = sum(r["consistency_pct"] for r in results) / len(results)

    print(f"{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Claims passing (≥80% consistent): {passing}/{len(results)}")
    print(f"  Average consistency: {avg_consistency:.1f}%")
    print(f"  Status: {'✅ PASS' if passing == len(results) else '⚠️ PARTIAL'}")
    print(f"{'='*70}\n")

    return {"passing": passing, "total": len(results), "avg_consistency": avg_consistency}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consistency Test")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend base URL")
    args = parser.parse_args()

    results = run_consistency_test(args.url)
    sys.exit(0 if results["passing"] == results["total"] else 1)
