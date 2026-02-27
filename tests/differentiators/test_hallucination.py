"""
Hallucination Detection Test

Verifies that the LLM never fabricates source citations. For each response,
parses the **Source Files** line and checks that every filename mentioned
actually exists in the returned source_documents array.

Requires a running backend instance.
Usage: python tests/differentiators/test_hallucination.py [--url URL]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

CLAIMS_FILE = Path(__file__).parent.parent / "quality" / "test_claims.json"
DEFAULT_URL = "http://localhost:8000"
TIMEOUT = 60


def extract_cited_files(chat_response: str) -> list[str]:
    """Parse filenames from the **Source Files** line in the response."""
    for line in chat_response.split("\n"):
        if "source files" in line.lower():
            # Remove the label and extract filenames
            content = re.sub(r"\*\*Source Files\*\*:?\s*", "", line, flags=re.IGNORECASE).strip()
            if content.lower() in ("n/a", "none", ""):
                return []
            # Split by comma, semicolon, or "and"
            files = re.split(r"[,;]|\band\b", content)
            return [f.strip().strip("`'\"") for f in files if f.strip()]
    return []


def run_hallucination_test(base_url: str) -> dict:
    with open(CLAIMS_FILE) as f:
        claims = json.load(f)

    test_claims = claims[:15]

    print(f"\n{'='*70}")
    print(f"  Hallucination Detection Test")
    print(f"  Target: {base_url}")
    print(f"  Claims: {len(test_claims)}")
    print(f"{'='*70}\n")

    total = 0
    clean = 0  # no hallucinated sources
    hallucinated = 0
    details = []
    client = httpx.Client(timeout=TIMEOUT)

    for i, claim in enumerate(test_claims, 1):
        print(f"[{i:2d}/{len(test_claims)}] {claim['claim'][:55]}...", end=" ", flush=True)

        try:
            r = client.post(f"{base_url}/rag/query", json={"query": claim["claim"], "top_k": 3})
            if r.status_code != 200:
                print(f"⚠️ HTTP {r.status_code}")
                continue

            data = r.json()
            total += 1

            # Get cited files from LLM response text
            cited_files = extract_cited_files(data["chat_response"])

            # Get actual source document filenames
            actual_files = {
                doc.get("metadata", {}).get("file_name", "")
                for doc in data.get("source_documents", [])
            }

            # Check if every cited file exists in actual sources
            fabricated = [f for f in cited_files if f and not any(f in af for af in actual_files)]

            if not fabricated:
                clean += 1
                print(f"✅ Cited: {len(cited_files)} | Actual: {len(actual_files)}")
            else:
                hallucinated += 1
                print(f"❌ HALLUCINATED: {fabricated}")

            details.append({
                "claim": claim["claim"],
                "cited_files": cited_files,
                "actual_files": list(actual_files),
                "fabricated": fabricated,
            })

        except Exception as e:
            print(f"⚠️ Error: {e}")

    client.close()

    clean_rate = clean / total * 100 if total else 0

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Clean responses:       {clean}/{total} ({clean_rate:.1f}%)")
    print(f"  Hallucinated sources:  {hallucinated}/{total}")
    print(f"  Status: {'✅ PASS' if hallucinated == 0 else '❌ FAIL'}  (target: 0 hallucinations)")
    print(f"{'='*70}\n")

    return {"total": total, "clean": clean, "hallucinated": hallucinated}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hallucination Detection Test")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend base URL")
    args = parser.parse_args()

    results = run_hallucination_test(args.url)
    sys.exit(0 if results["hallucinated"] == 0 else 1)
