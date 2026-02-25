"""
Off-Topic Rejection Test

Proves the system correctly returns the IRRELEVANT verdict for
queries that have nothing to do with health or medicine.

Requires a running backend instance.
Usage: python tests/differentiators/test_off_topic.py [--url URL]
"""

import argparse
import sys

import httpx

DEFAULT_URL = "http://localhost:8000"
TIMEOUT = 60

VALID_VERDICTS = [
    "PARTIALLY ACCURATE", "INACCURATE", "ACCURATE",
    "MISLEADING", "UNVERIFIABLE", "OUTDATED",
    "OPINION", "INCONCLUSIVE", "IRRELEVANT",
]

OFF_TOPIC_QUERIES = [
    "Who won the Super Bowl in 2024?",
    "What is the capital of France?",
    "What is the best programming language?",
    "How does blockchain technology work?",
    "What is the plot of the movie Inception?",
    "Who is the current president of the United States?",
    "What is the speed of light?",
    "How do I bake a chocolate cake?",
    "What are the rules of chess?",
    "Tell me about the history of the Roman Empire",
]


def extract_verdict(text: str) -> str | None:
    for line in text.split("\n"):
        if "verdict" in line.lower():
            for v in VALID_VERDICTS:
                if v.lower() in line.lower():
                    return v
    return None


def run_off_topic_test(base_url: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  Off-Topic Rejection Test")
    print(f"  Target: {base_url}")
    print(f"  Queries: {len(OFF_TOPIC_QUERIES)}")
    print(f"{'='*70}\n")

    client = httpx.Client(timeout=TIMEOUT)
    correctly_rejected = 0
    total = 0

    for i, query in enumerate(OFF_TOPIC_QUERIES, 1):
        print(f"[{i:2d}/{len(OFF_TOPIC_QUERIES)}] \"{query[:55]}\"...", end=" ", flush=True)

        try:
            r = client.post(
                f"{base_url}/rag/query",
                json={"query": query, "top_k": 3},
            )
            if r.status_code == 200:
                total += 1
                verdict = extract_verdict(r.json()["chat_response"])

                if verdict == "IRRELEVANT":
                    correctly_rejected += 1
                    print(f"✅ [{verdict}]")
                else:
                    print(f"❌ [{verdict}] (expected IRRELEVANT)")
            else:
                print(f"⚠️ HTTP {r.status_code}")
        except Exception as e:
            print(f"⚠️ Error: {e}")

    client.close()

    rejection_rate = correctly_rejected / total * 100 if total else 0

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Correctly rejected: {correctly_rejected}/{total} ({rejection_rate:.1f}%)")
    print(f"  Status: {'✅ PASS' if rejection_rate >= 90 else '❌ FAIL'}  (target: ≥90%)")
    print(f"{'='*70}\n")

    return {"correctly_rejected": correctly_rejected, "total": total, "rate": rejection_rate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Off-Topic Rejection Test")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend base URL")
    args = parser.parse_args()

    results = run_off_topic_test(args.url)
    sys.exit(0 if results["rate"] >= 90 else 1)
