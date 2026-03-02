"""
End-to-End Latency Benchmark

Runs 100 identical queries against the backend and reports latency statistics.
Measures the full round-trip: client → FastAPI → OpenRouter LLM → response.

Requires a running backend instance.
Usage: python tests/e2e/test_latency.py [--url URL] [--runs N]
"""

import argparse
import statistics
import sys
import time

import httpx

DEFAULT_URL = "http://localhost:8000"
TIMEOUT = 60
DEFAULT_RUNS = 100

STANDARD_QUERY = {
    "query": "Is ibuprofen safe during pregnancy?",
    "top_k": 3,
}


def run_latency_test(base_url: str, num_runs: int) -> dict:
    print(f"\n{'='*70}")
    print(f"  End-to-End Latency Benchmark")
    print(f"  Target: {base_url}")
    print(f"  Runs: {num_runs}")
    print(f"  Query: \"{STANDARD_QUERY['query']}\"")
    print(f"{'='*70}\n")

    client = httpx.Client(timeout=TIMEOUT)
    times_ms = []
    errors = 0

    for i in range(1, num_runs + 1):
        start = time.perf_counter()
        try:
            r = client.post(f"{base_url}/rag/query", json=STANDARD_QUERY)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if r.status_code == 200:
                times_ms.append(elapsed_ms)
                print(f"  Run {i:3d}: {elapsed_ms:7.0f} ms [OK]")
            else:
                errors += 1
                print(f"  Run {i:3d}: HTTP {r.status_code} [FAIL]")
        except Exception as e:
            errors += 1
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"  Run {i:3d}: Error ({elapsed_ms:.0f} ms) [FAIL] - {e}")

    client.close()

    if not times_ms:
        print("\n  [FAIL] No successful runs!")
        return {"p95": float("inf"), "errors": errors}

    times_ms.sort()
    p50 = times_ms[int(0.50 * len(times_ms))]
    p95 = times_ms[int(0.95 * len(times_ms))]
    mean = statistics.mean(times_ms)

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Successful runs: {len(times_ms)}/{num_runs}")
    print(f"  Errors:          {errors}")
    print(f"  Min:             {min(times_ms):,.0f} ms")
    print(f"  Mean:            {mean:,.0f} ms")
    print(f"  P50 (median):    {p50:,.0f} ms")
    print(f"  P95:             {p95:,.0f} ms  {'[PASS]' if p95 <= 3000 else '[FAIL]'} (target: <=3000 ms)")
    print(f"  Max:             {max(times_ms):,.0f} ms")
    print(f"{'='*70}\n")

    return {
        "successful": len(times_ms),
        "errors": errors,
        "min": min(times_ms),
        "mean": mean,
        "p50": p50,
        "p95": p95,
        "max": max(times_ms),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E Latency Benchmark")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend base URL")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of runs")
    args = parser.parse_args()

    results = run_latency_test(args.url, args.runs)
    sys.exit(0 if results["p95"] <= 3000 else 1)
