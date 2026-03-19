"""
RAG vs Vanilla LLM — F1 Score Comparison (Multi-Model)

A robust evaluation that compares the MedCheck RAG system against multiple
vanilla LLMs using proper classification metrics: Accuracy, Precision,
Recall, and F1 Score.

Scoring Pipeline — Equivalence Normalization:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Before computing any metrics, each model's raw verdict is     │
  │  checked against an equivalence table.  If a returned verdict  │
  │  is an acceptable alternative for the expected verdict, it is  │
  │  normalized to the expected verdict before scoring.            │
  │                                                                │
  │  Example: expected = ACCURATE, returned = PARTIALLY ACCURATE   │
  │           -> normalized to ACCURATE (acceptable equivalent)    │
  │                                                                │
  │  Metrics (Accuracy, Precision, Recall, F1) are then computed   │
  │  on the normalized fine-grained verdicts directly.             │
  └─────────────────────────────────────────────────────────────────┘

Models tested (via OpenRouter — all ~8B parameters for fair comparison):
  - MedCheck RAG system (pre-computed or live backend)
  - OpenAI GPT-4o-mini           (~8B)   (openai/gpt-4o-mini)
  - Meta Llama 3.1 8B Instruct   (8B)    (meta-llama/llama-3.1-8b-instruct)
  - Google Gemma 2 9B IT         (9B)    (google/gemma-2-9b-it)
  - Qwen 2.5 7B Instruct         (7.6B)  (qwen/qwen-2.5-7b-instruct)
  - Mistral 7B Instruct          (7.3B)  (mistralai/mistral-7b-instruct)

Usage:
  # Use pre-computed RAG results (skips RAG queries entirely):
  python tests/differentiators/test_rag_vs_vanilla_f1.py \\
      --rag-results tests/results/2026-03-01_203852/1_quality.json \\
      --api-key YOUR_OPENROUTER_KEY

  # Or query the live RAG backend:
  python tests/differentiators/test_rag_vs_vanilla_f1.py \\
      --url http://localhost:8000 \\
      --api-key YOUR_OPENROUTER_KEY

Results are saved to tests/results/<timestamp>/f1_comparison.json
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── Paths & Constants ────────────────────────────────────────────────

CLAIMS_FILE = Path(__file__).parent.parent / "quality" / "test_claims.json"
DEFAULT_URL = "http://localhost:8000"
TIMEOUT = 90  # seconds per request

# ── Verdict Definitions ──────────────────────────────────────────────

VALID_VERDICTS = [
    "PARTIALLY ACCURATE", "INACCURATE", "ACCURATE",
    "MISLEADING", "UNVERIFIABLE", "OUTDATED",
    "OPINION", "INCONCLUSIVE", "IRRELEVANT",
]

# Verdict equivalences: if a returned verdict is in the list for an
# expected verdict, it is treated as if the model returned the expected
# verdict.  This matches the quality test's scoring logic.
VERDICT_EQUIVALENCES = {
    "ACCURATE":            ["PARTIALLY ACCURATE"],
    "PARTIALLY ACCURATE":  ["ACCURATE", "MISLEADING"],
    "UNVERIFIABLE":        ["INCONCLUSIVE"],
    "INCONCLUSIVE":        ["UNVERIFIABLE"],
    "MISLEADING":          ["PARTIALLY ACCURATE", "INACCURATE"],
}

# ── Models to Evaluate ───────────────────────────────────────────────

VANILLA_MODELS = [
    {
        "id": "openai/gpt-4o-mini",
        "label": "OpenAI GPT-4o-mini",
    },
    {
        "id": "meta-llama/llama-3.1-8b-instruct",
        "label": "Meta Llama 3.1 8B",
    },
    {
        "id": "google/gemma-2-9b-it",
        "label": "Google Gemma 2 9B",
    },
    {
        "id": "qwen/qwen-2.5-7b-instruct",
        "label": "Qwen 2.5 7B",
    },
    {
        "id": "mistralai/mistral-7b-instruct-v0.1",
        "label": "Mistral 7B",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────


def extract_verdict(text: str) -> str | None:
    """Extract the verdict keyword from an LLM response."""
    for line in text.split("\n"):
        if "verdict" in line.lower():
            for v in VALID_VERDICTS:
                if v.lower() in line.lower():
                    return v
    return None


def normalize_verdict(expected: str, returned: str | None) -> str:
    """Apply equivalence normalization.

    If the returned verdict is an acceptable equivalent of the expected
    verdict, return the *expected* verdict (treating it as correct).
    Otherwise return the original returned verdict unchanged.
    """
    if returned is None:
        return "NONE"
    r = returned.upper().strip()
    e = expected.upper().strip()
    if r == e:
        return r  # exact match, no change needed
    if r in VERDICT_EQUIVALENCES.get(e, []):
        return e  # equivalent — normalize to expected
    return r  # not equivalent — keep as-is


def query_rag(client: httpx.Client, base_url: str, claim: str) -> str | None:
    """Query the MedCheck RAG backend."""
    try:
        r = client.post(
            f"{base_url}/rag/query",
            json={"query": claim, "top_k": 3},
        )
        if r.status_code == 200:
            return extract_verdict(r.json()["chat_response"])
        else:
            print(f"           [RAG] HTTP {r.status_code}: {r.text[:120]}", flush=True)
    except Exception as e:
        print(f"           [RAG] ERROR: {e}", flush=True)
    return None


def query_vanilla(api_key: str, model_id: str, claim: str) -> str | None:
    """Query a vanilla LLM via OpenRouter."""
    prompt = (
        "You are a medical fact-checker. Evaluate this health claim and respond with:\n"
        "**Verdict**: <ONE of: [ACCURATE] / [INACCURATE] / [PARTIALLY ACCURATE] / "
        "[MISLEADING] / [UNVERIFIABLE] / [OUTDATED] / [OPINION] / [INCONCLUSIVE]>\n\n"
        f"Claim: {claim}"
    )
    try:
        r = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            return extract_verdict(content)
        else:
            print(f"           [{model_id}] HTTP {r.status_code}: {r.text[:120]}", flush=True)
    except Exception as e:
        print(f"           [{model_id}] ERROR: {e}", flush=True)
    return None


# ── Metric Computation ───────────────────────────────────────────────


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Compute full classification metrics for a single model."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "classification_report": report,
        "confusion_matrix": cm,
    }


# ── Main Test Runner ─────────────────────────────────────────────────


def _load_rag_results(rag_results_path: str) -> dict[str, str]:
    """Load pre-computed RAG verdicts from a quality JSON file.

    Returns a dict mapping claim text -> returned verdict.
    """
    with open(rag_results_path) as f:
        data = json.load(f)

    # The quality JSON has a "details" list with "claim" and "returned" keys
    results = {}
    for entry in data.get("details", []):
        claim_text = entry["claim"]
        verdict = entry.get("returned", "").upper().strip()
        results[claim_text] = verdict
    return results


def run_f1_comparison(base_url: str, api_key: str, limit: int = None,
                      results_dir: Path = None,
                      rag_results_path: str = None) -> dict:
    """Run the full multi-model F1 comparison.

    Scoring pipeline:
      1. Each model's raw verdict is normalized via VERDICT_EQUIVALENCES.
      2. Metrics are computed on the normalized fine-grained verdicts.

    If rag_results_path is provided, RAG verdicts are loaded from the
    pre-computed quality JSON file instead of querying the backend live.

    Results are saved incrementally to results_dir/f1_comparison.json after
    every claim so that partial data is always available even if the run
    crashes or is interrupted.
    """

    with open(CLAIMS_FILE) as f:
        claims = json.load(f)

    if limit:
        claims = claims[:limit]

    # Load pre-computed RAG results if provided
    rag_cache = {}
    if rag_results_path:
        rag_cache = _load_rag_results(rag_results_path)
        print(f"  Loaded {len(rag_cache)} pre-computed RAG verdicts from:", flush=True)
        print(f"    {rag_results_path}", flush=True)

    n = len(claims)
    model_names = ["MedCheck RAG"] + [m["label"] for m in VANILLA_MODELS]

    rag_source = "pre-computed" if rag_results_path else base_url
    print(f"\n{'='*70}")
    print(f"  RAG vs Vanilla LLM -- F1 Score Comparison")
    print(f"  RAG source: {rag_source}")
    print(f"  Claims: {n}")
    print(f"  Models: {', '.join(model_names)}")
    if results_dir:
        print(f"  Saving to: {results_dir}")
    print(f"{'='*70}\n")

    # Storage: ground truth + normalized predictions per model
    ground_truth = []        # fine-grained expected verdicts
    predictions = {name: [] for name in model_names}
    per_claim_details = []

    # Incremental save helper
    out_file = results_dir / "f1_comparison.json" if results_dir else None

    def _save_incremental(completed: int):
        """Save current progress to disk after each claim."""
        if out_file is None:
            return
        snapshot = {
            "test": "RAG vs Vanilla LLM (F1 Multi-Model)",
            "status": f"in_progress ({completed}/{n})" if completed < n else "complete",
            "claims_total": n,
            "claims_completed": completed,
            "models_tested": model_names,
            "verdict_equivalences": VERDICT_EQUIVALENCES,
            "per_claim_details": per_claim_details,
        }
        with open(out_file, "w") as fp:
            json.dump(snapshot, fp, indent=2, default=str)

    # Save initial empty state
    _save_incremental(0)

    # Only create HTTP client for RAG if we're querying live
    client = httpx.Client(timeout=TIMEOUT) if not rag_results_path else None

    for i, claim in enumerate(claims, 1):
        expected = claim["expected_verdict"].upper().strip()
        ground_truth.append(expected)

        claim_text = claim["claim"]
        print(f"  [{i:3d}/{n}] {claim_text[:55]}...", flush=True)

        detail = {
            "claim": claim_text,
            "expected": expected,
        }

        # (a) RAG system — use cache if available, otherwise query live
        if rag_results_path and claim_text in rag_cache:
            rag_verdict = rag_cache[claim_text]
        elif client:
            rag_verdict = query_rag(client, base_url, claim_text)
        else:
            rag_verdict = None
            print(f"           [RAG] WARN: claim not found in pre-computed results", flush=True)

        # Normalize via equivalences
        rag_norm = normalize_verdict(expected, rag_verdict)
        predictions["MedCheck RAG"].append(rag_norm)
        detail["MedCheck RAG"] = {
            "verdict": rag_verdict,
            "normalized": rag_norm,
        }

        status_parts = [f"RAG:{rag_verdict or 'ERR':>20s}"]

        # (b) Each vanilla model
        for model in VANILLA_MODELS:
            verdict = query_vanilla(api_key, model["id"], claim_text)
            v_norm = normalize_verdict(expected, verdict)
            predictions[model["label"]].append(v_norm)
            detail[model["label"]] = {
                "verdict": verdict,
                "normalized": v_norm,
            }
            status_parts.append(f"{model['label'][:12]:>12s}:{verdict or 'ERR':>20s}")

        per_claim_details.append(detail)
        print(f"           {' | '.join(status_parts)}", flush=True)

        # Save after every completed claim
        _save_incremental(i)

    if client:
        client.close()

    # ── Compute Metrics ──────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  COMPUTING METRICS (on equivalence-normalized verdicts)")
    print(f"{'='*70}\n")

    all_verdicts = sorted(set(ground_truth))
    results = {"models": {}}

    for name in model_names:
        metrics = compute_metrics(
            ground_truth,
            predictions[name],
            labels=all_verdicts,
        )
        results["models"][name] = metrics

    # ── Print Summary Table ──────────────────────────────────────────

    header = f"{'Model':<30s} {'Accuracy':>9s} {'Prec':>7s} {'Recall':>7s} {'F1(M)':>7s} {'F1(W)':>7s}"
    sep = "-" * len(header)

    print(f"\n  Equivalence-Normalized Fine-Grained Metrics")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")

    best_f1 = -1
    best_model = ""

    for name in model_names:
        m = results["models"][name]
        acc_str = f"{m['accuracy']:.1%}"
        line = (
            f"  {name:<30s} {acc_str:>9s} {m['precision_macro']:>7.3f}"
            f" {m['recall_macro']:>7.3f} {m['f1_macro']:>7.3f} {m['f1_weighted']:>7.3f}"
        )
        print(line)
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_model = name

    print(f"  {sep}")
    print(f"\n  Best Macro F1: {best_model} ({best_f1:.3f})")

    # Per-verdict breakdown
    print(f"\n{'='*70}")
    print(f"  PER-VERDICT CLASSIFICATION REPORTS")
    print(f"{'='*70}")

    for name in model_names:
        report = results["models"][name]["classification_report"]
        print(f"\n  -- {name} --")
        print(f"  {'Verdict':<22s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'Support':>8s}")
        for label in all_verdicts:
            if label in report:
                r = report[label]
                print(
                    f"  {label:<22s} {r['precision']:>7.3f} {r['recall']:>7.3f}"
                    f" {r['f1-score']:>7.3f} {int(r['support']):>8d}"
                )
        if "macro avg" in report:
            r = report["macro avg"]
            print(f"  {'--- macro avg ---':<22s} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1-score']:>7.3f}")

    # ── Build Result Object ──────────────────────────────────────────

    results["test"] = "RAG vs Vanilla LLM (F1 Multi-Model)"
    results["total_claims"] = n
    results["models_tested"] = model_names
    results["best_model"] = best_model
    results["best_f1_macro"] = best_f1
    results["verdict_equivalences"] = VERDICT_EQUIVALENCES
    results["per_claim_details"] = per_claim_details

    # ── Determine pass/fail ──────────────────────────────────────────

    rag_f1 = results["models"]["MedCheck RAG"]["f1_macro"]
    vanilla_f1s = [
        results["models"][m["label"]]["f1_macro"]
        for m in VANILLA_MODELS
    ]
    avg_vanilla_f1 = sum(vanilla_f1s) / len(vanilla_f1s) if vanilla_f1s else 0
    delta = rag_f1 - avg_vanilla_f1

    results["rag_f1_macro"] = round(rag_f1, 4)
    results["avg_vanilla_f1_macro"] = round(avg_vanilla_f1, 4)
    results["delta_f1"] = round(delta, 4)
    results["pass"] = delta >= 0  # RAG should at least match vanilla average

    print(f"\n{'='*70}")
    print(f"  OVERALL RESULT")
    print(f"{'='*70}")
    print(f"  RAG F1 (Macro):          {rag_f1:.3f}")
    print(f"  Avg Vanilla F1 (Macro):  {avg_vanilla_f1:.3f}")
    print(f"  Delta (RAG - Vanilla):   {delta:+.3f}")
    print(f"  Status: {'PASS' if results['pass'] else 'FAIL'} (target: RAG >= vanilla average)")
    print(f"{'='*70}\n")

    return results


# ── Entry Point ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="RAG vs Vanilla LLM — F1 Score Comparison (Multi-Model)",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend base URL (used when --rag-results is not set)")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key")
    parser.add_argument("--limit", type=int, default=None, help="Max number of claims to test (default: all)")
    parser.add_argument("--rag-results", default=None,
                        help="Path to a pre-computed RAG quality JSON (e.g. 1_quality.json). "
                             "When provided, RAG verdicts are loaded from this file instead of "
                             "querying the backend, saving significant time.")
    args = parser.parse_args()

    api_key = (
        args.api_key
        or os.environ.get("APP_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not api_key:
        print("ERROR: Provide --api-key or set OPENROUTER_API_KEY env var")
        sys.exit(1)

    if args.rag_results and not Path(args.rag_results).exists():
        print(f"ERROR: RAG results file not found: {args.rag_results}")
        sys.exit(1)

    # Create results directory upfront so incremental saves work
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = Path(__file__).parent.parent / "results" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "f1_comparison.json"

    start = time.time()
    results = run_f1_comparison(args.url, api_key, limit=args.limit,
                                results_dir=results_dir,
                                rag_results_path=args.rag_results)
    results["elapsed_sec"] = round(time.time() - start, 1)

    # Final save with all metrics included
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Final results saved to: {out_file}")

    sys.exit(0 if results["pass"] else 1)


if __name__ == "__main__":
    main()
