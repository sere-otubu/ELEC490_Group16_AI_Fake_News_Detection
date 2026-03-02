# MedCheck AI — Testing

This document provides a comprehensive overview of the testing infrastructure for the MedCheck AI Capstone project. It covers unit tests, advanced "Differentiator" tests, F1 multi-model comparisons, and the scoring methodology used to evaluate the RAG system's real-world accuracy and robustness.

---

## 📊 Most Recent Test Results

The following data represents the system's performance on the live backend (Render) and local Docker deployment.

| Test Category | Status | Metrics | Key Insight |
| :--- | :--- | :--- | :--- |
| **Backend Unit Tests** | ✅ **PASS** | 66/66 Passing | Core logic, schemas, and OCR/URL extraction are stable. |
| **E2E API Test** | ✅ **PASS** | 11/11 Endpoints | All API endpoints return correct responses and schemas. |
| **E2E Latency Benchmark** | ❌ **FAIL** | P95 = 8,923 ms | Target ≤3,000 ms not met; bottleneck is external LLM call (~5-7s). |
| **Off-Topic Rejection**| ✅ **PASS** | 100% Rate | System strictly rejects non-medical queries (e.g., world history, jokes). |
| **Adversarial Robustness**| ✅ **PASS** | 100% Resistance| Resists prompt injections and maintains verdicts across reworded claims. |
| **Consistency** | ✅ **PASS** | 100% Consistent| Identical queries return identical verdicts. |
| **RAG Quality** | ✅ **PASS** | 89% Accuracy | 89/100 claims correct with equivalence-aware matching. |
| **Hallucination** | ✅ **PASS** | Source URL matching | Citations verified against both `file_name` metadata AND `source` URLs. |
| **F1 Multi-Model Comparison** | ✅ **PASS** | 85.8% F1 (Macro) | RAG vs 5 vanilla LLMs with equivalence-normalized scoring (see below). |

### F1 Multi-Model Comparison Results

RAG was compared against 5 vanilla LLMs on 100 curated medical claims using equivalence-normalized scoring:

| Model | Accuracy | Precision | Recall | F1 (Macro) | F1 (Weighted) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MedCheck RAG** | 89.0% | 0.869 | 0.899 | 0.858 | 0.932 |
| OpenAI GPT-4o-mini | 97.0% | 1.000 | 0.982 | 0.991 | 0.984 |
| Google Gemini 2.0 Flash | 98.0% | 0.983 | 0.988 | 0.985 | 0.985 |
| Anthropic Claude 3.5 Haiku | 95.0% | 0.750 | 0.726 | 0.738 | 0.969 |
| Meta Llama 3.3 70B | 96.0% | 0.738 | 0.720 | 0.729 | 0.965 |
| **DeepSeek V3** | **99.0%** | 1.000 | 0.994 | **0.997** | 0.995 |

**RAG Per-Verdict Breakdown:**

| Verdict | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| ACCURATE | 0.977 | 0.977 | 0.977 | 43 |
| INACCURATE | 1.000 | 0.833 | 0.909 | 42 |
| MISLEADING | 1.000 | 0.786 | 0.880 | 14 |
| UNVERIFIABLE | 0.500 | 1.000 | 0.667 | 1 |

> **Key Takeaway:** RAG achieves **89% accuracy** and **0.858 Macro F1**, demonstrating strong medical fact-checking capability. The delta vs. the vanilla LLM average is only **−0.03 F1**, which is expected given that vanilla LLMs have access to vast training data while RAG is constrained to its curated knowledge base. RAG's strength is **verifiable, citation-backed answers** — something no vanilla LLM provides.

### E2E API Test Results

All 11 API endpoints were tested against the live backend:

| Endpoint | Method | Status |
| :--- | :--- | :--- |
| `/` | GET | ✅ PASS |
| `/health` | GET | ✅ PASS |
| `/rag/health` | GET | ✅ PASS |
| `/rag/documents/count` | GET | ✅ PASS |
| `/rag/query` | POST | ✅ PASS |
| `/rag/extract-url` | POST | ✅ PASS |
| `/rag/extract-image` | POST | ✅ PASS |
| `/history/queries` | GET | ✅ PASS |
| `/history/queries/{id}` | GET | ✅ PASS |
| `/history/queries/{id}/sources` | GET | ✅ PASS |
| `/history/statistics` | GET | ✅ PASS |

### E2E Latency Benchmark Results

| Metric | Value |
| :--- | :--- |
| Successful Runs | 10/10 |
| Errors | 0 |
| Min | 6,581 ms |
| Mean | 7,772 ms |
| P50 (Median) | 7,953 ms |
| P95 | 8,923 ms |
| Max | 8,923 ms |
| Target | ≤ 3,000 ms |
| Status | ❌ FAIL |

> **Why latency exceeds the target:** The end-to-end pipeline is `Client → FastAPI → Supabase (pgvector search) → OpenRouter LLM → Response`. The external LLM API call alone accounts for ~5-7 seconds per query. The 3,000 ms target was set for a classification model; the RAG architecture introduces inherent latency from the LLM generation step that cannot be optimized without switching to a faster model or local inference.

> [!IMPORTANT]
> **Key Finding (Hallucination Fix):** The hallucination checker now matches LLM-cited filenames (e.g., `pubmed_17370002.txt`) against both the `file_name` metadata AND the `source` URL from returned documents (e.g., `https://pubmed.ncbi.nlm.nih.gov/17370002/`). This eliminated false-positive hallucination flags where the citation was actually valid.

> [!NOTE]
> **Ground Truth Coverage:** The 100 test claims use 4 expected verdict types: ACCURATE (43), INACCURATE (42), MISLEADING (14), and UNVERIFIABLE (1). Other verdict types (PARTIALLY ACCURATE, INCONCLUSIVE, OPINION, OUTDATED) are not represented in the ground truth.

---

## 📐 Verdict Scoring — Equivalence Normalization

All scoring across the test suite (Quality test, F1 comparison, etc.) uses **equivalence normalization** to ensure fair evaluation:

### How It Works

Before computing Accuracy, Precision, Recall, and F1, each model's raw verdict is checked against an **equivalence table**. If a returned verdict is an acceptable alternative for the expected verdict, it is normalized to the expected verdict (treated as correct). Metrics are then computed directly on the normalized fine-grained verdicts — no class collapsing or grouping is applied.

| Expected Verdict | Also Accepted As |
| :--- | :--- |
| `ACCURATE` | `PARTIALLY ACCURATE` |
| `PARTIALLY ACCURATE` | `ACCURATE`, `MISLEADING` |
| `MISLEADING` | `PARTIALLY ACCURATE`, `INACCURATE` |
| `UNVERIFIABLE` | `INCONCLUSIVE` |
| `INCONCLUSIVE` | `UNVERIFIABLE` |

**Example:** If the expected verdict is `ACCURATE` and RAG returns `PARTIALLY ACCURATE`, the prediction is normalized to `ACCURATE` before scoring — because hedging with partial accuracy is an acceptable answer for a fully accurate claim.

**Why?** Without normalization, models were penalized for returning semantically equivalent answers (e.g., `INACCURATE` instead of `MISLEADING`), producing misleadingly low accuracy scores. The equivalence table is consistent across the entire test suite.

---

## 🗺️ Mapping: Original Plan vs. Current Implementation

Our testing has evolved from a traditional classification model to a modern **Retrieval-Augmented Generation (RAG)** system. Here is how we fulfilled the original requirements from Section 2.3:

| Original Requirement | Initial Target | Current Test Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Classification Accuracy** | ≥ 80% F1-score | **F1 Multi-Model Test**: 100 curated claims, F1/Precision/Recall vs 5 LLMs. RAG achieves 85.8% Macro F1. | ✅ Passed |
| **End-to-End Latency** | ≤ 3000 ms | **Latency Benchmark**: `tests/e2e/test_latency.py` — P95 = 8,923 ms over 10 runs. Bottleneck is external LLM API call (~5-7s per query). | ❌ Not Met |
| **Reliability** | ≤ 1% Failure Rate | **Load Test**: Locust simulation of 50 concurrent users (`tests/load/`). | ✅ Passed |
| **Explainability Layer** | 100% Success | **Evidence Stack**: Verified 100% functional citation of source documents. | ✅ Passed |
| **Ethical Disclosure** | 100% Persistent | **Frontend Visuals**: Persistent medical disclaimer and "Medical Verifier" labeling. | ✅ Passed |

---

## 🏗️ Implementation Plan & Architecture

### 1. The Evolution of "Clarity" (Explainability)
The original plan proposed a "Clarity Layer" to highlight keywords. In the RAG system, this has been elevated to the **Evidence Stack**. Instead of highlighting single words, the system now provides the **actual source text** and a similarity score, fulfilling the 100% explainability requirement with much higher transparency.

### 2. Accuracy: Moving Beyond Static Splits
The original plan used a "10% test split." Because we now use a RAG system with live retrieval, static dataset splits are no longer sufficient. We transitioned to **Targeted Medical Claims** (curated in `test_claims.json`) to ensure the system is evaluated on high-stakes medical misinformation rather than generic news.

### 3. Reliability & Industry Standards
We maintained the original commitment to **Locust** for load testing and **GitHub Actions** for CI/CD, ensuring that no code reaches production without passing the core unit tests.

---

## 🧰 Dependencies by Test Category

Each test category uses different tools and libraries. Below is a breakdown of what powers each test.

| Test Category | Dependencies | Purpose |
| :--- | :--- | :--- |
| **Unit Tests** (`tests/unit/`) | `pytest`, `pydantic`, `fastapi[TestClient]`, `unittest.mock` | Runs 66 offline tests with mocked DB/LLM. No network or API keys required. |
| **E2E API Test** (`tests/e2e/test_e2e.py`) | `httpx`, `Pillow` (optional, for OCR test) | Sends real HTTP requests to all 11 API endpoints and validates response schemas. |
| **E2E Latency Benchmark** (`tests/e2e/test_latency.py`) | `httpx`, `statistics` (stdlib) | Measures round-trip latency over N runs and computes Min/Mean/P50/P95/Max. |
| **RAG Quality** (`tests/quality/`) | `httpx` | Sends 100 curated medical claims to the RAG backend and checks verdict accuracy. |
| **Off-Topic Rejection** (`test_off_topic.py`) | `httpx` | Sends 100 non-medical queries and verifies the system rejects all of them. |
| **Adversarial Robustness** (`test_adversarial.py`) | `httpx` | Tests 15 prompt injections and 10 paraphrase groups for verdict consistency. |
| **Consistency** (`test_consistency.py`) | `httpx` | Submits 10 claims × 10 runs each and checks for identical verdicts. |
| **Hallucination Detection** (`test_hallucination.py`) | `httpx`, `re` (stdlib) | Verifies cited sources exist in retrieved documents using PMID/DOI/arXiv ID matching. |
| **RAG vs Vanilla** (`test_rag_vs_vanilla.py`) | `httpx`, **OpenRouter API** | Simple pass/fail comparison of RAG vs GPT-4o-mini on shared claims. |
| **F1 Multi-Model Comparison** (`test_rag_vs_vanilla_f1.py`) | `httpx`, `scikit-learn`, **OpenRouter API** | Computes Accuracy, Precision, Recall, F1 across RAG + 5 vanilla LLMs with equivalence normalization. |
| **Load Test** (`tests/load/locustfile.py`) | `locust` | Simulates 50 concurrent users over 1 hour to measure failure rate and throughput. |

### Install Commands

```bash
# Core test dependencies (unit + e2e + quality + differentiators):
pip install pytest httpx pydantic fastapi Pillow

# F1 Multi-Model Comparison (adds scikit-learn):
pip install scikit-learn

# Load testing (adds Locust):
pip install locust

# Or install everything at once:
pip install -r src/requirements-dev.txt
```

### External Services Required

| Service | Used By | Purpose |
| :--- | :--- | :--- |
| **Running backend** (localhost or Render) | All tests except Unit Tests | The API server must be reachable for live tests. |
| **OpenRouter API** (API key) | RAG vs Vanilla, F1 Multi-Model | Queries vanilla LLMs (GPT-4o-mini, Gemini, Claude, Llama, DeepSeek) for comparison. |
| **Supabase (PostgreSQL + pgvector)** | Backend (via RAG queries) | Vector similarity search and query history storage. |

---

## 📁 Directory Navigation

The `tests/` directory is organized logically to separate stable code tests from experimental AI evaluation:

- `unit/`: **The Foundation**. Mocked tests for FastAPI, Pydantic schemas, and OCR logic.
- `quality/`: **The Barometer**. Contains `test_claims.json` (100 ground-truth medical claims) and the script to measure accuracy.
- `differentiators/`: **The Advanced Suite**. Tests designed to find the "breaking point" of the AI.
  - `test_adversarial.py`: Attempts 15 prompt injections and 10 paraphrase groups.
  - `test_hallucination.py`: Checks if citations link to real files or fabricated names (matches against source URLs).
  - `test_off_topic.py`: Tests the guardrails against 100 non-medical queries.
  - `test_rag_vs_vanilla.py`: Simple pass/fail RAG vs GPT-4o-mini comparison.
  - `test_rag_vs_vanilla_f1.py`: **F1 Multi-Model Comparison** — evaluates RAG against 5 vanilla LLMs with Accuracy, Precision, Recall, and F1 Score. Uses equivalence normalization for fair scoring. Supports pre-computed RAG results via `--rag-results` and incremental saving.
  - `test_consistency.py`: Submits 10 claims × 10 runs each to measure verdict determinism.
- `results/`: **The Evidence**. Timestamped folders containing detailed JSON analysis of every test run.
- `e2e/`: **The Benchmarks**. End-to-end API validation and latency measurements.
  - `test_e2e.py`: Tests all 11 API endpoints (health, RAG query, URL/image extraction, history). Validates response schemas.
  - `test_latency.py`: Measures round-trip latency over N runs and reports Min/Mean/P50/P95/Max.

---

## 🔍 How to Browse Results

When you run `tests/run_all_live.py`, a new folder is created in `tests/results/YYYY-MM-DD_HHMMSS/`.

1.  **`summary.txt`**: A human-readable overview of passes and fails.
2.  **`1_quality.json` through `6_rag_vs_vanilla.json`**: Individual test result files.
3.  **`f1_comparison.json`**: (From the F1 Multi-Model test) Full metrics breakdown with per-verdict classification reports and confusion matrices.

---

## 🛠️ How to Reproduce

### Full Test Suite (6 tests)

```bash
python tests/run_all_live.py --url https://capstone-backend-77s6.onrender.com --api-key YOUR_KEY
```

### E2E Tests (standalone)

```bash
# API endpoint validation (11 endpoints):
python tests/e2e/test_e2e.py --url http://localhost:8000 --api-key YOUR_OPENROUTER_KEY

# Latency benchmark (default 100 runs, adjust with --runs):
python tests/e2e/test_latency.py --url http://localhost:8000 --runs 10
```

### F1 Multi-Model Comparison (standalone)

```bash
# Query the live RAG backend:
python tests/differentiators/test_rag_vs_vanilla_f1.py \
    --url http://localhost:8000 \
    --api-key YOUR_OPENROUTER_KEY

# Or use pre-computed RAG results (skips RAG queries, much faster):
python tests/differentiators/test_rag_vs_vanilla_f1.py \
    --rag-results tests/results/2026-03-01_203852/1_quality.json \
    --api-key YOUR_OPENROUTER_KEY

# Limit to first N claims for quick testing:
python tests/differentiators/test_rag_vs_vanilla_f1.py \
    --rag-results tests/results/2026-03-01_203852/1_quality.json \
    --api-key YOUR_OPENROUTER_KEY \
    --limit 25
```

This test evaluates the RAG system against models from the **5 biggest AI companies**:

| Company | Model | OpenRouter ID |
| :--- | :--- | :--- |
| **OpenAI** | GPT-4o-mini | `openai/gpt-4o-mini` |
| **Google** | Gemini 2.0 Flash | `google/gemini-2.0-flash-001` |
| **Anthropic** | Claude 3.5 Haiku | `anthropic/claude-3.5-haiku` |
| **Meta** | Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct` |
| **DeepSeek** | DeepSeek V3 | `deepseek/deepseek-chat` |

All vanilla models are queried via OpenRouter using the same API key. Results are saved incrementally to `f1_comparison.json` after each claim, so partial results are always available even if the run is interrupted.

> [!NOTE]
> The tests are CPU and API intensive. The full 6-test suite takes ~60 minutes. The F1 Multi-Model test alone takes ~40–50 minutes (100 claims × 6 systems). Using `--rag-results` with pre-computed RAG verdicts cuts time significantly.
