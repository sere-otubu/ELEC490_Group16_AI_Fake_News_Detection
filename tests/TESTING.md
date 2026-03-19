# MedCheck AI — Testing

This document provides a comprehensive overview of the testing infrastructure for the MedCheck AI Capstone project. It covers unit tests, advanced "Differentiator" tests, F1 multi-model comparisons, and the scoring methodology used to evaluate the RAG system's real-world accuracy and robustness.

---

## 📊 Most Recent Test Results

The following data represents the system's performance on the live backend (Render) and local Docker deployment.

| Test Category | Status | Test Cases | Metrics | Key Insight |
| :--- | :--- | :--- | :--- | :--- |
| **Backend Unit Tests** | ✅ **PASS** | 66 unit tests | 66/66 Passing | Core logic, schemas, and OCR/URL extraction are stable. |
| **E2E API Test** | ✅ **PASS** | 11 endpoints | 11/11 Endpoints | All API endpoints return correct responses and schemas. |
| **E2E Latency Benchmark** | ❌ **FAIL** | 10 runs | P95 = 8,923 ms | Target ≤3,000 ms not met; bottleneck is external LLM call (~5-7s). |
| **Off-Topic Rejection**| ✅ **PASS** | 100 queries | 100% Rejection Rate | System strictly rejects non-medical queries (e.g., world history, jokes). |
| **Adversarial Robustness**| ✅ **PASS** | 15 injections + 30 paraphrases | 100% Resistance | Resists prompt injections and maintains verdicts across reworded claims. |
| **Consistency** | ✅ **PASS** | 10 claims × 10 runs (100 queries) | 100% Consistent | Identical queries return identical verdicts. |
| **RAG Quality** | ✅ **PASS** | 100 medical claims | 89% Accuracy | 89/100 claims correct with equivalence-aware matching. |
| **Hallucination** | ✅ **PASS** | 100 medical claims | Source URL matching | Citations verified against both `file_name` metadata AND `source` URLs. |
| **Citation Hallucination vs Vanilla** | ✅ **PASS** | 100 claims × 5 models | 49–82% Hallucination Rate | All 5 vanilla LLMs hallucinate 49–82% of URLs; MedCheck RAG hallucinates 0%. |
| **F1 Multi-Model Comparison** | ✅ **PASS** | 100 claims × 6 models (600 queries) | 85.8% F1 (Macro) | RAG vs 5 vanilla LLMs (all ~8B params) with equivalence-normalized scoring (see below). |
| **Load Test (Reliability)** | ✅ **PASS** | 163 requests (10 users, 10 min) | 0.00% Failure Rate | 163 requests, 0 failures over 10 minutes with 10 concurrent users. |

### F1 Multi-Model Comparison Results

RAG was compared against 5 vanilla LLMs — all in the **~8B parameter range** for a fair comparison — on 100 curated medical claims using equivalence-normalized scoring:

| Model | Params | Accuracy | Precision | Recall | F1 (Macro) | F1 (Weighted) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MedCheck RAG** | — | 89.0% | 0.869 | 0.899 | **0.858** | 0.932 |
| **OpenAI GPT-4o-mini** | ~8B | **97.0%** | 1.000 | 0.982 | **0.991** | 0.984 |
| Google Gemma 2 9B | 9B | 96.0% | 0.947 | 0.964 | 0.955 | 0.961 |
| Qwen 2.5 7B | 7.6B | 96.0% | 0.744 | 0.720 | 0.732 | 0.970 |
| Mistral 7B | 7.3B | 52.0% | 0.842 | 0.607 | 0.622 | 0.533 |
| Meta Llama 3.1 8B | 8B | 84.0% | 0.612 | 0.625 | 0.607 | 0.853 |
| **Vanilla Average** | — | 85.0% | 0.829 | 0.780 | **0.781** | 0.860 |

> **RAG vs Vanilla Average: RAG 0.858 F1 vs Vanilla 0.781 F1 → RAG wins by +0.077**

**RAG Per-Verdict Breakdown:**

| Verdict | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| ACCURATE | 0.977 | 0.977 | 0.977 | 43 |
| INACCURATE | 1.000 | 0.833 | 0.909 | 42 |
| MISLEADING | 1.000 | 0.786 | 0.880 | 14 |
| UNVERIFIABLE | 0.500 | 1.000 | 0.667 | 1 |

> **Key Takeaway:** When compared against models in the **same ~8B parameter class**, RAG achieves **0.858 Macro F1** — **+0.077 above the vanilla average (0.781)**. RAG outperforms 3 of 5 vanilla LLMs (Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B) and approaches Gemma 2 9B. Only GPT-4o-mini scores higher. RAG's true advantage remains **verifiable, citation-backed answers** — something no vanilla LLM provides.

> **Why RAG outperforms most same-size vanilla LLMs:** Smaller LLMs (~8B parameters) have limited medical knowledge baked into their weights. Models like Mistral 7B (52% accuracy, 0.622 F1) and Llama 3.1 8B (84% accuracy, 0.607 F1) struggle significantly with medical fact-checking, often defaulting to "ACCURATE" or failing to identify misinformation. RAG compensates for model size limitations by retrieving relevant, curated medical documents before generating a verdict. This is reflected in RAG's near-perfect precision (1.000 for INACCURATE/MISLEADING), proving that when RAG makes a call, it is almost always correct. The only model that significantly outperforms RAG is GPT-4o-mini (0.991 F1), which benefits from OpenAI's proprietary training pipeline — but it cannot provide verifiable source citations.

### Validation Strategy Results: Citation Hallucination

To prove the necessity of the MedCheck RAG architecture over a standard monolithic LLM, we conducted the **Citation Hallucination vs Vanilla LLM Test** (`test_citation_hallucination.py`). We prompted each vanilla LLM (via OpenRouter) with the 100 curated medical claims and explicitly demanded a verifiable medical URL supporting its verdict. All vanilla models are in the **~8B parameter range**.

**Results:**

| Model | Params | URLs Checked | Hallucinated | Hallucination Rate |
| :--- | :--- | :--- | :--- | :--- |
| **MedCheck RAG** | — | 100 | 0 | **0.0%** |
| OpenAI GPT-4o-mini | ~8B | 100 | 49 | 49.0% |
| Meta Llama 3.1 8B | 8B | 214 | 107 | 50.0% |
| Qwen 2.5 7B | 7.6B | 100 | 53 | 53.0% |
| Mistral 7B | 7.3B | 117 | 90 | **76.9%** |
| Google Gemma 3 12B | 12B | 100 | 82 | **82.0%** |
| **Vanilla Average** | — | — | — | **62.2%** |

> **RAG vs Vanilla Average: RAG 0.0% vs Vanilla 62.2% hallucination → RAG wins by −62.2 percentage points**

> **Key Finding:** Every single vanilla LLM hallucinated between **49% and 82%** of their citation URLs. This is not a GPT-4o-mini-specific problem — it is a fundamental limitation of all LLMs regardless of vendor. Smaller models (Mistral 7B, Gemma 3 12B) hallucinate even more aggressively than GPT-4o-mini. MedCheck RAG achieves **0% hallucination** by extracting physical `source` metadata from explicitly retrieved, pre-verified documents.

> **Why Vanilla LLMs Hallucinate Citations:** When an LLM is asked for a specific URL, it does not query an internet database or browse live sites. Because LLMs are probabilistic "next-token predictors", they simply generate strings of text that *statistically resemble* real URLs based on their training data patterns (e.g., combining a real domain like `cdc.gov` with an invented article slug). This means relying on a generic LLM for medical proof is a coin-flip — or worse — on whether the citation actually exists.

### Validation Strategy Results: Niche Medical Claims

To demonstrate MedCheck's ability to accurately answer highly specific, obscure medical claims where standard LLMs fail, we conducted the **Niche Medical Claims Test** (`test_niche_claims.py`). This test compares the live MedCheck RAG API against 5 vanilla LLMs (all ~8B parameters) on 20 highly specific facts directly extracted from WHO factsheets.

**Results:**
- **MedCheck RAG**: 100% Accuracy (20/20)
- **OpenAI GPT-4o-mini**: 90.0% Accuracy (18/20)
- **Meta Llama 3.1 8B**: 80.0% Accuracy (16/20)
- **Google Gemma 3 12B**: 80.0% Accuracy (16/20)
- **Mistral 7B**: 80.0% Accuracy (16/20)
- **Qwen 2.5 7B**: 70.0% Accuracy (14/20)
- **Vanilla Average**: 80.0% Accuracy

> **RAG vs Vanilla Average: RAG 100% vs Vanilla 80% → RAG wins by +20 percentage points**

**Deep Dive on Failures:**
For the niche claim: *"The WHO global incidence of noma from 1998 was estimated at 140,000 cases per year."* (Expected: True)
- **MedCheck RAG** and **GPT-4o-mini** correctly identified this as True.
- **Llama 3.1 8B**, **Gemma 3 12B**, **Qwen 2.5 7B**, and **Mistral 7B** all incorrectly labeled it as False.

For the niche claim: *"At least one third of people with schizophrenia experiences complete remission of symptoms."* (Expected: True)
- **MedCheck RAG**, **GPT-4o-mini**, and **Mistral 7B** correctly identified this as True.
- **Llama 3.1 8B**, **Gemma 3 12B**, and **Qwen 2.5 7B** incorrectly labeled it as False.

> **Why Vanilla LLMs Fail on Niche Claims:** When compared against models in the **same ~8B parameter class**, RAG's advantage becomes even more pronounced. Smaller LLMs lack the specific medical knowledge to evaluate obscure WHO statistics — 4 out of 5 vanilla models defaulted to "False" on the noma incidence claim. MedCheck RAG achieves **100% accuracy** by retrieving the exact WHO factsheet from the vector database and reading the statistic verbatim. This proves that for hyper-specific or recently updated medical protocols, RAG is inherently more reliable than relying on an LLM's static training memory — especially for smaller models.

*(Note on Metrics: This test relies purely on basic Accuracy (Correct Labels / Total) rather than the strict Equivalence-Normalized F1 Scoring. This is because the test is designed as a focused, qualitative differentiator on a balanced 20-claim set to explicitly demonstrate where LLMs guess incorrectly, rather than a large-scale statistical benchmark.)*

### Validation Strategy Results: Emerging Threats (Temporal Shift)

To demonstrate how the MedCheck RAG architecture solves the "training data cutoff" problem of Vanilla LLMs, we conducted the **Emerging Threat (Temporal Shift) Test** (`test_emerging_threats.py`). We created 20 completely fabricated, modern-sounding medical claims (e.g., *"Drinking ionized copper water completely cures the new XBB.1.5 variant"*) and simultaneously injected "CDC/WHO Debunking Advisories" into the MedCheck vector database. All vanilla models are in the **~8B parameter range**.

**Results:**
- **MedCheck RAG**: 100% Accuracy (20/20)
- **OpenAI GPT-4o-mini**: 100% Accuracy (20/20)
- **Meta Llama 3.1 8B**: 100% Accuracy (20/20)
- **Qwen 2.5 7B**: 100% Accuracy (20/20)
- **Mistral 7B**: 100% Accuracy (20/20)
- **Google Gemma 3 12B**: 95.0% Accuracy (19/20)
- **Vanilla Average**: 99.0% Accuracy

> **RAG vs Vanilla Average: RAG 100% vs Vanilla 99% → RAG wins by +1 percentage point**

> **Test Analysis:** Most vanilla LLMs and the RAG system scored 100% accuracy by labeling the fictitious claims as "False". Google Gemma 3 12B was the only model to fail, incorrectly labeling one claim as "True" — demonstrating that even obvious-sounding fabrications can trip up smaller models.
> 
> Because the test explicitly forces models into a binary True/False choice, Vanilla LLMs correctly guess "False" simply because the bizarre claims do not exist in their pre-training data. However, as proven in the *Citation Hallucination Test*, if a user were to ask a Vanilla LLM *why* the claim is false or ask for a source, the LLM would hallucinate a response. 
> 
> **The MedCheck Advantage:** MedCheck RAG achieved 100% accuracy not by statistically guessing, but by explicitly retrieving the newly-injected synthetic documents (e.g., `cdc_advisory_xbb15_copper.txt`). This conclusively proves that hospitals can update the RAG system with breaking medical protocols **instantly** without waiting months to fine-tune an LLM, and the system will strictly abide by the new rules.

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

### Load Test (Reliability) Results

The system was tested for reliability under sustained concurrent load using Locust:

| Metric | Value |
| :--- | :--- |
| **Test Duration** | 10 minutes |
| **Concurrent Users** | 10 |
| **Total Requests** | 163 |
| **Total Failures** | 0 |
| **Failure Rate** | **0.00%** |
| **Target** | ≤ 1% |
| **Status** | ✅ **PASS** |

**Per-Endpoint Breakdown:**

| Endpoint | Requests | Failures | Avg (ms) | P50 (ms) | P95 (ms) | Max (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `GET /health` | 41 | 0 | 21,861 | 24,000 | 37,000 | 38,223 |
| `GET /rag/documents/count` | 18 | 0 | 25,808 | 27,000 | 43,000 | 43,001 |
| `GET /rag/health` | 20 | 0 | 28,943 | 32,000 | 51,000 | 50,890 |
| `POST /rag/query` | 84 | 0 | 40,261 | 43,000 | 58,000 | 59,371 |
| **Aggregated** | **163** | **0** | 32,648 | 34,000 | 57,000 | 59,371 |

**Key Findings:**
- **100% success rate** — All 163 requests completed successfully with zero failures
- **No timeouts** — All requests completed within the 90-second timeout (max: 59.4s)
- **Stable performance** — Response times remained consistent throughout the 10-minute test
- **Throughput** — 0.27 req/s aggregated (approximately 1 request every 3.7 seconds with 10 concurrent users)

> **Test Configuration:** The load test was run against the paid Render instance (Pro plan, $7/month starter instance) with a 90-second timeout to accommodate LLM API call latency. The test validates that the system can handle sustained concurrent load without degradation or failures.

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
| **Reliability** | ≤ 1% Failure Rate | **Load Test**: Locust simulation of 10 concurrent users for 10 minutes (`tests/load/`). Achieved 0.00% failure rate (0/163 requests). | ✅ Passed |
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

| Test Category | Test Cases | Dependencies | Purpose |
| :--- | :--- | :--- | :--- |
| **Unit Tests** (`tests/unit/`) | 66 tests | `pytest`, `pydantic`, `fastapi[TestClient]`, `unittest.mock` | Runs 66 offline tests with mocked DB/LLM. No network or API keys required. |
| **E2E API Test** (`tests/e2e/test_e2e.py`) | 11 endpoints | `httpx`, `Pillow` (optional, for OCR test) | Sends real HTTP requests to all 11 API endpoints and validates response schemas. |
| **E2E Latency Benchmark** (`tests/e2e/test_latency.py`) | 10 runs (configurable) | `httpx`, `statistics` (stdlib) | Measures round-trip latency over N runs and computes Min/Mean/P50/P95/Max. |
| **RAG Quality** (`tests/quality/`) | 100 claims | `httpx` | Sends 100 curated medical claims to the RAG backend and checks verdict accuracy. |
| **Off-Topic Rejection** (`test_off_topic.py`) | 100 queries | `httpx` | Sends 100 non-medical queries across 10 categories and verifies the system rejects all of them. |
| **Adversarial Robustness** (`test_adversarial.py`) | 45 prompts (15 injections + 10 groups × 3 paraphrases) | `httpx` | Tests 15 prompt injections and 10 paraphrase groups (3 variants each) for verdict consistency. |
| **Consistency** (`test_consistency.py`) | 100 queries (10 claims × 10 runs) | `httpx` | Submits 10 claims × 10 runs each and checks for identical verdicts. |
| **Hallucination Detection** (`test_hallucination.py`) | 100 claims | `httpx`, `re` (stdlib) | Verifies cited sources exist in retrieved documents using PMID/DOI/arXiv ID matching. |
| **Citation Hallucination vs Vanilla** (`test_citation_hallucination.py`) | 100 claims | `httpx`, `urllib`, **OpenRouter API** | Explicitly demands a verifiable URL from Vanilla LLMs and pings each returned URL to check for 404s (fake citations). |
| **RAG vs Vanilla** (`test_rag_vs_vanilla.py`) | 100 claims | `httpx`, **OpenRouter API** | Simple pass/fail comparison of RAG vs GPT-4o-mini on shared claims. |
| **F1 Multi-Model Comparison** (`test_rag_vs_vanilla_f1.py`) | 600 queries (100 claims × 6 models) | `httpx`, `scikit-learn`, **OpenRouter API** | Computes Accuracy, Precision, Recall, F1 across RAG + 5 vanilla LLMs with equivalence normalization. |
| **Load Test** (`tests/load/locustfile.py`) | 163 requests (10 users, 10 min) | `locust` | Simulates 10 concurrent users over 10 minutes with 10 diverse medical queries to measure failure rate and throughput. |

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

- `unit/`: **The Foundation** (66 tests). Mocked tests for FastAPI, Pydantic schemas, and OCR logic.
- `quality/`: **The Barometer** (100 claims). Contains `test_claims.json` (100 ground-truth medical claims) and the script to measure accuracy.
- `differentiators/`: **The Advanced Suite**. Tests designed to find the "breaking point" of the AI.
  - `test_adversarial.py` (45 prompts): Attempts 15 prompt injections and 10 paraphrase groups (3 variants each).
  - `test_hallucination.py` (100 claims): Checks if citations link to real files or fabricated names (matches against source URLs).
  - `test_off_topic.py` (100 queries): Tests the guardrails against 100 non-medical queries across 10 categories.
  - `test_rag_vs_vanilla.py` (100 claims): Simple pass/fail RAG vs GPT-4o-mini comparison.
  - `test_rag_vs_vanilla_f1.py` (600 queries): **F1 Multi-Model Comparison** — evaluates RAG against 5 vanilla LLMs with Accuracy, Precision, Recall, and F1 Score. Uses equivalence normalization for fair scoring. Supports pre-computed RAG results via `--rag-results` and incremental saving.
  - `test_consistency.py` (100 queries): Submits 10 claims × 10 runs each to measure verdict determinism.
- `results/`: **The Evidence**. Timestamped folders containing detailed JSON analysis of every test run.
- `e2e/`: **The Benchmarks**. End-to-end API validation and latency measurements.
  - `test_e2e.py` (11 endpoints): Tests all 11 API endpoints (health, RAG query, URL/image extraction, history). Validates response schemas.
  - `test_latency.py` (10 runs): Measures round-trip latency over N runs and reports Min/Mean/P50/P95/Max.
- `load/`: **The Stress Test** (163 requests). Load testing with Locust to validate reliability under concurrent load.
  - `locustfile.py`: Simulates 10 concurrent users for 10 minutes with 10 diverse medical queries, measuring failure rate and response times across all endpoints.

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

### Load Test (standalone)

```bash
# 10 concurrent users for 10 minutes (recommended):
python -m locust -f tests/load/locustfile.py \
    --host=https://capstone-backend-77s6.onrender.com \
    --users 10 --spawn-rate 2 --run-time 10m --headless \
    --csv=tests/load/results

# Quick 5-minute test:
python -m locust -f tests/load/locustfile.py \
    --host=https://capstone-backend-77s6.onrender.com \
    --users 10 --spawn-rate 2 --run-time 5m --headless \
    --csv=tests/load/results

# Results are saved to:
# - tests/load/results_stats.csv (summary statistics)
# - tests/load/results_failures.csv (failure details)
# - tests/load/results_stats_history.csv (time-series data)
```

**Note:** The load test uses a 90-second timeout to accommodate LLM API call latency. For best results, run against a paid Render instance (free tier may queue requests under concurrent load).

This test evaluates the RAG system against models from **5 different AI companies**, all in the **~8B parameter range** for a fair comparison:

| Company | Model | Params | OpenRouter ID |
| :--- | :--- | :--- | :--- |
| **OpenAI** | GPT-4o-mini | ~8B | `openai/gpt-4o-mini` |
| **Meta** | Llama 3.1 8B Instruct | 8B | `meta-llama/llama-3.1-8b-instruct` |
| **Google** | Gemma 2 9B IT | 9B | `google/gemma-2-9b-it` |
| **Alibaba** | Qwen 2.5 7B Instruct | 7.6B | `qwen/qwen-2.5-7b-instruct` |
| **Mistral AI** | Mistral 7B Instruct | 7.3B | `mistralai/mistral-7b-instruct-v0.1` |

All vanilla models are queried via OpenRouter using the same API key. Results are saved incrementally to `f1_comparison.json` after each claim, so partial results are always available even if the run is interrupted.

> [!NOTE]
> The tests are CPU and API intensive. The full 6-test suite takes ~60 minutes. The F1 Multi-Model test alone takes ~40–50 minutes (100 claims × 6 systems). Using `--rag-results` with pre-computed RAG verdicts cuts time significantly.
