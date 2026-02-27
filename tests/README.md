# MedCheck AI — Test Suite

How to run each test category.

## Prerequisites

```bash
pip install pytest httpx Pillow
```

## 1. Unit Tests (offline, no backend needed)

Validates schemas, config parsing, input processing, and API endpoint contracts using mocked dependencies.

```bash
python -m pytest tests/unit/ -v --tb=short
```

## 2. RAG Quality Tests (requires running backend)

Runs 25 curated medical claims and checks verdict accuracy.

```bash
python tests/quality/test_rag_quality.py --url http://localhost:8000
```

## 3. Differentiator Tests (requires running backend)

### RAG vs Vanilla LLM Comparison
```bash
# Requires OPENROUTER_API_KEY env var
python tests/differentiators/test_rag_vs_vanilla.py --url http://localhost:8000
```

### Hallucination Detection
```bash
python tests/differentiators/test_hallucination.py --url http://localhost:8000
```

### Consistency Test
```bash
python tests/differentiators/test_consistency.py --url http://localhost:8000
```

### Adversarial Robustness
```bash
python tests/differentiators/test_adversarial.py --url http://localhost:8000
```

### Off-Topic Rejection
```bash
python tests/differentiators/test_off_topic.py --url http://localhost:8000
```

## 4. End-to-End Latency Benchmark (requires running backend)

```bash
python tests/e2e/test_latency.py --url http://localhost:8000 --runs 100
```

## 5. Load Test (requires running backend + Locust)

```bash
pip install locust
locust -f tests/load/locustfile.py --host=http://localhost:8000 \
       --users 50 --spawn-rate 5 --run-time 1h --headless \
       --csv=tests/load/results
```

Then check `tests/load/results_stats.csv` for failure rates (target: ≤ 1%).
