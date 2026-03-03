"""
Locust Load Test — MedCheck AI

Simulates 50 concurrent users submitting analysis requests over 1 hour.
Validates that the HTTP failure rate stays ≤ 1%.

Install: pip install locust
Usage:
  locust -f tests/load/locustfile.py --host=http://localhost:8000 \
         --users 50 --spawn-rate 5 --run-time 1h --headless \
         --csv=tests/load/results

  Then check results_stats.csv for failure rates.
"""

from locust import HttpUser, between, task


class MedCheckUser(HttpUser):
    """Simulates a typical MedCheck AI user."""

    wait_time = between(2, 5)  # seconds between requests

    # Diverse queries to simulate real usage
    QUERIES = [
        "Is ibuprofen safe during pregnancy?",
        "Does vitamin C prevent colds?",
        "Can vaccines cause autism?",
        "Is drinking lemon water good for health?",
        "Does smoking cause cancer?",
        "Is intermittent fasting safe?",
        "Can essential oils cure infections?",
        "Is aspirin good for heart health?",
        "Do antibiotics work on viruses?",
        "Is melatonin safe for children?",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._query_index = 0

    def _next_query(self) -> str:
        query = self.QUERIES[self._query_index % len(self.QUERIES)]
        self._query_index += 1
        return query

    @task(5)
    def query_rag(self):
        """Main task: submit a medical claim for analysis."""
        self.client.post(
            "/rag/query",
            json={"query": self._next_query(), "top_k": 3},
            timeout=90,
            name="/rag/query",
        )

    @task(2)
    def health_check(self):
        """Lightweight health check."""
        self.client.get("/health", name="/health")

    @task(1)
    def rag_health(self):
        """RAG system health check."""
        self.client.get("/rag/health", name="/rag/health")

    @task(1)
    def document_count(self):
        """Check document count."""
        self.client.get("/rag/documents/count", name="/rag/documents/count")
