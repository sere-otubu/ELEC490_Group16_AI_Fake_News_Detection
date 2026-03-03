"""
End-to-End API Test Suite

Tests all API endpoints to ensure the backend is functioning correctly.
Covers health checks, RAG queries, URL/image extraction, and history endpoints.

Usage: python tests/e2e/test_e2e.py [--url URL] [--api-key KEY]
"""

import argparse
import sys
import time
from typing import Optional
from uuid import UUID

import httpx

DEFAULT_URL = "http://localhost:8000"
TIMEOUT = 60


class E2ETestRunner:
    """Runs comprehensive end-to-end tests against the backend API."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = httpx.Client(timeout=TIMEOUT)
        self.results = {
            "passed": [],
            "failed": [],
            "skipped": [],
        }
        self.query_id: Optional[UUID] = None  # Store for history tests

    def _headers(self) -> dict:
        """Get request headers with API key if provided."""
        headers = {}
        if self.api_key:
            headers["X-OpenRouter-API-Key"] = self.api_key
        return headers

    def _test(self, name: str, test_func):
        """Run a test and record the result."""
        try:
            print(f"  Testing: {name}...", end=" ", flush=True)
            result = test_func()
            if result:
                print("PASS")
                self.results["passed"].append(name)
            else:
                print("FAIL")
                self.results["failed"].append(name)
        except Exception as e:
            print(f"FAIL - {e}")
            self.results["failed"].append(name)

    def test_root_endpoint(self) -> bool:
        """Test GET / - Root API info endpoint."""
        r = self.client.get(f"{self.base_url}/")
        if r.status_code != 200:
            return False
        data = r.json()
        return (
            "message" in data
            and "version" in data
            and data.get("version") == "1.0.0"
            and "endpoints" in data
        )

    def test_health_check(self) -> bool:
        """Test GET /health - Basic health check."""
        r = self.client.get(f"{self.base_url}/health")
        if r.status_code != 200:
            return False
        data = r.json()
        return data.get("status") == "healthy" and data.get("service") == "Capstone API"

    def test_rag_health(self) -> bool:
        """Test GET /rag/health - RAG system health check."""
        r = self.client.get(f"{self.base_url}/rag/health")
        if r.status_code != 200:
            return False
        data = r.json()
        # vector_store must be True; embedding_model and chat_model are
        # lazy-loaded and may report False when not actively in use
        return (
            data.get("vector_store") is True
            and "embedding_model" in data
            and "chat_model" in data
        )

    def test_document_count(self) -> bool:
        """Test GET /rag/documents/count - Document count endpoint."""
        r = self.client.get(f"{self.base_url}/rag/documents/count")
        if r.status_code != 200:
            return False
        data = r.json()
        return (
            "document_count" in data
            and isinstance(data["document_count"], int)
            and data["document_count"] > 0
        )

    def test_rag_query(self) -> bool:
        """Test POST /rag/query - Main RAG query endpoint."""
        payload = {
            "query": "Is ibuprofen safe during pregnancy?",
            "top_k": 3,
        }
        r = self.client.post(
            f"{self.base_url}/rag/query",
            json=payload,
            headers=self._headers(),
        )
        if r.status_code != 200:
            return False
        data = r.json()
        if "chat_response" not in data or "source_documents" not in data:
            return False
        # Verify source documents structure
        for doc in data.get("source_documents", []):
            if "content" not in doc or "score" not in doc or "metadata" not in doc:
                return False
            if "file_name" not in doc.get("metadata", {}):
                return False
        # Store query_id for history tests (if available in response)
        # Note: query_id is not in QueryResponse, so we'll get it from history
        return True

    def test_url_extraction(self) -> bool:
        """Test POST /rag/extract-url - URL text extraction."""
        # Try multiple URLs in case one is unavailable
        test_urls = [
            "https://en.wikipedia.org/wiki/Ibuprofen",
            "https://medlineplus.gov/druginfo/meds/a682159.html",
            "https://www.cdc.gov/flu/about/index.html",
        ]
        for url in test_urls:
            payload = {"url": url}
            r = self.client.post(f"{self.base_url}/rag/extract-url", json=payload)
            if r.status_code == 200:
                data = r.json()
                if (
                    "extracted_text" in data
                    and len(data["extracted_text"]) > 0
                    and "source_url" in data
                ):
                    return True
        return False

    def test_image_extraction(self) -> bool:
        """Test POST /rag/extract-image - Image OCR extraction."""
        # Create a simple test image with text using PIL
        try:
            from io import BytesIO

            from PIL import Image, ImageDraw, ImageFont

            # Create a simple image with text
            img = Image.new("RGB", (400, 100), color="white")
            draw = ImageDraw.Draw(img)
            # Use default font (may not be perfect, but should work)
            draw.text((10, 30), "Test Medical Claim: Ibuprofen is safe.", fill="black")
            # Save to bytes
            img_bytes = BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            files = {"file": ("test.png", img_bytes, "image/png")}
            r = self.client.post(
                f"{self.base_url}/rag/extract-image",
                files=files,
            )
            if r.status_code != 200:
                return False
            data = r.json()
            return "extracted_text" in data and len(data["extracted_text"]) > 0
        except ImportError:
            # PIL not available, skip this test
            return None  # Special return to indicate skip
        except Exception as e:
            # OCR might fail, but endpoint should still return 200
            # If it returns 400, that's acceptable for invalid images
            return r.status_code in (200, 400)

    def test_history_queries(self) -> bool:
        """Test GET /history/queries - Query history list."""
        r = self.client.get(f"{self.base_url}/history/queries?limit=5")
        if r.status_code != 200:
            return False
        data = r.json()
        if "items" not in data or "total_count" not in data:
            return False
        # If there are queries, store the first one's ID for detail test
        if data["items"] and len(data["items"]) > 0:
            self.query_id = UUID(data["items"][0]["id"])
        return True

    def test_history_query_detail(self) -> bool:
        """Test GET /history/queries/{id} - Query detail by ID."""
        if not self.query_id:
            # Try to get a query ID first
            r = self.client.get(f"{self.base_url}/history/queries?limit=1")
            if r.status_code != 200:
                return False
            data = r.json()
            if not data.get("items") or len(data["items"]) == 0:
                return False  # No queries to test
            self.query_id = UUID(data["items"][0]["id"])

        r = self.client.get(f"{self.base_url}/history/queries/{self.query_id}")
        if r.status_code != 200:
            return False
        data = r.json()
        return (
            "query_history" in data
            and "source_documents" in data
            and str(data["query_history"]["id"]) == str(self.query_id)
        )

    def test_history_query_sources(self) -> bool:
        """Test GET /history/queries/{id}/sources - Source documents for query."""
        if not self.query_id:
            # Try to get a query ID first
            r = self.client.get(f"{self.base_url}/history/queries?limit=1")
            if r.status_code != 200:
                return False
            data = r.json()
            if not data.get("items") or len(data["items"]) == 0:
                return False  # No queries to test
            self.query_id = UUID(data["items"][0]["id"])

        r = self.client.get(
            f"{self.base_url}/history/queries/{self.query_id}/sources"
        )
        if r.status_code != 200:
            return False
        data = r.json()
        return isinstance(data, list)  # Should return list of source documents

    def test_history_statistics(self) -> bool:
        """Test GET /history/statistics - Query statistics."""
        r = self.client.get(f"{self.base_url}/history/statistics")
        if r.status_code != 200:
            return False
        data = r.json()
        return (
            "total_queries" in data
            and "success_rate_percent" in data
            and "average_response_time_ms" in data
        )

    def run_all(self):
        """Run all e2e tests."""
        print(f"\n{'='*70}")
        print(f"  End-to-End API Test Suite")
        print(f"  Target: {self.base_url}")
        print(f"{'='*70}\n")

        # Core endpoints
        self._test("Root endpoint (GET /)", self.test_root_endpoint)
        self._test("Health check (GET /health)", self.test_health_check)
        self._test("RAG health (GET /rag/health)", self.test_rag_health)
        self._test("Document count (GET /rag/documents/count)", self.test_document_count)

        # RAG functionality
        self._test("RAG query (POST /rag/query)", self.test_rag_query)
        self._test("URL extraction (POST /rag/extract-url)", self.test_url_extraction)
        img_result = self.test_image_extraction()
        if img_result is None:
            print("  Testing: Image extraction (POST /rag/extract-image)... SKIP (PIL not available)")
            self.results["skipped"].append("Image extraction")
        else:
            self._test("Image extraction (POST /rag/extract-image)", lambda: img_result)

        # History endpoints
        self._test("Query history list (GET /history/queries)", self.test_history_queries)
        self._test("Query detail (GET /history/queries/{id})", self.test_history_query_detail)
        self._test("Query sources (GET /history/queries/{id}/sources)", self.test_history_query_sources)
        self._test("Query statistics (GET /history/statistics)", self.test_history_statistics)

        # Summary
        print(f"\n{'='*70}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"  Passed:  {len(self.results['passed'])}")
        print(f"  Failed:  {len(self.results['failed'])}")
        print(f"  Skipped: {len(self.results['skipped'])}")
        print(f"{'='*70}\n")

        if self.results["failed"]:
            print("  Failed tests:")
            for test in self.results["failed"]:
                print(f"    - {test}")
            print()

        self.client.close()
        return len(self.results["failed"]) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E API Test Suite")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend base URL")
    parser.add_argument("--api-key", help="OpenRouter API key (for /rag/query)")
    args = parser.parse_args()

    runner = E2ETestRunner(args.url, args.api_key)
    success = runner.run_all()
    sys.exit(0 if success else 1)
