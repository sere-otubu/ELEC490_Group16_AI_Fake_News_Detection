"""
Tests for FastAPI endpoint contracts.

Validates that all API endpoints return correct status codes and response
structures. Uses the mocked TestClient from conftest.py (no real DB/LLM).
"""

import pytest
from uuid import uuid4


# ── Root & Health ────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/").json()
        assert "message" in data
        assert "description" in data
        assert "version" in data
        assert "endpoints" in data
        assert isinstance(data["endpoints"], dict)


class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data


# ── RAG Endpoints ────────────────────────────────────────────────────

class TestRAGQueryEndpoint:
    def test_valid_query_returns_200(self, client):
        response = client.post(
            "/rag/query", json={"query": "Is ibuprofen safe?", "top_k": 3}
        )
        assert response.status_code == 200

    def test_response_has_chat_and_sources(self, client):
        data = client.post(
            "/rag/query", json={"query": "test claim"}
        ).json()
        assert "chat_response" in data
        assert "source_documents" in data

    def test_missing_query_returns_422(self, client):
        response = client.post("/rag/query", json={})
        assert response.status_code == 422

    def test_invalid_top_k_returns_422(self, client):
        response = client.post(
            "/rag/query", json={"query": "test", "top_k": 0}
        )
        assert response.status_code == 422

    def test_top_k_too_high_returns_422(self, client):
        response = client.post(
            "/rag/query", json={"query": "test", "top_k": 100}
        )
        assert response.status_code == 422


class TestRAGExtractUrlEndpoint:
    def test_valid_url_returns_200(self, client):
        from unittest.mock import patch

        mock_result = {
            "extracted_text": "Article text here",
            "source_url": "https://example.com",
            "page_title": "Example",
        }
        with patch(
            "src.rag.routes.extract_text_from_url", return_value=mock_result
        ):
            response = client.post(
                "/rag/extract-url", json={"url": "https://example.com"}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["extracted_text"] == "Article text here"

    def test_invalid_url_returns_400(self, client):
        from unittest.mock import patch

        with patch(
            "src.rag.routes.extract_text_from_url",
            side_effect=ValueError("Could not connect"),
        ):
            response = client.post(
                "/rag/extract-url", json={"url": "https://invalid.test"}
            )
        assert response.status_code == 400

    def test_missing_url_returns_422(self, client):
        response = client.post("/rag/extract-url", json={})
        assert response.status_code == 422


class TestRAGExtractImageEndpoint:
    def test_invalid_content_type_returns_400(self, client):
        response = client.post(
            "/rag/extract-image",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_missing_file_returns_422(self, client):
        response = client.post("/rag/extract-image")
        assert response.status_code == 422


class TestRAGHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/rag/health")
        assert response.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/rag/health").json()
        assert "vector_store" in data
        assert "embedding_model" in data
        assert "chat_model" in data


class TestDocumentCountEndpoint:
    def test_returns_200(self, client):
        response = client.get("/rag/documents/count")
        assert response.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/rag/documents/count").json()
        assert "document_count" in data
        assert "message" in data
        assert data["document_count"] == 100


# ── History Endpoints ────────────────────────────────────────────────

class TestHistoryQueriesEndpoint:
    def test_returns_200(self, client):
        response = client.get("/history/queries")
        assert response.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/history/queries").json()
        assert "items" in data
        assert "total_count" in data
        assert "limit" in data
        assert "offset" in data

    def test_pagination_params(self, client):
        response = client.get("/history/queries?limit=5&offset=10")
        assert response.status_code == 200


class TestHistoryQueryByIdEndpoint:
    def test_nonexistent_query_returns_404(self, client):
        fake_id = str(uuid4())
        response = client.get(f"/history/queries/{fake_id}")
        assert response.status_code == 404

    def test_invalid_uuid_returns_422(self, client):
        response = client.get("/history/queries/not-a-uuid")
        assert response.status_code == 422


class TestHistorySourceDocumentsEndpoint:
    def test_nonexistent_query_returns_404(self, client):
        fake_id = str(uuid4())
        response = client.get(f"/history/queries/{fake_id}/sources")
        assert response.status_code == 404


class TestHistoryStatisticsEndpoint:
    def test_returns_200(self, client):
        response = client.get("/history/statistics")
        assert response.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/history/statistics").json()
        assert "total_queries" in data
        assert "successful_queries" in data
        assert "success_rate_percent" in data
