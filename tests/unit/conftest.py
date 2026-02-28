"""
Shared test fixtures for MedCheck AI backend unit tests.

Provides a FastAPI TestClient with all external dependencies (database, LLM,
embedding model, history service) mocked out so tests run fast and offline.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Override settings to avoid needing a real .env file."""
    monkeypatch.setenv("APP_PG_HOST", "localhost")
    monkeypatch.setenv("APP_PG_PORT", "5432")
    monkeypatch.setenv("APP_PG_USER", "test")
    monkeypatch.setenv("APP_PG_PASSWORD", "test_password")
    monkeypatch.setenv("APP_PG_DATABASE", "test_db")
    monkeypatch.setenv("APP_OPENROUTER_API_KEY", "test-key")


@pytest.fixture
def mock_rag_service():
    """Create a fully mocked RAGService."""
    service = MagicMock()

    # Default: query returns a valid response
    service.query.return_value = MagicMock(
        chat_response="**Verdict**: [ACCURATE]\n\n**Reasoning**: Test.\n\n"
                       "**Confidence Score**: 0.95\n\n**Evidence**: test evidence\n\n"
                       "**Source Files**: test_doc.pdf",
        source_documents=[
            MagicMock(
                content="Sample content",
                score=0.85,
                metadata=MagicMock(file_name="test_doc.pdf", page=1, source="PubMed"),
            )
        ],
    )

    # Default: health check returns healthy
    service.get_health_status.return_value = {
        "vector_store": True,
        "embedding_model": True,
        "chat_model": True,
    }

    # Default: document count
    service.get_document_count.return_value = 100

    return service


@pytest.fixture
def mock_history_service():
    """Create a fully mocked HistoryService."""
    from src.history.schemas import (
        QueryHistoryListResponse,
        QueryStatisticsResponse,
    )

    service = MagicMock()

    service.get_query_history.return_value = QueryHistoryListResponse(
        items=[], total_count=0, limit=10, offset=0
    )
    service.get_query_by_id.return_value = None
    service.get_source_documents_for_query.return_value = []
    service.get_query_statistics.return_value = QueryStatisticsResponse(
        total_queries=5,
        successful_queries=4,
        success_rate_percent=80.0,
        average_response_time_ms=1500.0,
    )

    return service


@pytest.fixture
def client(mock_rag_service, mock_history_service):
    """
    Create a FastAPI TestClient with mocked dependencies.

    Overrides the dependency injection so no real DB, LLM, or embedding
    connections are needed.
    """
    from src.main import app
    from src.dependencies import get_rag_service
    from src.history.dependencies import get_history_service

    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    app.dependency_overrides[get_history_service] = lambda: mock_history_service

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
