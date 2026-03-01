"""
Tests for Pydantic schema validation.

Validates that all request/response schemas enforce correct types,
constraints, and defaults.
"""

import pytest
from pydantic import ValidationError

from src.schemas import (
    APIInfoResponse,
    DocumentCountResponse,
    DocumentMetadata,
    HealthCheckResponse,
    HealthStatusResponse,
    ImageExtractResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    URLExtractRequest,
    URLExtractResponse,
)


# ── QueryRequest ─────────────────────────────────────────────────────

class TestQueryRequest:
    def test_valid_minimal(self):
        req = QueryRequest(query="Is ibuprofen safe?")
        assert req.query == "Is ibuprofen safe?"
        assert req.top_k == 3  # default

    def test_valid_with_top_k(self):
        req = QueryRequest(query="test", top_k=5)
        assert req.top_k == 5

    def test_missing_query_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest()

    def test_top_k_too_low_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)

    def test_top_k_too_high_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=6)

    def test_top_k_minimum_boundary(self):
        req = QueryRequest(query="test", top_k=1)
        assert req.top_k == 1

    def test_top_k_maximum_boundary(self):
        req = QueryRequest(query="test", top_k=5)
        assert req.top_k == 5


# ── URLExtractRequest ────────────────────────────────────────────────

class TestURLExtractRequest:
    def test_valid_url(self):
        req = URLExtractRequest(url="https://example.com/article")
        assert req.url == "https://example.com/article"

    def test_missing_url_raises(self):
        with pytest.raises(ValidationError):
            URLExtractRequest()


# ── Response Models ──────────────────────────────────────────────────

class TestDocumentMetadata:
    def test_all_fields(self):
        meta = DocumentMetadata(file_name="test.pdf", page=3, source="PubMed")
        assert meta.file_name == "test.pdf"
        assert meta.page == 3
        assert meta.source == "PubMed"

    def test_optional_fields(self):
        meta = DocumentMetadata(file_name="test.pdf")
        assert meta.page is None
        assert meta.source is None


class TestSourceDocument:
    def test_valid(self):
        doc = SourceDocument(
            content="Test content",
            score=0.85,
            metadata=DocumentMetadata(file_name="test.pdf"),
        )
        assert doc.score == 0.85

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            SourceDocument(content="Test")  # missing score and metadata


class TestQueryResponse:
    def test_valid_with_documents(self):
        resp = QueryResponse(
            chat_response="Test response",
            source_documents=[
                SourceDocument(
                    content="c",
                    score=0.9,
                    metadata=DocumentMetadata(file_name="f.pdf"),
                )
            ],
        )
        assert len(resp.source_documents) == 1

    def test_default_empty_documents(self):
        resp = QueryResponse(chat_response="Test")
        assert resp.source_documents == []


class TestHealthStatusResponse:
    def test_valid(self):
        resp = HealthStatusResponse(
            vector_store=True, embedding_model=True, chat_model=False
        )
        assert resp.chat_model is False
        assert resp.index_status is None


class TestDocumentCountResponse:
    def test_valid(self):
        resp = DocumentCountResponse(document_count=42, message="42 docs")
        assert resp.document_count == 42


class TestAPIInfoResponse:
    def test_valid(self):
        resp = APIInfoResponse(
            message="Welcome",
            description="Test API",
            version="1.0.0",
            endpoints={"docs": "/docs"},
        )
        assert resp.version == "1.0.0"


class TestHealthCheckResponse:
    def test_valid(self):
        resp = HealthCheckResponse(
            status="healthy", service="Test", version="1.0.0"
        )
        assert resp.status == "healthy"


class TestURLExtractResponse:
    def test_valid(self):
        resp = URLExtractResponse(
            extracted_text="Hello world",
            source_url="https://example.com",
            page_title="Example",
        )
        assert resp.source_url == "https://example.com"

    def test_default_page_title(self):
        resp = URLExtractResponse(
            extracted_text="Hello", source_url="https://example.com"
        )
        assert resp.page_title == ""


class TestImageExtractResponse:
    def test_valid(self):
        resp = ImageExtractResponse(extracted_text="OCR text here")
        assert resp.extracted_text == "OCR text here"
