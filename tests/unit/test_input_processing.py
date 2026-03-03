"""
Tests for input processing (URL extraction and OCR).

Uses mocks for network requests and OCR to run offline.
"""

from unittest.mock import MagicMock, patch

import pytest


# Helper: patch both the SSRF validator (needs DNS) and requests.get
_PATCH_SSRF = patch("src.rag.input_processing._validate_url_target")


# ── URL Extraction ───────────────────────────────────────────────────

class TestExtractTextFromUrl:
    @_PATCH_SSRF
    @patch("src.rag.input_processing.requests.get")
    def test_valid_article_extraction(self, mock_get, _mock_ssrf):
        """Extracts text from a well-structured HTML page."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <head><title>Test Article</title></head>
            <body>
                <article>
                    <p>This is a test paragraph with enough content to pass the length filter easily.</p>
                    <p>Another paragraph with medical information about vaccine safety and efficacy.</p>
                </article>
            </body>
        </html>
        """
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from src.rag.input_processing import extract_text_from_url

        result = extract_text_from_url("https://example.com/article")

        assert "extracted_text" in result
        assert "source_url" in result
        assert "page_title" in result
        assert result["source_url"] == "https://example.com/article"
        assert result["page_title"] == "Test Article"
        assert len(result["extracted_text"]) > 0

    @_PATCH_SSRF
    @patch("src.rag.input_processing.requests.get")
    def test_empty_page_raises(self, mock_get, _mock_ssrf):
        """Raises ValueError when page has no extractable text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from src.rag.input_processing import extract_text_from_url

        with pytest.raises(ValueError, match="No text content"):
            extract_text_from_url("https://example.com/empty")

    @_PATCH_SSRF
    @patch("src.rag.input_processing.requests.get")
    def test_timeout_raises(self, mock_get, _mock_ssrf):
        """Raises ValueError on request timeout."""
        import requests

        mock_get.side_effect = requests.exceptions.Timeout()

        from src.rag.input_processing import extract_text_from_url

        with pytest.raises(ValueError, match="timed out"):
            extract_text_from_url("https://example.com/slow")

    @_PATCH_SSRF
    @patch("src.rag.input_processing.requests.get")
    def test_connection_error_raises(self, mock_get, _mock_ssrf):
        """Raises ValueError on connection failure."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError()

        from src.rag.input_processing import extract_text_from_url

        with pytest.raises(ValueError, match="Could not connect"):
            extract_text_from_url("https://nonexistent.invalid")

    @_PATCH_SSRF
    @patch("src.rag.input_processing.requests.get")
    def test_http_error_raises(self, mock_get, _mock_ssrf):
        """Raises ValueError for HTTP error status codes."""
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        from src.rag.input_processing import extract_text_from_url

        with pytest.raises(ValueError, match="HTTP error"):
            extract_text_from_url("https://example.com/missing")

    @_PATCH_SSRF
    @patch("src.rag.input_processing.requests.get")
    def test_long_article_truncated(self, mock_get, _mock_ssrf):
        """Articles longer than 8000 chars are truncated."""
        long_paragraph = "<p>" + ("A" * 200 + " ") * 50 + "</p>"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = f"<html><body><article>{long_paragraph}</article></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from src.rag.input_processing import extract_text_from_url

        result = extract_text_from_url("https://example.com/long")
        assert "[Article truncated...]" in result["extracted_text"]

    # ── SSRF Protection Tests ─────────────────────────────────────────

    def test_ssrf_blocks_localhost(self):
        """SSRF guard blocks localhost URLs."""
        from src.rag.input_processing import _validate_url_target

        with pytest.raises(ValueError, match="not allowed"):
            _validate_url_target("http://localhost:8080/admin")

    @patch("src.rag.input_processing.socket.getaddrinfo")
    def test_ssrf_blocks_private_ip(self, mock_dns):
        """SSRF guard blocks URLs that resolve to private IPs."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("192.168.1.1", 0)),
        ]
        from src.rag.input_processing import _validate_url_target

        with pytest.raises(ValueError, match="private/internal"):
            _validate_url_target("https://evil-redirect.com/steal")


# ── Image / OCR Extraction ───────────────────────────────────────────

class TestExtractTextFromImage:
    def test_valid_image_returns_text(self):
        """Extracts text from a valid image."""
        import io
        import sys
        from unittest.mock import MagicMock

        from PIL import Image

        # pytesseract is imported lazily inside the function, so we
        # inject a mock into sys.modules before calling it
        mock_tesseract = MagicMock()
        mock_tesseract.image_to_string.return_value = "Extracted OCR text"
        sys.modules["pytesseract"] = mock_tesseract

        try:
            from src.rag.input_processing import extract_text_from_image

            img = Image.new("RGB", (100, 30), color="white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")

            result = extract_text_from_image(buf.getvalue())
            assert result == "Extracted OCR text"
        finally:
            del sys.modules["pytesseract"]

    def test_blank_image_raises(self):
        """Raises ValueError when OCR produces no text."""
        import io
        import sys
        from unittest.mock import MagicMock

        from PIL import Image

        mock_tesseract = MagicMock()
        mock_tesseract.image_to_string.return_value = ""
        sys.modules["pytesseract"] = mock_tesseract

        try:
            from src.rag.input_processing import extract_text_from_image

            img = Image.new("RGB", (10, 10), color="white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")

            with pytest.raises(ValueError, match="No text could be detected"):
                extract_text_from_image(buf.getvalue())
        finally:
            del sys.modules["pytesseract"]

    def test_invalid_image_bytes_raises(self):
        """Raises ValueError for invalid image data."""
        from src.rag.input_processing import extract_text_from_image

        with pytest.raises(ValueError):
            extract_text_from_image(b"not an image at all")
