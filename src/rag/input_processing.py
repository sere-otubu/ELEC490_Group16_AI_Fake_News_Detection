"""
Input processing services for multi-modal input.

Handles extracting text from URLs (articles) and images (OCR).
The extracted text can then be sent through the existing RAG pipeline.
"""

import logging
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def extract_text_from_url(url: str) -> dict[str, str]:
    """
    Fetch and extract the main text content from a URL.

    Args:
        url: The URL of the article/page to extract text from.

    Returns:
        dict with keys: extracted_text, source_url, page_title
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract page title
        page_title = ""
        if soup.title and soup.title.string:
            page_title = soup.title.string.strip()

        # Remove unwanted elements
        for tag in soup(
            ["script", "style", "nav", "footer", "header", "aside",
             "iframe", "noscript", "form", "button", "input", "select",
             "textarea", "meta", "link"]
        ):
            tag.decompose()

        # Remove common ad/sidebar class patterns
        for element in soup.find_all(
            class_=re.compile(
                r"(ad|sidebar|comment|social|share|related|popup|modal|cookie|banner|menu|nav)",
                re.IGNORECASE,
            )
        ):
            element.decompose()

        # Try to find the main article content
        article_content = None

        # Look for common article containers
        for selector in ["article", "main", "[role='main']", ".article-body",
                         ".post-content", ".entry-content", ".story-body",
                         "#article-body", ".article-content"]:
            article_content = soup.select_one(selector)
            if article_content:
                break

        # Fall back to body if no article container found
        if not article_content:
            article_content = soup.body or soup

        # Extract text from paragraphs for cleaner output
        tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
        paragraphs = article_content.find_all(tags)
        
        if paragraphs:
            text_parts = []
            for p in paragraphs:
                # If it's an LI and it contains other block tags, skip it 
                # (because those blocks are also in paragraphs list and will be processed separately)
                if p.name == 'li' and p.find(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
                    continue
                
                # Use separator to avoid joining text without spaces (e.g. "HeaderContent")
                text = p.get_text(separator=" ", strip=True)
                if len(text) > 20:  # Skip very short fragments
                    text_parts.append(text)
            extracted_text = "\n\n".join(text_parts)
        else:
            # Fallback: get all text
            extracted_text = article_content.get_text(separator="\n", strip=True)

        # Clean up whitespace
        extracted_text = re.sub(r"\n{3,}", "\n\n", extracted_text).strip()

        if not extracted_text:
            raise ValueError("No text content could be extracted from the URL.")

        # Truncate very long articles to avoid overwhelming the LLM
        max_chars = 8000
        if len(extracted_text) > max_chars:
            extracted_text = extracted_text[:max_chars] + "\n\n[Article truncated...]"

        logger.info(
            f"Extracted {len(extracted_text)} chars from URL: {url}"
        )

        return {
            "extracted_text": extracted_text,
            "source_url": url,
            "page_title": page_title,
        }

    except requests.exceptions.Timeout:
        raise ValueError(f"Request to {url} timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise ValueError(f"Could not connect to {url}. Please check the URL.")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error fetching {url}: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Error extracting text from URL {url}: {e}")
        raise ValueError(f"Failed to extract text from URL: {str(e)}")


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from an image using OCR (pytesseract).

    Args:
        image_bytes: Raw bytes of the image file.

    Returns:
        The extracted text string.
    """
    try:
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")

        try:
            import pytesseract

            extracted_text = pytesseract.image_to_string(image).strip()
        except ImportError:
            raise ValueError(
                "pytesseract is not installed. "
                "Install it with: pip install pytesseract "
                "and ensure Tesseract OCR is installed on the system."
            )
        except Exception as e:
            raise ValueError(f"OCR processing failed: {str(e)}")

        if not extracted_text:
            raise ValueError(
                "No text could be detected in the image. "
                "Please ensure the image contains readable text."
            )

        logger.info(f"Extracted {len(extracted_text)} chars from image via OCR")
        return extracted_text

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        raise ValueError(f"Failed to process image: {str(e)}")
