from fastapi import APIRouter, Depends, Header, UploadFile, File, HTTPException
from typing import Optional
from src.dependencies import get_rag_service
from src.schemas import (
    DocumentCountResponse,
    HealthStatusResponse,
    QueryRequest,
    QueryResponse,
    URLExtractRequest,
    URLExtractResponse,
    ImageExtractResponse,
)

from .services import RAGService
from .input_processing import extract_text_from_url, extract_text_from_image

rag_router = APIRouter(prefix="/rag", tags=["RAG"])


@rag_router.post("/query", response_model=QueryResponse)
async def query(
    query_request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    x_openrouter_api_key: Optional[str] = Header(None, alias="X-OpenRouter-API-Key"),
) -> QueryResponse:
    """
    Query the RAG system with a text query.

    Args:
        query_request: The query request containing the query text and top_k
        rag_service: The RAG service dependency

    Returns:
        QueryResponse: The response containing chat response and source documents
    """
    result = rag_service.query(query_request, api_key=x_openrouter_api_key)
    return result


@rag_router.post("/extract-url", response_model=URLExtractResponse)
async def extract_url(request: URLExtractRequest) -> URLExtractResponse:
    """
    Extract text content from a URL (article, webpage).
    Returns extracted text that can be reviewed before querying.
    """
    try:
        result = extract_text_from_url(request.url)
        return URLExtractResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@rag_router.post("/extract-image", response_model=ImageExtractResponse)
async def extract_image(
    file: UploadFile = File(..., description="Image file to extract text from"),
) -> ImageExtractResponse:
    """
    Extract text from an uploaded image using OCR.
    Returns extracted text that can be reviewed before querying.
    """
    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(allowed_types)}",
        )

    # Limit file size (10MB)
    max_size = 10 * 1024 * 1024
    image_bytes = await file.read()
    if len(image_bytes) > max_size:
        raise HTTPException(status_code=400, detail="Image file too large. Maximum size is 10MB.")

    try:
        extracted_text = extract_text_from_image(image_bytes)
        return ImageExtractResponse(extracted_text=extracted_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@rag_router.get("/health", response_model=HealthStatusResponse)
async def get_health_status(
    include_index: bool = False,
    rag_service: RAGService = Depends(get_rag_service),
) -> HealthStatusResponse:
    """
    Get health status of the RAG system components.

    Args:
        include_index: Whether to include vector store index status in health check
        rag_service: The RAG service dependency

    Returns:
        HealthStatusResponse: Health status of vector store, embedding model, and chat model
    """
    health_status = rag_service.get_health_status(include_index=include_index)
    return HealthStatusResponse(
        vector_store=health_status.get("vector_store", False),
        embedding_model=health_status.get("embedding_model", False),
        chat_model=health_status.get("chat_model", False),
        index_status=health_status.get("index_status") if include_index else None,
    )


@rag_router.get("/documents/count", response_model=DocumentCountResponse)
async def get_document_count(
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentCountResponse:
    """
    Get the total number of documents in the vector store.

    Args:
        rag_service: The RAG service dependency

    Returns:
        DocumentCountResponse: Total document count
    """
    count = rag_service.get_document_count()
    return DocumentCountResponse(
        document_count=count, message=f"Vector store contains {count} documents"
    )