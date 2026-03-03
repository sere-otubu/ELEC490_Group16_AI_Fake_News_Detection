from .config import settings
from .schemas import QueryRequest, QueryResponse

__all__ = [
	"QueryRequest",
	"QueryResponse",
	"RAGService",
	"get_rag_service",
	"settings",
]


def __getattr__(name: str):
	if name == "RAGService":
		from .rag.services import RAGService

		return RAGService
	if name == "get_rag_service":
		from .dependencies import get_rag_service

		return get_rag_service
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")