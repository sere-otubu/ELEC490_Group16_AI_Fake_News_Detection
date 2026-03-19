"""
Dependency injection setup for RAG components.
"""

from src.history import get_history_service
from src.rag import RAGRepository, get_rag_repository_instance, RAGService

def get_rag_repository() -> RAGRepository:
    return get_rag_repository_instance()


def get_rag_service() -> RAGService:
    repository = get_rag_repository_instance()
    return RAGService(
        rag_repository=repository, history_service=get_history_service()
    )