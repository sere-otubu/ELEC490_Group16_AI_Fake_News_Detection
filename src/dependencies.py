"""
Dependency injection setup for RAG components.
"""

from src.history import get_history_service
from src.rag import RAGRepository, rag_repository, RAGService

def get_rag_repository() -> RAGRepository:
    return rag_repository


def get_rag_service() -> RAGService:
    return RAGService(
        rag_repository=rag_repository, history_service=get_history_service()
    )