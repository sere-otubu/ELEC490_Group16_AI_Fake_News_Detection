"""
Python package for Retrieval-Augmented Generation (RAG) functionalities.
"""

from .repositories import RAGRepository, get_rag_repository_instance # data access layer for RAG
from .services import RAGService # business logic layer for RAG

# Only expose the RAGService and RAGRepository at the package level
__all__ = ["RAGService", "RAGRepository", "get_rag_repository_instance"]