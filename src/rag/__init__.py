"""
Python package for Retrieval-Augmented Generation (RAG) functionalities.
"""

from .repositories import RAGRepository, rag_repository # data access layer for RAG
from .services import RAGService # business logic layer for RAG

# Only expose the RAGService and RAGRepository at the package level
__all__ = ["RAGService", "RAGRepository", "rag_repository"]