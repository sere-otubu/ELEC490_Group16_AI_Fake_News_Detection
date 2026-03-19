"""
Dependency injection for the history module.
"""

from .repositories import HistoryRepository
from .services import HistoryService

# Lazy singleton - only created when first accessed
_history_repository: HistoryRepository | None = None


def get_history_repository() -> HistoryRepository:
    global _history_repository
    if _history_repository is None:
        _history_repository = HistoryRepository()
    return _history_repository


def get_history_service() -> HistoryService:
    return HistoryService(repository=get_history_repository())