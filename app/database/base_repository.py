"""Base repository pattern for database access.

Provides an abstract base class with common CRUD operations,
making it easier to add new repositories and maintain consistency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository with standard CRUD interface.

    Subclasses should implement:
    - ``_table_name`` class attribute
    - ``_from_row(row)`` to convert DB rows to model objects
    - Any domain-specific query methods

    Common operations (create, get_by_id, list, delete) are provided
    as abstract methods for consistent interfaces.
    """

    _table_name: str = ""

    @staticmethod
    @abstractmethod
    def _from_row(row) -> Optional[T]:
        """Convert a database row to a model instance."""
        ...

    @staticmethod
    @abstractmethod
    def get_by_id(entity_id: int) -> Optional[T]:
        """Retrieve a single entity by its primary key."""
        ...

    @staticmethod
    @abstractmethod
    def get_all(**kwargs) -> List[T]:
        """Retrieve all entities, optionally filtered."""
        ...

    @staticmethod
    @abstractmethod
    def create(**kwargs) -> Any:
        """Create a new entity and return its ID or the entity."""
        ...

    @staticmethod
    def delete(entity_id: int) -> bool:
        """Delete an entity by ID. Override in subclasses that support deletion."""
        raise NotImplementedError(f"delete() not implemented for this repository")

    @staticmethod
    def update(entity_id: int, **kwargs) -> bool:
        """Update an entity by ID. Override in subclasses that support updates."""
        raise NotImplementedError(f"update() not implemented for this repository")
