"""Database initialization and connection management.

Provides SQLite database setup, migrations, connection pooling,
and base repository pattern for data access.
"""

from .connection import get_db_connection, init_db, ensure_database_directory
from .base_repository import BaseRepository
from .models import (
    DetectionLog,
    User,
    ConfigHistory,
    LogSettings,
    RetailCatalogItem,
    Invoice,
    PCBDefect,
)
from .repositories import (
    DetectionLogRepository,
    UserRepository,
    ConfigHistoryRepository,
    LogSettingsRepository,
    RetailCatalogRepository,
    InvoiceRepository,
    PCBDefectRepository,
)

__all__ = [
    # Connection management
    "get_db_connection",
    "init_db",
    "ensure_database_directory",
    # Base repository
    "BaseRepository",
    # Models
    "DetectionLog",
    "User",
    "ConfigHistory",
    "LogSettings",
    "RetailCatalogItem",
    "Invoice",
    "PCBDefect",
    # Repositories
    "DetectionLogRepository",
    "UserRepository",
    "ConfigHistoryRepository",
    "LogSettingsRepository",
    "RetailCatalogRepository",
    "InvoiceRepository",
    "PCBDefectRepository",
]
