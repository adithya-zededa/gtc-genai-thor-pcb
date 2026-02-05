"""Database initialization and connection management.

Provides SQLite database setup, migrations, and connection pooling.
"""

from .connection import get_db_connection, init_db, ensure_database_directory
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
