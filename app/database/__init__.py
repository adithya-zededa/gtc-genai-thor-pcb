"""Database initialization and connection management.

Provides SQLite database setup, migrations, connection pooling,
and base repository pattern for data access.
"""

from .base_repository import BaseRepository
from .connection import ensure_database_directory, get_db_connection, init_db
from .models import (
    ConfigHistory,
    DetectionLog,
    Invoice,
    LogSettings,
    PCBDefect,
    User,
)
from .repositories import (
    ConfigHistoryRepository,
    DetectionLogRepository,
    InvoiceRepository,
    LogSettingsRepository,
    PCBDefectRepository,
    UserRepository,
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
    "Invoice",
    "PCBDefect",
    # Repositories
    "DetectionLogRepository",
    "UserRepository",
    "ConfigHistoryRepository",
    "LogSettingsRepository",
    "InvoiceRepository",
    "PCBDefectRepository",
]
