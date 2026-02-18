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
    PCBFrameStore,
    PCBInspection,
    User,
)
from .repositories import (
    ChatHistoryRepository,
    ConfigHistoryRepository,
    DetectionLogRepository,
    InvoiceRepository,
    LogSettingsRepository,
    PCBDefectRepository,
    PCBFrameStoreRepository,
    PCBInspectionRepository,
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
    "PCBFrameStore",
    "PCBInspection",
    # Repositories
    "ChatHistoryRepository",
    "DetectionLogRepository",
    "UserRepository",
    "ConfigHistoryRepository",
    "LogSettingsRepository",
    "InvoiceRepository",
    "PCBDefectRepository",
    "PCBFrameStoreRepository",
    "PCBInspectionRepository",
]
