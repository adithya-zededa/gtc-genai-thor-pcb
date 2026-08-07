"""Database initialization and connection management.

Provides SQLite database setup, migrations, connection pooling,
and the repository layer for data access.
"""

from .connection import ensure_database_directory, get_db_connection, init_db
from .models import (
    ConfigHistory,
    DetectionLog,
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
    # Models
    "DetectionLog",
    "User",
    "ConfigHistory",
    "LogSettings",
    "PCBDefect",
    "PCBFrameStore",
    "PCBInspection",
    # Repositories
    "ChatHistoryRepository",
    "DetectionLogRepository",
    "UserRepository",
    "ConfigHistoryRepository",
    "LogSettingsRepository",
    "PCBDefectRepository",
    "PCBFrameStoreRepository",
    "PCBInspectionRepository",
]
