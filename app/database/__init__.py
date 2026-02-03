"""Database initialization and connection management.

Provides SQLite database setup, migrations, and connection pooling.
"""

from .connection import get_db_connection, init_db, ensure_database_directory
from .models import DetectionLog, User, ConfigHistory, LogSettings
from .repositories import (
    DetectionLogRepository,
    UserRepository,
    ConfigHistoryRepository,
    LogSettingsRepository,
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
    # Repositories
    "DetectionLogRepository",
    "UserRepository",
    "ConfigHistoryRepository",
    "LogSettingsRepository",
]
