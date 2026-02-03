"""Database repositories for data access operations.

Each repository handles CRUD operations for a specific domain model.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from .connection import get_db_connection
from .models import DetectionLog, User, ConfigHistory, LogSettings

logger = get_logger(__name__)

# Default log settings
DEFAULT_LOG_SETTINGS = {
    "log_level": "INFO",
    "log_retention": 30,
    "max_log_size": 100,
    "log_to_file": True,
    "log_to_console": True,
    "log_database": False,
}


class UserRepository:
    """Repository for User data operations."""
    
    @staticmethod
    def get_all(active_only: bool = True) -> List[User]:
        """Get all users, optionally filtering by active status."""
        with get_db_connection() as conn:
            query = "SELECT * FROM users"
            if active_only:
                query += " WHERE active = 1"
            query += " ORDER BY created_at DESC"
            rows = conn.execute(query).fetchall()
        return [User.from_row(row) for row in rows]
    
    @staticmethod
    def get_by_id(user_id: int) -> Optional[User]:
        """Get a user by ID."""
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return User.from_row(row)
    
    @staticmethod
    def create(email: str, name: str, role: str = "user") -> int:
        """Create a new user and return the ID."""
        with get_db_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (email, name, role) VALUES (?, ?, ?)",
                (email, name, role),
            )
            conn.commit()
            return cursor.lastrowid
    
    @staticmethod
    def update(user_id: int, email: str, name: str, role: str = "user") -> bool:
        """Update an existing user."""
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE users SET email = ?, name = ?, role = ? WHERE id = ?",
                (email, name, role, user_id),
            )
            conn.commit()
        return True
    
    @staticmethod
    def deactivate(user_id: int) -> bool:
        """Soft delete a user by setting active = 0."""
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE users SET active = 0 WHERE id = ?", (user_id,)
            )
            conn.commit()
        return True
    
    @staticmethod
    def get_active_emails() -> List[str]:
        """Get email addresses of all active users."""
        with get_db_connection() as conn:
            rows = conn.execute(
                'SELECT email FROM users WHERE active = 1 AND email IS NOT NULL AND email != ""'
            ).fetchall()
        return [row["email"].strip() for row in rows if row["email"]]


class DetectionLogRepository:
    """Repository for DetectionLog data operations."""
    
    @staticmethod
    def get_paginated(
        page: int = 1,
        per_page: int = 50,
        detected_only: bool = False,
    ) -> tuple[List[DetectionLog], int]:
        """Get paginated detection logs.
        
        Returns:
            Tuple of (logs list, total count).
        """
        offset = (page - 1) * per_page
        
        with get_db_connection() as conn:
            # Get total count
            count_query = "SELECT COUNT(*) FROM detection_logs"
            if detected_only:
                count_query += " WHERE confidence > 0"
            total_count = conn.execute(count_query).fetchone()[0]
            
            # Get paginated results
            query = """
                SELECT id, timestamp, confidence, response, image_path,
                       frame_number, reason, vision_description, decision_details, tool_trace
                FROM detection_logs
            """
            if detected_only:
                query += " WHERE confidence > 0"
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            
            rows = conn.execute(query, (per_page, offset)).fetchall()
        
        return [DetectionLog.from_row(row) for row in rows], total_count
    
    @staticmethod
    def get_by_id(log_id: int) -> Optional[DetectionLog]:
        """Get a detection log by ID."""
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM detection_logs WHERE id = ?", (log_id,)
            ).fetchone()
        return DetectionLog.from_row(row)
    
    @staticmethod
    def create(
        timestamp: str,
        confidence: float,
        response: str,
        image_path: str = "",
        frame_number: Optional[int] = None,
        reason: str = "",
        vision_description: str = "",
        decision_details: Optional[Dict] = None,
        tool_trace: Optional[List] = None,
    ) -> int:
        """Create a new detection log entry and return the ID."""
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO detection_logs (
                    timestamp, confidence, response, image_path, frame_number,
                    reason, vision_description, decision_details, tool_trace
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    confidence,
                    response,
                    image_path,
                    frame_number,
                    reason,
                    vision_description,
                    json.dumps(decision_details or {}),
                    json.dumps(tool_trace or []),
                ),
            )
            conn.commit()
            return cursor.lastrowid
    
    @staticmethod
    def delete(log_id: int) -> bool:
        """Delete a specific log entry."""
        with get_db_connection() as conn:
            conn.execute("DELETE FROM detection_logs WHERE id = ?", (log_id,))
            conn.commit()
        return True
    
    @staticmethod
    def delete_all() -> bool:
        """Delete all detection logs."""
        with get_db_connection() as conn:
            conn.execute("DELETE FROM detection_logs")
            conn.commit()
        return True
    
    @staticmethod
    def get_all_for_export() -> List[DetectionLog]:
        """Get all logs for export."""
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, confidence, response, image_path, frame_number,
                       reason, vision_description, decision_details, tool_trace
                FROM detection_logs
                ORDER BY timestamp DESC
                """
            ).fetchall()
        return [DetectionLog.from_row(row) for row in rows]


class ConfigHistoryRepository:
    """Repository for ConfigHistory data operations."""
    
    @staticmethod
    def create(config_type: str, changes: str, user_email: str = "") -> int:
        """Record a configuration change."""
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO config_history (config_type, changes, user_email)
                VALUES (?, ?, ?)
                """,
                (config_type, changes, user_email),
            )
            conn.commit()
            return cursor.lastrowid


class LogSettingsRepository:
    """Repository for LogSettings data operations."""
    
    @staticmethod
    def get() -> LogSettings:
        """Get current log settings."""
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT log_level, log_retention, max_log_size, log_to_file, log_to_console, log_database
                FROM log_settings
                WHERE id = 1
                """
            ).fetchone()
        return LogSettings.from_row(row)
    
    @staticmethod
    def update(settings: LogSettings) -> LogSettings:
        """Update log settings."""
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE log_settings
                SET log_level = ?,
                    log_retention = ?,
                    max_log_size = ?,
                    log_to_file = ?,
                    log_to_console = ?,
                    log_database = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
                """,
                (
                    settings.log_level,
                    settings.log_retention,
                    settings.max_log_size,
                    int(settings.log_to_file),
                    int(settings.log_to_console),
                    int(settings.log_database),
                ),
            )
            conn.commit()
        return settings
