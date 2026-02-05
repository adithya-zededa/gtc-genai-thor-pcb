"""Database repositories for data access operations.

Each repository handles CRUD operations for a specific domain model.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from .connection import get_db_connection
from .models import DetectionLog, User, ConfigHistory, LogSettings, RetailCatalogItem, Invoice, PCBDefect

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


class RetailCatalogRepository:
    """Repository for RetailCatalogItem data operations."""

    @staticmethod
    def get_all() -> List[RetailCatalogItem]:
        """Get all catalog items."""
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM retail_catalog ORDER BY category, item_name"
            ).fetchall()
        return [RetailCatalogItem.from_row(row) for row in rows]

    @staticmethod
    def get_by_id(item_id: int) -> Optional[RetailCatalogItem]:
        """Get a catalog item by ID."""
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM retail_catalog WHERE id = ?", (item_id,)
            ).fetchone()
        return RetailCatalogItem.from_row(row)

    @staticmethod
    def get_by_sku(sku: str) -> Optional[RetailCatalogItem]:
        """Get a catalog item by SKU."""
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM retail_catalog WHERE sku = ?", (sku,)
            ).fetchone()
        return RetailCatalogItem.from_row(row)

    @staticmethod
    def search_by_name(query: str) -> List[RetailCatalogItem]:
        """Search catalog items by name (case-insensitive fuzzy match)."""
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM retail_catalog WHERE LOWER(item_name) LIKE ? ORDER BY item_name",
                (f"%{query.lower()}%",),
            ).fetchall()
        return [RetailCatalogItem.from_row(row) for row in rows]

    @staticmethod
    def search_by_category(category: str) -> List[RetailCatalogItem]:
        """Get all items in a given category."""
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM retail_catalog WHERE LOWER(category) = ? ORDER BY item_name",
                (category.lower(),),
            ).fetchall()
        return [RetailCatalogItem.from_row(row) for row in rows]

    @staticmethod
    def create(item_name: str, sku: str, price: float, category: str = "other") -> int:
        """Create a new catalog item and return the ID."""
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO retail_catalog (item_name, sku, price, category)
                VALUES (?, ?, ?, ?)
                """,
                (item_name, sku, price, category),
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def update(item_id: int, **fields) -> bool:
        """Update a catalog item's fields."""
        allowed = {"item_name", "sku", "price", "category"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return False

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [item_id]

        with get_db_connection() as conn:
            conn.execute(
                f"UPDATE retail_catalog SET {set_clause} WHERE id = ?",
                values,
            )
            conn.commit()
        return True

    @staticmethod
    def delete(item_id: int) -> bool:
        """Delete a catalog item."""
        with get_db_connection() as conn:
            conn.execute("DELETE FROM retail_catalog WHERE id = ?", (item_id,))
            conn.commit()
        return True

    @staticmethod
    def bulk_create(items: List[Dict[str, Any]]) -> int:
        """Bulk insert catalog items. Returns count of inserted items."""
        count = 0
        with get_db_connection() as conn:
            for item in items:
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO retail_catalog (item_name, sku, price, category)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            item.get("item_name", ""),
                            item.get("sku", ""),
                            float(item.get("price", 0.0)),
                            item.get("category", "other"),
                        ),
                    )
                    count += 1
                except Exception as e:
                    logger.warning("Skipping catalog item: %s", e)
            conn.commit()
        return count


class InvoiceRepository:
    """Repository for Invoice data operations."""

    @staticmethod
    def create(
        recipient_email: str,
        items_json: str,
        subtotal: float,
        tax: float,
        total: float,
        status: str = "draft",
    ) -> int:
        """Create a new invoice and return the ID."""
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO invoices (
                    timestamp, recipient_email, items_json,
                    subtotal, tax, total, status
                )
                VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
                """,
                (recipient_email, items_json, subtotal, tax, total, status),
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_by_id(invoice_id: int) -> Optional[Invoice]:
        """Get an invoice by ID."""
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM invoices WHERE id = ?", (invoice_id,)
            ).fetchone()
        return Invoice.from_row(row)

    @staticmethod
    def get_paginated(
        page: int = 1,
        per_page: int = 50,
        status: Optional[str] = None,
    ) -> tuple:
        """Get paginated invoices. Returns (invoices, total_count)."""
        offset = (page - 1) * per_page

        with get_db_connection() as conn:
            count_query = "SELECT COUNT(*) FROM invoices"
            data_query = "SELECT * FROM invoices"

            if status:
                count_query += " WHERE status = ?"
                data_query += " WHERE status = ?"
                params_count = (status,)
                params_data = (status, per_page, offset)
            else:
                params_count = ()
                params_data = (per_page, offset)

            total = conn.execute(count_query, params_count).fetchone()[0]
            data_query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            rows = conn.execute(data_query, params_data).fetchall()

        return [Invoice.from_row(row) for row in rows], total

    @staticmethod
    def update_status(invoice_id: int, status: str) -> bool:
        """Update an invoice's status."""
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE invoices SET status = ? WHERE id = ?",
                (status, invoice_id),
            )
            conn.commit()
        return True


class PCBDefectRepository:
    """Repository for PCBDefect data operations."""

    @staticmethod
    def create(
        board_type: str,
        defect_type: str,
        severity: str = "low",
        confidence: float = 0.0,
        image_path: str = "",
        description: str = "",
    ) -> int:
        """Record a new PCB defect and return the ID."""
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO pcb_defects (
                    timestamp, board_type, defect_type, severity,
                    confidence, image_path, description
                )
                VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
                """,
                (board_type, defect_type, severity, confidence, image_path, description),
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_by_id(defect_id: int) -> Optional[PCBDefect]:
        """Get a defect record by ID."""
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM pcb_defects WHERE id = ?", (defect_id,)
            ).fetchone()
        return PCBDefect.from_row(row)

    @staticmethod
    def get_paginated(
        page: int = 1,
        per_page: int = 50,
        board_type: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> tuple:
        """Get paginated defect records. Returns (defects, total_count)."""
        offset = (page - 1) * per_page
        conditions = []
        params: List[Any] = []

        if board_type:
            conditions.append("LOWER(board_type) = ?")
            params.append(board_type.lower())
        if severity:
            conditions.append("severity = ?")
            params.append(severity)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        with get_db_connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM pcb_defects{where}", params
            ).fetchone()[0]

            rows = conn.execute(
                f"SELECT * FROM pcb_defects{where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                params + [per_page, offset],
            ).fetchall()

        return [PCBDefect.from_row(row) for row in rows], total

    @staticmethod
    def get_by_board_type(board_type: str) -> List[PCBDefect]:
        """Get all defects for a specific board type."""
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM pcb_defects WHERE LOWER(board_type) = ? ORDER BY timestamp DESC",
                (board_type.lower(),),
            ).fetchall()
        return [PCBDefect.from_row(row) for row in rows]

    @staticmethod
    def get_summary() -> Dict[str, Any]:
        """Get a summary of all defect records."""
        with get_db_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM pcb_defects").fetchone()[0]
            by_severity = conn.execute(
                "SELECT severity, COUNT(*) as cnt FROM pcb_defects GROUP BY severity"
            ).fetchall()
            by_board = conn.execute(
                "SELECT board_type, COUNT(*) as cnt FROM pcb_defects GROUP BY board_type"
            ).fetchall()
            by_type = conn.execute(
                "SELECT defect_type, COUNT(*) as cnt FROM pcb_defects GROUP BY defect_type ORDER BY cnt DESC"
            ).fetchall()

        return {
            "total_defects": total,
            "by_severity": {row["severity"]: row["cnt"] for row in by_severity},
            "by_board_type": {row["board_type"]: row["cnt"] for row in by_board},
            "by_defect_type": {row["defect_type"]: row["cnt"] for row in by_type},
        }
