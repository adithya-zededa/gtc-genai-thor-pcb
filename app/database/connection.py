"""Database connection management and initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from core.config import get_config
from core.logging import get_logger
from core.utils import ensure_directory

logger = get_logger(__name__)


def ensure_database_directory() -> None:
    """Ensure the database directory exists before connecting."""
    config = get_config()
    try:
        ensure_directory(config.database.path.parent)
    except Exception as exc:
        logger.error(
            "Failed to prepare database directory %s: %s",
            config.database.path.parent,
            exc,
        )


def get_db_connection() -> sqlite3.Connection:
    """Get database connection with row factory.
    
    Returns:
        sqlite3.Connection: A new database connection.
        
    Note:
        Caller is responsible for closing the connection.
        Prefer using 'with get_db_connection() as conn:' pattern.
    """
    config = get_config()
    ensure_database_directory()
    
    conn = sqlite3.connect(config.database.path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    conn.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
    return conn


def init_db() -> None:
    """Initialize SQLite database schema.
    
    Creates all required tables and indexes if they don't exist.
    Handles schema migrations for existing databases.
    """
    ensure_database_directory()
    config = get_config()
    
    conn = sqlite3.connect(config.database.path)
    cursor = conn.cursor()
    
    # Enable WAL mode for better concurrent access
    cursor.execute("PRAGMA journal_mode=WAL")

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Detection logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            response TEXT,
            image_path TEXT,
            frame_number INTEGER,
            reason TEXT,
            vision_description TEXT,
            decision_details TEXT,
            tool_trace TEXT
        )
    """)
    
    # Create indexes for frequently queried columns
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detection_logs_timestamp 
        ON detection_logs(timestamp DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detection_logs_confidence 
        ON detection_logs(confidence)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_active 
        ON users(active)
    """)

    # Handle schema migrations for existing databases
    _apply_migrations(cursor)

    # Configuration history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS config_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            config_type TEXT,
            changes TEXT,
            user_email TEXT
        )
    """)

    # Log settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS log_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            log_level TEXT NOT NULL DEFAULT 'INFO',
            log_retention INTEGER NOT NULL DEFAULT 30,
            max_log_size INTEGER NOT NULL DEFAULT 100,
            log_to_file INTEGER NOT NULL DEFAULT 1,
            log_to_console INTEGER NOT NULL DEFAULT 1,
            log_database INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Seed default log settings
    _seed_log_settings(cursor)

    # Retail catalog table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retail_catalog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL,
            sku TEXT UNIQUE NOT NULL,
            price REAL NOT NULL DEFAULT 0.0,
            category TEXT DEFAULT 'other',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_retail_catalog_sku
        ON retail_catalog(sku)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_retail_catalog_category
        ON retail_catalog(category)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_retail_catalog_item_name
        ON retail_catalog(item_name)
    """)

    # Invoices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            recipient_email TEXT,
            items_json TEXT DEFAULT '[]',
            subtotal REAL NOT NULL DEFAULT 0.0,
            tax REAL NOT NULL DEFAULT 0.0,
            total REAL NOT NULL DEFAULT 0.0,
            status TEXT DEFAULT 'draft',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_invoices_status
        ON invoices(status)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_invoices_timestamp
        ON invoices(timestamp DESC)
    """)

    # PCB defects table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pcb_defects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            board_type TEXT DEFAULT 'unknown',
            defect_type TEXT NOT NULL,
            severity TEXT DEFAULT 'low',
            confidence REAL DEFAULT 0.0,
            image_path TEXT DEFAULT '',
            description TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pcb_defects_board_type
        ON pcb_defects(board_type)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pcb_defects_severity
        ON pcb_defects(severity)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pcb_defects_timestamp
        ON pcb_defects(timestamp DESC)
    """)

    conn.commit()
    conn.close()
    
    logger.info("Database initialized at %s", config.database.path)


def _apply_migrations(cursor: sqlite3.Cursor) -> None:
    """Apply database schema migrations for existing tables."""
    # Add vision_description column if missing
    try:
        cursor.execute(
            "ALTER TABLE detection_logs ADD COLUMN vision_description TEXT"
        )
    except sqlite3.OperationalError:
        pass

    # Add decision_details column if missing
    try:
        cursor.execute(
            "ALTER TABLE detection_logs ADD COLUMN decision_details TEXT"
        )
    except sqlite3.OperationalError:
        pass
    
    # Add tool_trace column for agentic mode
    try:
        cursor.execute(
            "ALTER TABLE detection_logs ADD COLUMN tool_trace TEXT"
        )
    except sqlite3.OperationalError:
        pass


def _seed_log_settings(cursor: sqlite3.Cursor) -> None:
    """Seed default log settings if not present."""
    from .repositories import DEFAULT_LOG_SETTINGS
    
    cursor.execute("""
        INSERT INTO log_settings (id, log_level, log_retention, max_log_size, log_to_file, log_to_console, log_database)
        SELECT 1, ?, ?, ?, ?, ?, ?
        WHERE NOT EXISTS (SELECT 1 FROM log_settings WHERE id = 1)
    """, (
        DEFAULT_LOG_SETTINGS["log_level"],
        DEFAULT_LOG_SETTINGS["log_retention"],
        DEFAULT_LOG_SETTINGS["max_log_size"],
        int(DEFAULT_LOG_SETTINGS["log_to_file"]),
        int(DEFAULT_LOG_SETTINGS["log_to_console"]),
        int(DEFAULT_LOG_SETTINGS["log_database"]),
    ))
