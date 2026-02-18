"""Notification preference management for the PCB monitoring agent.

Provides configurable, chat-controllable notification settings:
- Enable / disable email notifications
- Set severity threshold for notifications
- Configure recipient lists
- Per-channel preferences (email, with future extensibility)

Preferences are persisted to a SQLite table so they survive restarts.
Both the monitoring loop and the chat interface access these preferences
through a thread-safe singleton.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)

_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class NotificationPreferences:
    """User notification preferences."""

    email_enabled: bool = True
    email_recipients: List[str] = field(default_factory=list)
    min_severity: str = "medium"  # low, medium, high
    notify_on_threshold_exceeded: bool = True
    quiet_hours_start: Optional[str] = None   # HH:MM format
    quiet_hours_end: Optional[str] = None
    daily_digest_enabled: bool = False
    daily_digest_time: str = "08:00"
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email_enabled": self.email_enabled,
            "email_recipients": self.email_recipients,
            "min_severity": self.min_severity,
            "notify_on_threshold_exceeded": self.notify_on_threshold_exceeded,
            "quiet_hours_start": self.quiet_hours_start,
            "quiet_hours_end": self.quiet_hours_end,
            "daily_digest_enabled": self.daily_digest_enabled,
            "daily_digest_time": self.daily_digest_time,
            "updated_at": self.updated_at,
        }

    def should_notify(self, severity: str = "medium") -> bool:
        """Determine if a notification should be sent for a given severity.

        Checks:
        1. Email notifications are enabled
        2. Severity meets minimum threshold
        3. Not in quiet hours (if configured)
        """
        if not self.email_enabled:
            return False

        sev_level = _SEVERITY_ORDER.get(severity.lower(), 0)
        min_level = _SEVERITY_ORDER.get(self.min_severity.lower(), 1)
        if sev_level < min_level:
            return False

        # Check quiet hours
        if self.quiet_hours_start and self.quiet_hours_end:
            try:
                now = datetime.now().strftime("%H:%M")
                if self.quiet_hours_start <= self.quiet_hours_end:
                    if self.quiet_hours_start <= now <= self.quiet_hours_end:
                        return False
                else:
                    # Wraps midnight
                    if now >= self.quiet_hours_start or now <= self.quiet_hours_end:
                        return False
            except (ValueError, TypeError):
                pass

        return True


# ---------------------------------------------------------------------------
# Persistence layer
# ---------------------------------------------------------------------------

def _ensure_table() -> None:
    """Create the notification_preferences table if it doesn't exist."""
    from app.database.connection import get_db_connection

    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notification_preferences (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                email_enabled INTEGER DEFAULT 1,
                email_recipients TEXT DEFAULT '[]',
                min_severity TEXT DEFAULT 'medium',
                notify_on_threshold_exceeded INTEGER DEFAULT 1,
                quiet_hours_start TEXT,
                quiet_hours_end TEXT,
                daily_digest_enabled INTEGER DEFAULT 0,
                daily_digest_time TEXT DEFAULT '08:00',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Seed default row if missing
        conn.execute("""
            INSERT OR IGNORE INTO notification_preferences (id)
            VALUES (1)
        """)
        conn.commit()


def load_preferences() -> NotificationPreferences:
    """Load notification preferences from the database."""
    _ensure_table()
    from app.database.connection import get_db_connection

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM notification_preferences WHERE id = 1"
        ).fetchone()

    if row is None:
        return NotificationPreferences()

    recipients = []
    try:
        recipients = json.loads(row["email_recipients"] or "[]")
    except (json.JSONDecodeError, TypeError):
        pass

    return NotificationPreferences(
        email_enabled=bool(row["email_enabled"]),
        email_recipients=recipients,
        min_severity=row["min_severity"] or "medium",
        notify_on_threshold_exceeded=bool(row["notify_on_threshold_exceeded"]),
        quiet_hours_start=row["quiet_hours_start"],
        quiet_hours_end=row["quiet_hours_end"],
        daily_digest_enabled=bool(row["daily_digest_enabled"]),
        daily_digest_time=row["daily_digest_time"] or "08:00",
        updated_at=row["updated_at"],
    )


def save_preferences(prefs: NotificationPreferences) -> None:
    """Persist notification preferences to the database."""
    _ensure_table()
    from app.database.connection import get_db_connection

    prefs.updated_at = datetime.now().isoformat()

    with get_db_connection() as conn:
        conn.execute("""
            UPDATE notification_preferences SET
                email_enabled = ?,
                email_recipients = ?,
                min_severity = ?,
                notify_on_threshold_exceeded = ?,
                quiet_hours_start = ?,
                quiet_hours_end = ?,
                daily_digest_enabled = ?,
                daily_digest_time = ?,
                updated_at = ?
            WHERE id = 1
        """, (
            int(prefs.email_enabled),
            json.dumps(prefs.email_recipients),
            prefs.min_severity,
            int(prefs.notify_on_threshold_exceeded),
            prefs.quiet_hours_start,
            prefs.quiet_hours_end,
            int(prefs.daily_digest_enabled),
            prefs.daily_digest_time,
            prefs.updated_at,
        ))
        conn.commit()

    logger.info(
        "Notification preferences saved: enabled=%s, min_severity=%s, recipients=%d",
        prefs.email_enabled, prefs.min_severity, len(prefs.email_recipients),
    )


# ---------------------------------------------------------------------------
# Thread-safe singleton access
# ---------------------------------------------------------------------------

_preferences: Optional[NotificationPreferences] = None
_prefs_lock = threading.Lock()


def get_notification_preferences() -> NotificationPreferences:
    """Return the current notification preferences (cached singleton)."""
    global _preferences
    if _preferences is None:
        with _prefs_lock:
            if _preferences is None:
                _preferences = load_preferences()
    return _preferences


def update_notification_preferences(**kwargs: Any) -> NotificationPreferences:
    """Update specific notification preference fields.

    Accepted keyword arguments match NotificationPreferences fields.
    Returns the updated preferences.
    """
    global _preferences
    with _prefs_lock:
        prefs = get_notification_preferences()

        for key, value in kwargs.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)

        save_preferences(prefs)
        _preferences = prefs

    return prefs


def reset_preferences_cache() -> None:
    """Force a refresh of cached preferences from the DB."""
    global _preferences
    with _prefs_lock:
        _preferences = None
