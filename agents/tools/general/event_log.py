"""Log-event tool implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from core.logging import get_logger

logger = get_logger(__name__)


def tool_log_event(
    event_type: str,
    description: str,
    severity: str = "info",
    details: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Log an event to the persistent database.

    Records the event in the ``detection_logs`` table and also logs it via
    the application logger so it appears both in the database and in the
    console / log file for operational visibility.
    """
    try:
        from app.database import DetectionLogRepository

        log_id = DetectionLogRepository.create(
            timestamp=datetime.now().isoformat(),
            confidence=0.0,
            response=description,
            reason=event_type,
            vision_description=description,
            decision_details=details or {},
        )

        logger.info(
            "Event logged: id=%d type=%s severity=%s desc=%s",
            log_id, event_type, severity, description[:120],
        )

        return {
            "success": True,
            "message": f"Event logged (type: {event_type}, severity: {severity})",
            "data": {
                "log_id": log_id,
                "event_type": event_type,
                "severity": severity,
            },
        }
    except Exception as e:
        logger.error("Failed to log event (type=%s): %s", event_type, e)
        return {"success": False, "message": f"Failed to log event: {e}"}
