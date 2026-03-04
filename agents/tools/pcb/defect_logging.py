"""PCB defect logging tool."""

from __future__ import annotations

from typing import Any, Dict

from core.logging import get_logger
from agents.tools.validation import (
    safe_error as _safe_error,
    sanitise_severity as _sanitise_severity,
)

logger = get_logger(__name__)


def tool_log_defect(
    board_type: str = "unknown",
    defect_type: str = "",
    severity: str = "low",
    confidence: float = 0.0,
    description: str = "",
    image_path: str = "",
) -> Dict[str, Any]:
    """Record a PCB defect to the database."""
    logger.info(
        "tool_log_defect invoked (board_type=%s, defect_type=%s, severity=%s)",
        board_type, defect_type, severity,
    )

    severity = _sanitise_severity(severity)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return {"success": False, "message": "confidence must be a number between 0.0 and 1.0"}
    confidence = max(0.0, min(1.0, confidence_value))

    from services.domains.pcb.service import record_defect

    try:
        result = record_defect(
            board_type=board_type,
            defect_type=defect_type,
            severity=severity,
            confidence=confidence,
            image_path=image_path,
            description=description,
        )
        if result.get("success"):
            logger.info(
                "Defect logged: id=%s type=%s severity=%s board=%s confidence=%.2f",
                result.get("defect_id"), defect_type, severity, board_type, confidence,
            )
        else:
            logger.warning("Defect log returned failure: %s", result.get("error"))
        return result
    except Exception as exc:
        logger.error("Failed to log defect (type=%s, board=%s): %s", defect_type, board_type, exc)
        return _safe_error("Failed to log defect", exc=exc)
