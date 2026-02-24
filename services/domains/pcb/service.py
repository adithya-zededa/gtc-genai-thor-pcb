"""PCB defect inspection service.

Provides business logic for:
- Recording defects to the database
- Board-type-specific alert filtering
- Defect report generation / summarisation
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)

_service_lock = threading.Lock()

# Board types that trigger alerts on ANY defect (configurable via env)
ALERT_BOARD_TYPES = os.getenv(
    "PCB_ALERT_BOARD_TYPES", "arduino,raspberry pi,esp32,stm32"
).lower().split(",")

# Minimum severity to trigger alert: low, medium, high
ALERT_MIN_SEVERITY = os.getenv("PCB_ALERT_MIN_SEVERITY", "medium").lower()

_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


def record_defect(
    board_type: str,
    defect_type: str,
    severity: str = "low",
    confidence: float = 0.0,
    image_path: str = "",
    description: str = "",
) -> Dict[str, Any]:
    """Persist a PCB defect record to the database.

    Args:
        board_type: Identified board type (e.g. "Arduino Uno").
        defect_type: Defect classification string.
        severity: "low", "medium", or "high".
        confidence: Detection confidence 0‒1.
        image_path: Path to the saved evidence image.
        description: Human-readable defect description.

    Returns:
        Dict with ``defect_id`` and success status.
    """
    from app.database.repositories import PCBDefectRepository

    try:
        defect_id = PCBDefectRepository.create(
            board_type=board_type,
            defect_type=defect_type,
            severity=severity,
            confidence=confidence,
            image_path=image_path,
            description=description,
        )
        logger.info(
            "Defect recorded: id=%d board=%s type=%s severity=%s "
            "confidence=%.2f image=%s desc=%s",
            defect_id, board_type, defect_type, severity,
            confidence, bool(image_path), description[:80] if description else "",
        )
        return {"success": True, "defect_id": defect_id}
    except Exception as e:
        logger.error(
            "Failed to record defect (board=%s type=%s): %s",
            board_type, defect_type, e,
        )
        return {"success": False, "error": str(e)}


def should_alert(
    board_type: str,
    defect_type: str = "",
    severity: str = "low",
) -> bool:
    """Determine whether a defect should trigger an alert.

    Alert logic:
    1. Severity must be >= ``PCB_ALERT_MIN_SEVERITY`` (env-configurable).
    2. If ``PCB_ALERT_BOARD_TYPES`` is set, the board_type (case-insensitive
       substring match) must be in the list.

    Args:
        board_type: Board type string from detection.
        defect_type: Defect classification (currently unused in filter).
        severity: "low", "medium", or "high".

    Returns:
        True if an alert should be sent.
    """
    sev_level = _SEVERITY_ORDER.get(severity.lower(), 0)
    min_level = _SEVERITY_ORDER.get(ALERT_MIN_SEVERITY, 1)

    if sev_level < min_level:
        return False

    # Check board type filter
    board_lower = board_type.lower()
    for allowed_board in ALERT_BOARD_TYPES:
        if allowed_board.strip() and allowed_board.strip() in board_lower:
            return True

    # If no board type filter configured, alert on any board
    if not any(b.strip() for b in ALERT_BOARD_TYPES):
        return True

    return False


def generate_defect_report(board_type: Optional[str] = None) -> Dict[str, Any]:
    """Generate a summary report of PCB defects.

    Args:
        board_type: Optional filter by board type.

    Returns:
        Dict with summary statistics and recent defects.
    """
    from app.database.repositories import PCBDefectRepository

    try:
        summary = PCBDefectRepository.get_summary()

        if board_type:
            recent = PCBDefectRepository.get_by_board_type(board_type)
        else:
            recent, _ = PCBDefectRepository.get_paginated(page=1, per_page=20)

        return {
            "success": True,
            "summary": summary,
            "recent_defects": [d.to_dict() for d in recent],
            "report_generated_at": datetime.now().isoformat(),
            "filter_board_type": board_type,
        }
    except Exception as e:
        logger.error("Failed to generate defect report: %s", e)
        return {"success": False, "error": str(e)}


def classify_board_from_analysis(analysis_result: Dict[str, Any]) -> str:
    """Extract board type from a VLM analysis result dict.

    Looks for ``board_type`` key in the analysis data.

    Args:
        analysis_result: Dict from VLM analysis.

    Returns:
        Board type string, or "unknown".
    """
    return analysis_result.get("board_type", "unknown")


def extract_defects_from_analysis(analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract defect list from a VLM analysis result dict.

    Args:
        analysis_result: Dict from VLM analysis.

    Returns:
        List of defect dicts.
    """
    return analysis_result.get("defects", [])
