"""PCB inspection domain tools for the MCP agent.

Provides thin-adapter tool handler functions used by the PCB MCP executor.
Each function validates its inputs, delegates to pcb_service / email_tools,
and returns a sanitised result dict.

Tools:
- inspect_pcb: Analyze current frame for PCB defects via VLM
- classify_board: Identify the board type from the current frame
- send_defect_alert: Send email alert for a detected defect
- log_defect: Record a defect to the database
- generate_defect_report: Produce a summary report of logged defects
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
MAX_EMAIL_LEN = 254
VALID_SEVERITIES = frozenset(["low", "medium", "high"])


def _validate_email(email: str) -> Optional[str]:
    """Return *None* if valid, or an error message."""
    if not email:
        return "Email address is required"
    if len(email) > MAX_EMAIL_LEN:
        return f"Email address too long (max {MAX_EMAIL_LEN} chars)"
    if not _EMAIL_RE.match(email):
        return f"Invalid email format: {email}"
    return None


def _validate_emails(emails: List[str]) -> Optional[str]:
    """Return *None* if every address in the list is valid, or an error."""
    if not emails:
        return "At least one recipient email is required"
    for em in emails:
        err = _validate_email(em)
        if err:
            return err
    return None


def _safe_error(internal_msg: str, *, exc: Optional[Exception] = None) -> Dict[str, Any]:
    """Return a user-safe error dict and log the internal detail."""
    if exc:
        logger.error("%s: %s", internal_msg, exc, exc_info=True)
    else:
        logger.error(internal_msg)
    return {"success": False, "message": "An internal error occurred. Please try again."}


def _sanitise_severity(raw: str) -> str:
    """Normalise severity to one of the allowed values."""
    s = raw.strip().lower() if raw else "medium"
    return s if s in VALID_SEVERITIES else "medium"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def tool_inspect_pcb(
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze the current camera frame for PCB defects.

    Uses the VLM with ``PCB_INSPECTION`` task type.
    """
    logger.info("tool_inspect_pcb invoked (query=%s)", query and query[:80])

    from services.monitoring_service import get_monitoring_service
    from agents.vlm.task_types import TaskType

    service = get_monitoring_service()
    if not service:
        return {"success": False, "message": "Monitoring service not available"}

    try:
        event = service.analyze_single_frame(
            task_type=TaskType.PCB_INSPECTION,
            custom_prompt=query or None,
        )
    except Exception as exc:
        return _safe_error("VLM PCB inspection failed", exc=exc)

    if not event:
        return {
            "success": False,
            "message": service.last_error or "PCB inspection failed — no frame available",
        }

    return {
        "success": True,
        "message": "PCB inspection complete",
        "data": {
            "detected": event.detected,
            "confidence": event.confidence,
            "description": event.vision_description,
            "should_alert": event.should_alert,
            "full_response": event.full_response,
        },
    }


def tool_classify_board() -> Dict[str, Any]:
    """Identify the board type visible in the current frame."""
    logger.info("tool_classify_board invoked")

    from services.monitoring_service import get_monitoring_service
    from services.pcb_service import classify_board_from_analysis
    from agents.vlm.task_types import TaskType
    import json

    service = get_monitoring_service()
    if not service:
        return {"success": False, "message": "Monitoring service not available"}

    try:
        event = service.analyze_single_frame(task_type=TaskType.PCB_INSPECTION)
    except Exception as exc:
        return _safe_error("VLM board classification failed", exc=exc)

    if not event:
        return {"success": False, "message": "Board classification failed — no frame"}

    analysis: Dict[str, Any] = {}
    if event.full_response and isinstance(event.full_response, str):
        try:
            analysis = json.loads(event.full_response)
            if not isinstance(analysis, dict):
                analysis = {}
        except (json.JSONDecodeError, TypeError):
            analysis = {}

    board_type = classify_board_from_analysis(analysis)

    return {
        "success": True,
        "message": f"Board identified as: {board_type}",
        "data": {
            "board_type": board_type,
            "board_markings": analysis.get("board_markings", ""),
            "confidence": event.confidence,
        },
    }


def tool_send_defect_alert(
    recipients: Optional[List[str]] = None,
    board_type: str = "unknown",
    defect_summary: str = "",
    severity: str = "medium",
    include_image: bool = True,
    image_data: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Send an email alert about a detected PCB defect."""
    logger.info(
        "tool_send_defect_alert invoked (recipients=%s, severity=%s)",
        recipients, severity,
    )

    if not recipients:
        return {"success": False, "message": "No recipients specified"}

    err = _validate_emails(recipients)
    if err:
        return {"success": False, "message": err}

    severity = _sanitise_severity(severity)

    from agents.email_tools import send_email

    subject = f"[PCB ALERT] {severity.upper()} defect on {board_type}"
    body = (
        f"PCB Defect Alert\n"
        f"================\n\n"
        f"Board Type: {board_type}\n"
        f"Severity:   {severity}\n"
        f"Time:       {datetime.now().isoformat()}\n\n"
        f"Description:\n{defect_summary}\n"
    )

    payload: Dict[str, Any] = {
        "to": recipients,
        "subject": subject,
        "body": body,
    }

    if include_image and image_data:
        payload["image_data"] = image_data
        payload["image_filename"] = f"pcb_defect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    try:
        result = send_email(payload)
        return {
            "success": True,
            "message": f"Defect alert sent to {len(recipients)} recipient(s)",
            "email_result": result,
        }
    except Exception as exc:
        return _safe_error("Failed to send defect alert", exc=exc)


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
    confidence = max(0.0, min(1.0, float(confidence)))

    from services.pcb_service import record_defect

    try:
        return record_defect(
            board_type=board_type,
            defect_type=defect_type,
            severity=severity,
            confidence=confidence,
            image_path=image_path,
            description=description,
        )
    except Exception as exc:
        return _safe_error("Failed to log defect", exc=exc)


def tool_generate_defect_report(
    board_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a summary report of all logged PCB defects."""
    logger.info("tool_generate_defect_report invoked (board_type=%s)", board_type)

    from services.pcb_service import generate_defect_report

    try:
        return generate_defect_report(board_type=board_type)
    except Exception as exc:
        return _safe_error("Failed to generate defect report", exc=exc)
