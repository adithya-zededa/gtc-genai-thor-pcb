"""PCB inspection domain tools for the MCP agent.

Provides tool handler functions used by the PCB MCP executor:
- inspect_pcb: Analyze current frame for PCB defects via VLM
- classify_board: Identify the board type from the current frame
- send_defect_alert: Send email alert for a detected defect
- log_defect: Record a defect to the database
- generate_defect_report: Produce a summary report of logged defects
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)


def tool_inspect_pcb(
    query: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Analyze the current camera frame for PCB defects.

    Uses the VLM with ``PCB_INSPECTION`` task type to inspect the frame.

    Args:
        query: Optional additional question to ask about the frame.

    Returns:
        Analysis result dict with defect information.
    """
    from services.monitoring_service import get_monitoring_service
    from agents.vlm.task_types import TaskType

    service = get_monitoring_service()
    if not service:
        return {"success": False, "message": "Monitoring service not available"}

    custom_prompt = query if query else None
    event = service.analyze_single_frame(
        task_type=TaskType.PCB_INSPECTION,
        custom_prompt=custom_prompt,
    )

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


def tool_classify_board(**kwargs) -> Dict[str, Any]:
    """Identify the board type visible in the current frame.

    Runs the PCB inspection prompt and extracts the ``board_type`` field.

    Returns:
        Dict with identified board type.
    """
    from services.monitoring_service import get_monitoring_service
    from services.pcb_service import classify_board_from_analysis
    from agents.vlm.task_types import TaskType
    import json

    service = get_monitoring_service()
    if not service:
        return {"success": False, "message": "Monitoring service not available"}

    event = service.analyze_single_frame(task_type=TaskType.PCB_INSPECTION)
    if not event:
        return {"success": False, "message": "Board classification failed — no frame"}

    # Parse the VLM response for structured data
    try:
        analysis = json.loads(event.full_response) if isinstance(event.full_response, str) else {}
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
    recipients: List[str],
    board_type: str = "unknown",
    defect_summary: str = "",
    severity: str = "medium",
    include_image: bool = True,
    image_data: Optional[bytes] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Send an email alert about a detected PCB defect.

    Args:
        recipients: Email addresses to alert.
        board_type: Identified board type.
        defect_summary: Human-readable defect description.
        severity: Defect severity level.
        include_image: Whether to attach the current frame.
        image_data: Raw image bytes (from execution context).

    Returns:
        Email send result dict.
    """
    from agents.email_tools import send_email

    if not recipients:
        return {"success": False, "error": "No recipients specified"}

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
    except Exception as e:
        logger.error("Failed to send defect alert: %s", e)
        return {"success": False, "error": str(e)}


def tool_log_defect(
    board_type: str = "unknown",
    defect_type: str = "",
    severity: str = "low",
    confidence: float = 0.0,
    description: str = "",
    image_path: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """Record a PCB defect to the database.

    Args:
        board_type: Board type string.
        defect_type: Defect classification (e.g. solder_bridge).
        severity: low / medium / high.
        confidence: Detection confidence 0‒1.
        description: Defect description text.
        image_path: Path to saved evidence image.

    Returns:
        Dict with defect_id and success status.
    """
    from services.pcb_service import record_defect

    return record_defect(
        board_type=board_type,
        defect_type=defect_type,
        severity=severity,
        confidence=confidence,
        image_path=image_path,
        description=description,
    )


def tool_generate_defect_report(
    board_type: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Generate a summary report of all logged PCB defects.

    Args:
        board_type: Optional filter by board type.

    Returns:
        Report dict with summary statistics and recent defects.
    """
    from services.pcb_service import generate_defect_report

    return generate_defect_report(board_type=board_type)
