"""PCB inspection domain tools for the MCP agent.

Provides thin-adapter tool handler functions used by the PCB MCP executor.
Each function validates its inputs, delegates to pcb_service / email_tools,
and returns a sanitised result dict.

Architecture
~~~~~~~~~~~~
Frame capture is **decoupled** from the LLM.  The proactive monitoring agent
continuously stores frames in ``pcb_frame_store`` (DB + disk) when a PCB is
detected and motion is low.  The LLM/agent accesses these stored frames via
tool calls — it never touches the camera directly.

**Automatic logging** — inspection tools that detect defects now auto-log
them to the persistent defect table.  The LLM no longer needs to issue a
separate ``log_defect`` call; it's still available for manual logging.

Tools:
- get_latest_pcb_frames: Retrieve metadata for recently stored PCB frames
- inspect_pcb_frame: Send a stored frame to the VLM and return analysis
- inspect_pcb: Analyze the *live* frame for PCB defects via VLM
- classify_board: Identify the board type from the current frame
- send_defect_alert: Send email alert (simple sender — the LLM decides)
- log_defect: Record a defect to the database
- generate_defect_report: Produce a summary report of logged defects
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from agents.tools.validation import (
    validate_emails as _validate_emails,
    safe_error as _safe_error,
    sanitise_severity as _sanitise_severity,
)

logger = get_logger(__name__)


DEFAULT_PCB_DEFECT_INSPECTION_PROMPT = (
    "Inspect this PCB image for manufacturing defects and output only JSON. "
    "This is a FAIL-focused inspection: if evidence of a defect exists, detected must be true. "
    "Mandatory connector checks (do these first): "
    "(1) DC/power barrel jack presence and integrity. Verify the connector body exists where expected and is not bent/deformed. "
    "Missing connector body, missing black plastic housing, or bent jack = DEFECT. "
    "(2) UART/header cap presence on both expected sides. Missing cap on one side = DEFECT. "
    "Do not assume parts are present from prior frames or typical board layout; use only visible evidence in this image. "
    "If the power jack region is not clearly visible or cannot be positively confirmed intact, treat as defect_suspected and set detected=true. "
    "Also check solder bridges, missing components, cold solder joints, trace damage, lifted pads, and connector misalignment. "
    "Return JSON with keys: detected (bool), confidence (0-1), reasoning (string), should_alert (bool), details (object). "
    "details must include: power_jack_status (present_intact|missing|damaged|uncertain), uart_cap_status (present_both_sides|missing_one_side|missing_both_sides|uncertain), defects (list of {type,severity,location,description}). "
    "Set should_alert=true for medium/high severity defects."
)


# ---------------------------------------------------------------------------
# Automatic defect logging helper
# ---------------------------------------------------------------------------

def _auto_log_defect(
    detected: bool,
    full_response: Any,
    description: str,
    confidence: float,
    image_path: str = "",
) -> None:
    """Automatically log defect data when an inspection finds issues.

    This removes the need for the LLM to issue a separate ``log_defect``
    tool call after every inspection — the persistence is a side-effect
    of inspection itself.  The LLM can still override or amend the log
    via the explicit ``tool_log_defect`` tool.

    Also triggers email notifications if enabled in notification preferences.
    """
    if not detected:
        return  # Nothing to log for clean inspections

    import json as _json

    analysis: Dict[str, Any] = {}
    if full_response and isinstance(full_response, str):
        try:
            analysis = _json.loads(full_response)
            if not isinstance(analysis, dict):
                analysis = {}
        except (ValueError, TypeError):
            analysis = {}

    board_type = analysis.get("board_type", "unknown")
    defect_list = analysis.get("defects", [])
    defect_type = "general_defect"
    severity = "medium"

    if defect_list and isinstance(defect_list, list):
        first = defect_list[0] if defect_list else {}
        if isinstance(first, dict):
            defect_type = first.get("type", defect_type)
            sev = first.get("severity", "").lower()
            if sev in ("low", "medium", "high"):
                severity = sev

        # Escalate to "high" if any single defect is high/critical
        for d in defect_list:
            if isinstance(d, dict) and d.get("severity", "").lower() in ("high", "critical"):
                severity = "high"
                break

    try:
        from services.domains.pcb.service import record_defect
        record_defect(
            board_type=board_type,
            defect_type=defect_type,
            severity=severity,
            confidence=confidence,
            image_path=image_path,
            description=description or "Auto-logged from VLM inspection",
        )
        logger.info("Auto-logged defect: type=%s severity=%s board=%s", defect_type, severity, board_type)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Auto-log defect failed (non-fatal): %s", exc)

    # Trigger notification if preferences allow
    _auto_notify_defect(
        board_type=board_type,
        defect_type=defect_type,
        severity=severity,
        description=description,
        confidence=confidence,
    )


def _auto_notify_defect(
    board_type: str,
    defect_type: str,
    severity: str,
    description: str,
    confidence: float,
) -> None:
    """Send an automatic email notification if preferences are enabled.

    Respects the user's notification preferences at all times:
    - Only sends if email_enabled is True
    - Only sends if severity meets min_severity threshold
    - Only sends to configured recipients
    """
    try:
        from services.domains.pcb.notification_preferences import get_notification_preferences

        prefs = get_notification_preferences()
        if not prefs.should_notify(severity):
            return

        if not prefs.email_recipients:
            logger.debug("Auto-notify: no recipients configured, skipping")
            return

        from agents.tools.email import send_email

        subject = f"[PCB ALERT] {severity.upper()} — {defect_type} on {board_type}"
        body = (
            f"Automated PCB Defect Notification\n"
            f"{'=' * 40}\n\n"
            f"Board Type:  {board_type}\n"
            f"Defect Type: {defect_type}\n"
            f"Severity:    {severity}\n"
            f"Confidence:  {confidence:.2f}\n"
            f"Time:        {datetime.now().isoformat()}\n\n"
            f"Description:\n{description}\n\n"
            f"---\n"
            f"This is an automated notification from the PCB monitoring agent.\n"
            f"Manage notification preferences via the chat interface.\n"
        )

        send_email({
            "to": prefs.email_recipients,
            "subject": subject,
            "body": body,
        })
        logger.info(
            "Auto-notification sent for %s defect to %d recipient(s)",
            severity, len(prefs.email_recipients),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Auto-notification failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def tool_inspect_pcb(
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze the current camera frame for PCB defects."""
    logger.info("tool_inspect_pcb invoked (query=%s)", query and query[:80])

    from services.core.monitoring import get_monitoring_service
    from agents.vlm.task_types import TaskType

    service = get_monitoring_service()
    if not service:
        return {"success": False, "message": "Monitoring service not available"}

    try:
        event = service.analyze_single_frame(
            task_type=TaskType.CUSTOM,
            custom_prompt=query or DEFAULT_PCB_DEFECT_INSPECTION_PROMPT,
        )
    except Exception as exc:
        return _safe_error("VLM PCB inspection failed", exc=exc)

    if not event:
        return {
            "success": False,
            "message": service.last_error or "PCB inspection failed — no frame available",
        }

    # Auto-log defect if detected (no LLM round-trip needed)
    _auto_log_defect(
        detected=event.detected,
        full_response=event.full_response,
        description=event.vision_description or "",
        confidence=event.confidence,
    )

    return {
        "success": True,
        "message": "PCB inspection complete",
        "data": {
            "detected": event.detected,
            "confidence": event.confidence,
            "description": event.vision_description,
            "should_alert": event.should_alert,
            "full_response": event.full_response,
            "auto_logged": event.detected,  # Inform LLM defect was auto-logged
        },
    }


def tool_classify_board() -> Dict[str, Any]:
    """Identify the board type visible in the current frame."""
    logger.info("tool_classify_board invoked")

    from services.core.monitoring import get_monitoring_service
    from services.domains.pcb.service import classify_board_from_analysis
    from agents.vlm.task_types import TaskType
    import json

    service = get_monitoring_service()
    if not service:
        return {"success": False, "message": "Monitoring service not available"}

    try:
        event = service.analyze_single_frame(
            task_type=TaskType.CUSTOM,
            custom_prompt="Identify the board type visible in this image. Report board_type, board_markings, and any visible text or logos. Respond with ONLY valid JSON.",
        )
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


# ---------------------------------------------------------------------------
# Stored-frame tools (decoupled from camera)
# ---------------------------------------------------------------------------

def tool_get_latest_pcb_frames(
    limit: int = 5,
    unconsumed_only: bool = True,
) -> Dict[str, Any]:
    """Retrieve metadata for the most recently stored PCB frames.

    These frames were automatically captured by the monitoring pipeline
    whenever a PCB was detected on-camera with low motion (< threshold).
    The LLM agent uses this tool to see what frames are available before
    deciding whether to inspect them.
    """
    logger.info(
        "tool_get_latest_pcb_frames invoked (limit=%d, unconsumed_only=%s)",
        limit, unconsumed_only,
    )

    from app.database import PCBFrameStoreRepository

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"success": False, "message": "limit must be an integer"}
    limit = max(1, min(50, limit))

    try:
        if unconsumed_only:
            frames = PCBFrameStoreRepository.get_latest_unconsumed(limit=limit)
        else:
            frames = PCBFrameStoreRepository.get_latest(limit=limit)
    except Exception as exc:
        return _safe_error("Failed to retrieve stored PCB frames", exc=exc)

    if not frames:
        return {
            "success": True,
            "message": "No stored PCB frames available. The camera may not have detected a PCB yet, or monitoring may not be active.",
            "data": {"frames": [], "count": 0},
        }

    frame_list = [f.to_dict() for f in frames]
    return {
        "success": True,
        "message": f"Found {len(frame_list)} stored PCB frame(s).",
        "data": {"frames": frame_list, "count": len(frame_list)},
    }


def tool_inspect_pcb_frame(
    frame_id: Optional[int] = None,
    query: Optional[str] = None,
    auto_log: bool = False,
) -> Dict[str, Any]:
    """Send a stored PCB frame to the VLM for defect inspection.

    If *frame_id* is omitted the most recent unconsumed frame is used.
    The VLM analysis result is returned to the LLM so it can reason about
    the findings and decide on next steps (e.g., send an alert, log a
    defect, or do nothing).

    By default this tool only analyses.  If ``auto_log=True``, detected
    defects are automatically persisted (and may trigger notifications
    based on notification preferences).
    """
    logger.info(
        "tool_inspect_pcb_frame invoked (frame_id=%s, query=%s)",
        frame_id, query and query[:80],
    )

    import cv2
    from app.database import PCBFrameStoreRepository
    from services.core.monitoring import get_monitoring_service
    from agents.vlm.task_types import TaskType

    # ── Resolve frame ──────────────────────────────────────────────────
    frame_record = None
    if frame_id is not None:
        try:
            frame_record = PCBFrameStoreRepository.get_by_id(int(frame_id))
        except Exception as exc:
            return _safe_error("Failed to look up frame", exc=exc)
    else:
        try:
            unconsumed = PCBFrameStoreRepository.get_latest_unconsumed(limit=1)
            frame_record = unconsumed[0] if unconsumed else None
        except Exception as exc:
            return _safe_error("Failed to retrieve latest frame", exc=exc)

    if frame_record is None:
        return {
            "success": False,
            "message": "No stored PCB frame available to inspect. Wait for the camera to detect a PCB with low motion.",
        }

    # ── Load image from disk ───────────────────────────────────────────
    try:
        frame_img = cv2.imread(frame_record.image_path)
        if frame_img is None:
            return {"success": False, "message": f"Could not read stored frame image: {frame_record.image_path}"}
    except Exception as exc:
        return _safe_error("Failed to load stored frame", exc=exc)

    # ── Run VLM analysis on the stored frame ───────────────────────────
    service = get_monitoring_service()
    if not service or not service.agent:
        return {"success": False, "message": "Monitoring service / agent not available"}

    try:
        event = service.agent.analyze_with_prompt(
            frame=frame_img,
            task_type=TaskType.CUSTOM,
            custom_prompt=query or DEFAULT_PCB_DEFECT_INSPECTION_PROMPT,
        )
    except Exception as exc:
        return _safe_error("VLM analysis of stored frame failed", exc=exc)

    if not event:
        return {"success": False, "message": "VLM returned no result for stored frame"}

    # Mark frame as consumed so it doesn't get re-inspected
    try:
        if frame_record.id is not None:
            PCBFrameStoreRepository.mark_consumed(frame_record.id)
    except Exception as exc:
        logger.warning("Failed to mark frame as consumed (frame_id=%s): %s", frame_record.id, exc)

    # Optional side-effect: persist detected defects
    if auto_log:
        _auto_log_defect(
            detected=event.detected,
            full_response=event.full_response,
            description=event.vision_description or "",
            confidence=event.confidence,
            image_path=frame_record.image_path or "",
        )

    return {
        "success": True,
        "message": "Stored PCB frame inspected by VLM",
        "data": {
            "frame_id": frame_record.id,
            "frame_timestamp": frame_record.timestamp,
            "board_signature": frame_record.board_signature,
            "detected": event.detected,
            "confidence": event.confidence,
            "description": event.vision_description,
            "should_alert": event.should_alert,
            "full_response": event.full_response,
            "image_path": frame_record.image_path,
            "auto_logged": bool(auto_log and event.detected),
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
    """Send an email alert about a detected PCB defect.

    This is a simple email sender.  The LLM agent is responsible for
    first inspecting the frame (via ``inspect_pcb_frame``), reasoning
    about the findings, and only then calling this tool if it decides
    an alert is warranted.
    """
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

    # If image_data not passed explicitly, try to load from a stored frame
    if include_image and not image_data:
        try:
            import cv2
            from app.database import PCBFrameStoreRepository
            latest = PCBFrameStoreRepository.get_latest(limit=1)
            if latest and latest[0].image_path:
                img = cv2.imread(latest[0].image_path)
                if img is not None:
                    ok, buf = cv2.imencode(".jpg", img)
                    if ok:
                        image_data = bytes(buf)
        except Exception as exc:
            logger.debug("Could not load stored frame image for alert: %s", exc)

    from agents.tools.email import send_email

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
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return {"success": False, "message": "confidence must be a number between 0.0 and 1.0"}
    confidence = max(0.0, min(1.0, confidence_value))

    from services.domains.pcb.service import record_defect

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

    from services.domains.pcb.service import generate_defect_report

    try:
        return generate_defect_report(board_type=board_type)
    except Exception as exc:
        return _safe_error("Failed to generate defect report", exc=exc)


# ---------------------------------------------------------------------------
# Continuous defect monitoring tools
# ---------------------------------------------------------------------------

def tool_start_defect_monitoring(
    recipients: Optional[List[str]] = None,
    polling_interval: float = 10.0,
    auto_alert: bool = True,
) -> Dict[str, Any]:
    """Deprecated — the monitoring loop and the LLM now handle inspection.

    The proactive monitoring loop observes the camera feed and delegates
    inspection decisions to the LLM via MCP tools.  Use
    ``start_monitoring_session`` instead.
    """
    logger.info(
        "tool_start_defect_monitoring is deprecated — use start_monitoring_session "
        "(ignored args: recipients=%s, polling_interval=%s, auto_alert=%s)",
        recipients,
        polling_interval,
        auto_alert,
    )
    return {
        "success": False,
        "message": (
            "Deprecated: continuous defect monitoring is now handled by the "
            "proactive monitoring loop.  Use `start_monitoring_session` to "
            "begin monitoring.  The LLM will decide when to inspect and alert. "
            "Provided parameters were ignored."
        ),
    }


def tool_stop_defect_monitoring() -> Dict[str, Any]:
    """Deprecated — see tool_start_defect_monitoring."""
    logger.info("tool_stop_defect_monitoring is deprecated")
    return {
        "success": True,
        "message": "Defect monitoring is managed by the proactive monitoring loop.",
    }


# ---------------------------------------------------------------------------
# PCB inspection history query (for Q&A)
# ---------------------------------------------------------------------------

def tool_query_pcb_inspections(
    limit: int = 10,
    result_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Query past PCB inspection results from the database.

    This gives you access to historical inspection data so you can
    answer user questions about detected defects, pass rates, etc.

    Parameters
    ----------
    limit : int
        Max number of records to return (default 10).
    result_filter : str, optional
        Filter by result: ``"PASS"``, ``"FAIL"``, or ``None`` for all.
    """
    logger.info(
        "tool_query_pcb_inspections invoked (limit=%d, result_filter=%s)",
        limit, result_filter,
    )

    from app.database import PCBInspectionRepository

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"success": False, "message": "limit must be an integer"}
    limit = max(1, min(50, limit))

    try:
        inspections, total = PCBInspectionRepository.get_paginated(
            page=1,
            per_page=limit,
            result=result_filter,
        )
    except Exception as exc:
        return _safe_error("Failed to query PCB inspections", exc=exc)

    records = [insp.to_dict() for insp in inspections]

    # Build a human-readable summary for the LLM
    pass_count = sum(1 for r in records if r.get("result") == "PASS")
    fail_count = sum(1 for r in records if r.get("result") == "FAIL")

    summary = (
        f"Found {total} total inspection(s). "
        f"Showing {len(records)} most recent. "
        f"Pass: {pass_count}, Fail: {fail_count}."
    )

    return {
        "success": True,
        "message": summary,
        "data": {
            "inspections": records,
            "total": total,
            "showing": len(records),
            "pass_count": pass_count,
            "fail_count": fail_count,
        },
    }


# ---------------------------------------------------------------------------
# Chat query tools — monitoring status, defect analytics, notifications
# ---------------------------------------------------------------------------

def tool_get_monitoring_status() -> Dict[str, Any]:
    """Get current and historical monitoring status including defect counts.

    Returns agent state, session info, total defects, and recent activity.
    """
    logger.info("tool_get_monitoring_status invoked")

    from services.core.monitoring import get_monitoring_service
    from services.domains.pcb.defect_store import count_defects, get_latest_defect

    service = get_monitoring_service()
    monitoring_active = False
    monitoring_mode = "idle"
    if service:
        monitoring_mode = service.get_active_monitoring_mode() if hasattr(service, "get_active_monitoring_mode") else "unknown"
        monitoring_active = monitoring_mode != "idle"

    total_defects = count_defects()
    defects_24h = count_defects(hours=24)
    defects_1h = count_defects(hours=1)
    latest = get_latest_defect()

    return {
        "success": True,
        "message": f"Monitoring is {'active' if monitoring_active else 'inactive'}. "
                   f"{defects_24h} defect(s) in the last 24h, {total_defects} total.",
        "data": {
            "monitoring_active": monitoring_active,
            "monitoring_mode": monitoring_mode,
            "total_defects": total_defects,
            "defects_last_24h": defects_24h,
            "defects_last_1h": defects_1h,
            "latest_defect": latest,
        },
    }


def tool_toggle_email_notifications(
    enabled: Optional[bool] = None,
    min_severity: Optional[str] = None,
    recipients: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Enable or disable email notifications for detected defects.

    Parameters
    ----------
    enabled : bool, optional
        Set to True to enable, False to disable. If omitted, toggles current state.
    min_severity : str, optional
        Minimum severity to trigger notifications: "low", "medium", "high".
    recipients : list of str, optional
        Email addresses to receive notifications.
    """
    logger.info(
        "tool_toggle_email_notifications invoked (enabled=%s, min_severity=%s)",
        enabled, min_severity,
    )

    from services.domains.pcb.notification_preferences import (
        get_notification_preferences,
        update_notification_preferences,
    )

    prefs = get_notification_preferences()
    updates = {}

    if enabled is not None:
        updates["email_enabled"] = bool(enabled)
    elif enabled is None and min_severity is None and recipients is None:
        # Toggle
        updates["email_enabled"] = not prefs.email_enabled

    if min_severity:
        sev = min_severity.strip().lower()
        if sev in ("low", "medium", "high"):
            updates["min_severity"] = sev

    if recipients is not None:
        from agents.tools.validation import validate_emails
        err = validate_emails(recipients)
        if err:
            return {"success": False, "message": err}
        updates["email_recipients"] = recipients

    if updates:
        prefs = update_notification_preferences(**updates)

    status = "enabled" if prefs.email_enabled else "disabled"
    return {
        "success": True,
        "message": f"Email notifications are now {status} "
                   f"(min severity: {prefs.min_severity}, "
                   f"recipients: {len(prefs.email_recipients)}).",
        "data": prefs.to_dict(),
    }


def tool_get_defect_summary(
    hours: Optional[float] = None,
    board_type: Optional[str] = None,
    severity: Optional[str] = None,
) -> Dict[str, Any]:
    """Get a summary of defects identified within a time range or for a board type.

    Parameters
    ----------
    hours : float, optional
        Time window in hours (e.g., 24 for last day, 168 for last week).
    board_type : str, optional
        Filter by specific board type.
    severity : str, optional
        Filter by severity level.
    """
    logger.info(
        "tool_get_defect_summary invoked (hours=%s, board_type=%s, severity=%s)",
        hours, board_type, severity,
    )

    from services.domains.pcb.defect_store import (
        get_defects_in_range,
        _time_window,
    )

    start_time = None
    if hours is not None:
        if hours <= 0:
            return {"success": False, "message": "hours must be > 0"}
        start_time, _ = _time_window(hours)

    defects = get_defects_in_range(
        start_time=start_time,
        board_type=board_type,
        severity=severity,
    )

    # Match breakdown to the same filtered defect set used for totals
    type_counts: Dict[str, int] = {}
    for defect in defects:
        defect_type = defect.get("defect_type", "unknown")
        type_counts[defect_type] = type_counts.get(defect_type, 0) + 1

    total_for_pct = len(defects) or 1
    type_breakdown = [
        {
            "defect_type": defect_type,
            "count": count,
            "percentage": round((count / total_for_pct) * 100, 2),
        }
        for defect_type, count in sorted(type_counts.items(), key=lambda item: item[1], reverse=True)
    ]

    # Severity counts
    sev_counts = {}
    for d in defects:
        s = d.get("severity", "unknown")
        sev_counts[s] = sev_counts.get(s, 0) + 1

    time_desc = f"last {hours}h" if hours else "all time"
    filters = []
    if board_type:
        filters.append(f"board={board_type}")
    if severity:
        filters.append(f"severity={severity}")
    filter_str = f" ({', '.join(filters)})" if filters else ""

    return {
        "success": True,
        "message": f"{len(defects)} defect(s) found for {time_desc}{filter_str}.",
        "data": {
            "total": len(defects),
            "time_window": time_desc,
            "severity_breakdown": sev_counts,
            "defect_types": type_breakdown,
            "recent_defects": [d for d in defects[:10]],
        },
    }


def tool_count_defective_pcbs(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Count how many defective PCBs were detected overall or within a time window.

    Parameters
    ----------
    hours : float, optional
        Time window in hours. Omit for all-time count.
    """
    logger.info("tool_count_defective_pcbs invoked (hours=%s)", hours)

    from services.domains.pcb.defect_store import count_defects

    total = count_defects(hours=hours)
    high = count_defects(hours=hours, severity="high")
    medium = count_defects(hours=hours, severity="medium")
    low = count_defects(hours=hours, severity="low")

    time_desc = f"last {hours}h" if hours else "all time"
    return {
        "success": True,
        "message": f"{total} defective PCB(s) detected ({time_desc}): "
                   f"{high} high, {medium} medium, {low} low severity.",
        "data": {
            "total": total,
            "high": high,
            "medium": medium,
            "low": low,
            "time_window": time_desc,
        },
    }


def tool_get_latest_defect() -> Dict[str, Any]:
    """Get the most recently identified defect with full details."""
    logger.info("tool_get_latest_defect invoked")

    from services.domains.pcb.defect_store import get_latest_defect

    defect = get_latest_defect()
    if not defect:
        return {
            "success": True,
            "message": "No defects have been recorded yet.",
            "data": None,
        }

    return {
        "success": True,
        "message": f"Most recent defect: {defect.get('defect_type', 'unknown')} "
                   f"(severity: {defect.get('severity', 'unknown')}) at {defect.get('timestamp', 'unknown')}.",
        "data": defect,
    }


def tool_get_defect_type_breakdown(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Get what types of defects have been identified and how frequently each occurs.

    Parameters
    ----------
    hours : float, optional
        Time window in hours. Omit for all-time breakdown.
    """
    logger.info("tool_get_defect_type_breakdown invoked (hours=%s)", hours)

    from services.domains.pcb.defect_store import get_defect_type_breakdown

    breakdown = get_defect_type_breakdown(hours=hours)
    if not breakdown:
        return {
            "success": True,
            "message": "No defects recorded yet.",
            "data": {"types": []},
        }

    time_desc = f"last {hours}h" if hours else "all time"
    summary_parts = [f"{t['defect_type']}: {t['count']} ({t['percentage']}%)" for t in breakdown[:5]]

    return {
        "success": True,
        "message": f"Defect type breakdown ({time_desc}): {'; '.join(summary_parts)}.",
        "data": {"types": breakdown, "time_window": time_desc},
    }


def tool_get_defect_trend(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Analyze whether defect rates are increasing, decreasing, or stable.

    Parameters
    ----------
    hours : float, optional
        Analysis window in hours (default: 168 = 1 week).
    """
    logger.info("tool_get_defect_trend invoked (hours=%s)", hours)

    from services.domains.pcb.defect_store import get_defect_trend

    window = hours or 168.0
    trend = get_defect_trend(window_hours=window)

    return {
        "success": True,
        "message": f"Defect trend over the last {window}h: {trend['trend'].upper()}. "
                   f"First half rate: {trend['first_half_rate']}/period, "
                   f"second half rate: {trend['second_half_rate']}/period. "
                   f"Total: {trend['total_defects']} defects.",
        "data": trend,
    }


def tool_get_most_severe_defect(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Get the most severe / highest-priority defect detected.

    Parameters
    ----------
    hours : float, optional
        Time window in hours. Omit for all-time.
    """
    logger.info("tool_get_most_severe_defect invoked (hours=%s)", hours)

    from services.domains.pcb.defect_store import get_most_severe_defect

    defect = get_most_severe_defect(hours=hours)
    if not defect:
        time_desc = f"last {hours}h" if hours else "all time"
        return {
            "success": True,
            "message": f"No defects found ({time_desc}).",
            "data": None,
        }

    return {
        "success": True,
        "message": f"Most severe defect: {defect.get('defect_type', 'unknown')} "
                   f"(severity: {defect.get('severity')}, confidence: {defect.get('confidence')}) "
                   f"on {defect.get('board_type', 'unknown')} at {defect.get('timestamp')}.",
        "data": defect,
    }


def tool_get_top_defect_sources(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Get which logs or sources (board types) produce the most defects.

    Parameters
    ----------
    hours : float, optional
        Time window in hours. Omit for all-time.
    """
    logger.info("tool_get_top_defect_sources invoked (hours=%s)", hours)

    from services.domains.pcb.defect_store import get_top_defect_sources

    sources = get_top_defect_sources(hours=hours)
    if not sources:
        return {
            "success": True,
            "message": "No defect sources recorded yet.",
            "data": {"sources": []},
        }

    time_desc = f"last {hours}h" if hours else "all time"
    summary_parts = [f"{s['board_type']}: {s['count']} defects" for s in sources[:5]]

    return {
        "success": True,
        "message": f"Top defect sources ({time_desc}): {'; '.join(summary_parts)}.",
        "data": {"sources": sources, "time_window": time_desc},
    }


def tool_generate_summary_report(
    period: str = "daily",
    board_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a daily, weekly, or on-demand summary report of monitoring results.

    Parameters
    ----------
    period : str
        Report period: "daily" (24h), "weekly" (168h), or "all".
    board_type : str, optional
        Filter report to a specific board type.
    """
    logger.info("tool_generate_summary_report invoked (period=%s, board_type=%s)", period, board_type)

    from services.domains.pcb.defect_store import generate_summary_report

    try:
        report = generate_summary_report(period=period, board_type=board_type)
    except Exception as exc:
        return _safe_error("Failed to generate summary report", exc=exc)

    total = report.get("total_defects", 0)
    trend = report.get("trend", {}).get("trend", "unknown")

    return {
        "success": True,
        "message": f"{period.capitalize()} report: {total} defect(s), trend: {trend}.",
        "data": report,
    }


def tool_check_threshold_alerts(
    rate_threshold: Optional[float] = None,
    window_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Check whether any defects exceeded predefined thresholds or alerts.

    Parameters
    ----------
    rate_threshold : float, optional
        Max acceptable defects per window (default: 10).
    window_hours : float, optional
        Time window in hours (default: 24).
    """
    logger.info(
        "tool_check_threshold_alerts invoked (rate_threshold=%s, window_hours=%s)",
        rate_threshold, window_hours,
    )

    from services.domains.pcb.defect_store import check_threshold_alerts

    check = check_threshold_alerts(
        rate_threshold=rate_threshold or 10.0,
        window_hours=window_hours or 24.0,
    )

    if check["threshold_exceeded"]:
        alert_msgs = [a["message"] for a in check["alerts"]]
        msg = "THRESHOLD EXCEEDED: " + "; ".join(alert_msgs)
    else:
        msg = f"All thresholds OK. {check['total_defects']} defect(s) in the last {check['window_hours']}h."

    return {
        "success": True,
        "message": msg,
        "data": check,
    }


def tool_get_defect_insights(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Get recommendations and insights based on observed defect patterns.

    Parameters
    ----------
    hours : float, optional
        Analysis window in hours. Omit for all-time analysis.
    """
    logger.info("tool_get_defect_insights invoked (hours=%s)", hours)

    from services.domains.pcb.defect_store import get_defect_insights

    try:
        insights = get_defect_insights(hours=hours)
    except Exception as exc:
        return _safe_error("Failed to generate defect insights", exc=exc)

    recommendations = insights.get("recommendations", [])
    risk = insights.get("risk_level", "unknown")

    return {
        "success": True,
        "message": f"Risk level: {risk.upper()}. "
                   f"{len(recommendations)} recommendation(s). "
                   f"Trend: {insights.get('trend_direction', 'unknown')}.",
        "data": insights,
    }


def tool_get_notification_preferences() -> Dict[str, Any]:
    """Get current notification preferences and configuration."""
    logger.info("tool_get_notification_preferences invoked")

    from services.domains.pcb.notification_preferences import get_notification_preferences

    prefs = get_notification_preferences()
    status = "enabled" if prefs.email_enabled else "disabled"

    return {
        "success": True,
        "message": f"Email notifications: {status}, "
                   f"min severity: {prefs.min_severity}, "
                   f"recipients: {len(prefs.email_recipients)}.",
        "data": prefs.to_dict(),
    }
