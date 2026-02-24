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
    "You are a professional PCB quality control inspector.\n\n"
    "You will be given ONE image of an Arduino Uno R4 Minima PCB.\n"
    "Your task is to determine whether the PCB is defective.\n\n"
    "IMPORTANT:\n"
    "- Output ONLY valid JSON.\n"
    "- Do NOT include any extra text outside the JSON.\n"
    "- Be strict: if a required connector is missing, partially visible, mechanically damaged, or clearly not populated, mark it as defective.\n"
    "- If image quality prevents certainty, use \"uncertain\" and lower confidence.\n\n"
    "Inspection Instructions:\n\n"
    "Q1 – LEFT EDGE (Upper Area – Power Input Region):\n\n"
    "You MUST visually confirm the presence of a tall black cylindrical barrel jack.\n\n"
    "Do NOT assume it exists because this is an Arduino board.\n"
    "Only mark \"present_intact\" if you clearly see:\n"
    "- A black cylindrical body\n"
    "- Significant vertical height relative to surrounding components\n"
    "- A hollow circular opening at the end\n\n"
    "If the barrel jack is not clearly visible as a tall cylindrical connector,\n"
    "even if mounting holes are present,\n"
    "then classify it as \"missing\".\n\n"
    "If you are unsure due to image angle or blur, classify as \"uncertain\".\n\n"
    "Never infer presence from typical board design.\n"
    "Only report what is directly visible in the image.\n\n"
    "Q2 – TOP EDGE (USB Connector):\n"
    "Inspect the top edge of the PCB.\n"
    "Check for a USB connector.\n\n"
    "A correct USB connector should:\n"
    "- Be metallic (silver)\n"
    "- Rectangular\n"
    "- Protrude from the board edge\n"
    "- Have a visible port opening\n\n"
    "Classify as:\n"
    "- present_intact\n"
    "- missing\n"
    "- damaged\n"
    "- uncertain\n\n"
    "Q3 – Header Pins (Plastic Base Presence):\n"
    "Inspect:\n"
    "- LEFT EDGE header row (below power area)\n"
    "- BOTTOM EDGE header row\n\n"
    "Check whether both header strips have black plastic spacer bases.\n\n"
    "Classify as:\n"
    "- present_both_sides\n"
    "- missing_one_side\n"
    "- missing_both_sides\n"
    "- damaged\n"
    "- uncertain\n\n"
    "Q4 – Additional Defects:\n"
    "Inspect entire PCB for:\n"
    "- Missing components\n"
    "- Misaligned components\n"
    "- Solder bridges\n"
    "- Cold solder joints\n"
    "- Burn marks\n"
    "- Lifted pads\n"
    "- Trace damage\n"
    "- Mechanical cracks\n"
    "- Bent connectors\n\n"
    "List all observed issues.\n\n"
    "Final Decision Rule:\n"
    "Set \"detected\" to true if ANY of the following:\n"
    "- A required connector is missing or damaged\n"
    "- Header bases missing/damaged\n"
    "- Any additional defect observed\n\n"
    "Set \"should_alert\" equal to detected.\n\n"
    "Return JSON: {detected: true/false, confidence: 0-1, "
    "reasoning: your observations, should_alert: true/false, "
    "details: {power_jack_status: present_intact|missing|damaged|uncertain, "
    "usb_connector_status: present_intact|missing|damaged|uncertain, "
    "header_pin_status: present_both_sides|missing_one_side|missing_both_sides|damaged|uncertain, "
    "defects: [{type, severity, location, description}]}}"
)


def get_inspection_prompt() -> str:
    """Return the active PCB defect inspection prompt.

    Reads from ``config.yaml`` (``detection.inspection_prompt``).  If that
    field is empty or missing, falls back to the hardcoded default above.
    This allows users to customise the prompt from the Settings UI without
    redeploying.
    """
    try:
        from services.infrastructure.config import load_camera_config

        cfg = load_camera_config()
        custom = (cfg.get("detection") or {}).get("inspection_prompt", "") or ""
        if isinstance(custom, str) and custom.strip():
            return custom.strip()
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return DEFAULT_PCB_DEFECT_INSPECTION_PROMPT


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

    Every individual defect discovered is logged as a separate record so
    that counts, trends, and breakdowns reflect the actual number of
    distinct issues — not just "one row per inspection".
    """
    if not detected:
        logger.debug("_auto_log_defect: no defect detected, skipping")
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

    # Also try to extract board type from nested details
    details = analysis.get("details", {})
    if board_type == "unknown" and isinstance(details, dict):
        board_type = details.get("board_type", "unknown")

    defect_list = analysis.get("defects", [])
    # Fallback: extract defects from details.defects if top-level is empty
    if not defect_list and isinstance(details, dict):
        defect_list = details.get("defects", [])

    # Build a rich description from the VLM output rather than a generic string
    rich_description = description or ""
    if not rich_description and analysis.get("reasoning"):
        rich_description = str(analysis["reasoning"])

    # Determine overall severity and defect type
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

    # Also infer severity from component statuses in details
    if isinstance(details, dict):
        for status_key in ("power_jack_status", "usb_connector_status", "header_pin_status"):
            status_val = details.get(status_key, "")
            if status_val in ("missing", "damaged"):
                severity = "high"
                if defect_type == "general_defect":
                    defect_type = f"{status_key.replace('_status', '')}_{status_val}"
                break

    logged_count = 0
    from services.domains.pcb.service import record_defect

    # Log each individual defect as a separate record
    if defect_list and isinstance(defect_list, list) and len(defect_list) > 0:
        for idx, defect_item in enumerate(defect_list):
            if not isinstance(defect_item, dict):
                continue
            d_type = defect_item.get("type", defect_type)
            d_sev = defect_item.get("severity", severity).lower()
            if d_sev not in ("low", "medium", "high"):
                d_sev = severity
            d_loc = defect_item.get("location", "")
            d_desc = defect_item.get("description", "")
            combined_desc = f"{d_desc}" if d_desc else rich_description
            if d_loc:
                combined_desc = f"[{d_loc}] {combined_desc}"
            try:
                record_defect(
                    board_type=board_type,
                    defect_type=d_type,
                    severity=d_sev,
                    confidence=confidence,
                    image_path=image_path,
                    description=combined_desc or rich_description or "Auto-logged from VLM inspection",
                )
                logged_count += 1
                logger.info(
                    "Auto-logged defect %d/%d: type=%s severity=%s board=%s location=%s",
                    idx + 1, len(defect_list), d_type, d_sev, board_type, d_loc,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Auto-log defect %d failed (non-fatal): %s", idx + 1, exc)
    else:
        # No structured defect list — log a single record with best-available data
        try:
            record_defect(
                board_type=board_type,
                defect_type=defect_type,
                severity=severity,
                confidence=confidence,
                image_path=image_path,
                description=rich_description or "Auto-logged from VLM inspection",
            )
            logged_count = 1
            logger.info("Auto-logged defect: type=%s severity=%s board=%s", defect_type, severity, board_type)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Auto-log defect failed (non-fatal): %s", exc)

    logger.info(
        "Auto-log summary: %d defect(s) persisted for board=%s (confidence=%.2f, image=%s)",
        logged_count, board_type, confidence, bool(image_path),
    )

    # Trigger notification if preferences allow
    _auto_notify_defect(
        board_type=board_type,
        defect_type=defect_type,
        severity=severity,
        description=rich_description or description,
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
            custom_prompt=query or get_inspection_prompt(),
        )
    except Exception as exc:
        logger.error("VLM PCB inspection failed: %s", exc)
        return _safe_error("VLM PCB inspection failed", exc=exc)

    if not event:
        logger.warning("PCB inspection returned no event: %s", service.last_error)
        return {
            "success": False,
            "message": service.last_error or "PCB inspection failed — no frame available",
        }

    # Resolve the saved image path so auto-logged defects can reference it
    inspection_image_path = ""
    if hasattr(event, "image_path") and event.image_path:
        inspection_image_path = event.image_path

    logger.info(
        "PCB inspection result: detected=%s confidence=%.2f should_alert=%s image=%s",
        event.detected, event.confidence, event.should_alert, bool(inspection_image_path),
    )

    # Auto-log defect if detected (no LLM round-trip needed)
    _auto_log_defect(
        detected=event.detected,
        full_response=event.full_response,
        description=event.vision_description or "",
        confidence=event.confidence,
        image_path=inspection_image_path,
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
            "image_path": inspection_image_path,
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
            custom_prompt=query or get_inspection_prompt(),
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


def tool_generate_defect_report(
    board_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive summary report of all logged PCB defects.

    Includes severity breakdown, type breakdown, trend analysis, and
    recent defect list for complete reporting.
    """
    logger.info("tool_generate_defect_report invoked (board_type=%s)", board_type)

    from services.domains.pcb.service import generate_defect_report
    from services.domains.pcb.defect_store import (
        count_defects,
        get_defect_trend,
        get_defect_type_breakdown,
        get_defect_insights,
    )

    try:
        report = generate_defect_report(board_type=board_type)
    except Exception as exc:
        logger.error("Failed to generate defect report: %s", exc)
        return _safe_error("Failed to generate defect report", exc=exc)

    # Enrich with trend and insight data
    try:
        total_all = count_defects()
        total_24h = count_defects(hours=24)
        total_week = count_defects(hours=168)
        trend = get_defect_trend(window_hours=168.0)
        type_breakdown = get_defect_type_breakdown()
        insights = get_defect_insights()

        report["enriched"] = {
            "total_all_time": total_all,
            "total_last_24h": total_24h,
            "total_last_week": total_week,
            "trend": trend,
            "type_breakdown": type_breakdown,
            "risk_level": insights.get("risk_level", "unknown"),
            "recommendations": insights.get("recommendations", []),
        }
        logger.info(
            "Defect report generated: total=%d, 24h=%d, week=%d, trend=%s, risk=%s",
            total_all, total_24h, total_week,
            trend.get("trend", "unknown"),
            insights.get("risk_level", "unknown"),
        )
    except Exception as exc:
        logger.warning("Failed to enrich defect report with analytics: %s", exc)

    return report


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

    # Collect defect types from failed inspections
    fail_types: Dict[str, int] = {}
    for r in records:
        if r.get("result") == "FAIL" and r.get("defect_type"):
            dt = r["defect_type"]
            fail_types[dt] = fail_types.get(dt, 0) + 1

    summary = (
        f"Found {total} total inspection(s). "
        f"Showing {len(records)} most recent. "
        f"Pass: {pass_count}, Fail: {fail_count}."
    )

    logger.info(
        "PCB inspections query: total=%d, showing=%d, pass=%d, fail=%d, "
        "fail_types=%s, filter=%s",
        total, len(records), pass_count, fail_count, fail_types, result_filter,
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
            "fail_defect_types": fail_types,
        },
    }


# ---------------------------------------------------------------------------
# Chat query tools — monitoring status, defect analytics, notifications
# ---------------------------------------------------------------------------

def tool_get_monitoring_status() -> Dict[str, Any]:
    """Get current and historical monitoring status including defect counts.

    Returns agent state, session info, total defects, and recent activity.
    Also includes trend direction and threshold status for quick overview.
    """
    logger.info("tool_get_monitoring_status invoked")

    from services.core.monitoring import get_monitoring_service
    from services.domains.pcb.defect_store import (
        count_defects,
        get_latest_defect,
        get_defect_trend,
        check_threshold_alerts,
    )

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

    # Add trend and threshold status
    trend_info = None
    threshold_info = None
    try:
        trend = get_defect_trend(window_hours=168.0)
        trend_info = {
            "direction": trend.get("trend", "unknown"),
            "total_in_window": trend.get("total_defects", 0),
        }
    except Exception:
        pass

    try:
        threshold = check_threshold_alerts(window_hours=24.0)
        threshold_info = {
            "exceeded": threshold.get("threshold_exceeded", False),
            "alerts": [a["message"] for a in threshold.get("alerts", [])],
        }
    except Exception:
        pass

    logger.info(
        "Monitoring status: active=%s mode=%s total_defects=%d 24h=%d 1h=%d trend=%s",
        monitoring_active, monitoring_mode, total_defects, defects_24h, defects_1h,
        trend_info.get("direction") if trend_info else "N/A",
    )

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
            "trend": trend_info,
            "threshold_status": threshold_info,
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
        count_defects,
        get_defect_trend,
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
    sev_counts: Dict[str, int] = {}
    for d in defects:
        s = d.get("severity", "unknown")
        sev_counts[s] = sev_counts.get(s, 0) + 1

    # Include trend data for context
    trend_info = None
    try:
        trend_data = get_defect_trend(window_hours=hours or 168.0)
        trend_info = {
            "direction": trend_data.get("trend", "unknown"),
            "first_half_rate": trend_data.get("first_half_rate", 0),
            "second_half_rate": trend_data.get("second_half_rate", 0),
        }
    except Exception:
        pass

    # Compare to previous period for delta
    delta_info = None
    if hours:
        try:
            prev_count = count_defects(hours=hours * 2) - len(defects)
            if prev_count > 0:
                change_pct = round(((len(defects) - prev_count) / prev_count) * 100, 1)
            else:
                change_pct = 100.0 if len(defects) > 0 else 0.0
            delta_info = {
                "current_period": len(defects),
                "previous_period": prev_count,
                "change_percent": change_pct,
            }
        except Exception:
            pass

    time_desc = f"last {hours}h" if hours else "all time"
    filters = []
    if board_type:
        filters.append(f"board={board_type}")
    if severity:
        filters.append(f"severity={severity}")
    filter_str = f" ({', '.join(filters)})" if filters else ""

    logger.info(
        "Defect summary: %d defect(s) for %s%s, severities=%s, types=%d",
        len(defects), time_desc, filter_str, sev_counts, len(type_counts),
    )

    return {
        "success": True,
        "message": f"{len(defects)} defect(s) found for {time_desc}{filter_str}.",
        "data": {
            "total": len(defects),
            "time_window": time_desc,
            "severity_breakdown": sev_counts,
            "defect_types": type_breakdown,
            "trend": trend_info,
            "period_comparison": delta_info,
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

    logger.info(
        "Defect count (%s): total=%d, high=%d, medium=%d, low=%d",
        time_desc, total, high, medium, low,
    )

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
        logger.info("No defects recorded yet")
        return {
            "success": True,
            "message": "No defects have been recorded yet.",
            "data": None,
        }

    logger.info(
        "Latest defect: type=%s severity=%s board=%s confidence=%.2f timestamp=%s",
        defect.get("defect_type", "unknown"),
        defect.get("severity", "unknown"),
        defect.get("board_type", "unknown"),
        defect.get("confidence", 0.0),
        defect.get("timestamp", "unknown"),
    )

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
        logger.info("Defect type breakdown: no defects recorded")
        return {
            "success": True,
            "message": "No defects recorded yet.",
            "data": {"types": []},
        }

    time_desc = f"last {hours}h" if hours else "all time"
    summary_parts = [f"{t['defect_type']}: {t['count']} ({t['percentage']}%)" for t in breakdown[:5]]
    total_types = len(breakdown)

    logger.info(
        "Defect type breakdown (%s): %d type(s), top=%s",
        time_desc, total_types, summary_parts[:3],
    )

    return {
        "success": True,
        "message": f"Defect type breakdown ({time_desc}): {'; '.join(summary_parts)}.",
        "data": {"types": breakdown, "time_window": time_desc, "total_types": total_types},
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

    logger.info(
        "Defect trend (%sh window): direction=%s, total=%d, "
        "first_half_rate=%.2f, second_half_rate=%.2f, buckets=%d",
        window, trend["trend"], trend["total_defects"],
        trend["first_half_rate"], trend["second_half_rate"],
        len(trend.get("buckets", [])),
    )

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
        logger.info("No defects found (%s)", time_desc)
        return {
            "success": True,
            "message": f"No defects found ({time_desc}).",
            "data": None,
        }

    logger.info(
        "Most severe defect: type=%s severity=%s confidence=%.2f board=%s",
        defect.get("defect_type", "unknown"),
        defect.get("severity", "unknown"),
        defect.get("confidence", 0.0),
        defect.get("board_type", "unknown"),
    )

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
        logger.info("No defect sources recorded")
        return {
            "success": True,
            "message": "No defect sources recorded yet.",
            "data": {"sources": []},
        }

    time_desc = f"last {hours}h" if hours else "all time"
    summary_parts = [f"{s['board_type']}: {s['count']} defects" for s in sources[:5]]

    logger.info(
        "Top defect sources (%s): %d source(s), top=%s",
        time_desc, len(sources), summary_parts[:3],
    )

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
        logger.error("Failed to generate summary report: %s", exc)
        return _safe_error("Failed to generate summary report", exc=exc)

    total = report.get("total_defects", 0)
    trend = report.get("trend", {}).get("trend", "unknown")
    sev_breakdown = report.get("severity_breakdown", {})
    threshold_exceeded = report.get("threshold_check", {}).get("threshold_exceeded", False)

    logger.info(
        "Summary report (%s): total=%d, trend=%s, severities=%s, "
        "threshold_exceeded=%s, board_type=%s",
        period, total, trend, sev_breakdown, threshold_exceeded, board_type,
    )

    # Build a richer message for the LLM
    sev_parts = [f"{k}: {v}" for k, v in sev_breakdown.items() if v > 0]
    sev_summary = ", ".join(sev_parts) if sev_parts else "none"

    msg = (
        f"{period.capitalize()} report: {total} defect(s), "
        f"trend: {trend}, severity: [{sev_summary}]"
    )
    if threshold_exceeded:
        msg += " ⚠️ THRESHOLD EXCEEDED"

    return {
        "success": True,
        "message": msg,
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
        logger.warning(
            "Threshold exceeded! total=%d, high=%d, window=%.1fh, alerts=%s",
            check["total_defects"], check["high_severity_count"],
            check["window_hours"], alert_msgs,
        )
    else:
        msg = f"All thresholds OK. {check['total_defects']} defect(s) in the last {check['window_hours']}h."
        logger.info(
            "Threshold check OK: total=%d, high=%d, window=%.1fh",
            check["total_defects"], check["high_severity_count"],
            check["window_hours"],
        )

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
        logger.error("Failed to generate defect insights: %s", exc)
        return _safe_error("Failed to generate defect insights", exc=exc)

    recommendations = insights.get("recommendations", [])
    risk = insights.get("risk_level", "unknown")
    trend_direction = insights.get("trend_direction", "unknown")
    total_analyzed = insights.get("total_defects_analyzed", 0)

    logger.info(
        "Defect insights: risk=%s, trend=%s, total_analyzed=%d, "
        "recommendations=%d, threshold_exceeded=%s",
        risk, trend_direction, total_analyzed,
        len(recommendations), insights.get("threshold_exceeded", False),
    )

    return {
        "success": True,
        "message": f"Risk level: {risk.upper()}. "
                   f"{len(recommendations)} recommendation(s). "
                   f"Trend: {trend_direction}. "
                   f"Total analyzed: {total_analyzed}.",
        "data": insights,
    }


def tool_get_notification_preferences() -> Dict[str, Any]:
    """Get current notification preferences and configuration."""
    logger.info("tool_get_notification_preferences invoked")

    from services.domains.pcb.notification_preferences import get_notification_preferences

    prefs = get_notification_preferences()
    status = "enabled" if prefs.email_enabled else "disabled"

    logger.info(
        "Notification preferences: %s, min_severity=%s, recipients=%d",
        status, prefs.min_severity, len(prefs.email_recipients),
    )

    return {
        "success": True,
        "message": f"Email notifications: {status}, "
                   f"min severity: {prefs.min_severity}, "
                   f"recipients: {len(prefs.email_recipients)}.",
        "data": prefs.to_dict(),
    }


# ---------------------------------------------------------------------------
# Unified detection & defect log query tool
# ---------------------------------------------------------------------------

def tool_query_detection_logs(
    limit: int = 20,
    hours: Optional[float] = None,
    detected_only: bool = False,
    include_defects: bool = True,
) -> Dict[str, Any]:
    """Query detection logs enriched with defect information.

    Returns a unified view of detection events from ``detection_logs``
    combined with defect-specific data from ``pcb_defects``, giving a
    complete picture of what was inspected and what defects were found.

    Parameters
    ----------
    limit : int
        Maximum number of records to return (default 20).
    hours : float, optional
        Time window in hours. Omit for all records.
    detected_only : bool
        Only return detections with confidence > 0.
    include_defects : bool
        If True (default), also return recent defect records.
    """
    logger.info(
        "tool_query_detection_logs invoked (limit=%d, hours=%s, "
        "detected_only=%s, include_defects=%s)",
        limit, hours, detected_only, include_defects,
    )

    from app.database import DetectionLogRepository

    try:
        limit = max(1, min(100, int(limit)))
    except (TypeError, ValueError):
        return {"success": False, "message": "limit must be an integer"}

    try:
        logs, total_logs = DetectionLogRepository.get_paginated(
            page=1,
            per_page=limit,
            detected_only=detected_only,
        )
    except Exception as exc:
        logger.error("Failed to query detection logs: %s", exc)
        return _safe_error("Failed to query detection logs", exc=exc)

    log_entries = [log.to_dict() for log in logs]

    # Filter by time window if requested
    if hours is not None and hours > 0:
        from services.domains.pcb.defect_store import _time_window
        cutoff, _ = _time_window(hours)
        log_entries = [
            e for e in log_entries
            if e.get("timestamp", "") >= cutoff
        ]

    result: Dict[str, Any] = {
        "success": True,
        "message": f"Retrieved {len(log_entries)} detection log(s) (total: {total_logs}).",
        "data": {
            "detection_logs": log_entries,
            "total_detection_logs": total_logs,
            "showing": len(log_entries),
        },
    }

    if include_defects:
        try:
            from services.domains.pcb.defect_store import (
                get_defects_in_range,
                count_defects,
                get_latest_defect,
                get_defect_type_breakdown,
                _time_window,
            )

            start_time = None
            if hours is not None and hours > 0:
                start_time, _ = _time_window(hours)

            recent_defects = get_defects_in_range(
                start_time=start_time,
                limit=limit,
            )
            total_defects = count_defects(hours=hours)
            latest_defect = get_latest_defect()
            type_breakdown = get_defect_type_breakdown(hours=hours)

            # Severity breakdown from recent defects
            sev_counts: Dict[str, int] = {}
            for d in recent_defects:
                s = d.get("severity", "unknown")
                sev_counts[s] = sev_counts.get(s, 0) + 1

            result["data"]["defects"] = {
                "recent_defects": recent_defects,
                "total_defects": total_defects,
                "latest_defect": latest_defect,
                "severity_breakdown": sev_counts,
                "type_breakdown": type_breakdown,
            }

            logger.info(
                "Detection logs enriched: %d logs, %d defects, severities=%s",
                len(log_entries), total_defects, sev_counts,
            )
        except Exception as exc:
            logger.warning("Failed to enrich detection logs with defect data: %s", exc)
            result["data"]["defects"] = {"error": str(exc)}

    return result
