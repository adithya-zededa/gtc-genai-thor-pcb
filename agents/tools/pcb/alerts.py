"""PCB alert and notification tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from agents.tools.validation import (
    validate_emails as _validate_emails,
    safe_error as _safe_error,
    sanitise_severity as _sanitise_severity,
)
from ._helpers import _auto_generate_defect_summary

logger = get_logger(__name__)


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

    If *defect_summary* is not provided, a summary is automatically
    generated from the defect database.
    """
    logger.info(
        "tool_send_defect_alert invoked (recipients=%s, severity=%s)",
        recipients, severity,
    )

    if not recipients:
        return {"success": False, "message": "No recipients specified"}

    # Auto-populate defect_summary from DB when not supplied
    if not defect_summary:
        defect_summary = _auto_generate_defect_summary()

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
            "data": {
                "recipients_count": len(recipients),
                "email_result": str(result),
            },
        }
    except Exception as exc:
        return _safe_error("Failed to send defect alert", exc=exc)


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
