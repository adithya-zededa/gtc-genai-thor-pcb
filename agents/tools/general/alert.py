"""Send-alert-email tool implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)


def tool_send_alert_email(
    recipients: List[str],
    subject: str,
    body: str,
    priority: str = "normal",
    include_image: bool = True,
    image_data: Optional[bytes] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Send an alert email to specified recipients."""
    from agents.tools.email import send_email

    if not recipients:
        return {"success": False, "error": "No recipients specified"}

    payload: Dict[str, Any] = {
        "to": recipients,
        "subject": subject,
        "body": body,
    }

    if include_image is not False and image_data:
        payload["image_data"] = image_data
        payload["image_filename"] = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    try:
        result = send_email(payload)
        return {
            "success": True,
            "message": f"Alert email sent to {len(recipients)} recipient(s)",
            "data": {
                "recipients_count": len(recipients),
                "email_result": str(result),
            },
        }
    except Exception as e:
        logger.error("Failed to send alert email: %s", e)
        return {"success": False, "message": f"Failed to send alert email: {e}"}
