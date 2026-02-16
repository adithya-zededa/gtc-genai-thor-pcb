"""Notification routing components for the monitoring agent."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Any, Dict, TYPE_CHECKING

from core.utils import coerce_bool
from core.logging import get_logger

logger = get_logger(__name__)

try:
    from plyer import notification as plyer_notification
except ImportError:
    plyer_notification = None

if TYPE_CHECKING:
    from agents.state import DetectionEvent


class AlertManager:
    """Handles alert notifications via email and desktop channels."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the alert manager."""
        self.config = config
        notifications = (
            config.get("notifications", {})
            if isinstance(config, dict) else {}
        )
        self.email_config: Dict[str, Any] = (
            notifications.get("email", {})
            if isinstance(notifications, dict) else {}
        )
        self.desktop_config: Dict[str, Any] = (
            notifications.get("desktop", {})
            if isinstance(notifications, dict) else {}
        )
        logger.info("Alert manager initialized")

    def refresh_config(self, config: Dict[str, Any]) -> None:
        """Refresh configuration."""
        self.config = config
        notifications = (
            config.get("notifications", {})
            if isinstance(config, dict) else {}
        )
        self.email_config = (
            notifications.get("email", {})
            if isinstance(notifications, dict) else {}
        )
        self.desktop_config = (
            notifications.get("desktop", {})
            if isinstance(notifications, dict) else {}
        )

    def send_email_alert(  # pylint: disable=too-many-locals,too-many-return-statements
        self,
        event: "DetectionEvent",
        rule: Dict[str, Any],
        image_data: bytes | None = None,
    ) -> bool:
        """Send an email alert for a detection event."""
        if not coerce_bool(self.email_config.get("enabled"), False):
            logger.info("Email alerts disabled")
            return False

        try:
            msg = EmailMessage()
            actions_email = (
                rule.get("actions", {}).get("email")
                if isinstance(rule.get("actions"), dict) else None
            )
            legacy_email = (
                rule.get("email", {})
                if isinstance(rule.get("email"), dict) else {}
            )
            email_action = (
                actions_email if actions_email is not None else legacy_email
            )

            if not email_action:
                logger.warning("Rule missing email configuration")
                return False

            enabled = coerce_bool(
                email_action.get("enabled"),
                coerce_bool(legacy_email.get("enabled"), True)
            )
            if not enabled:
                return False

            subject_template = (
                email_action.get("subject") or legacy_email.get("subject")
                or "PCB Inspection Alert"
            )
            body_template = (
                email_action.get("body") or legacy_email.get("body")
                or "A PCB inspection alert condition was detected."
            )

            template_vars = {
                "timestamp": event.timestamp,
                "confidence": f"{event.confidence:.2f}",
                "primary_label": event.primary_label,
                "full_response": event.full_response,
                "device": f"/dev/video{os.getenv('CAMERA_INDEX', '0')}",
                "pcb_stable": (
                    "yes" if event.pcb_stable
                    else ("no" if event.pcb_stable is not None else "unknown")
                ),
                "should_alert": str(event.should_alert).lower(),
            }

            subject = subject_template
            body = body_template
            for key, value in template_vars.items():
                subject = subject.replace(f"{{{{{key}}}}}", str(value))
                body = body.replace(f"{{{{{key}}}}}", str(value))

            msg["Subject"] = subject

            sender_email = os.getenv("EMAIL_USER") or self.email_config.get("sender_email")
            if not sender_email:
                logger.error("Email sender address missing")
                return False

            sender_password = os.getenv("EMAIL_PASS")
            if not sender_password:
                logger.error("Email password missing")
                return False

            msg["From"] = os.getenv("EMAIL_FROM", sender_email)

            recipients = (
                email_action.get("to") or legacy_email.get("to")
                or self.email_config.get("recipients")
            )
            if not recipients:
                logger.warning("No email recipients configured")
                return False

            if isinstance(recipients, str):
                recipients = [recipients]
            msg["To"] = ", ".join(recipients)
            msg.set_content(body)

            # Attach image if provided
            if image_data:
                msg.add_attachment(
                    image_data,
                    maintype="image",
                    subtype="jpeg",
                    filename="detection.jpg"
                )

            # Send via SMTP
            smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))

            with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
                use_tls = os.getenv("EMAIL_USE_TLS", "true").lower()
                if use_tls in {"1", "true", "yes"}:
                    server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            logger.info("Email alert sent to %s", recipients)
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to send email alert: %s", e)
            return False

    def send_desktop_notification(self, event: "DetectionEvent") -> bool:
        """Send a desktop notification."""
        if plyer_notification is None:
            logger.debug("Desktop notifications not available")
            return False

        if not coerce_bool(self.desktop_config.get("enabled"), False):
            return False

        try:
            message = (
                f"Detection: {event.primary_label} "
                f"(Confidence: {event.confidence:.2%})"
            )
            plyer_notification.notify(
                title="Camera Agent Alert",
                message=message,
                timeout=10,
            )
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Desktop notification failed: %s", e)
            return False
