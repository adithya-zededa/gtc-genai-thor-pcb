"""Notification routing components for the monitoring agent."""

from __future__ import annotations

import logging
import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from agent_runtime.utils import coerce_bool as _coerce_bool

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from plyer import notification as plyer_notification  # type: ignore
except ImportError:  # pragma: no cover - desktop notifications optional
    plyer_notification = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from agent_runtime.state import DetectionEvent


class AlertManager:
    """Handles alert notifications via email and desktop channels."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        notifications = config.get("notifications", {}) if isinstance(config, dict) else {}
        self.email_config = notifications.get("email", {}) if isinstance(notifications, dict) else {}
        self.desktop_config = notifications.get("desktop", {}) if isinstance(notifications, dict) else {}
        logger.info("Alert manager initialized")

    def refresh_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        notifications = config.get("notifications", {}) if isinstance(config, dict) else {}
        self.email_config = notifications.get("email", {}) if isinstance(notifications, dict) else {}
        self.desktop_config = notifications.get("desktop", {}) if isinstance(notifications, dict) else {}

    def send_email_alert(self, event: "DetectionEvent", rule: Dict[str, Any]) -> bool:
        if not _coerce_bool(self.email_config.get("enabled"), False):
            logger.info("Email alerts disabled")
            return False

        try:
            msg = EmailMessage()
            actions_email = rule.get("actions", {}).get("email") if isinstance(rule.get("actions"), dict) else None
            legacy_email = rule.get("email", {}) if isinstance(rule.get("email"), dict) else {}
            email_action = actions_email if actions_email is not None else legacy_email

            if not email_action:
                logger.warning(
                    "Rule %s missing email configuration; skipping alert",
                    rule.get("id", "unknown"),
                )
                return False

            if not _coerce_bool(
                email_action.get("enabled"),
                _coerce_bool(legacy_email.get("enabled"), True),
            ):
                logger.info("Email action disabled for rule %s", rule.get("id", "unknown"))
                return False

            subject_template = (
                email_action.get("subject")
                or legacy_email.get("subject")
                or "Unlabeled Packaging Box Detected"
            )
            body_template = (
                email_action.get("body")
                or legacy_email.get("body")
                or "An unlabeled packaging box was detected in the camera feed."
            )

            template_vars = {
                "timestamp": event.timestamp,
                "confidence": f"{event.confidence:.2f}",
                "primary_label": event.primary_label,
                "full_response": event.full_response,
                "device": f"/dev/video{os.getenv('CAMERA_INDEX', '0')}",
                "shipping_label_present": (
                    "yes"
                    if event.shipping_label_present
                    else (
                        "no"
                        if event.shipping_label_present is not None
                        else "unknown"
                    )
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
                logger.error(
                    "Email sender address missing; configure EMAIL_USER or notifications.email.sender_email"
                )
                return False

            sender_password = os.getenv("EMAIL_PASS")
            if not sender_password:
                logger.error("Email password missing; set EMAIL_PASS environment variable")
                return False

            msg["From"] = os.getenv("EMAIL_FROM", sender_email)

            recipients: Any = (
                email_action.get("to")
                or legacy_email.get("to")
                or self.email_config.get("recipients")
                or []
            )
            if isinstance(recipients, str):
                recipients = [recipients]
            if not recipients:
                logger.error("No email recipients configured in rule")
                return False

            msg["To"] = ", ".join(recipients)
            msg.set_content(body)

            if event.image_path:
                try:
                    image_path = Path(event.image_path).expanduser()
                    if image_path.exists():
                        mime_type, _ = mimetypes.guess_type(str(image_path))
                        if mime_type:
                            maintype, subtype = mime_type.split("/", 1)
                        else:
                            maintype, subtype = "application", "octet-stream"
                        with image_path.open("rb") as img_file:
                            img_bytes = img_file.read()
                        msg.add_attachment(
                            img_bytes,
                            maintype=maintype,
                            subtype=subtype,
                            filename=image_path.name,
                        )
                        logger.debug("Attached detection image %s to email alert", image_path)
                    else:
                        logger.warning("Detection image path %s not found; skipping attachment", image_path)
                except Exception as exc:
                    logger.error("Failed to attach detection image %s: %s", event.image_path, exc)

            smtp_server = self.email_config.get("smtp_server")
            if not smtp_server:
                logger.error("SMTP server not configured; set notifications.email.smtp_server")
                return False
            try:
                smtp_port = int(self.email_config.get("smtp_port"))
            except (TypeError, ValueError):
                logger.error("Invalid SMTP port configuration")
                return False

            use_tls_config = _coerce_bool(
                email_action.get("use_tls"),
                _coerce_bool(self.email_config.get("use_tls"), True),
            )
            use_tls = _coerce_bool(os.getenv("EMAIL_USE_TLS"), use_tls_config)

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if use_tls:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            logger.info("Email alert sent successfully to %s", msg["To"])
            return True
        except Exception as exc:
            logger.error("Failed to send email alert: %s", exc)
            return False

    def send_desktop_notification(self, event: "DetectionEvent") -> bool:
        if not _coerce_bool(self.desktop_config.get("enabled"), False):
            logger.debug("Desktop notifications disabled")
            return False
        if plyer_notification is None:
            logger.debug("plyer not available for desktop notifications")
            return False
        try:
            try:
                timeout = int(self.desktop_config.get("timeout", 10))
            except (TypeError, ValueError):
                timeout = 10
            title = self.desktop_config.get("title", "ZEDEDA Camera Alert")
            message_template = self.desktop_config.get(
                "message", "Unlabeled packaging box detected at {timestamp}"
            )
            label_status = (
                "yes"
                if event.shipping_label_present
                else (
                    "no"
                    if event.shipping_label_present is not None
                    else "unknown"
                )
            )
            message = message_template.format(
                timestamp=event.timestamp,
                confidence=f"{event.confidence:.2f}",
                label=event.primary_label,
                shipping_label_present=label_status,
            )
            plyer_notification.notify(title=title, message=message, timeout=timeout)
            logger.info("Desktop notification sent")
            return True
        except Exception as exc:
            logger.error("Failed to send desktop notification: %s", exc)
            return False
