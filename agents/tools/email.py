"""Email utility functions for sending emails via SMTP."""

from __future__ import annotations

import logging
import mimetypes
import os
import smtplib
from email.message import EmailMessage
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)


def _ensure_list(value: Optional[List[str]]) -> List[str]:
    """Ensure value is a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _email_settings() -> Dict[str, Any]:
    """Resolve SMTP settings: environment > config.yaml > default.

    The Settings page writes ``notifications.email.{smtp_server,smtp_port,
    sender_email}`` into config.yaml. Those keys were previously read by
    nothing — SMTP came from the environment alone — so configuring a mail
    server through the UI silently had no effect. The YAML layer is
    consulted here, with the environment still winning so a deployment can
    override without editing a mounted file.
    """
    yaml_email: Dict[str, Any] = {}
    try:
        from services.infrastructure.config import load_camera_config

        yaml_email = (load_camera_config().get("notifications", {})
                      .get("email", {})) or {}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("Could not read email settings from config.yaml: %s", exc)

    def _yaml(key, default):
        value = yaml_email.get(key)
        return default if value in (None, "") else value

    try:
        port = int(os.getenv("EMAIL_SMTP_PORT") or _yaml("smtp_port", 587))
    except (TypeError, ValueError):
        port = 587

    use_tls_raw = os.getenv("EMAIL_USE_TLS")
    if use_tls_raw is None:
        use_tls = bool(_yaml("use_tls", True))
    else:
        use_tls = use_tls_raw.lower() in {"1", "true", "yes"}

    return {
        "host": os.getenv("EMAIL_SMTP_SERVER") or _yaml("smtp_server", "smtp.gmail.com"),
        "port": port,
        "use_tls": use_tls,
        "username": os.getenv("EMAIL_USER"),
        "password": os.getenv("EMAIL_PASS"),
        "sender": (
            os.getenv("EMAIL_FROM")
            or _yaml("sender_email", "")
            or os.getenv("EMAIL_USER")
            or "noreply@example.com"
        ),
    }


def _send_email_via_smtp(message: EmailMessage) -> str:
    """Send email via SMTP using the resolved configuration."""
    settings = _email_settings()
    host = settings["host"]
    port = settings["port"]
    use_tls = settings["use_tls"]
    username = settings["username"]
    password = settings["password"]

    if not username or not password:
        logger.warning("Email credentials not configured; skipping send.")
        return "Email skipped (credentials missing)."

    with smtplib.SMTP(host=host, port=port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls()
        smtp.login(username, password)
        smtp.send_message(message)

    return "Email sent via SMTP."


def send_email(payload: Dict[str, object]) -> str:
    """Send an email using SMTP credentials provided via environment variables.
    
    Args:
        payload: Dictionary containing:
            - 'to': List of recipient emails (required)
            - 'subject': Email subject line
            - 'body': Email body text
            - 'cc': List of CC recipients (optional)
            - 'bcc': List of BCC recipients (optional)
            - 'attachments': List of attachment dicts (optional)
            - 'image_data': bytes of image to attach (shorthand)
            - 'image_filename': filename for the image
        
    Returns:
        Status message indicating success or reason for skipping.
    """
    to = _ensure_list(payload.get("to"))
    if not to:
        raise ValueError("At least one recipient is required in 'to'.")

    subject = str(payload.get("subject", "Automated Notification"))
    body = str(payload.get("body", ""))
    cc = _ensure_list(payload.get("cc"))
    bcc = _ensure_list(payload.get("bcc"))

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = _email_settings()["sender"]
    message["To"] = ", ".join(to)

    if cc:
        message["Cc"] = ", ".join(cc)

    message.set_content(body)
    
    # Handle attachments
    attachments: List[Dict[str, Any]] = list(payload.get("attachments", []))
    
    # Handle shorthand image_data parameter
    image_data = payload.get("image_data")
    if image_data and isinstance(image_data, bytes):
        image_filename = str(payload.get("image_filename", "detection.jpg"))
        attachments.append({
            "data": image_data,
            "filename": image_filename,
            "mime_type": "image/jpeg",
        })
    
    # Attach all files
    for attachment in attachments:
        data = attachment.get("data")
        if not data or not isinstance(data, bytes):
            continue
            
        filename = str(attachment.get("filename", "attachment"))
        mime_type = attachment.get("mime_type")
        
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or "application/octet-stream"
        
        maintype, subtype = mime_type.split("/", 1)
        message.add_attachment(
            data,
            maintype=maintype,
            subtype=subtype,
            filename=filename,
        )
        logger.debug("Attached file: %s (%s)", filename, mime_type)

    recipients = to + cc + bcc
    logger.info("Sending email to %s with subject '%s'", recipients, subject)
    result = _send_email_via_smtp(message)
    logger.debug("Email send result: %s", result)
    return result
