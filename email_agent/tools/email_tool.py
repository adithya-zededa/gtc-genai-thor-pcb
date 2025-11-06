"""Email sending helper for the automation agent."""
from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


def _ensure_list(value: Optional[List[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _send_email_via_smtp(message: EmailMessage) -> str:
    host = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    use_tls = os.getenv("EMAIL_USE_TLS", "true").lower() in {"1", "true", "yes"}
    username = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")

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
    """Send an email using SMTP credentials provided via environment variables."""

    to = _ensure_list(payload.get("to"))
    if not to:
        raise ValueError("At least one recipient is required in 'to'.")

    subject = str(payload.get("subject", "Automated Notification"))
    body = str(payload.get("body", ""))
    cc = _ensure_list(payload.get("cc"))
    bcc = _ensure_list(payload.get("bcc"))

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = os.getenv("EMAIL_FROM", os.getenv("EMAIL_USER", "noreply@example.com"))
    message["To"] = ", ".join(to)

    if cc:
        message["Cc"] = ", ".join(cc)
    recipients = to + cc + bcc

    message.set_content(body)

    logger.info("Sending email to %s with subject '%s'", recipients, subject)
    result = _send_email_via_smtp(message)
    logger.debug("Email send result: %s", result)
    return result
