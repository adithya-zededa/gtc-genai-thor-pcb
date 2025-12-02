"""Email automation tools including sending logic and rule evaluation."""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# --- Email Sending Tools ---

def _ensure_list(value: Optional[List[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _send_email_via_smtp(message: EmailMessage) -> str:
    host = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    use_tls = os.getenv("EMAIL_USE_TLS", "true").lower() in {
        "1",
        "true",
        "yes",
    }
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
    message["From"] = os.getenv(
        "EMAIL_FROM", os.getenv("EMAIL_USER", "noreply@example.com")
    )
    message["To"] = ", ".join(to)

    if cc:
        message["Cc"] = ", ".join(cc)
    recipients = to + cc + bcc

    message.set_content(body)

    logger.info("Sending email to %s with subject '%s'", recipients, subject)
    result = _send_email_via_smtp(message)
    logger.debug("Email send result: %s", result)
    return result


# --- Rule Evaluation Tools ---

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "/app/config.yaml"))


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        logger.warning(
            "Configuration file %s not found; using empty rules list.",
            CONFIG_PATH,
        )
        return {"rules": []}

    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {"rules": []}


def _render_template(template: str, payload: Dict[str, Any]) -> str:
    rendered = template
    for key, value in payload.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def _compare(value: Any, expected: Any, operator: str) -> bool:
    try:
        if operator in {"lt", "less_than"}:
            return float(value) < float(expected)
        if operator in {"lte", "less_than_or_equal"}:
            return float(value) <= float(expected)
        if operator in {"gt", "greater_than"}:
            return float(value) > float(expected)
        if operator in {"gte", "greater_than_or_equal"}:
            return float(value) >= float(expected)
        if operator in {"ne", "not_equal"}:
            return value != expected
        # default equality comparison
        return value == expected
    except (TypeError, ValueError):
        return False


def _build_email_payload(
    rule: Dict[str, Any], event: Dict[str, Any]
) -> Dict[str, Any]:
    email_cfg = rule.get("email", {})
    body_template = email_cfg.get("body", "")
    subject_template = email_cfg.get(
        "subject", "Notification from Email Agent"
    )

    return {
        "to": email_cfg.get("to", []),
        "cc": email_cfg.get("cc", []),
        "bcc": email_cfg.get("bcc", []),
        "subject": _render_template(subject_template, event),
        "body": _render_template(body_template, event),
    }


def evaluate_rules(event: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate configured rules against the incoming event payload."""

    config = _load_config()
    triggered: List[Dict[str, Any]] = []

    for rule in config.get("rules", []):
        condition = rule.get("condition", {})
        field = condition.get("field")
        operator = condition.get("operator", "eq")
        expected = condition.get("value")

        if not field:
            logger.debug("Skipping rule %s - no field defined", rule.get("id"))
            continue

        value = event.get(field)
        if value is None:
            continue

        if _compare(value, expected, operator):
            triggered.append(
                {
                    "rule_id": rule.get("id"),
                    "description": rule.get("description"),
                    "email_payload": _build_email_payload(rule, event),
                }
            )

    logger.info("evaluate_rules triggered %d rule(s)", len(triggered))
    return {"triggered": triggered}
