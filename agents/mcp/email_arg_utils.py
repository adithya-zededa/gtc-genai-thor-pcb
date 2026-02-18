"""Helpers for robust email recipient extraction and fallback resolution."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from core.logging import get_logger

logger = get_logger(__name__)

_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def normalize_recipients(values: List[Any]) -> List[str]:
    recipients: List[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        candidate = value.strip().strip(".,;:")
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        recipients.append(candidate)
    return recipients


def extract_emails_from_text(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    return normalize_recipients(_EMAIL_PATTERN.findall(text))


def recipients_from_params(params: Dict[str, Any]) -> List[str]:
    collected: List[Any] = []

    recipients = params.get("recipients")
    if isinstance(recipients, list):
        collected.extend(recipients)
    elif isinstance(recipients, str):
        collected.append(recipients)

    emails = params.get("emails")
    if isinstance(emails, list):
        collected.extend(emails)
    elif isinstance(emails, str):
        collected.append(emails)

    recipient_email = params.get("recipient_email")
    if isinstance(recipient_email, str):
        collected.append(recipient_email)

    return normalize_recipients(collected)


def default_recipients() -> List[str]:
    recipients: List[str] = []

    try:
        from services.infrastructure.config import extract_recipients, load_camera_config

        cfg = load_camera_config() or {}
        recipients.extend(extract_recipients(cfg))
    except Exception as exc:
        logger.debug("Unable to read recipients from camera config: %s", exc)

    if recipients:
        return normalize_recipients(recipients)

    try:
        from app.database import UserRepository

        recipients.extend(UserRepository.get_active_emails())
    except Exception as exc:
        logger.debug("Unable to read recipients from user repository: %s", exc)

    return normalize_recipients(recipients)


def resolve_recipients(params: Dict[str, Any], user_message: str) -> List[str]:
    explicit = recipients_from_params(params)
    if explicit:
        return explicit

    from_message = extract_emails_from_text(user_message)
    if from_message:
        return from_message

    return default_recipients()
