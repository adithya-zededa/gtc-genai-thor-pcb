"""Shared validation helpers for domain tool handlers.

Consolidates duplicate email-validation, error-formatting, and severity
sanitisation logic used by PCB domain tools.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)

# ── Email validation ───────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
MAX_EMAIL_LEN = 254


def validate_email(email: str) -> Optional[str]:
    """Return *None* if valid, or an error message."""
    if not email:
        return "Email address is required"
    if len(email) > MAX_EMAIL_LEN:
        return f"Email address too long (max {MAX_EMAIL_LEN} chars)"
    if not _EMAIL_RE.match(email):
        return f"Invalid email format: {email}"
    return None


def validate_emails(emails: List[str]) -> Optional[str]:
    """Return *None* if every address in the list is valid, or an error."""
    if not emails:
        return "At least one recipient email is required"
    for em in emails:
        err = validate_email(em)
        if err:
            return err
    return None


# ── Safe error responses ──────────────────────────────────────────────────

def safe_error(internal_msg: str, *, exc: Optional[Exception] = None) -> Dict[str, Any]:
    """Return a user-safe error dict and log the internal detail."""
    if exc:
        logger.error("%s: %s", internal_msg, exc, exc_info=True)
    else:
        logger.error(internal_msg)
    return {"success": False, "message": "An internal error occurred. Please try again."}


# ── Severity sanitisation ─────────────────────────────────────────────────

VALID_SEVERITIES = frozenset(["low", "medium", "high"])


def sanitise_severity(raw: str) -> str:
    """Normalise severity to one of the allowed values."""
    s = raw.strip().lower() if raw else "medium"
    return s if s in VALID_SEVERITIES else "medium"
