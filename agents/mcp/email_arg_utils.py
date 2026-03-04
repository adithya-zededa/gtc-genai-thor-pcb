"""Email recipient helpers — backward-compatible re-export.

The canonical implementation now lives in ``agents.tools._email_utils``.
This module re-exports all public symbols so existing imports continue to work.
"""

from agents.tools._email_utils import (  # noqa: F401 — re-export
    normalize_recipients,
    extract_emails_from_text,
    recipients_from_params,
    default_recipients,
    resolve_recipients,
)
