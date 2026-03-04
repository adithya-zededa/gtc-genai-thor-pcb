"""Shared helpers for PCB tool handlers.

Internal module — not part of the public API.  Tool submodules import
from here to avoid circular dependencies and code duplication.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from agents.tools.validation import (
    validate_emails as _validate_emails,
    safe_error as _safe_error,
    sanitise_severity as _sanitise_severity,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Fingerprint-based dedup cache for _auto_log_defect
# ---------------------------------------------------------------------------
_DEFECT_FINGERPRINT_CACHE: Dict[str, float] = {}
_DEDUP_TTL_SECONDS: float = 60.0


def _compute_defect_fingerprint(
    board_type: str,
    defect_type: str,
    severity: str,
    description: str,
) -> str:
    """Return a hex digest uniquely identifying this defect record."""
    key = f"{board_type}\x00{defect_type}\x00{severity}\x00{description[:200]}"
    return hashlib.md5(key.encode("utf-8", errors="replace")).hexdigest()


def get_inspection_prompt() -> str:
    """Return the active PCB defect inspection prompt.

    Reads from ``config.yaml`` (``detection.inspection_prompt``).  If that
    field is empty or missing, falls back to
    ``agents.vlm.prompts.DEFAULT_MONITORING_DEFECT_PROMPT``.
    This allows users to customise the prompt from the Settings UI without
    redeploying.
    """
    try:
        from services.infrastructure.config import load_camera_config

        cfg = load_camera_config()
        custom = (cfg.get("detection") or {}).get("inspection_prompt", "") or ""
        if isinstance(custom, str) and custom.strip():
            return custom.strip()
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    from agents.vlm.prompts import DEFAULT_MONITORING_DEFECT_PROMPT
    return DEFAULT_MONITORING_DEFECT_PROMPT


# ---------------------------------------------------------------------------
# Automatic defect notification helper
# ---------------------------------------------------------------------------

def _auto_notify_defect(
    board_type: str,
    defect_type: str,
    severity: str,
    description: str,
    confidence: float,
) -> None:
    """Send an automatic email notification if preferences are enabled.

    Respects the user's notification preferences at all times:
    - Only sends if email_enabled is True
    - Only sends if severity meets min_severity threshold
    - Only sends to configured recipients
    """
    try:
        from services.domains.pcb.notification_preferences import get_notification_preferences

        prefs = get_notification_preferences()
        if not prefs.should_notify(severity):
            return

        if not prefs.email_recipients:
            logger.debug("Auto-notify: no recipients configured, skipping")
            return

        from agents.tools.email import send_email

        subject = f"[PCB ALERT] {severity.upper()} — {defect_type} on {board_type}"
        body = (
            f"Automated PCB Defect Notification\n"
            f"{'=' * 40}\n\n"
            f"Board Type:  {board_type}\n"
            f"Defect Type: {defect_type}\n"
            f"Severity:    {severity}\n"
            f"Confidence:  {confidence:.2f}\n"
            f"Time:        {datetime.now().isoformat()}\n\n"
            f"Description:\n{description}\n\n"
            f"---\n"
            f"This is an automated notification from the PCB monitoring agent.\n"
            f"Manage notification preferences via the chat interface.\n"
        )

        send_email({
            "to": prefs.email_recipients,
            "subject": subject,
            "body": body,
        })
        logger.info(
            "Auto-notification sent for %s defect to %d recipient(s)",
            severity, len(prefs.email_recipients),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Auto-notification failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Automatic defect logging helper
# ---------------------------------------------------------------------------

def _auto_log_defect(
    detected: bool,
    full_response: Any,
    description: str,
    confidence: float,
    image_path: str = "",
) -> None:
    """Automatically log defect data when an inspection finds issues.

    This removes the need for the LLM to issue a separate ``log_defect``
    tool call after every inspection — the persistence is a side-effect
    of inspection itself.  The LLM can still override or amend the log
    via the explicit ``tool_log_defect`` tool.

    Also triggers email notifications if enabled in notification preferences.

    Every individual defect discovered is logged as a separate record so
    that counts, trends, and breakdowns reflect the actual number of
    distinct issues — not just "one row per inspection".
    """
    if not detected:
        logger.debug("_auto_log_defect: no defect detected, skipping")
        return  # Nothing to log for clean inspections

    import json as _json

    analysis: Dict[str, Any] = {}
    if full_response and isinstance(full_response, str):
        try:
            analysis = _json.loads(full_response)
            if not isinstance(analysis, dict):
                analysis = {}
        except (ValueError, TypeError):
            analysis = {}

    board_type = analysis.get("board_type", "unknown")

    # Also try to extract board type from nested details
    details = analysis.get("details", {})
    if board_type == "unknown" and isinstance(details, dict):
        board_type = details.get("board_type", "unknown")

    defect_list = analysis.get("defects", [])
    # Fallback: extract defects from details.defects if top-level is empty
    if not defect_list and isinstance(details, dict):
        defect_list = details.get("defects", [])

    # Build a rich description from the VLM output rather than a generic string
    rich_description = description or ""
    if not rich_description and analysis.get("reasoning"):
        rich_description = str(analysis["reasoning"])

    # Determine overall severity and defect type
    defect_type = "general_defect"
    severity = "medium"

    if defect_list and isinstance(defect_list, list):
        first = defect_list[0] if defect_list else {}
        if isinstance(first, dict):
            defect_type = first.get("type", defect_type)
            sev = first.get("severity", "").lower()
            if sev in ("low", "medium", "high"):
                severity = sev

        # Escalate to "high" if any single defect is high/critical
        for d in defect_list:
            if isinstance(d, dict) and d.get("severity", "").lower() in ("high", "critical"):
                severity = "high"
                break

    # Also infer severity from component statuses in details
    if isinstance(details, dict):
        for status_key in ("power_jack_status", "usb_connector_status", "header_pin_status"):
            status_val = details.get(status_key, "")
            if status_val in ("missing", "damaged"):
                severity = "high"
                if defect_type == "general_defect":
                    defect_type = f"{status_key.replace('_status', '')}_{status_val}"
                break

    # ── Dedup: skip if identical fingerprint logged within TTL window ──
    _now = time.monotonic()
    stale = [k for k, ts in _DEFECT_FINGERPRINT_CACHE.items()
             if _now - ts > _DEDUP_TTL_SECONDS]
    for k in stale:
        del _DEFECT_FINGERPRINT_CACHE[k]

    fp = _compute_defect_fingerprint(
        board_type=board_type,
        defect_type=defect_type,
        severity=severity,
        description=rich_description or description,
    )
    if fp in _DEFECT_FINGERPRINT_CACHE:
        logger.debug(
            "_auto_log_defect: skipping duplicate (fp=%s, board=%s, type=%s)",
            fp[:8], board_type, defect_type,
        )
        return
    _DEFECT_FINGERPRINT_CACHE[fp] = _now
    # ──────────────────────────────────────────────────────────────────

    logged_count = 0
    from services.domains.pcb.service import record_defect

    # Log each individual defect as a separate record
    if defect_list and isinstance(defect_list, list) and len(defect_list) > 0:
        for idx, defect_item in enumerate(defect_list):
            if not isinstance(defect_item, dict):
                continue
            d_type = defect_item.get("type", defect_type)
            d_sev = defect_item.get("severity", severity).lower()
            if d_sev not in ("low", "medium", "high"):
                d_sev = severity
            d_loc = defect_item.get("location", "")
            d_desc = defect_item.get("description", "")
            combined_desc = f"{d_desc}" if d_desc else rich_description
            if d_loc:
                combined_desc = f"[{d_loc}] {combined_desc}"
            try:
                record_defect(
                    board_type=board_type,
                    defect_type=d_type,
                    severity=d_sev,
                    confidence=confidence,
                    image_path=image_path,
                    description=combined_desc or rich_description or "Auto-logged from VLM inspection",
                )
                logged_count += 1
                logger.info(
                    "Auto-logged defect %d/%d: type=%s severity=%s board=%s location=%s",
                    idx + 1, len(defect_list), d_type, d_sev, board_type, d_loc,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Auto-log defect %d failed (non-fatal): %s", idx + 1, exc)
    else:
        # No structured defect list — log a single record with best-available data
        try:
            record_defect(
                board_type=board_type,
                defect_type=defect_type,
                severity=severity,
                confidence=confidence,
                image_path=image_path,
                description=rich_description or "Auto-logged from VLM inspection",
            )
            logged_count = 1
            logger.info("Auto-logged defect: type=%s severity=%s board=%s", defect_type, severity, board_type)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Auto-log defect failed (non-fatal): %s", exc)

    logger.info(
        "Auto-log summary: %d defect(s) persisted for board=%s (confidence=%.2f, image=%s)",
        logged_count, board_type, confidence, bool(image_path),
    )

    # Trigger notification if preferences allow
    _auto_notify_defect(
        board_type=board_type,
        defect_type=defect_type,
        severity=severity,
        description=rich_description or description,
        confidence=confidence,
    )


def _auto_generate_defect_summary() -> str:
    """Build a human-readable defect summary from recent defect logs."""
    try:
        from services.domains.pcb.defect_store import get_defects_in_range

        defects = get_defects_in_range(start_time=None)
        if not defects:
            return "No defects recorded."

        lines: list[str] = [f"Total defects on record: {len(defects)}\n"]

        # Severity breakdown
        sev_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for d in defects:
            s = d.get("severity", "unknown")
            sev_counts[s] = sev_counts.get(s, 0) + 1
            t = d.get("defect_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        if sev_counts:
            lines.append("Severity breakdown:")
            for sev in ("high", "medium", "low"):
                if sev in sev_counts:
                    lines.append(f"  - {sev}: {sev_counts[sev]}")

        if type_counts:
            lines.append("\nDefect types:")
            for dtype, cnt in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {dtype}: {cnt}")

        # Include details from the most recent defects (up to 10)
        recent = defects[:10]
        if recent:
            lines.append("\nRecent defective events:")
            for i, d in enumerate(recent, 1):
                ts = d.get("timestamp", "N/A")
                dtype = d.get("defect_type", "unknown")
                sev = d.get("severity", "unknown")
                desc = d.get("description", "")
                board = d.get("board_type", "unknown")
                entry = f"  {i}. [{ts}] {dtype} ({sev}) on {board}"
                if desc:
                    entry += f" — {desc[:120]}"
                lines.append(entry)

        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Failed to auto-generate defect summary: %s", exc)
        return "Unable to retrieve defect logs automatically."
