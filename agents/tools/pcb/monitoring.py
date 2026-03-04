"""Deprecated PCB monitoring lifecycle tools."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)


def tool_start_defect_monitoring(
    recipients: Optional[List[str]] = None,
    polling_interval: float = 10.0,
    auto_alert: bool = True,
) -> Dict[str, Any]:
    """Deprecated — the monitoring loop and the LLM now handle inspection.

    The proactive monitoring loop observes the camera feed and delegates
    inspection decisions to the LLM via MCP tools.  Use
    ``start_monitoring_session`` instead.
    """
    logger.info(
        "tool_start_defect_monitoring is deprecated — use start_monitoring_session "
        "(ignored args: recipients=%s, polling_interval=%s, auto_alert=%s)",
        recipients,
        polling_interval,
        auto_alert,
    )
    return {
        "success": False,
        "message": (
            "Deprecated: continuous defect monitoring is now handled by the "
            "proactive monitoring loop.  Use `start_monitoring_session` to "
            "begin monitoring.  The LLM will decide when to inspect and alert. "
            "Provided parameters were ignored."
        ),
    }


def tool_stop_defect_monitoring() -> Dict[str, Any]:
    """Deprecated — see tool_start_defect_monitoring."""
    logger.info("tool_stop_defect_monitoring is deprecated")
    return {
        "success": True,
        "message": "Defect monitoring is managed by the proactive monitoring loop.",
    }
