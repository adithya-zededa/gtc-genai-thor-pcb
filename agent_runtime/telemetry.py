"""Shared telemetry helpers (logging and metrics stubs)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

_LOG_LEVEL = os.getenv("AGENT_TELEMETRY_LEVEL", "INFO").upper()
logging.basicConfig(level=_LOG_LEVEL)
telemetry_logger = logging.getLogger("agent.runtime")


def emit_metric(name: str, value: float, tags: Dict[str, Any] | None = None) -> None:
    """Emit a structured metric (placeholder for Prometheus/OTel integration)."""
    if telemetry_logger.isEnabledFor(logging.DEBUG):
        telemetry_logger.debug("metric %s=%s tags=%s", name, value, tags or {})
