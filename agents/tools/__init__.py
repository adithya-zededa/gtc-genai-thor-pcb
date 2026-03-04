"""Tool calling DTOs and handler functions for the agent system.

Canonical tool implementations live in domain-specific subpackages:
- ``general/`` — alert, evidence, event_log, history
- ``pcb/``     — PCB defect inspection, analytics, reporting
"""

from .base import ToolCall, ToolResult
from .email import send_email

# General-domain tools (re-exported for backward compatibility)
from .general.alert import tool_send_alert_email
from .general.evidence import tool_save_evidence
from .general.event_log import tool_log_event
from .general.history import tool_query_history

__all__ = [
    "ToolCall",
    "ToolResult",
    "send_email",
    "tool_send_alert_email",
    "tool_save_evidence",
    "tool_log_event",
    "tool_query_history",
]
