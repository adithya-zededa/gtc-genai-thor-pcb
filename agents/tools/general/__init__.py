"""General-domain tool implementations.

These tools handle cross-domain concerns: alerting, evidence capture,
event logging, and history queries. They are used by the general-domain
MCP executor and can also be called directly.
"""

from .alert import tool_send_alert_email
from .evidence import tool_save_evidence
from .event_log import tool_log_event
from .history import tool_query_history

__all__ = [
    "tool_send_alert_email",
    "tool_save_evidence",
    "tool_log_event",
    "tool_query_history",
]
