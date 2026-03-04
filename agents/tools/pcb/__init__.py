"""PCB inspection domain tools package.

Re-exports all public tool functions and helpers so that existing
imports are fully preserved::

    from agents.tools.pcb import tool_inspect_pcb, tool_log_defect, ...
"""

from ._helpers import get_inspection_prompt

# Analysis tools
from .analysis import (
    tool_inspect_pcb,
    tool_classify_board,
    tool_get_latest_pcb_frames,
    tool_inspect_pcb_frame,
)

# Alert / notification tools
from .alerts import (
    tool_send_defect_alert,
    tool_toggle_email_notifications,
    tool_get_notification_preferences,
)

# Defect logging
from .defect_logging import tool_log_defect

# Analytics / query tools
from .analytics import (
    tool_get_monitoring_status,
    tool_get_defect_summary,
    tool_count_defective_pcbs,
    tool_get_latest_defect,
    tool_get_defect_type_breakdown,
    tool_get_defect_trend,
    tool_get_most_severe_defect,
    tool_get_top_defect_sources,
    tool_check_threshold_alerts,
    tool_get_defect_insights,
    tool_query_detection_logs,
)

# Reporting tools
from .reporting import (
    tool_generate_defect_report,
    tool_generate_summary_report,
    tool_query_pcb_inspections,
)

# Deprecated monitoring stubs
from .monitoring import (
    tool_start_defect_monitoring,
    tool_stop_defect_monitoring,
)

__all__ = [
    "get_inspection_prompt",
    # Analysis
    "tool_inspect_pcb",
    "tool_classify_board",
    "tool_get_latest_pcb_frames",
    "tool_inspect_pcb_frame",
    # Alerts
    "tool_send_defect_alert",
    "tool_toggle_email_notifications",
    "tool_get_notification_preferences",
    # Logging
    "tool_log_defect",
    # Analytics
    "tool_get_monitoring_status",
    "tool_get_defect_summary",
    "tool_count_defective_pcbs",
    "tool_get_latest_defect",
    "tool_get_defect_type_breakdown",
    "tool_get_defect_trend",
    "tool_get_most_severe_defect",
    "tool_get_top_defect_sources",
    "tool_check_threshold_alerts",
    "tool_get_defect_insights",
    "tool_query_detection_logs",
    # Reporting
    "tool_generate_defect_report",
    "tool_generate_summary_report",
    "tool_query_pcb_inspections",
    # Deprecated
    "tool_start_defect_monitoring",
    "tool_stop_defect_monitoring",
]
