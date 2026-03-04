"""PCB Inspection domain — tool executor.

Dispatches approved PCB tool call proposals to the corresponding
``agents.tools.pcb`` handler functions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.mcp.executor_base import BaseDomainExecutor
from agents.mcp.globals import get_agent_state_machine, get_audit_log

from .tool_defs import PCBToolRegistry, TOOL_PARAM_ALLOWLIST

from core.logging import get_logger

logger = get_logger(__name__)


class PCBExecutor(BaseDomainExecutor):
    """Executes approved PCB tool call proposals."""

    _domain_label = "pcb"

    def __init__(
        self,
        registry: Optional[PCBToolRegistry] = None,
        state_machine=None,
        context: Optional[Dict[str, Any]] = None,
        invoke_timeout: int = 60,
    ):
        reg = registry or PCBToolRegistry()
        super().__init__(
            registry=reg,
            state_machine=state_machine or get_agent_state_machine(),
            audit_log=get_audit_log(),
            context=context,
            invoke_timeout=invoke_timeout,
        )

    def _invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from agents.tools.pcb import (
            tool_get_latest_pcb_frames,
            tool_inspect_pcb_frame,
            tool_inspect_pcb,
            tool_classify_board,
            tool_send_defect_alert,
            tool_log_defect,
            tool_generate_defect_report,
            tool_start_defect_monitoring,
            tool_stop_defect_monitoring,
            tool_query_pcb_inspections,
            # Monitoring analytics & chat queries
            tool_get_monitoring_status,
            tool_toggle_email_notifications,
            tool_get_defect_summary,
            tool_count_defective_pcbs,
            tool_get_latest_defect,
            tool_get_defect_type_breakdown,
            tool_get_defect_trend,
            tool_get_most_severe_defect,
            tool_get_top_defect_sources,
            tool_generate_summary_report,
            tool_check_threshold_alerts,
            tool_get_defect_insights,
            tool_get_notification_preferences,
            tool_query_detection_logs,
        )

        handlers = {
            "get_latest_pcb_frames": tool_get_latest_pcb_frames,
            "inspect_pcb_frame": tool_inspect_pcb_frame,
            "inspect_pcb": tool_inspect_pcb,
            "classify_board": tool_classify_board,
            "send_defect_alert": tool_send_defect_alert,
            "log_defect": tool_log_defect,
            "generate_defect_report": tool_generate_defect_report,
            "start_defect_monitoring": tool_start_defect_monitoring,
            "stop_defect_monitoring": tool_stop_defect_monitoring,
            "query_pcb_inspections": tool_query_pcb_inspections,
            # Monitoring analytics & chat queries
            "get_monitoring_status": tool_get_monitoring_status,
            "toggle_email_notifications": tool_toggle_email_notifications,
            "get_defect_summary": tool_get_defect_summary,
            "count_defective_pcbs": tool_count_defective_pcbs,
            "get_latest_defect": tool_get_latest_defect,
            "get_defect_type_breakdown": tool_get_defect_type_breakdown,
            "get_defect_trend": tool_get_defect_trend,
            "get_most_severe_defect": tool_get_most_severe_defect,
            "get_top_defect_sources": tool_get_top_defect_sources,
            "generate_summary_report": tool_generate_summary_report,
            "check_threshold_alerts": tool_check_threshold_alerts,
            "get_defect_insights": tool_get_defect_insights,
            "get_notification_preferences": tool_get_notification_preferences,
            "query_detection_logs": tool_query_detection_logs,
        }

        handler = handlers.get(tool_name)
        if not handler:
            raise ValueError(f"No PCB handler for: {tool_name}")

        allowed = TOOL_PARAM_ALLOWLIST.get(tool_name, frozenset())
        filtered = {k: v for k, v in arguments.items() if k in allowed}
        return handler(**filtered)
