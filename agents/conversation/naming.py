"""User-facing labels for MCP tool names."""

from __future__ import annotations

from typing import Dict


TOOL_DISPLAY_NAMES: Dict[str, str] = {
    "start_monitoring_session": "Start Monitoring",
    "end_session": "End Monitoring Session",
    "go_idle": "Pause Monitoring",
    "get_agent_status": "Check Agent Status",
    "analyze_current_frame": "Analyze Live Frame",
    "query_history": "Review Event History",
    "get_session_summary": "Generate Session Summary",
    "set_detection_task": "Configure Detection Task",
    "send_alert_email": "Send Alert Email",
    "save_evidence": "Save Evidence Snapshot",
    "log_event": "Record Monitoring Event",
    "shutdown_agent": "Shutdown Agent",
    "acknowledge_error": "Acknowledge Error",
    "get_latest_pcb_frames": "List Captured PCB Frames",
    "inspect_pcb_frame": "Inspect Stored PCB Frame",
    "inspect_pcb": "Inspect Live PCB",
    "classify_board": "Identify Board Type",
    "send_defect_alert": "Send Defect Alert",
    "log_defect": "Record Defect",
    "generate_defect_report": "Generate Defect Report",
    "query_pcb_inspections": "Query PCB Inspections",
    "get_monitoring_status": "Check Monitoring Status",
    "toggle_email_notifications": "Update Email Notifications",
    "get_defect_summary": "Summarize Defects",
    "count_defective_pcbs": "Count Defective PCBs",
    "get_latest_defect": "Get Latest Defect",
    "get_defect_type_breakdown": "Defect Type Breakdown",
    "get_defect_trend": "Analyze Defect Trend",
    "get_most_severe_defect": "Find Most Severe Defect",
    "get_top_defect_sources": "Top Defect Sources",
    "generate_summary_report": "Generate Monitoring Report",
    "check_threshold_alerts": "Check Threshold Alerts",
    "get_defect_insights": "Generate Defect Insights",
    "get_notification_preferences": "View Notification Settings",
}


def friendly_tool_name(tool_name: str) -> str:
    """Return a polished user-facing tool/action label."""
    if tool_name in TOOL_DISPLAY_NAMES:
        return TOOL_DISPLAY_NAMES[tool_name]

    compact = "".join(ch if ch.isalnum() else " " for ch in (tool_name or ""))
    tokens = [t for t in compact.split() if t]
    if not tokens and tool_name:
        tokens = [tool_name]
    return " ".join(t.capitalize() for t in tokens) if tokens else "Requested Action"
