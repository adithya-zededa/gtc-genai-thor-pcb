"""PCB Inspection domain MCP — tool definitions.

Defines all PCB-specific MCP tool schemas, the ``PCBToolRegistry``,
and supporting constants (parameter allowlists, hours-aware tool sets).
"""

from __future__ import annotations

from typing import Dict, FrozenSet

from agents.mcp.registry import MCPToolDefinition, MCPToolRegistry
from agents.mcp.schema import MCPSchemaType, MCPParameterSchema, MCPOutputSchema, standard_output_schema
from agents.mcp.state_machine import AgentState


# ── Tool definitions ──────────────────────────────────────────────────────

TOOL_GET_LATEST_PCB_FRAMES = MCPToolDefinition(
    name="get_latest_pcb_frames",
    description="Retrieve metadata for the most recently stored PCB frames. Frames are auto-captured by the monitoring pipeline when a PCB is detected with low motion. Use this to see what frames are available before inspecting them.",
    category="pcb_analysis",
    input_schema=[
        MCPParameterSchema(name="limit", type=MCPSchemaType.INTEGER, description="Max number of frames to return", required=False, default=5, min_value=1, max_value=20),
        MCPParameterSchema(name="unconsumed_only", type=MCPSchemaType.BOOLEAN, description="Only return frames not yet inspected", required=False, default=True),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)

TOOL_INSPECT_PCB_FRAME = MCPToolDefinition(
    name="inspect_pcb_frame",
    description="Send a stored PCB frame to the VLM for defect inspection. Returns analysis so you can reason about next steps. By default this tool is analysis-only; set auto_log=true to persist detected defects. If frame_id is omitted, uses the most recent unconsumed frame.",
    category="pcb_analysis",
    input_schema=[
        MCPParameterSchema(name="frame_id", type=MCPSchemaType.INTEGER, description="ID of a stored frame from get_latest_pcb_frames. If omitted, uses the newest unconsumed frame.", required=False),
        MCPParameterSchema(name="query", type=MCPSchemaType.STRING, description="Optional specific question about the PCB", required=False, max_length=500),
        MCPParameterSchema(name="auto_log", type=MCPSchemaType.BOOLEAN, description="If true, automatically log detected defects", required=False, default=False),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)

TOOL_INSPECT_PCB = MCPToolDefinition(
    name="inspect_pcb",
    description="Inspect the current camera frame for PCB defects including solder bridges, missing components, trace damage, and more. Defects are auto-logged to the database — you do NOT need to call log_defect afterwards.",
    category="pcb_analysis",
    input_schema=[
        MCPParameterSchema(name="query", type=MCPSchemaType.STRING, description="Optional specific question about the PCB", required=False, max_length=500),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING),
)

TOOL_CLASSIFY_BOARD = MCPToolDefinition(
    name="classify_board",
    description="Identify the type of PCB board visible in the current frame.",
    category="pcb_analysis",
    input_schema=[],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING),
)

TOOL_SEND_DEFECT_ALERT = MCPToolDefinition(
    name="send_defect_alert",
    description="Send an email alert about a detected PCB defect to specified recipients. You should first inspect a frame using inspect_pcb_frame and only call this tool if you determined a defect is present.",
    category="pcb_alerts",
    input_schema=[
        MCPParameterSchema(name="recipients", type=MCPSchemaType.ARRAY, description="List of email addresses to alert", required=True, items_type=MCPSchemaType.STRING),
        MCPParameterSchema(name="board_type", type=MCPSchemaType.STRING, description="Board type (from inspection)", required=False, default="unknown"),
        MCPParameterSchema(name="defect_summary", type=MCPSchemaType.STRING, description="Description of the defect. If omitted, a summary is auto-generated from recent defect logs.", required=False, max_length=5000),
        MCPParameterSchema(name="severity", type=MCPSchemaType.STRING, description="Defect severity", required=False, enum=("low", "medium", "high"), default="medium"),
        MCPParameterSchema(name="include_image", type=MCPSchemaType.BOOLEAN, description="Attach the PCB frame image", required=False, default=True),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=True,
    confirmation_message="Send a PCB defect alert email to {recipients}?",
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ALERTING),
)

TOOL_LOG_DEFECT = MCPToolDefinition(
    name="log_defect",
    description="Manually record a PCB defect to the persistent database. Use when the user explicitly requests manual logging, or when inspect_pcb_frame was run with auto_log=false.",
    category="pcb_logging",
    input_schema=[
        MCPParameterSchema(name="board_type", type=MCPSchemaType.STRING, description="Board type", required=True),
        MCPParameterSchema(name="defect_type", type=MCPSchemaType.STRING, description="Defect classification", required=True),
        MCPParameterSchema(name="severity", type=MCPSchemaType.STRING, description="Severity level", required=False, enum=("low", "medium", "high"), default="low"),
        MCPParameterSchema(name="confidence", type=MCPSchemaType.NUMBER, description="Detection confidence 0-1", required=False, min_value=0.0, max_value=1.0),
        MCPParameterSchema(name="description", type=MCPSchemaType.STRING, description="Human-readable defect description", required=False, max_length=2000),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING, AgentState.ALERTING),
)

TOOL_GENERATE_DEFECT_REPORT = MCPToolDefinition(
    name="generate_defect_report",
    description="Generate a summary report of all recorded PCB defects.",
    category="pcb_reporting",
    input_schema=[
        MCPParameterSchema(name="board_type", type=MCPSchemaType.STRING, description="Filter report to a specific board type", required=False),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)

TOOL_QUERY_PCB_INSPECTIONS = MCPToolDefinition(
    name="query_pcb_inspections",
    description="Query past PCB inspection results from the database. Use this to answer user questions about detected defects, pass rates, board types seen, and inspection history. Supports time-window filtering.",
    category="pcb_reporting",
    input_schema=[
        MCPParameterSchema(name="limit", type=MCPSchemaType.INTEGER, description="Max records to return", required=False, default=10, min_value=1, max_value=50),
        MCPParameterSchema(name="result_filter", type=MCPSchemaType.STRING, description="Filter by result: PASS, FAIL, or omit for all", required=False, enum=("PASS", "FAIL")),
        MCPParameterSchema(name="hours", type=MCPSchemaType.NUMBER, description="Time window in hours (e.g., 24 for last day, 168 for last week)", required=False),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)


# ── Monitoring analytics tools ────────────────────────────────────────────

_ANALYTICS_STATES = (AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING)

TOOL_GET_MONITORING_STATUS = MCPToolDefinition(
    name="get_monitoring_status",
    description="Get the current and historical monitoring status including whether monitoring is active, total defect counts, and recent activity summary.",
    category="pcb_analytics",
    input_schema=[],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ANALYTICS_STATES,
)

TOOL_TOGGLE_EMAIL_NOTIFICATIONS = MCPToolDefinition(
    name="toggle_email_notifications",
    description="Enable or disable email notifications for detected defects. Can also set minimum severity threshold and configure recipient list.",
    category="pcb_notifications",
    input_schema=[
        MCPParameterSchema(name="enabled", type=MCPSchemaType.BOOLEAN, description="True to enable, False to disable. Omit to toggle.", required=False),
        MCPParameterSchema(name="min_severity", type=MCPSchemaType.STRING, description="Minimum severity to trigger notifications", required=False, enum=("low", "medium", "high")),
        MCPParameterSchema(name="recipients", type=MCPSchemaType.ARRAY, description="Email addresses for notifications", required=False, items_type=MCPSchemaType.STRING),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=tuple(AgentState),
)

TOOL_GET_DEFECT_SUMMARY = MCPToolDefinition(
    name="get_defect_summary",
    description="Get a summary of defects identified within a specific time range, log, or board type. Use when user asks about defects in a period or for a specific source.",
    category="pcb_analytics",
    input_schema=[
        MCPParameterSchema(name="hours", type=MCPSchemaType.NUMBER, description="Time window in hours (e.g., 24 for last day, 168 for last week)", required=False),
        MCPParameterSchema(name="board_type", type=MCPSchemaType.STRING, description="Filter by board type", required=False),
        MCPParameterSchema(name="severity", type=MCPSchemaType.STRING, description="Filter by severity", required=False, enum=("low", "medium", "high")),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ANALYTICS_STATES,
)

TOOL_COUNT_DEFECTIVE_PCBS = MCPToolDefinition(
    name="count_defective_pcbs",
    description="Count how many defective PCBs were detected overall or within a given time window. Use when user asks 'how many defects' or 'how many bad PCBs'.",
    category="pcb_analytics",
    input_schema=[
        MCPParameterSchema(name="hours", type=MCPSchemaType.NUMBER, description="Time window in hours. Omit for all-time count.", required=False),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ANALYTICS_STATES,
)

TOOL_GET_LATEST_DEFECT = MCPToolDefinition(
    name="get_latest_defect",
    description="Get the most recently identified defect with full details. Use when user asks 'when was the last defect' or 'most recent defect'.",
    category="pcb_analytics",
    input_schema=[],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ANALYTICS_STATES,
)

TOOL_GET_DEFECT_TYPE_BREAKDOWN = MCPToolDefinition(
    name="get_defect_type_breakdown",
    description="Get what types of defects have been identified and how frequently each occurs. Use when user asks about defect categories, types, or frequencies.",
    category="pcb_analytics",
    input_schema=[
        MCPParameterSchema(name="hours", type=MCPSchemaType.NUMBER, description="Time window in hours", required=False),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ANALYTICS_STATES,
)

TOOL_GET_DEFECT_TREND = MCPToolDefinition(
    name="get_defect_trend",
    description="Analyze whether defect rates are increasing, decreasing, or stable over time. Use when user asks about trends or rate changes.",
    category="pcb_analytics",
    input_schema=[
        MCPParameterSchema(name="hours", type=MCPSchemaType.NUMBER, description="Analysis window in hours (default: 168 = 1 week)", required=False),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ANALYTICS_STATES,
)

TOOL_GET_MOST_SEVERE_DEFECT = MCPToolDefinition(
    name="get_most_severe_defect",
    description="Get the most severe or highest-priority defect detected. Use when user asks about worst defect, highest severity, or most critical issue.",
    category="pcb_analytics",
    input_schema=[
        MCPParameterSchema(name="hours", type=MCPSchemaType.NUMBER, description="Time window in hours", required=False),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ANALYTICS_STATES,
)

TOOL_GET_TOP_DEFECT_SOURCES = MCPToolDefinition(
    name="get_top_defect_sources",
    description="Get which board types or sources produce the most defects. Use when user asks which boards have the most problems.",
    category="pcb_analytics",
    input_schema=[
        MCPParameterSchema(name="hours", type=MCPSchemaType.NUMBER, description="Time window in hours", required=False),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ANALYTICS_STATES,
)

TOOL_GENERATE_SUMMARY_REPORT = MCPToolDefinition(
    name="generate_summary_report",
    description="Generate a comprehensive daily, weekly, or on-demand summary report of monitoring results including defect statistics, trends, and threshold status.",
    category="pcb_reporting",
    input_schema=[
        MCPParameterSchema(name="period", type=MCPSchemaType.STRING, description="Report period", required=False, enum=("daily", "weekly", "all"), default="daily"),
        MCPParameterSchema(name="board_type", type=MCPSchemaType.STRING, description="Filter to a specific board type", required=False),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ANALYTICS_STATES,
)

TOOL_CHECK_THRESHOLD_ALERTS = MCPToolDefinition(
    name="check_threshold_alerts",
    description="Check whether any defects exceeded predefined thresholds or alert conditions. Use when user asks about threshold violations or alert conditions.",
    category="pcb_analytics",
    input_schema=[
        MCPParameterSchema(name="rate_threshold", type=MCPSchemaType.NUMBER, description="Max acceptable defects per window (default: 10)", required=False),
        MCPParameterSchema(name="window_hours", type=MCPSchemaType.NUMBER, description="Time window in hours (default: 24)", required=False),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ANALYTICS_STATES,
)

TOOL_GET_DEFECT_INSIGHTS = MCPToolDefinition(
    name="get_defect_insights",
    description="Get AI-driven recommendations and insights based on observed defect patterns. Use when user asks for analysis, recommendations, or 'what should we do'.",
    category="pcb_analytics",
    input_schema=[
        MCPParameterSchema(name="hours", type=MCPSchemaType.NUMBER, description="Analysis window in hours", required=False),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ANALYTICS_STATES,
)

TOOL_GET_NOTIFICATION_PREFERENCES = MCPToolDefinition(
    name="get_notification_preferences",
    description="Get the current notification preferences and configuration. Use when user asks about their notification settings.",
    category="pcb_notifications",
    input_schema=[],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=tuple(AgentState),
)

TOOL_QUERY_DETECTION_LOGS = MCPToolDefinition(
    name="query_detection_logs",
    description="Query detection logs enriched with defect information. Provides a unified view of detection events and defect records. Use when user asks to see logs, detection history, or what was inspected.",
    category="pcb_reporting",
    input_schema=[
        MCPParameterSchema(name="limit", type=MCPSchemaType.INTEGER, description="Max records to return (default 20)", required=False, default=20, min_value=1, max_value=100),
        MCPParameterSchema(name="hours", type=MCPSchemaType.NUMBER, description="Time window in hours", required=False),
        MCPParameterSchema(name="detected_only", type=MCPSchemaType.BOOLEAN, description="Only return detections with confidence > 0", required=False, default=False),
        MCPParameterSchema(name="include_defects", type=MCPSchemaType.BOOLEAN, description="Include defect records alongside detection logs", required=False, default=True),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ANALYTICS_STATES,
)


# ── All PCB tools list (for registry) ─────────────────────────────────────

ALL_PCB_TOOLS = [
    # Core PCB analysis
    TOOL_GET_LATEST_PCB_FRAMES,
    TOOL_INSPECT_PCB_FRAME,
    TOOL_INSPECT_PCB,
    TOOL_CLASSIFY_BOARD,
    TOOL_SEND_DEFECT_ALERT,
    TOOL_LOG_DEFECT,
    TOOL_GENERATE_DEFECT_REPORT,
    TOOL_QUERY_PCB_INSPECTIONS,
    # Monitoring analytics & chat queries
    TOOL_GET_MONITORING_STATUS,
    TOOL_TOGGLE_EMAIL_NOTIFICATIONS,
    TOOL_GET_DEFECT_SUMMARY,
    TOOL_COUNT_DEFECTIVE_PCBS,
    TOOL_GET_LATEST_DEFECT,
    TOOL_GET_DEFECT_TYPE_BREAKDOWN,
    TOOL_GET_DEFECT_TREND,
    TOOL_GET_MOST_SEVERE_DEFECT,
    TOOL_GET_TOP_DEFECT_SOURCES,
    TOOL_GENERATE_SUMMARY_REPORT,
    TOOL_CHECK_THRESHOLD_ALERTS,
    TOOL_GET_DEFECT_INSIGHTS,
    TOOL_GET_NOTIFICATION_PREFERENCES,
    TOOL_QUERY_DETECTION_LOGS,
]


# ── Registry ──────────────────────────────────────────────────────────────

class PCBToolRegistry(MCPToolRegistry):
    """Registry containing only PCB-domain tools."""

    def _register_default_tools(self) -> None:
        for tool in ALL_PCB_TOOLS:
            self.register(tool)


# ── Context allowlist ─────────────────────────────────────────────────────

TOOL_PARAM_ALLOWLIST: Dict[str, FrozenSet[str]] = {
    "get_latest_pcb_frames": frozenset(["limit", "unconsumed_only"]),
    "inspect_pcb_frame": frozenset(["frame_id", "query", "auto_log"]),
    "inspect_pcb": frozenset(["query"]),
    "classify_board": frozenset(),
    "send_defect_alert": frozenset(["recipients", "board_type", "defect_summary", "severity", "include_image", "image_data"]),
    "log_defect": frozenset(["board_type", "defect_type", "severity", "confidence", "description", "image_path"]),
    "generate_defect_report": frozenset(["board_type"]),
    "query_pcb_inspections": frozenset(["limit", "result_filter", "hours"]),
    # Monitoring analytics & chat queries
    "get_monitoring_status": frozenset(),
    "toggle_email_notifications": frozenset(["enabled", "min_severity", "recipients"]),
    "get_defect_summary": frozenset(["hours", "board_type", "severity"]),
    "count_defective_pcbs": frozenset(["hours"]),
    "get_latest_defect": frozenset(),
    "get_defect_type_breakdown": frozenset(["hours"]),
    "get_defect_trend": frozenset(["hours"]),
    "get_most_severe_defect": frozenset(["hours"]),
    "get_top_defect_sources": frozenset(["hours"]),
    "generate_summary_report": frozenset(["period", "board_type"]),
    "check_threshold_alerts": frozenset(["rate_threshold", "window_hours"]),
    "get_defect_insights": frozenset(["hours"]),
    "get_notification_preferences": frozenset(),
    "query_detection_logs": frozenset(["limit", "hours", "detected_only", "include_defects"]),
}

HOURS_AWARE_TOOLS: FrozenSet[str] = frozenset({
    "get_defect_summary",
    "count_defective_pcbs",
    "get_defect_type_breakdown",
    "get_defect_trend",
    "get_most_severe_defect",
    "get_top_defect_sources",
    "get_defect_insights",
    "query_detection_logs",
    "query_pcb_inspections",
})
