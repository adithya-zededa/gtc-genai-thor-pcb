"""PCB Inspection domain MCP — registry, interpreter, and executor.

Provides a self-contained MCP for PCB defect detection workflows.
Reuses the core MCP infrastructure via ``BaseDomainExecutor`` and
registers only PCB-specific tools and intent phrases.

Capabilities
~~~~~~~~~~~~
- **PCB analysis**: get/inspect stored frames, live inspection, board classification.
- **Defect management**: log, report, and query defects.
- **Monitoring analytics**: status, trends, severity ranking, source analysis.
- **Notification control**: enable/disable/configure email notifications via chat.
- **Reporting**: daily/weekly/on-demand summary reports with threshold checking.
- **Insights**: AI-driven recommendations based on observed defect patterns.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, FrozenSet, List, Optional

from agents.mcp.audit import AuditEventType, AuditLogEntry
from agents.mcp.email_arg_utils import resolve_recipients
from agents.mcp.executor_base import BaseDomainExecutor
from agents.mcp.globals import get_agent_state_machine, get_audit_log
from agents.mcp.lifecycle import MCPToolCallProposal
from agents.mcp.registry import MCPToolDefinition, MCPToolRegistry
from agents.mcp.schema import MCPSchemaType, MCPParameterSchema, MCPOutputSchema, standard_output_schema
from agents.mcp.state_machine import AgentState

from core.logging import get_logger

logger = get_logger(__name__)


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
        MCPParameterSchema(name="defect_summary", type=MCPSchemaType.STRING, description="Description of the defect (from inspection)", required=True, min_length=1, max_length=2000),
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

TOOL_START_DEFECT_MONITORING = MCPToolDefinition(
    name="start_defect_monitoring",
    description="Deprecated: continuous monitoring is handled by the proactive monitoring loop. Use start_monitoring_session instead.",
    category="pcb_monitoring",
    input_schema=[],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING),
)

TOOL_STOP_DEFECT_MONITORING = MCPToolDefinition(
    name="stop_defect_monitoring",
    description="Stop the continuous defect monitoring loop. Returns stats on frames inspected and defects found.",
    category="pcb_monitoring",
    input_schema=[],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING, AgentState.ALERTING),
)

TOOL_QUERY_PCB_INSPECTIONS = MCPToolDefinition(
    name="query_pcb_inspections",
    description="Query past PCB inspection results from the database. Use this to answer user questions about detected defects, pass rates, board types seen, and inspection history.",
    category="pcb_reporting",
    input_schema=[
        MCPParameterSchema(name="limit", type=MCPSchemaType.INTEGER, description="Max records to return", required=False, default=10, min_value=1, max_value=50),
        MCPParameterSchema(name="result_filter", type=MCPSchemaType.STRING, description="Filter by result: PASS, FAIL, or omit for all", required=False, enum=("PASS", "FAIL")),
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


# ── Registry ──────────────────────────────────────────────────────────────

class PCBToolRegistry(MCPToolRegistry):
    """Registry containing only PCB-domain tools."""

    def _register_default_tools(self) -> None:
        for tool in [
            # Core PCB analysis
            TOOL_GET_LATEST_PCB_FRAMES,
            TOOL_INSPECT_PCB_FRAME,
            TOOL_INSPECT_PCB,
            TOOL_CLASSIFY_BOARD,
            TOOL_SEND_DEFECT_ALERT,
            TOOL_LOG_DEFECT,
            TOOL_GENERATE_DEFECT_REPORT,
            TOOL_START_DEFECT_MONITORING,
            TOOL_STOP_DEFECT_MONITORING,
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
        ]:
            self.register(tool)


# ── Context allowlist ─────────────────────────────────────────────────────

_TOOL_PARAM_ALLOWLIST: Dict[str, FrozenSet[str]] = {
    "get_latest_pcb_frames": frozenset(["limit", "unconsumed_only"]),
    "inspect_pcb_frame": frozenset(["frame_id", "query", "auto_log"]),
    "inspect_pcb": frozenset(["query"]),
    "classify_board": frozenset(),
    "send_defect_alert": frozenset(["recipients", "board_type", "defect_summary", "severity", "include_image", "image_data"]),
    "log_defect": frozenset(["board_type", "defect_type", "severity", "confidence", "description", "image_path"]),
    "generate_defect_report": frozenset(["board_type"]),
    "start_defect_monitoring": frozenset(),
    "stop_defect_monitoring": frozenset(),
    "query_pcb_inspections": frozenset(["limit", "result_filter"]),
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
}


# ── Interpreter ───────────────────────────────────────────────────────────

class PCBInterpreter:
    """Intent interpreter for PCB inspection — LLM-first with keyword fallback."""

    # All tools registered in the PCB registry are available to the agent.
    # No static whitelist — the registry is the single source of truth.

    def __init__(self, registry: Optional[PCBToolRegistry] = None):
        self.registry = registry or PCBToolRegistry()
        self.audit_log = get_audit_log()

    def interpret(
        self,
        user_message: str,
        agent_state: AgentState,
        session_id: Optional[str] = None,
    ) -> Optional[MCPToolCallProposal]:
        start_time = time.time()

        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.INTENT_DETECTED,
            details={"message": user_message[:500], "agent_state": agent_state.value, "domain": "pcb"},
            session_id=session_id,
        ))

        from agents.classifiers.llm_classifier import get_classifier

        result = get_classifier().classify(user_message)
        logger.info(
            "PCB interpreter: LLM classified as domain=%s tool=%s confidence=%.2f source=%s",
            result.domain, result.tool, result.confidence, result.source,
        )

        if result.domain != "pcb" or not result.tool:
            return None

        if result.tool in ("start_defect_monitoring", "stop_defect_monitoring"):
            logger.info(
                "PCB interpreter: deprecated tool '%s' suggested; deferring to general session tools",
                result.tool,
            )
            return None

        tool_def = self.registry.get(result.tool)
        if not tool_def:
            return None

        # Runtime state policy is enforced by the executor.

        arguments: Dict[str, Any] = {}
        allowed = _TOOL_PARAM_ALLOWLIST.get(result.tool, frozenset())
        for key, value in result.params.items():
            if key in allowed and value is not None:
                arguments[key] = value

        if result.tool == "inspect_pcb" and "query" not in arguments:
            arguments["query"] = user_message

        if result.tool in ("send_defect_alert", "start_defect_monitoring"):
            recipients = resolve_recipients(arguments, user_message)
            if recipients:
                arguments["recipients"] = recipients

        proposal = MCPToolCallProposal.create(
            tool_name=result.tool,
            arguments=arguments,
            rationale=result.rationale or f"LLM classified as {result.tool}",
            confidence=result.confidence,
            requires_confirmation=tool_def.requires_confirmation,
            session_id=session_id,
        )

        latency_ms = (time.time() - start_time) * 1000
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_PROPOSED,
            details={"proposal_id": proposal.id, "tool_name": proposal.tool_name, "arguments": proposal.arguments, "domain": "pcb"},
            session_id=session_id,
            latency_ms=latency_ms,
        ))
        return proposal


# ── Executor ──────────────────────────────────────────────────────────────

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
        }

        handler = handlers.get(tool_name)
        if not handler:
            raise ValueError(f"No PCB handler for: {tool_name}")

        allowed = _TOOL_PARAM_ALLOWLIST.get(tool_name, frozenset())
        filtered = {k: v for k, v in arguments.items() if k in allowed}
        return handler(**filtered)


# ── Singletons ────────────────────────────────────────────────────────────

_pcb_registry: Optional[PCBToolRegistry] = None
_pcb_interpreter: Optional[PCBInterpreter] = None
_pcb_executor: Optional[PCBExecutor] = None
_pcb_lock = threading.RLock()


def get_pcb_registry() -> PCBToolRegistry:
    global _pcb_registry
    with _pcb_lock:
        if _pcb_registry is None:
            _pcb_registry = PCBToolRegistry()
        return _pcb_registry


def get_pcb_interpreter() -> PCBInterpreter:
    global _pcb_interpreter
    with _pcb_lock:
        if _pcb_interpreter is None:
            _pcb_interpreter = PCBInterpreter(get_pcb_registry())
        return _pcb_interpreter


def get_pcb_executor() -> PCBExecutor:
    global _pcb_executor
    with _pcb_lock:
        if _pcb_executor is None:
            _pcb_executor = PCBExecutor(get_pcb_registry())
        return _pcb_executor
