"""General-domain MCP tool definitions.

These are the built-in tools for session management, frame analysis,
alerting, evidence, logging, history, and agent control.
"""

from __future__ import annotations

from .schema import MCPSchemaType, MCPParameterSchema, MCPOutputSchema, standard_output_schema
from .state_machine import AgentState
from .registry import MCPToolDefinition, MCPToolRegistry

# All states — the LLM makes the decisions, not state filters.
_ALL_STATES = tuple(AgentState)


# ── Session tools ──────────────────────────────────────────────────────────

TOOL_START_MONITORING_SESSION = MCPToolDefinition(
    name="start_monitoring_session",
    description="Start a new monitoring session. The agent will begin actively watching the camera feed.",
    category="session",
    input_schema=[
        MCPParameterSchema(
            name="description",
            type=MCPSchemaType.STRING,
            description="Optional description of what to monitor for",
            required=False,
            max_length=500,
        ),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)

TOOL_END_SESSION = MCPToolDefinition(
    name="end_session",
    description="End the current session. Monitoring will stop and a summary will be provided.",
    category="session",
    input_schema=[],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)

TOOL_GET_SESSION_SUMMARY = MCPToolDefinition(
    name="get_session_summary",
    description="Get a summary of the current or previous session including events and statistics.",
    category="session",
    input_schema=[
        MCPParameterSchema(
            name="session_id",
            type=MCPSchemaType.STRING,
            description="Optional session ID. If not provided, summarizes the current/most recent session.",
            required=False,
        ),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)

# ── Status tools ───────────────────────────────────────────────────────────

TOOL_GET_AGENT_STATUS = MCPToolDefinition(
    name="get_agent_status",
    description="Get the current status of the monitoring agent including state, session info, and statistics.",
    category="status",
    input_schema=[],
    output_schema=MCPOutputSchema(
        type=MCPSchemaType.OBJECT,
        description="Agent status information",
        properties={
            "state": {"type": "string", "description": "Current agent state"},
            "session": {"type": "object", "description": "Current session info"},
            "stats": {"type": "object", "description": "Monitoring statistics"},
        },
        required_properties=("state",),
    ),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)

# ── Analysis tools ─────────────────────────────────────────────────────────

TOOL_ANALYZE_FRAME = MCPToolDefinition(
    name="analyze_current_frame",
    description="Analyze the current camera frame and describe what is visible. Can include a specific question.",
    category="analysis",
    input_schema=[
        MCPParameterSchema(
            name="query",
            type=MCPSchemaType.STRING,
            description="Optional specific question about the frame",
            required=False,
            max_length=500,
        ),
    ],
    output_schema=MCPOutputSchema(
        type=MCPSchemaType.OBJECT,
        description="Frame analysis result",
        properties={
            "detected": {"type": "boolean", "description": "Whether objects of interest were detected"},
            "confidence": {"type": "number", "description": "Detection confidence 0-1"},
            "description": {"type": "string", "description": "Description of what was observed"},
            "should_alert": {"type": "boolean", "description": "Whether this warrants an alert"},
        },
        required_properties=("detected", "description"),
    ),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)

# ── Alert tools ────────────────────────────────────────────────────────────

TOOL_SEND_ALERT_EMAIL = MCPToolDefinition(
    name="send_alert_email",
    description="Send an alert email to specified recipients with optional image attachment.",
    category="alerts",
    input_schema=[
        MCPParameterSchema(
            name="recipients",
            type=MCPSchemaType.ARRAY,
            description="List of email addresses to send the alert to",
            required=True,
            items_type=MCPSchemaType.STRING,
        ),
        MCPParameterSchema(
            name="subject",
            type=MCPSchemaType.STRING,
            description="Email subject line",
            required=True,
            min_length=1,
            max_length=200,
        ),
        MCPParameterSchema(
            name="body",
            type=MCPSchemaType.STRING,
            description="Email body content",
            required=True,
            min_length=1,
            max_length=5000,
        ),
        MCPParameterSchema(
            name="include_image",
            type=MCPSchemaType.BOOLEAN,
            description="Whether to attach the current camera frame",
            required=False,
            default=True,
        ),
        MCPParameterSchema(
            name="priority",
            type=MCPSchemaType.STRING,
            description="Email priority level",
            required=False,
            enum=("low", "normal", "high"),
            default="normal",
        ),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=True,
    confirmation_message="This will send an email to {recipients}. Do you want to proceed?",
    allowed_in_states=_ALL_STATES,
)

# ── Evidence tools ─────────────────────────────────────────────────────────

TOOL_SAVE_EVIDENCE = MCPToolDefinition(
    name="save_evidence",
    description="Save the current frame as evidence for later review.",
    category="evidence",
    input_schema=[
        MCPParameterSchema(
            name="label",
            type=MCPSchemaType.STRING,
            description="Label to identify this evidence",
            required=True,
            min_length=1,
            max_length=100,
        ),
        MCPParameterSchema(
            name="notes",
            type=MCPSchemaType.STRING,
            description="Additional notes about the evidence",
            required=False,
            max_length=1000,
        ),
    ],
    output_schema=MCPOutputSchema(
        type=MCPSchemaType.OBJECT,
        description="Evidence save result",
        properties={
            "success": {"type": "boolean"},
            "filepath": {"type": "string", "description": "Path to saved evidence"},
            "timestamp": {"type": "string", "description": "When the evidence was saved"},
        },
        required_properties=("success",),
    ),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)

# ── Logging tools ──────────────────────────────────────────────────────────

TOOL_LOG_EVENT = MCPToolDefinition(
    name="log_event",
    description="Log an event to the persistent audit log.",
    category="logging",
    input_schema=[
        MCPParameterSchema(
            name="event_type",
            type=MCPSchemaType.STRING,
            description="Type of event",
            required=True,
            enum=("observation", "detection", "alert", "system", "user_action"),
        ),
        MCPParameterSchema(
            name="description",
            type=MCPSchemaType.STRING,
            description="Description of the event",
            required=True,
            min_length=1,
            max_length=1000,
        ),
        MCPParameterSchema(
            name="severity",
            type=MCPSchemaType.STRING,
            description="Event severity level",
            required=False,
            enum=("info", "warning", "error"),
            default="info",
        ),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)

# ── History tools ──────────────────────────────────────────────────────────

TOOL_QUERY_HISTORY = MCPToolDefinition(
    name="query_history",
    description="Query recent detection history and events.",
    category="history",
    input_schema=[
        MCPParameterSchema(
            name="limit",
            type=MCPSchemaType.INTEGER,
            description="Maximum number of events to return",
            required=False,
            default=10,
            min_value=1,
            max_value=100,
        ),
        MCPParameterSchema(
            name="event_type",
            type=MCPSchemaType.STRING,
            description="Filter by event type",
            required=False,
            enum=("detection", "alert", "all"),
        ),
    ],
    output_schema=MCPOutputSchema(
        type=MCPSchemaType.OBJECT,
        description="Query results",
        properties={
            "events": {"type": "array", "description": "List of matching events"},
            "total": {"type": "integer", "description": "Total matching events"},
        },
        required_properties=("events", "total"),
    ),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)

# ── Configuration tools ───────────────────────────────────────────────────

TOOL_SET_DETECTION_TASK = MCPToolDefinition(
    name="set_detection_task",
    description="Configure what the agent should look for during monitoring.",
    category="configuration",
    input_schema=[
        MCPParameterSchema(
            name="task_type",
            type=MCPSchemaType.STRING,
            description="The type of detection task",
            required=True,
            enum=("package_detection", "ppe_detection", "person_counting", "scene_description", "custom"),
        ),
        MCPParameterSchema(
            name="custom_instructions",
            type=MCPSchemaType.STRING,
            description="Custom instructions for the detection task (required for custom task type)",
            required=False,
            max_length=2000,
        ),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)

# ── Control tools ──────────────────────────────────────────────────────────

TOOL_GO_IDLE = MCPToolDefinition(
    name="go_idle",
    description="Pause active monitoring and enter idle state. Camera monitoring will stop.",
    category="control",
    input_schema=[],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)

TOOL_SHUTDOWN_AGENT = MCPToolDefinition(
    name="shutdown_agent",
    description="Completely shut down the agent. All monitoring will stop.",
    category="control",
    input_schema=[],
    output_schema=standard_output_schema(),
    requires_confirmation=True,
    confirmation_message="This will completely shut down the agent. Are you sure?",
    allowed_in_states=_ALL_STATES,
)

TOOL_ACKNOWLEDGE_ERROR = MCPToolDefinition(
    name="acknowledge_error",
    description="Acknowledge an error state and attempt to recover to idle.",
    category="control",
    input_schema=[],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)


class GeneralToolRegistry(MCPToolRegistry):
    """Registry containing the general-domain tools."""

    def _register_default_tools(self) -> None:
        for tool in [
            TOOL_START_MONITORING_SESSION,
            TOOL_END_SESSION,
            TOOL_GET_SESSION_SUMMARY,
            TOOL_GET_AGENT_STATUS,
            TOOL_ANALYZE_FRAME,
            TOOL_SEND_ALERT_EMAIL,
            TOOL_SAVE_EVIDENCE,
            TOOL_LOG_EVENT,
            TOOL_QUERY_HISTORY,
            TOOL_SET_DETECTION_TASK,
            TOOL_GO_IDLE,
            TOOL_SHUTDOWN_AGENT,
            TOOL_ACKNOWLEDGE_ERROR,
        ]:
            self.register(tool)
