"""PCB Inspection domain MCP — registry, interpreter, and executor.

Provides a self-contained MCP (Model Context Protocol) for PCB defect
detection workflows.  Reuses the core MCP infrastructure (state machine,
audit log, lifecycle types) from ``agents.mcp`` but registers only
PCB-specific tools and intent phrases.
"""

from __future__ import annotations

import json
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.mcp import (
    # Schema types
    MCPSchemaType,
    MCPParameterSchema,
    MCPOutputSchema,
    # Lifecycle
    ToolLifecycleState,
    MCPToolCallProposal,
    MCPToolResult,
    # State machine
    AgentState,
    AgentStateMachine,
    # Session
    SessionType,
    MCPSession,
    # Tool definition
    MCPToolDefinition,
    MCPToolRegistry,
    # Audit
    AuditEventType,
    AuditLogEntry,
    MCPAuditLog,
    # Globals
    get_agent_state_machine,
    get_audit_log,
)
from core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _std_output() -> MCPOutputSchema:
    return MCPOutputSchema(
        type=MCPSchemaType.OBJECT,
        description="Tool execution result",
        properties={
            "success": {"type": "boolean", "description": "Whether the operation succeeded"},
            "message": {"type": "string", "description": "Human-readable result message"},
            "data": {"type": "object", "description": "Additional result data"},
        },
        required_properties=("success",),
    )


# ---------------------------------------------------------------------------
# PCB Tool Definitions
# ---------------------------------------------------------------------------

TOOL_INSPECT_PCB = MCPToolDefinition(
    name="inspect_pcb",
    description="Inspect the current camera frame for PCB defects including solder bridges, missing components, trace damage, and more.",
    category="pcb_analysis",
    input_schema=[
        MCPParameterSchema(
            name="query",
            type=MCPSchemaType.STRING,
            description="Optional specific question about the PCB",
            required=False,
            max_length=500,
        ),
    ],
    output_schema=_std_output(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING),
)

TOOL_CLASSIFY_BOARD = MCPToolDefinition(
    name="classify_board",
    description="Identify the type of PCB board visible in the current frame (e.g. Arduino Uno, Raspberry Pi, custom PCB).",
    category="pcb_analysis",
    input_schema=[],
    output_schema=_std_output(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING),
)

TOOL_SEND_DEFECT_ALERT = MCPToolDefinition(
    name="send_defect_alert",
    description="Send an email alert about a detected PCB defect to specified recipients.",
    category="pcb_alerts",
    input_schema=[
        MCPParameterSchema(
            name="recipients",
            type=MCPSchemaType.ARRAY,
            description="List of email addresses to alert",
            required=True,
            items_type=MCPSchemaType.STRING,
        ),
        MCPParameterSchema(
            name="board_type",
            type=MCPSchemaType.STRING,
            description="Board type (e.g. Arduino Uno)",
            required=False,
            default="unknown",
        ),
        MCPParameterSchema(
            name="defect_summary",
            type=MCPSchemaType.STRING,
            description="Description of the defect",
            required=True,
            min_length=1,
            max_length=2000,
        ),
        MCPParameterSchema(
            name="severity",
            type=MCPSchemaType.STRING,
            description="Defect severity",
            required=False,
            enum=("low", "medium", "high"),
            default="medium",
        ),
        MCPParameterSchema(
            name="include_image",
            type=MCPSchemaType.BOOLEAN,
            description="Attach the current camera frame",
            required=False,
            default=True,
        ),
    ],
    output_schema=_std_output(),
    requires_confirmation=True,
    confirmation_message="Send a PCB defect alert email to {recipients}?",
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ALERTING),
)

TOOL_LOG_DEFECT = MCPToolDefinition(
    name="log_defect",
    description="Record a PCB defect to the persistent database for tracking and reporting.",
    category="pcb_logging",
    input_schema=[
        MCPParameterSchema(
            name="board_type",
            type=MCPSchemaType.STRING,
            description="Board type",
            required=True,
        ),
        MCPParameterSchema(
            name="defect_type",
            type=MCPSchemaType.STRING,
            description="Defect classification (solder_bridge, missing_component, cold_joint, trace_damage, misalignment, burn_mark, lifted_pad, other)",
            required=True,
        ),
        MCPParameterSchema(
            name="severity",
            type=MCPSchemaType.STRING,
            description="Severity level",
            required=False,
            enum=("low", "medium", "high"),
            default="low",
        ),
        MCPParameterSchema(
            name="confidence",
            type=MCPSchemaType.NUMBER,
            description="Detection confidence 0-1",
            required=False,
            min_value=0.0,
            max_value=1.0,
        ),
        MCPParameterSchema(
            name="description",
            type=MCPSchemaType.STRING,
            description="Human-readable defect description",
            required=False,
            max_length=2000,
        ),
    ],
    output_schema=_std_output(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING, AgentState.ALERTING),
)

TOOL_GENERATE_DEFECT_REPORT = MCPToolDefinition(
    name="generate_defect_report",
    description="Generate a summary report of all recorded PCB defects, optionally filtered by board type.",
    category="pcb_reporting",
    input_schema=[
        MCPParameterSchema(
            name="board_type",
            type=MCPSchemaType.STRING,
            description="Filter report to a specific board type",
            required=False,
        ),
    ],
    output_schema=_std_output(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)


# ---------------------------------------------------------------------------
# PCB Tool Registry
# ---------------------------------------------------------------------------

class PCBToolRegistry(MCPToolRegistry):
    """Registry containing only PCB-domain tools."""

    def _register_default_tools(self) -> None:
        """Override: register PCB-specific tools instead of generic ones."""
        for tool in [
            TOOL_INSPECT_PCB,
            TOOL_CLASSIFY_BOARD,
            TOOL_SEND_DEFECT_ALERT,
            TOOL_LOG_DEFECT,
            TOOL_GENERATE_DEFECT_REPORT,
        ]:
            self.register(tool)


# ---------------------------------------------------------------------------
# PCB Interpreter
# ---------------------------------------------------------------------------

class PCBInterpreter:
    """Intent interpreter tuned for PCB inspection phrases."""

    INSPECT_PHRASES = frozenset([
        "inspect pcb",
        "inspect the pcb",
        "inspect board",
        "inspect the board",
        "check for defects",
        "check pcb",
        "check the pcb",
        "pcb inspection",
        "look for defects",
        "scan pcb",
        "scan the pcb",
        "analyze pcb",
        "analyze the pcb",
        "any defects",
        "is there a defect",
        "is this defective",
        "defect check",
        "quality check",
        "quality inspection",
        "solder check",
        "see a pcb",
    ])

    CLASSIFY_PHRASES = frozenset([
        "classify board",
        "what board is this",
        "identify board",
        "identify the board",
        "board type",
        "what type of board",
        "is this an arduino",
        "is it an arduino",
        "is this a raspberry pi",
        "what pcb is this",
        "recognize board",
    ])

    ALERT_PHRASES = frozenset([
        "send defect alert",
        "send an alert",
        "alert about defect",
        "email defect",
        "notify about defect",
        "send alert if",
        "send an email",
        "alert when",
        "email when",
        "notify when",
    ])

    LOG_PHRASES = frozenset([
        "log defect",
        "record defect",
        "save defect",
        "store defect",
        "log this defect",
    ])

    REPORT_PHRASES = frozenset([
        "defect report",
        "generate report",
        "show defects",
        "defect summary",
        "defect history",
        "pcb report",
        "all defects",
        "list defects",
    ])

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
        msg = user_message.lower().strip()

        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.INTENT_DETECTED,
            details={"message": user_message[:500], "agent_state": agent_state.value, "domain": "pcb"},
            session_id=session_id,
        ))

        # --- Alert (check first — contains "send" + context words) ---
        for phrase in self.ALERT_PHRASES:
            if phrase in msg:
                tool = self.registry.get("send_defect_alert")
                if tool and agent_state in tool.allowed_in_states:
                    # Try to extract recipients from message
                    recipients = self._extract_emails(user_message)
                    board_filter = self._extract_board_type(msg)
                    return self._finalize(MCPToolCallProposal.create(
                        tool_name="send_defect_alert",
                        arguments={
                            "recipients": recipients,
                            "defect_summary": user_message,
                            "board_type": board_filter,
                            "severity": "medium",
                        },
                        rationale=f"User requested defect alert: '{msg}'",
                        confidence=0.90,
                        requires_confirmation=True,
                        session_id=session_id,
                    ), start_time, session_id)

        # --- Inspect ---
        for phrase in self.INSPECT_PHRASES:
            if phrase in msg:
                tool = self.registry.get("inspect_pcb")
                if tool and agent_state in tool.allowed_in_states:
                    return self._finalize(MCPToolCallProposal.create(
                        tool_name="inspect_pcb",
                        arguments={"query": user_message},
                        rationale=f"User requested PCB inspection: '{msg}'",
                        confidence=0.90,
                        requires_confirmation=False,
                        session_id=session_id,
                    ), start_time, session_id)

        # --- Classify ---
        for phrase in self.CLASSIFY_PHRASES:
            if phrase in msg:
                tool = self.registry.get("classify_board")
                if tool and agent_state in tool.allowed_in_states:
                    return self._finalize(MCPToolCallProposal.create(
                        tool_name="classify_board",
                        arguments={},
                        rationale=f"User requested board classification: '{msg}'",
                        confidence=0.85,
                        requires_confirmation=False,
                        session_id=session_id,
                    ), start_time, session_id)

        # --- Log ---
        for phrase in self.LOG_PHRASES:
            if phrase in msg:
                tool = self.registry.get("log_defect")
                if tool and agent_state in tool.allowed_in_states:
                    board_type = self._extract_board_type(msg)
                    return self._finalize(MCPToolCallProposal.create(
                        tool_name="log_defect",
                        arguments={
                            "board_type": board_type,
                            "defect_type": "other",
                            "description": user_message,
                        },
                        rationale=f"User requested to log defect: '{msg}'",
                        confidence=0.85,
                        requires_confirmation=False,
                        session_id=session_id,
                    ), start_time, session_id)

        # --- Report ---
        for phrase in self.REPORT_PHRASES:
            if phrase in msg:
                tool = self.registry.get("generate_defect_report")
                if tool and agent_state in tool.allowed_in_states:
                    board_type = self._extract_board_type(msg)
                    return self._finalize(MCPToolCallProposal.create(
                        tool_name="generate_defect_report",
                        arguments={"board_type": board_type} if board_type != "unknown" else {},
                        rationale=f"User requested defect report: '{msg}'",
                        confidence=0.85,
                        requires_confirmation=False,
                        session_id=session_id,
                    ), start_time, session_id)

        return None

    # -- helpers --

    @staticmethod
    def _extract_emails(text: str) -> List[str]:
        """Extract email addresses from text."""
        import re
        return re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)

    @staticmethod
    def _extract_board_type(text: str) -> str:
        """Try to detect a board type mentioned in the text."""
        known = [
            "arduino uno", "arduino mega", "arduino nano", "arduino",
            "raspberry pi", "esp32", "esp8266", "stm32", "teensy",
            "nodemcu", "micro:bit", "beaglebone",
        ]
        for board in known:
            if board in text:
                return board.title()
        return "unknown"

    def _finalize(
        self,
        proposal: MCPToolCallProposal,
        start_time: float,
        session_id: Optional[str],
    ) -> MCPToolCallProposal:
        latency_ms = (time.time() - start_time) * 1000
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_PROPOSED,
            details={
                "proposal_id": proposal.id,
                "tool_name": proposal.tool_name,
                "arguments": proposal.arguments,
                "domain": "pcb",
            },
            session_id=session_id,
            latency_ms=latency_ms,
        ))
        return proposal


# ---------------------------------------------------------------------------
# PCB Executor
# ---------------------------------------------------------------------------

class PCBExecutor:
    """Executes approved PCB tool call proposals.

    Follows the same two-phase pattern as the generic ``MCPExecutor`` but
    routes tool invocations to ``agents.pcb_tools`` handlers.
    """

    def __init__(
        self,
        registry: Optional[PCBToolRegistry] = None,
        state_machine: Optional[AgentStateMachine] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.registry = registry or PCBToolRegistry()
        self.state_machine = state_machine or get_agent_state_machine()
        self.audit_log = get_audit_log()
        self.context = context or {}

        self._current_session: Optional[MCPSession] = None
        self._sessions: List[MCPSession] = []
        self._session_lock = threading.Lock()

        self._pending_proposals: Dict[str, MCPToolCallProposal] = {}
        self._proposals_lock = threading.Lock()

    # -- public API (mirrors MCPExecutor interface) --

    def update_context(self, **kwargs) -> None:
        self.context.update(kwargs)

    @property
    def current_session(self) -> Optional[MCPSession]:
        with self._session_lock:
            return self._current_session

    def submit_proposal(self, proposal: MCPToolCallProposal) -> Dict[str, Any]:
        tool = self.registry.get(proposal.tool_name)
        if not tool:
            proposal.reject(f"Unknown PCB tool: {proposal.tool_name}")
            self._log_rejection(proposal)
            return {"status": "rejected", "reason": proposal.rejection_reason, "proposal": proposal.to_dict()}

        is_valid, error = tool.validate_input(proposal.arguments)
        if not is_valid:
            proposal.reject(f"Validation error: {error}")
            self._log_rejection(proposal)
            return {"status": "rejected", "reason": error, "proposal": proposal.to_dict()}

        current_state = self.state_machine.state
        if current_state not in tool.allowed_in_states:
            proposal.reject(
                f"Tool not allowed in state '{current_state.value}'. "
                f"Allowed: {[s.value for s in tool.allowed_in_states]}"
            )
            self._log_rejection(proposal)
            return {"status": "rejected", "reason": proposal.rejection_reason, "proposal": proposal.to_dict()}

        if tool.requires_confirmation and not tool.can_auto_approve(proposal.arguments):
            with self._proposals_lock:
                self._pending_proposals[proposal.id] = proposal
            proposal.state = ToolLifecycleState.PENDING_APPROVAL
            self.audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.TOOL_PENDING_APPROVAL,
                details={"proposal_id": proposal.id, "tool_name": proposal.tool_name, "domain": "pcb"},
                session_id=proposal.session_id,
            ))
            return {
                "status": "pending_approval",
                "proposal_id": proposal.id,
                "tool_name": proposal.tool_name,
                "confirmation_message": self._format_confirmation(tool, proposal),
                "proposal": proposal.to_dict(),
            }

        proposal.approve(approved_by="policy")
        self._log_approval(proposal)
        return self._execute_proposal(proposal, tool)

    def approve_proposal(self, proposal_id: str) -> Dict[str, Any]:
        with self._proposals_lock:
            proposal = self._pending_proposals.pop(proposal_id, None)
        if not proposal:
            return {"status": "error", "reason": f"No pending PCB proposal: {proposal_id}"}
        tool = self.registry.get(proposal.tool_name)
        if not tool:
            return {"status": "error", "reason": f"Tool gone: {proposal.tool_name}"}
        proposal.approve(approved_by="user")
        self._log_approval(proposal)
        return self._execute_proposal(proposal, tool)

    def reject_proposal(self, proposal_id: str, reason: str = "User rejected") -> Dict[str, Any]:
        with self._proposals_lock:
            proposal = self._pending_proposals.pop(proposal_id, None)
        if not proposal:
            return {"status": "error", "reason": f"No pending PCB proposal: {proposal_id}"}
        proposal.reject(reason)
        self._log_rejection(proposal)
        return {"status": "rejected", "proposal_id": proposal_id, "reason": reason}

    def get_pending_proposals(self) -> List[Dict[str, Any]]:
        with self._proposals_lock:
            return [p.to_dict() for p in self._pending_proposals.values()]

    # -- execution --

    def _execute_proposal(self, proposal: MCPToolCallProposal, tool: MCPToolDefinition) -> Dict[str, Any]:
        start = time.time()
        proposal.start_execution()
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_EXECUTING,
            details={"proposal_id": proposal.id, "tool_name": proposal.tool_name, "domain": "pcb"},
            session_id=proposal.session_id,
        ))

        try:
            result = self._invoke(proposal.tool_name, proposal.arguments)
            duration = (time.time() - start) * 1000
            proposal.complete(success=True)
            tr = MCPToolResult(
                proposal_id=proposal.id, tool_name=proposal.tool_name,
                success=True, output=result, duration_ms=duration,
            )
            self.audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.TOOL_SUCCEEDED,
                details={"proposal_id": proposal.id, "tool_name": proposal.tool_name, "output_preview": str(result)[:500]},
                session_id=proposal.session_id, latency_ms=duration,
            ))
            return {"status": "executed", "result": tr.to_dict(), "proposal": proposal.to_dict()}

        except Exception as e:
            duration = (time.time() - start) * 1000
            proposal.complete(success=False, error=str(e))
            tr = MCPToolResult(
                proposal_id=proposal.id, tool_name=proposal.tool_name,
                success=False, output=None, error=str(e), duration_ms=duration,
            )
            self.audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.TOOL_FAILED,
                details={"proposal_id": proposal.id, "error": str(e)},
                session_id=proposal.session_id, latency_ms=duration,
            ))
            return {"status": "failed", "error": str(e), "result": tr.to_dict(), "proposal": proposal.to_dict()}

    def _invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from agents.pcb_tools import (
            tool_inspect_pcb,
            tool_classify_board,
            tool_send_defect_alert,
            tool_log_defect,
            tool_generate_defect_report,
        )

        handlers = {
            "inspect_pcb": tool_inspect_pcb,
            "classify_board": tool_classify_board,
            "send_defect_alert": tool_send_defect_alert,
            "log_defect": tool_log_defect,
            "generate_defect_report": tool_generate_defect_report,
        }

        handler = handlers.get(tool_name)
        if not handler:
            raise ValueError(f"No PCB handler for: {tool_name}")

        # Merge execution context
        merged = {**self.context, **arguments}
        return handler(**merged)

    # -- helpers --

    def _format_confirmation(self, tool: MCPToolDefinition, proposal: MCPToolCallProposal) -> str:
        if not tool.confirmation_message:
            return f"Execute PCB tool '{tool.name}'?"
        msg = tool.confirmation_message
        for k, v in proposal.arguments.items():
            msg = msg.replace(f"{{{k}}}", str(v))
        return msg

    def _log_approval(self, proposal: MCPToolCallProposal) -> None:
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_APPROVED,
            details={"proposal_id": proposal.id, "tool_name": proposal.tool_name, "domain": "pcb"},
            session_id=proposal.session_id,
        ))

    def _log_rejection(self, proposal: MCPToolCallProposal) -> None:
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_REJECTED,
            details={"proposal_id": proposal.id, "tool_name": proposal.tool_name, "reason": proposal.rejection_reason, "domain": "pcb"},
            session_id=proposal.session_id,
        ))


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_pcb_registry: Optional[PCBToolRegistry] = None
_pcb_interpreter: Optional[PCBInterpreter] = None
_pcb_executor: Optional[PCBExecutor] = None
_pcb_lock = threading.Lock()


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
