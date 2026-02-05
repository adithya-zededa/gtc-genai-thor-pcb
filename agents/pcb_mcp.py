"""PCB Inspection domain MCP — registry, interpreter, and executor.

Provides a self-contained MCP (Model Context Protocol) for PCB defect
detection workflows.  Reuses the core MCP infrastructure (state machine,
audit log, lifecycle types) from ``agents.mcp`` but registers only
PCB-specific tools and intent phrases.

Hardened executor features:
- Session-scoped inspection state
- Deduplication window to reject repeated identical proposals
- Context allowlist to prevent pollution from arbitrary keys
- Timeout on tool invocation with configurable limit
- LLM-powered intent interpretation with keyword fallback
"""

from __future__ import annotations

import hashlib
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Optional

from agents.mcp import (
    MCPSchemaType,
    MCPParameterSchema,
    MCPOutputSchema,
    ToolLifecycleState,
    MCPToolCallProposal,
    MCPToolResult,
    AgentState,
    AgentStateMachine,
    SessionType,
    MCPSession,
    MCPToolDefinition,
    MCPToolRegistry,
    AuditEventType,
    AuditLogEntry,
    MCPAuditLog,
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
        for tool in [
            TOOL_INSPECT_PCB,
            TOOL_CLASSIFY_BOARD,
            TOOL_SEND_DEFECT_ALERT,
            TOOL_LOG_DEFECT,
            TOOL_GENERATE_DEFECT_REPORT,
        ]:
            self.register(tool)


# ---------------------------------------------------------------------------
# Context allowlist
# ---------------------------------------------------------------------------

_TOOL_PARAM_ALLOWLIST: Dict[str, FrozenSet[str]] = {
    "inspect_pcb": frozenset(["query"]),
    "classify_board": frozenset(),
    "send_defect_alert": frozenset(["recipients", "board_type", "defect_summary", "severity", "include_image", "image_data"]),
    "log_defect": frozenset(["board_type", "defect_type", "severity", "confidence", "description", "image_path"]),
    "generate_defect_report": frozenset(["board_type"]),
}


# ---------------------------------------------------------------------------
# PCB Interpreter (LLM-first, keyword fallback)
# ---------------------------------------------------------------------------

class PCBInterpreter:
    """Intent interpreter for PCB inspection — LLM-first with keyword fallback."""

    _VALID_TOOLS: FrozenSet[str] = frozenset([
        "inspect_pcb", "classify_board", "send_defect_alert",
        "log_defect", "generate_defect_report",
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

        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.INTENT_DETECTED,
            details={"message": user_message[:500], "agent_state": agent_state.value, "domain": "pcb"},
            session_id=session_id,
        ))

        # --- Use LLM classifier ---
        from agents.llm_classifier import get_classifier

        result = get_classifier().classify(user_message)
        logger.info(
            "PCB interpreter: LLM classified as domain=%s tool=%s confidence=%.2f source=%s",
            result.domain, result.tool, result.confidence, result.source,
        )

        if result.domain != "pcb" or result.tool not in self._VALID_TOOLS:
            return None

        tool_def = self.registry.get(result.tool)
        if not tool_def or agent_state not in tool_def.allowed_in_states:
            return None

        # Build arguments from classifier params
        arguments: Dict[str, Any] = {}
        allowed = _TOOL_PARAM_ALLOWLIST.get(result.tool, frozenset())
        for key, value in result.params.items():
            if key in allowed and value:
                arguments[key] = value

        # For inspect, pass user message as query
        if result.tool == "inspect_pcb" and "query" not in arguments:
            arguments["query"] = user_message

        # For alert, ensure defect_summary is present
        if result.tool == "send_defect_alert" and "defect_summary" not in arguments:
            arguments["defect_summary"] = user_message

        proposal = MCPToolCallProposal.create(
            tool_name=result.tool,
            arguments=arguments,
            rationale=result.rationale or f"LLM classified as {result.tool}",
            confidence=result.confidence,
            requires_confirmation=tool_def.requires_confirmation,
            session_id=session_id,
        )

        return self._finalize(proposal, start_time, session_id)

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
# PCB Executor (hardened)
# ---------------------------------------------------------------------------

_DEFAULT_INVOKE_TIMEOUT = 60
_DEDUP_WINDOW = 5.0


class PCBExecutor:
    """Executes approved PCB tool call proposals.

    Hardened features match RetailExecutor: session-scoped state,
    dedup, context allowlist, timeout.
    """

    def __init__(
        self,
        registry: Optional[PCBToolRegistry] = None,
        state_machine: Optional[AgentStateMachine] = None,
        context: Optional[Dict[str, Any]] = None,
        invoke_timeout: int = _DEFAULT_INVOKE_TIMEOUT,
    ):
        self.registry = registry or PCBToolRegistry()
        self.state_machine = state_machine or get_agent_state_machine()
        self.audit_log = get_audit_log()
        self.context = context or {}
        self._invoke_timeout = invoke_timeout

        self._current_session: Optional[MCPSession] = None
        self._sessions: List[MCPSession] = []
        self._session_lock = threading.Lock()

        self._pending_proposals: Dict[str, MCPToolCallProposal] = {}
        self._proposals_lock = threading.Lock()

        # Dedup tracking
        self._recent_hashes: Dict[str, float] = {}
        self._dedup_lock = threading.Lock()

        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pcb-exec")

    # -- dedup --

    def _is_duplicate(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        sig = hashlib.sha256(
            json.dumps({"t": tool_name, "a": arguments}, sort_keys=True, default=str).encode()
        ).hexdigest()
        now = time.time()
        with self._dedup_lock:
            expired = [h for h, ts in self._recent_hashes.items() if now - ts > _DEDUP_WINDOW]
            for h in expired:
                del self._recent_hashes[h]
            if sig in self._recent_hashes:
                return True
            self._recent_hashes[sig] = now
        return False

    # -- public API --

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

        # Deduplication check
        if self._is_duplicate(proposal.tool_name, proposal.arguments):
            proposal.reject("Duplicate request (already submitted recently)")
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
            future = self._pool.submit(self._invoke, proposal.tool_name, proposal.arguments)
            result = future.result(timeout=self._invoke_timeout)
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

        except FuturesTimeout:
            duration = (time.time() - start) * 1000
            error_msg = f"Tool '{proposal.tool_name}' timed out after {self._invoke_timeout}s"
            proposal.complete(success=False, error=error_msg)
            logger.error(error_msg)
            tr = MCPToolResult(
                proposal_id=proposal.id, tool_name=proposal.tool_name,
                success=False, output=None, error="Operation timed out", duration_ms=duration,
            )
            self.audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.TOOL_FAILED,
                details={"proposal_id": proposal.id, "error": error_msg},
                session_id=proposal.session_id, latency_ms=duration,
            ))
            return {"status": "failed", "error": "Operation timed out", "result": tr.to_dict(), "proposal": proposal.to_dict()}

        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error("PCB tool execution failed: %s", e, exc_info=True)
            proposal.complete(success=False, error=str(e))
            tr = MCPToolResult(
                proposal_id=proposal.id, tool_name=proposal.tool_name,
                success=False, output=None, error="An internal error occurred", duration_ms=duration,
            )
            self.audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.TOOL_FAILED,
                details={"proposal_id": proposal.id, "error": str(e)},
                session_id=proposal.session_id, latency_ms=duration,
            ))
            return {"status": "failed", "error": "An internal error occurred", "result": tr.to_dict(), "proposal": proposal.to_dict()}

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

        # Filter through allowlist — no context pollution
        allowed = _TOOL_PARAM_ALLOWLIST.get(tool_name, frozenset())
        filtered = {k: v for k, v in arguments.items() if k in allowed}
        return handler(**filtered)

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
