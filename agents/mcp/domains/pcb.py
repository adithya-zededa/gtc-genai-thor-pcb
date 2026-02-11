"""PCB Inspection domain MCP — registry, interpreter, and executor.

Provides a self-contained MCP for PCB defect detection workflows.
Reuses the core MCP infrastructure via ``BaseDomainExecutor`` and
registers only PCB-specific tools and intent phrases.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, FrozenSet, List, Optional

from agents.mcp.audit import AuditEventType, AuditLogEntry
from agents.mcp.executor_base import BaseDomainExecutor
from agents.mcp.globals import get_agent_state_machine, get_audit_log
from agents.mcp.lifecycle import MCPToolCallProposal
from agents.mcp.registry import MCPToolDefinition, MCPToolRegistry
from agents.mcp.schema import MCPSchemaType, MCPParameterSchema, MCPOutputSchema, standard_output_schema
from agents.mcp.state_machine import AgentState

from core.logging import get_logger

logger = get_logger(__name__)


# ── Tool definitions ──────────────────────────────────────────────────────

TOOL_INSPECT_PCB = MCPToolDefinition(
    name="inspect_pcb",
    description="Inspect the current camera frame for PCB defects including solder bridges, missing components, trace damage, and more.",
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
    description="Send an email alert about a detected PCB defect to specified recipients.",
    category="pcb_alerts",
    input_schema=[
        MCPParameterSchema(name="recipients", type=MCPSchemaType.ARRAY, description="List of email addresses to alert", required=True, items_type=MCPSchemaType.STRING),
        MCPParameterSchema(name="board_type", type=MCPSchemaType.STRING, description="Board type", required=False, default="unknown"),
        MCPParameterSchema(name="defect_summary", type=MCPSchemaType.STRING, description="Description of the defect", required=True, min_length=1, max_length=2000),
        MCPParameterSchema(name="severity", type=MCPSchemaType.STRING, description="Defect severity", required=False, enum=("low", "medium", "high"), default="medium"),
        MCPParameterSchema(name="include_image", type=MCPSchemaType.BOOLEAN, description="Attach the current camera frame", required=False, default=True),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=True,
    confirmation_message="Send a PCB defect alert email to {recipients}?",
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ALERTING),
)

TOOL_LOG_DEFECT = MCPToolDefinition(
    name="log_defect",
    description="Record a PCB defect to the persistent database for tracking and reporting.",
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


# ── Registry ──────────────────────────────────────────────────────────────

class PCBToolRegistry(MCPToolRegistry):
    """Registry containing only PCB-domain tools."""

    def _register_default_tools(self) -> None:
        for tool in [TOOL_INSPECT_PCB, TOOL_CLASSIFY_BOARD, TOOL_SEND_DEFECT_ALERT, TOOL_LOG_DEFECT, TOOL_GENERATE_DEFECT_REPORT]:
            self.register(tool)


# ── Context allowlist ─────────────────────────────────────────────────────

_TOOL_PARAM_ALLOWLIST: Dict[str, FrozenSet[str]] = {
    "inspect_pcb": frozenset(["query"]),
    "classify_board": frozenset(),
    "send_defect_alert": frozenset(["recipients", "board_type", "defect_summary", "severity", "include_image", "image_data"]),
    "log_defect": frozenset(["board_type", "defect_type", "severity", "confidence", "description", "image_path"]),
    "generate_defect_report": frozenset(["board_type"]),
}


# ── Interpreter ───────────────────────────────────────────────────────────

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

        from agents.classifiers.llm_classifier import get_classifier

        result = get_classifier().classify(user_message)
        logger.info(
            "PCB interpreter: LLM classified as domain=%s tool=%s confidence=%.2f source=%s",
            result.domain, result.tool, result.confidence, result.source,
        )

        if result.domain != "pcb" or result.tool not in self._VALID_TOOLS:
            return None

        tool_def = self.registry.get(result.tool)
        if not tool_def:
            return None

        # NOTE: No state-based filtering — the LLM decides all actions.

        arguments: Dict[str, Any] = {}
        allowed = _TOOL_PARAM_ALLOWLIST.get(result.tool, frozenset())
        for key, value in result.params.items():
            if key in allowed and value:
                arguments[key] = value

        if result.tool == "inspect_pcb" and "query" not in arguments:
            arguments["query"] = user_message

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
