"""Retail Billing domain MCP — registry, interpreter, and executor.

Provides a self-contained MCP (Model Context Protocol) for retail
billing/invoicing workflows.  Reuses the core MCP infrastructure
(state machine, audit log, lifecycle types) from ``agents.mcp`` but
registers only retail-specific tools and intent phrases.

Hardened executor features:
- Session-scoped pipeline state (scan → bill → invoice → send)
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
from typing import Any, Dict, FrozenSet, List, Optional, Set

from agents.mcp.base import (
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
# Retail Tool Definitions
# ---------------------------------------------------------------------------

TOOL_SCAN_TRAY_ITEMS = MCPToolDefinition(
    name="scan_tray_items",
    description="Analyze the current camera frame to identify, classify, and count items placed on a tray or counter for billing.",
    category="retail_analysis",
    input_schema=[
        MCPParameterSchema(
            name="query",
            type=MCPSchemaType.STRING,
            description="Optional additional scanning instructions",
            required=False,
            max_length=500,
        ),
    ],
    output_schema=_std_output(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING),
)

TOOL_LOOKUP_ITEM_PRICE = MCPToolDefinition(
    name="lookup_item_price",
    description="Look up an item's price from the retail catalog database by name or SKU.",
    category="retail_catalog",
    input_schema=[
        MCPParameterSchema(
            name="item_name",
            type=MCPSchemaType.STRING,
            description="Product name to search for (fuzzy match)",
            required=False,
            max_length=200,
        ),
        MCPParameterSchema(
            name="sku",
            type=MCPSchemaType.STRING,
            description="Exact SKU code to look up",
            required=False,
            max_length=50,
        ),
    ],
    output_schema=_std_output(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.OFF, AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)

TOOL_CREATE_BILL = MCPToolDefinition(
    name="create_bill",
    description="Assemble a bill from detected items. Matches items against the catalog for prices, calculates subtotal, tax, and grand total.",
    category="retail_billing",
    input_schema=[
        MCPParameterSchema(
            name="items",
            type=MCPSchemaType.ARRAY,
            description="List of item dicts with name, quantity, price. If omitted, uses the last scan result.",
            required=False,
            items_type=MCPSchemaType.OBJECT,
        ),
    ],
    output_schema=_std_output(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)

TOOL_GENERATE_INVOICE = MCPToolDefinition(
    name="generate_invoice",
    description="Render an HTML invoice from the current bill and save it to the database.",
    category="retail_billing",
    input_schema=[
        MCPParameterSchema(
            name="recipient_email",
            type=MCPSchemaType.STRING,
            description="Recipient email for the invoice header",
            required=False,
            max_length=200,
        ),
    ],
    output_schema=_std_output(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)

TOOL_SEND_INVOICE_EMAIL = MCPToolDefinition(
    name="send_invoice_email",
    description="Email the generated invoice to a specified recipient.",
    category="retail_billing",
    input_schema=[
        MCPParameterSchema(
            name="recipient_email",
            type=MCPSchemaType.STRING,
            description="Email address to send the invoice to",
            required=True,
            max_length=200,
        ),
    ],
    output_schema=_std_output(),
    requires_confirmation=True,
    confirmation_message="Send the invoice to {recipient_email}?",
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ALERTING),
)


# ---------------------------------------------------------------------------
# Retail Tool Registry
# ---------------------------------------------------------------------------

class RetailToolRegistry(MCPToolRegistry):
    """Registry containing only retail-domain tools."""

    def _register_default_tools(self) -> None:
        for tool in [
            TOOL_SCAN_TRAY_ITEMS,
            TOOL_LOOKUP_ITEM_PRICE,
            TOOL_CREATE_BILL,
            TOOL_GENERATE_INVOICE,
            TOOL_SEND_INVOICE_EMAIL,
        ]:
            self.register(tool)


# ---------------------------------------------------------------------------
# Context allowlist — only these keys are forwarded to tool handlers
# ---------------------------------------------------------------------------

_TOOL_PARAM_ALLOWLIST: Dict[str, FrozenSet[str]] = {
    "scan_tray_items": frozenset(["query"]),
    "lookup_item_price": frozenset(["item_name", "sku"]),
    "create_bill": frozenset(["items", "scan_result"]),
    "generate_invoice": frozenset(["bill", "recipient_email"]),
    "send_invoice_email": frozenset(["recipient_email", "bill", "invoice_id"]),
}


# ---------------------------------------------------------------------------
# Retail Interpreter (LLM-first, keyword fallback)
# ---------------------------------------------------------------------------

class RetailInterpreter:
    """Intent interpreter for retail billing — LLM-first with keyword fallback."""

    # Retail tool names for validation
    _VALID_TOOLS: FrozenSet[str] = frozenset([
        "scan_tray_items", "lookup_item_price", "create_bill",
        "generate_invoice", "send_invoice_email",
    ])

    def __init__(self, registry: Optional[RetailToolRegistry] = None):
        self.registry = registry or RetailToolRegistry()
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
            details={"message": user_message[:500], "agent_state": agent_state.value, "domain": "retail"},
            session_id=session_id,
        ))

        # --- Use LLM classifier ---
        from agents.classifiers.llm_classifier import get_classifier

        result = get_classifier().classify(user_message)
        logger.info(
            "Retail interpreter: LLM classified as domain=%s tool=%s confidence=%.2f source=%s",
            result.domain, result.tool, result.confidence, result.source,
        )

        # Only accept if classifier picked a valid retail tool
        if result.domain != "retail" or result.tool not in self._VALID_TOOLS:
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

        # For scan/inspect, pass user message as query
        if result.tool == "scan_tray_items" and "query" not in arguments:
            arguments["query"] = user_message

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
                "domain": "retail",
            },
            session_id=session_id,
            latency_ms=latency_ms,
        ))
        return proposal


# ---------------------------------------------------------------------------
# Pipeline State (session-scoped)
# ---------------------------------------------------------------------------

class _PipelineState:
    """Session-scoped state that flows data through scan -> bill -> invoice -> send."""

    __slots__ = ("scan_result", "bill", "invoice_id", "updated_at")

    def __init__(self) -> None:
        self.scan_result: Optional[Dict[str, Any]] = None
        self.bill: Optional[Dict[str, Any]] = None
        self.invoice_id: Optional[int] = None
        self.updated_at: float = time.time()


# ---------------------------------------------------------------------------
# Retail Executor (hardened)
# ---------------------------------------------------------------------------

# Timeout for individual tool calls (seconds)
_DEFAULT_INVOKE_TIMEOUT = 60
# Deduplication window (seconds)
_DEDUP_WINDOW = 5.0


class RetailExecutor:
    """Executes approved retail tool call proposals.

    Hardened features:
    - **Session-scoped pipeline state**: each session_id gets its own
      ``_PipelineState`` so concurrent users don't overwrite each other.
    - **Deduplication**: identical (tool + args) proposals within
      ``_DEDUP_WINDOW`` seconds are rejected.
    - **Context allowlist**: only declared parameter names are forwarded
      to tool handlers.
    - **Timeout**: tool invocations are wrapped in a thread-pool future
      with a configurable timeout.
    """

    def __init__(
        self,
        registry: Optional[RetailToolRegistry] = None,
        state_machine: Optional[AgentStateMachine] = None,
        context: Optional[Dict[str, Any]] = None,
        invoke_timeout: int = _DEFAULT_INVOKE_TIMEOUT,
    ):
        self.registry = registry or RetailToolRegistry()
        self.state_machine = state_machine or get_agent_state_machine()
        self.audit_log = get_audit_log()
        self.context = context or {}
        self._invoke_timeout = invoke_timeout

        self._current_session: Optional[MCPSession] = None
        self._sessions: List[MCPSession] = []
        self._session_lock = threading.Lock()

        self._pending_proposals: Dict[str, MCPToolCallProposal] = {}
        self._proposals_lock = threading.Lock()

        # Session-scoped pipeline state
        self._pipeline: Dict[str, _PipelineState] = {}
        self._pipeline_lock = threading.Lock()

        # Dedup tracking: hash -> timestamp
        self._recent_hashes: Dict[str, float] = {}
        self._dedup_lock = threading.Lock()

        # Shared thread pool for timeout-wrapped invocations
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retail-exec")

    # -- pipeline helpers --

    def _get_pipeline(self, session_id: Optional[str] = None) -> _PipelineState:
        key = session_id or "__default__"
        with self._pipeline_lock:
            if key not in self._pipeline:
                self._pipeline[key] = _PipelineState()
            return self._pipeline[key]

    # -- dedup --

    def _is_duplicate(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Return True if the same tool+args were submitted within the dedup window."""
        sig = hashlib.sha256(
            json.dumps({"t": tool_name, "a": arguments}, sort_keys=True, default=str).encode()
        ).hexdigest()
        now = time.time()
        with self._dedup_lock:
            # Prune expired entries
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
            proposal.reject(f"Unknown retail tool: {proposal.tool_name}")
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
                details={"proposal_id": proposal.id, "tool_name": proposal.tool_name, "domain": "retail"},
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
            return {"status": "error", "reason": f"No pending retail proposal: {proposal_id}"}
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
            return {"status": "error", "reason": f"No pending retail proposal: {proposal_id}"}
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
            details={"proposal_id": proposal.id, "tool_name": proposal.tool_name, "domain": "retail"},
            session_id=proposal.session_id,
        ))

        try:
            future = self._pool.submit(self._invoke, proposal.tool_name, proposal.arguments, proposal.session_id)
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
            logger.error("Retail tool execution failed: %s", e, exc_info=True)
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

    def _invoke(self, tool_name: str, arguments: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        from agents.tools.retail import (
            tool_scan_tray_items,
            tool_lookup_item_price,
            tool_create_bill,
            tool_generate_invoice,
            tool_send_invoice_email,
        )

        # Build a filtered kwargs dict using the allowlist
        allowed = _TOOL_PARAM_ALLOWLIST.get(tool_name, frozenset())
        filtered = {k: v for k, v in arguments.items() if k in allowed}

        pipeline = self._get_pipeline(session_id)

        if tool_name == "scan_tray_items":
            result = tool_scan_tray_items(**filtered)
            if result.get("success"):
                pipeline.scan_result = result.get("data", {})
                pipeline.updated_at = time.time()
            return result

        elif tool_name == "lookup_item_price":
            return tool_lookup_item_price(**filtered)

        elif tool_name == "create_bill":
            # Auto-feed last scan result if no explicit items provided
            if "items" not in filtered or not filtered.get("items"):
                if pipeline.scan_result:
                    filtered["scan_result"] = pipeline.scan_result
            result = tool_create_bill(**filtered)
            if result.get("success"):
                pipeline.bill = result
                pipeline.updated_at = time.time()
            return result

        elif tool_name == "generate_invoice":
            # Auto-feed last bill if not provided
            if "bill" not in filtered or not filtered.get("bill"):
                if pipeline.bill:
                    filtered["bill"] = pipeline.bill
            result = tool_generate_invoice(**filtered)
            if result.get("success"):
                pipeline.invoice_id = result.get("data", {}).get("invoice_id")
                pipeline.updated_at = time.time()
            return result

        elif tool_name == "send_invoice_email":
            # Auto-feed last bill or invoice_id
            if "bill" not in filtered and "invoice_id" not in filtered:
                if pipeline.invoice_id:
                    filtered["invoice_id"] = pipeline.invoice_id
                elif pipeline.bill:
                    filtered["bill"] = pipeline.bill
            return tool_send_invoice_email(**filtered)

        raise ValueError(f"No retail handler for: {tool_name}")

    # -- helpers --

    def _format_confirmation(self, tool: MCPToolDefinition, proposal: MCPToolCallProposal) -> str:
        if not tool.confirmation_message:
            return f"Execute retail tool '{tool.name}'?"
        msg = tool.confirmation_message
        for k, v in proposal.arguments.items():
            msg = msg.replace(f"{{{k}}}", str(v))
        return msg

    def _log_approval(self, proposal: MCPToolCallProposal) -> None:
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_APPROVED,
            details={"proposal_id": proposal.id, "tool_name": proposal.tool_name, "domain": "retail"},
            session_id=proposal.session_id,
        ))

    def _log_rejection(self, proposal: MCPToolCallProposal) -> None:
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_REJECTED,
            details={"proposal_id": proposal.id, "tool_name": proposal.tool_name, "reason": proposal.rejection_reason, "domain": "retail"},
            session_id=proposal.session_id,
        ))


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_retail_registry: Optional[RetailToolRegistry] = None
_retail_interpreter: Optional[RetailInterpreter] = None
_retail_executor: Optional[RetailExecutor] = None
_retail_lock = threading.Lock()


def get_retail_registry() -> RetailToolRegistry:
    global _retail_registry
    with _retail_lock:
        if _retail_registry is None:
            _retail_registry = RetailToolRegistry()
        return _retail_registry


def get_retail_interpreter() -> RetailInterpreter:
    global _retail_interpreter
    with _retail_lock:
        if _retail_interpreter is None:
            _retail_interpreter = RetailInterpreter(get_retail_registry())
        return _retail_interpreter


def get_retail_executor() -> RetailExecutor:
    global _retail_executor
    with _retail_lock:
        if _retail_executor is None:
            _retail_executor = RetailExecutor(get_retail_registry())
        return _retail_executor
