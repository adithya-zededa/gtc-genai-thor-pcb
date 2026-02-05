"""Retail Billing domain MCP — registry, interpreter, and executor.

Provides a self-contained MCP (Model Context Protocol) for retail
billing/invoicing workflows.  Reuses the core MCP infrastructure
(state machine, audit log, lifecycle types) from ``agents.mcp`` but
registers only retail-specific tools and intent phrases.
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
# Retail Interpreter
# ---------------------------------------------------------------------------

class RetailInterpreter:
    """Intent interpreter tuned for retail billing phrases."""

    SCAN_PHRASES = frozenset([
        "scan tray",
        "scan the tray",
        "scan items",
        "scan the items",
        "look at the items",
        "look at the tray",
        "what items",
        "what's on the tray",
        "count items",
        "count the items",
        "identify items",
        "identify the items",
        "items on the tray",
        "items placed",
        "take a look at the items",
        "check the tray",
        "see the items",
        "detect items",
        "retail scan",
    ])

    LOOKUP_PHRASES = frozenset([
        "lookup price",
        "look up price",
        "price of",
        "how much is",
        "how much does",
        "what does it cost",
        "price check",
        "item price",
        "check price",
        "find price",
    ])

    BILL_PHRASES = frozenset([
        "create bill",
        "create a bill",
        "make a bill",
        "generate bill",
        "calculate bill",
        "prepare bill",
        "billing",
        "make bill",
        "total bill",
        "compute bill",
        "tally up",
        "add up",
        "ring up",
    ])

    INVOICE_PHRASES = frozenset([
        "generate invoice",
        "generate an invoice",
        "create invoice",
        "create an invoice",
        "make invoice",
        "prepare invoice",
        "render invoice",
    ])

    SEND_INVOICE_PHRASES = frozenset([
        "send invoice",
        "send the invoice",
        "email invoice",
        "email the invoice",
        "mail invoice",
        "mail the invoice",
        "send bill",
        "send the bill",
        "email bill",
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
        msg = user_message.lower().strip()

        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.INTENT_DETECTED,
            details={"message": user_message[:500], "agent_state": agent_state.value, "domain": "retail"},
            session_id=session_id,
        ))

        # --- Send invoice (check before generic invoice) ---
        for phrase in self.SEND_INVOICE_PHRASES:
            if phrase in msg:
                tool = self.registry.get("send_invoice_email")
                if tool and agent_state in tool.allowed_in_states:
                    email = self._extract_email(user_message)
                    return self._finalize(MCPToolCallProposal.create(
                        tool_name="send_invoice_email",
                        arguments={"recipient_email": email},
                        rationale=f"User requested to send invoice: '{msg}'",
                        confidence=0.90,
                        requires_confirmation=True,
                        session_id=session_id,
                    ), start_time, session_id)

        # --- Generate invoice ---
        for phrase in self.INVOICE_PHRASES:
            if phrase in msg:
                tool = self.registry.get("generate_invoice")
                if tool and agent_state in tool.allowed_in_states:
                    email = self._extract_email(user_message)
                    return self._finalize(MCPToolCallProposal.create(
                        tool_name="generate_invoice",
                        arguments={"recipient_email": email},
                        rationale=f"User requested invoice generation: '{msg}'",
                        confidence=0.90,
                        requires_confirmation=False,
                        session_id=session_id,
                    ), start_time, session_id)

        # --- Create bill ---
        for phrase in self.BILL_PHRASES:
            if phrase in msg:
                tool = self.registry.get("create_bill")
                if tool and agent_state in tool.allowed_in_states:
                    return self._finalize(MCPToolCallProposal.create(
                        tool_name="create_bill",
                        arguments={},
                        rationale=f"User requested bill creation: '{msg}'",
                        confidence=0.90,
                        requires_confirmation=False,
                        session_id=session_id,
                    ), start_time, session_id)

        # --- Scan tray ---
        for phrase in self.SCAN_PHRASES:
            if phrase in msg:
                tool = self.registry.get("scan_tray_items")
                if tool and agent_state in tool.allowed_in_states:
                    return self._finalize(MCPToolCallProposal.create(
                        tool_name="scan_tray_items",
                        arguments={"query": user_message},
                        rationale=f"User requested tray scan: '{msg}'",
                        confidence=0.90,
                        requires_confirmation=False,
                        session_id=session_id,
                    ), start_time, session_id)

        # --- Lookup price ---
        for phrase in self.LOOKUP_PHRASES:
            if phrase in msg:
                tool = self.registry.get("lookup_item_price")
                if tool and agent_state in tool.allowed_in_states:
                    item_name = self._extract_item_name(msg, phrase)
                    return self._finalize(MCPToolCallProposal.create(
                        tool_name="lookup_item_price",
                        arguments={"item_name": item_name},
                        rationale=f"User requested price lookup: '{msg}'",
                        confidence=0.85,
                        requires_confirmation=False,
                        session_id=session_id,
                    ), start_time, session_id)

        return None

    # -- helpers --

    @staticmethod
    def _extract_email(text: str) -> str:
        import re
        matches = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        return matches[0] if matches else ""

    @staticmethod
    def _extract_item_name(msg: str, trigger_phrase: str) -> str:
        """Extract item name from the text after the trigger phrase."""
        idx = msg.find(trigger_phrase)
        if idx >= 0:
            remainder = msg[idx + len(trigger_phrase):].strip()
            # Remove common prepositions
            for prefix in ["of ", "for ", "the ", "a "]:
                if remainder.startswith(prefix):
                    remainder = remainder[len(prefix):]
            return remainder.strip("?. ") or msg
        return msg

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
# Retail Executor
# ---------------------------------------------------------------------------

class RetailExecutor:
    """Executes approved retail tool call proposals.

    Maintains a ``_last_scan_result`` and ``_last_bill`` in context so that
    sequential tool calls (scan → bill → invoice → send) can flow data
    through the pipeline automatically.
    """

    def __init__(
        self,
        registry: Optional[RetailToolRegistry] = None,
        state_machine: Optional[AgentStateMachine] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.registry = registry or RetailToolRegistry()
        self.state_machine = state_machine or get_agent_state_machine()
        self.audit_log = get_audit_log()
        self.context = context or {}

        self._current_session: Optional[MCPSession] = None
        self._sessions: List[MCPSession] = []
        self._session_lock = threading.Lock()

        self._pending_proposals: Dict[str, MCPToolCallProposal] = {}
        self._proposals_lock = threading.Lock()

        # Pipeline state: carry data between sequential tool calls
        self._last_scan_result: Optional[Dict[str, Any]] = None
        self._last_bill: Optional[Dict[str, Any]] = None
        self._last_invoice_id: Optional[int] = None

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
        from agents.retail_tools import (
            tool_scan_tray_items,
            tool_lookup_item_price,
            tool_create_bill,
            tool_generate_invoice,
            tool_send_invoice_email,
        )

        merged = {**self.context, **arguments}

        if tool_name == "scan_tray_items":
            result = tool_scan_tray_items(**merged)
            if result.get("success"):
                self._last_scan_result = result.get("data", {})
            return result

        elif tool_name == "lookup_item_price":
            return tool_lookup_item_price(**merged)

        elif tool_name == "create_bill":
            # Auto-feed last scan result if no explicit items provided
            if "items" not in merged or not merged.get("items"):
                if self._last_scan_result:
                    merged["scan_result"] = self._last_scan_result
            result = tool_create_bill(**merged)
            if result.get("success"):
                self._last_bill = result
            return result

        elif tool_name == "generate_invoice":
            # Auto-feed last bill if not provided
            if "bill" not in merged or not merged.get("bill"):
                if self._last_bill:
                    merged["bill"] = self._last_bill
            result = tool_generate_invoice(**merged)
            if result.get("success"):
                self._last_invoice_id = result.get("data", {}).get("invoice_id")
            return result

        elif tool_name == "send_invoice_email":
            # Auto-feed last bill or invoice_id
            if "bill" not in merged and "invoice_id" not in merged:
                if self._last_invoice_id:
                    merged["invoice_id"] = self._last_invoice_id
                elif self._last_bill:
                    merged["bill"] = self._last_bill
            return tool_send_invoice_email(**merged)

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
