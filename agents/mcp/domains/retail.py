"""Retail Billing domain MCP — registry, interpreter, and executor.

Provides a self-contained MCP for retail billing/invoicing workflows.
Reuses the core MCP infrastructure via ``BaseDomainExecutor`` and
registers only retail-specific tools and intent phrases.
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
from agents.mcp.schema import MCPSchemaType, MCPParameterSchema, standard_output_schema
from agents.mcp.session import MCPSession
from agents.mcp.state_machine import AgentState

from core.logging import get_logger

logger = get_logger(__name__)


# ── Tool definitions ──────────────────────────────────────────────────────

TOOL_SCAN_TRAY_ITEMS = MCPToolDefinition(
    name="scan_tray_items",
    description="Analyze the current camera frame to identify, classify, and count items placed on a tray or counter for billing.",
    category="retail_analysis",
    input_schema=[
        MCPParameterSchema(name="query", type=MCPSchemaType.STRING, description="Optional additional scanning instructions", required=False, max_length=500),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING),
)

TOOL_LOOKUP_ITEM_PRICE = MCPToolDefinition(
    name="lookup_item_price",
    description="Look up an item's price from the retail catalog database by name or SKU.",
    category="retail_catalog",
    input_schema=[
        MCPParameterSchema(name="item_name", type=MCPSchemaType.STRING, description="Product name to search for", required=False, max_length=200),
        MCPParameterSchema(name="sku", type=MCPSchemaType.STRING, description="Exact SKU code to look up", required=False, max_length=50),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.OFF, AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)

TOOL_CREATE_BILL = MCPToolDefinition(
    name="create_bill",
    description="Assemble a bill from detected items. Matches items against the catalog for prices.",
    category="retail_billing",
    input_schema=[
        MCPParameterSchema(name="items", type=MCPSchemaType.ARRAY, description="List of item dicts. If omitted, uses the last scan result.", required=False, items_type=MCPSchemaType.OBJECT),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)

TOOL_GENERATE_INVOICE = MCPToolDefinition(
    name="generate_invoice",
    description="Render an HTML invoice from the current bill and save it to the database.",
    category="retail_billing",
    input_schema=[
        MCPParameterSchema(name="recipient_email", type=MCPSchemaType.STRING, description="Recipient email for the invoice header", required=False, max_length=200),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)

TOOL_SEND_INVOICE_EMAIL = MCPToolDefinition(
    name="send_invoice_email",
    description="Email the generated invoice to a specified recipient.",
    category="retail_billing",
    input_schema=[
        MCPParameterSchema(name="recipient_email", type=MCPSchemaType.STRING, description="Email address to send the invoice to", required=True, max_length=200),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=True,
    confirmation_message="Send the invoice to {recipient_email}?",
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ALERTING),
)


# ── Registry ──────────────────────────────────────────────────────────────

class RetailToolRegistry(MCPToolRegistry):
    """Registry containing only retail-domain tools."""

    def _register_default_tools(self) -> None:
        for tool in [TOOL_SCAN_TRAY_ITEMS, TOOL_LOOKUP_ITEM_PRICE, TOOL_CREATE_BILL, TOOL_GENERATE_INVOICE, TOOL_SEND_INVOICE_EMAIL]:
            self.register(tool)


# ── Context allowlist ─────────────────────────────────────────────────────

_TOOL_PARAM_ALLOWLIST: Dict[str, FrozenSet[str]] = {
    "scan_tray_items": frozenset(["query"]),
    "lookup_item_price": frozenset(["item_name", "sku"]),
    "create_bill": frozenset(["items", "scan_result"]),
    "generate_invoice": frozenset(["bill", "recipient_email"]),
    "send_invoice_email": frozenset(["recipient_email", "bill", "invoice_id"]),
}


# ── Interpreter ───────────────────────────────────────────────────────────

class RetailInterpreter:
    """Intent interpreter for retail billing — LLM-first with keyword fallback."""

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

        from agents.classifiers.llm_classifier import get_classifier

        result = get_classifier().classify(user_message)
        logger.info(
            "Retail interpreter: LLM classified as domain=%s tool=%s confidence=%.2f source=%s",
            result.domain, result.tool, result.confidence, result.source,
        )

        if result.domain != "retail" or result.tool not in self._VALID_TOOLS:
            return None

        tool_def = self.registry.get(result.tool)
        if not tool_def or agent_state not in tool_def.allowed_in_states:
            return None

        arguments: Dict[str, Any] = {}
        allowed = _TOOL_PARAM_ALLOWLIST.get(result.tool, frozenset())
        for key, value in result.params.items():
            if key in allowed and value:
                arguments[key] = value

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

        latency_ms = (time.time() - start_time) * 1000
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_PROPOSED,
            details={"proposal_id": proposal.id, "tool_name": proposal.tool_name, "arguments": proposal.arguments, "domain": "retail"},
            session_id=session_id,
            latency_ms=latency_ms,
        ))
        return proposal


# ── Pipeline state (session-scoped) ───────────────────────────────────────

class _PipelineState:
    """Session-scoped state that flows data through scan → bill → invoice → send."""

    __slots__ = ("scan_result", "bill", "invoice_id", "updated_at")

    def __init__(self) -> None:
        self.scan_result: Optional[Dict[str, Any]] = None
        self.bill: Optional[Dict[str, Any]] = None
        self.invoice_id: Optional[int] = None
        self.updated_at: float = time.time()


# ── Executor ──────────────────────────────────────────────────────────────

class RetailExecutor(BaseDomainExecutor):
    """Executes approved retail tool call proposals with session-scoped pipeline state."""

    _domain_label = "retail"

    def __init__(
        self,
        registry: Optional[RetailToolRegistry] = None,
        state_machine=None,
        context: Optional[Dict[str, Any]] = None,
        invoke_timeout: int = 60,
    ):
        reg = registry or RetailToolRegistry()
        super().__init__(
            registry=reg,
            state_machine=state_machine or get_agent_state_machine(),
            audit_log=get_audit_log(),
            context=context,
            invoke_timeout=invoke_timeout,
        )
        self._pipeline: Dict[str, _PipelineState] = {}
        self._pipeline_lock = threading.Lock()

    def _get_pipeline(self, session_id: Optional[str] = None) -> _PipelineState:
        key = session_id or "__default__"
        with self._pipeline_lock:
            if key not in self._pipeline:
                self._pipeline[key] = _PipelineState()
            return self._pipeline[key]

    def _invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from agents.tools.retail import (
            tool_scan_tray_items,
            tool_lookup_item_price,
            tool_create_bill,
            tool_generate_invoice,
            tool_send_invoice_email,
        )

        allowed = _TOOL_PARAM_ALLOWLIST.get(tool_name, frozenset())
        filtered = {k: v for k, v in arguments.items() if k in allowed}
        pipeline = self._get_pipeline()

        if tool_name == "scan_tray_items":
            result = tool_scan_tray_items(**filtered)
            if result.get("success"):
                pipeline.scan_result = result.get("data", {})
                pipeline.updated_at = time.time()
            return result

        if tool_name == "lookup_item_price":
            return tool_lookup_item_price(**filtered)

        if tool_name == "create_bill":
            if "items" not in filtered or not filtered.get("items"):
                if pipeline.scan_result:
                    filtered["scan_result"] = pipeline.scan_result
            result = tool_create_bill(**filtered)
            if result.get("success"):
                pipeline.bill = result
                pipeline.updated_at = time.time()
            return result

        if tool_name == "generate_invoice":
            if "bill" not in filtered or not filtered.get("bill"):
                if pipeline.bill:
                    filtered["bill"] = pipeline.bill
            result = tool_generate_invoice(**filtered)
            if result.get("success"):
                pipeline.invoice_id = result.get("data", {}).get("invoice_id")
                pipeline.updated_at = time.time()
            return result

        if tool_name == "send_invoice_email":
            if "bill" not in filtered and "invoice_id" not in filtered:
                if pipeline.invoice_id:
                    filtered["invoice_id"] = pipeline.invoice_id
                elif pipeline.bill:
                    filtered["bill"] = pipeline.bill
            return tool_send_invoice_email(**filtered)

        raise ValueError(f"No retail handler for: {tool_name}")


# ── Singletons ────────────────────────────────────────────────────────────

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
