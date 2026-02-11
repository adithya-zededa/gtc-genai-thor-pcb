"""Retail Billing domain MCP — registry, interpreter, and executor.

Provides a self-contained MCP for retail billing/invoicing workflows.
Reuses the core MCP infrastructure via ``BaseDomainExecutor`` and
registers only retail-specific tools and intent phrases.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, FrozenSet, Optional

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

# All states — the LLM makes the decisions, not state filters.
_ALL_STATES = tuple(AgentState)

TOOL_SCAN_TRAY_ITEMS = MCPToolDefinition(
    name="scan_tray_items",
    description="Analyze the current camera frame to identify, classify, and count items placed on a tray or counter for billing.",
    category="retail_analysis",
    input_schema=[
        MCPParameterSchema(name="query", type=MCPSchemaType.STRING, description="Optional additional scanning instructions", required=False, max_length=500),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
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
    allowed_in_states=_ALL_STATES,
)

TOOL_CREATE_BILL = MCPToolDefinition(
    name="create_bill",
    description="STEP 1: Assemble a bill from detected items and look up prices in the catalog. This creates the bill data structure but does NOT generate the final invoice document. After this, you must call generate_invoice to create the PDF and announce the total.",
    category="retail_billing",
    input_schema=[
        MCPParameterSchema(name="items", type=MCPSchemaType.ARRAY, description="List of item dicts. If omitted, uses the last scan result.", required=False, items_type=MCPSchemaType.OBJECT),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)

TOOL_GENERATE_INVOICE = MCPToolDefinition(
    name="generate_invoice",
    description="STEP 2: Generate the final customer invoice with HTML + PDF formats, announce the total via speaker, and save to database. This creates the complete invoice document that customers can view/download. Always call this after create_bill when the user requests an invoice.",
    category="retail_billing",
    input_schema=[
        MCPParameterSchema(name="recipient_email", type=MCPSchemaType.STRING, description="Recipient email or name for the invoice", required=False, max_length=200),
        MCPParameterSchema(name="output_path", type=MCPSchemaType.STRING, description="Optional custom output path for the PDF file", required=False, max_length=500),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=_ALL_STATES,
)

TOOL_SEND_INVOICE_EMAIL = MCPToolDefinition(
    name="send_invoice_email",
    description="Email the generated invoice to a specified recipient. Use this only when the customer explicitly requests email delivery.",
    category="retail_billing",
    input_schema=[
        MCPParameterSchema(name="recipient_email", type=MCPSchemaType.STRING, description="Email address to send the invoice to", required=True, max_length=200),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=True,
    confirmation_message="Send the invoice to {recipient_email}?",
    allowed_in_states=_ALL_STATES,
)


# ── Registry ──────────────────────────────────────────────────────────────

class RetailToolRegistry(MCPToolRegistry):
    """Registry containing only retail-domain tools."""

    def _register_default_tools(self) -> None:
        for tool in [TOOL_SCAN_TRAY_ITEMS, TOOL_LOOKUP_ITEM_PRICE, TOOL_CREATE_BILL, TOOL_GENERATE_INVOICE, TOOL_SEND_INVOICE_EMAIL]:
            self.register(tool)


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
        if not tool_def:
            return None

        # NOTE: No state-based filtering here — the LLM decides all actions.
        # State checks remain in tool definitions for documentation purposes only.

        # Build arguments from classifier result, filtering to valid schema params
        valid_param_names = {p.name for p in tool_def.input_schema} if tool_def.input_schema else set()
        arguments: Dict[str, Any] = {}
        for key, value in result.params.items():
            if value and key in valid_param_names:
                arguments[key] = value

        # Special case: for scan_tray_items, default query to user message if not provided
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
    """Session-scoped orchestration state for the retail billing pipeline.

    Stores intermediate results as tools execute in sequence:
    scan → bill → invoice → send / speak

    This is pure orchestration state — no business logic lives here.
    """

    __slots__ = ("scan_result", "bill", "invoice_id", "updated_at")

    def __init__(self) -> None:
        self.scan_result: Optional[Dict[str, Any]] = None
        self.bill: Optional[Dict[str, Any]] = None
        self.invoice_id: Optional[int] = None
        self.updated_at: float = time.time()


# ── Executor ──────────────────────────────────────────────────────────────

class RetailExecutor(BaseDomainExecutor):
    """Executes approved retail tool call proposals with session-scoped pipeline state.

    Orchestrates tool execution and manages the data flow through the retail
    billing pipeline. Does NOT contain business logic — that lives in the
    service and tool layers.
    """

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
        """Get or create session-scoped pipeline state."""
        key = session_id or "__default__"
        with self._pipeline_lock:
            if key not in self._pipeline:
                self._pipeline[key] = _PipelineState()
            return self._pipeline[key]

    def _invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a retail tool with orchestration support.

        Orchestration behavior:
        - Auto-inject pipeline state into tools when appropriate
        - Store results back into pipeline state for downstream tools
        - Auto-invoke TTS after invoice generation (explicit policy)

        All business logic is delegated to tools and services.
        """
        import inspect
        from agents.tools.retail import (
            tool_scan_tray_items,
            tool_lookup_item_price,
            tool_create_bill,
            tool_generate_invoice,
            tool_send_invoice_email,
        )

        def _strip(fn, args: Dict[str, Any]) -> Dict[str, Any]:
            """Strip keys not accepted by *fn* to prevent unexpected-kwarg errors."""
            sig = inspect.signature(fn)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                return args  # function accepts **kwargs, nothing to strip
            valid = set(sig.parameters)
            dropped = set(args) - valid
            if dropped:
                logger.debug("Stripping unexpected args %s before calling %s", dropped, fn.__name__)
            return {k: v for k, v in args.items() if k in valid}

        pipeline = self._get_pipeline()

        # Defence-in-depth: strip any arguments not accepted by the target tool.
        # The interpreter already filters to schema params, but the executor may
        # inject additional keys (bill, scan_result, etc.) that ARE expected.
        # _strip() removes anything not in the function signature.

        # ── scan_tray_items ───────────────────────────────────────────────

        if tool_name == "scan_tray_items":
            result = tool_scan_tray_items(**_strip(tool_scan_tray_items, arguments))
            if result.get("success"):
                pipeline.scan_result = result.get("data", {})
                pipeline.updated_at = time.time()
            return result

        # ── lookup_item_price ─────────────────────────────────────────────

        if tool_name == "lookup_item_price":
            return tool_lookup_item_price(**_strip(tool_lookup_item_price, arguments))

        # ── create_bill ───────────────────────────────────────────────────

        if tool_name == "create_bill":
            # Auto-inject scan_result if no explicit items provided
            if "items" not in arguments or not arguments.get("items"):
                if pipeline.scan_result:
                    arguments = {**arguments, "scan_result": pipeline.scan_result}
                else:
                    # Auto-pipeline: scan first when no scan result exists.
                    # If a recent general-domain frame analysis exists, pass its
                    # description as context so the VLM scan prompt can reference
                    # the items already identified visually.
                    logger.info("create_bill: no scan result available, auto-running scan_tray_items")
                    scan_query = arguments.get("query", "Identify and count all items on the tray for billing")
                    try:
                        from agents.mcp.globals import get_mcp_executor as _get_general_executor
                        last_analysis = _get_general_executor().context.get("last_frame_analysis")
                        if last_analysis and last_analysis.get("description"):
                            scan_query += (
                                f"\n\nA recent frame analysis found these items: "
                                f"{last_analysis['description']}"
                            )
                            logger.info("create_bill: enriching auto-scan with prior frame analysis")
                    except Exception:
                        pass  # General executor unavailable — proceed without context
                    scan_result = tool_scan_tray_items(query=scan_query)
                    if scan_result.get("success"):
                        pipeline.scan_result = scan_result.get("data", {})
                        pipeline.updated_at = time.time()
                        arguments = {**arguments, "scan_result": pipeline.scan_result}
                    else:
                        return {"success": False, "message": f"Auto-pipeline: scan_tray_items failed — {scan_result.get('message', 'unknown error')}"}

            result = tool_create_bill(**_strip(tool_create_bill, arguments))
            if result.get("success"):
                pipeline.bill = result
                pipeline.updated_at = time.time()
            return result

        # ── generate_invoice ──────────────────────────────────────────────

        if tool_name == "generate_invoice":
            # Auto-inject bill if not provided
            if "bill" not in arguments or not arguments.get("bill"):
                if pipeline.bill:
                    arguments = {**arguments, "bill": pipeline.bill}
                else:
                    # Auto-pipeline: scan → bill → invoice when no bill exists
                    logger.info("generate_invoice: no bill available, running auto-pipeline (scan → bill → invoice)")
                    scan_query = arguments.get("query", "Identify and count all items on the tray for billing")
                    try:
                        from agents.mcp.globals import get_mcp_executor as _get_general_executor
                        last_analysis = _get_general_executor().context.get("last_frame_analysis")
                        if last_analysis and last_analysis.get("description"):
                            scan_query += (
                                f"\n\nA recent frame analysis found these items: "
                                f"{last_analysis['description']}"
                            )
                    except Exception:
                        pass
                    scan_result = tool_scan_tray_items(query=scan_query)
                    if scan_result.get("success"):
                        pipeline.scan_result = scan_result.get("data", {})
                        pipeline.updated_at = time.time()

                        bill_result = tool_create_bill(scan_result=pipeline.scan_result)
                        if bill_result.get("success"):
                            pipeline.bill = bill_result
                            pipeline.updated_at = time.time()
                            arguments = {**arguments, "bill": pipeline.bill}
                        else:
                            return {"success": False, "message": f"Auto-pipeline: create_bill failed — {bill_result.get('message', 'unknown error')}"}
                    else:
                        return {"success": False, "message": f"Auto-pipeline: scan_tray_items failed — {scan_result.get('message', 'unknown error')}"}

            result = tool_generate_invoice(**_strip(tool_generate_invoice, arguments))
            if result.get("success"):
                pipeline.invoice_id = result.get("data", {}).get("invoice_id")
                pipeline.updated_at = time.time()

            return result

        # ── send_invoice_email ────────────────────────────────────────────

        if tool_name == "send_invoice_email":
            # Auto-inject invoice_id or bill if not provided
            if "bill" not in arguments and "invoice_id" not in arguments:
                if pipeline.invoice_id:
                    arguments = {**arguments, "invoice_id": pipeline.invoice_id}
                elif pipeline.bill:
                    arguments = {**arguments, "bill": pipeline.bill}

            return tool_send_invoice_email(**_strip(tool_send_invoice_email, arguments))

        raise ValueError(f"No retail handler for: {tool_name}")


# ── Singletons ────────────────────────────────────────────────────────────

_retail_registry: Optional[RetailToolRegistry] = None
_retail_interpreter: Optional[RetailInterpreter] = None
_retail_executor: Optional[RetailExecutor] = None
_retail_lock = threading.RLock()


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
