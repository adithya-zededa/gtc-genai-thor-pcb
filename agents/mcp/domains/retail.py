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

TOOL_SPEAK_INVOICE = MCPToolDefinition(
    name="speak_invoice",
    description="Convert a finalized invoice into spoken audio. Produces a short, human-friendly spoken summary of the invoice.",
    category="retail_billing",
    input_schema=[
        MCPParameterSchema(name="text", type=MCPSchemaType.STRING, description="The spoken-friendly summary text to convert to audio", required=True, max_length=2000),
    ],
    output_schema=standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)


# ── Registry ──────────────────────────────────────────────────────────────

class RetailToolRegistry(MCPToolRegistry):
    """Registry containing only retail-domain tools."""

    def _register_default_tools(self) -> None:
        for tool in [TOOL_SCAN_TRAY_ITEMS, TOOL_LOOKUP_ITEM_PRICE, TOOL_CREATE_BILL, TOOL_GENERATE_INVOICE, TOOL_SEND_INVOICE_EMAIL, TOOL_SPEAK_INVOICE]:
            self.register(tool)


# ── Interpreter ───────────────────────────────────────────────────────────

class RetailInterpreter:
    """Intent interpreter for retail billing — LLM-first with keyword fallback."""

    _VALID_TOOLS: FrozenSet[str] = frozenset([
        "scan_tray_items", "lookup_item_price", "create_bill",
        "generate_invoice", "send_invoice_email", "speak_invoice",
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

        # Build arguments from classifier result
        arguments: Dict[str, Any] = {}
        for key, value in result.params.items():
            if value:
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
        from agents.tools.retail import (
            tool_scan_tray_items,
            tool_lookup_item_price,
            tool_create_bill,
            tool_generate_invoice,
            tool_send_invoice_email,
            tool_speak_invoice,
        )

        pipeline = self._get_pipeline()

        # ── scan_tray_items ───────────────────────────────────────────────

        if tool_name == "scan_tray_items":
            result = tool_scan_tray_items(**arguments)
            if result.get("success"):
                pipeline.scan_result = result.get("data", {})
                pipeline.updated_at = time.time()
            return result

        # ── lookup_item_price ─────────────────────────────────────────────

        if tool_name == "lookup_item_price":
            return tool_lookup_item_price(**arguments)

        # ── create_bill ───────────────────────────────────────────────────

        if tool_name == "create_bill":
            # Auto-inject scan_result if no explicit items provided
            if "items" not in arguments or not arguments.get("items"):
                if pipeline.scan_result:
                    arguments = {**arguments, "scan_result": pipeline.scan_result}

            result = tool_create_bill(**arguments)
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

            result = tool_generate_invoice(**arguments)
            if result.get("success"):
                pipeline.invoice_id = result.get("data", {}).get("invoice_id")
                pipeline.updated_at = time.time()

                # Auto-invoke TTS after successful invoice generation
                # This is explicit orchestration policy, not hidden business logic
                self._auto_invoke_tts(result, pipeline)

            return result

        # ── send_invoice_email ────────────────────────────────────────────

        if tool_name == "send_invoice_email":
            # Auto-inject invoice_id or bill if not provided
            if "bill" not in arguments and "invoice_id" not in arguments:
                if pipeline.invoice_id:
                    arguments = {**arguments, "invoice_id": pipeline.invoice_id}
                elif pipeline.bill:
                    arguments = {**arguments, "bill": pipeline.bill}

            return tool_send_invoice_email(**arguments)

        # ── speak_invoice ─────────────────────────────────────────────────

        if tool_name == "speak_invoice":
            return tool_speak_invoice(**arguments)

        raise ValueError(f"No retail handler for: {tool_name}")

    # ── TTS auto-invocation (explicit orchestration policy) ───────────────

    def _auto_invoke_tts(
        self,
        invoice_result: Dict[str, Any],
        pipeline: _PipelineState,
    ) -> None:
        """Auto-invoke TTS after successful invoice generation.

        This is an explicit orchestration policy:
        "After generating an invoice, automatically speak it unless it fails."

        The business logic for determining WHAT to speak lives in the service layer.
        This method just orchestrates the call sequence.

        On failure, log and continue — invoice generation is never affected.
        """
        try:
            from services.domains.retail.invoice_service import format_invoice_for_speech
            from agents.tools.retail import tool_speak_invoice

            # Delegate text composition to service layer (business logic)
            speech_text = format_invoice_for_speech(
                invoice_result,
                pipeline.bill,
                full_narration=False,  # Default: short summary
            )

            if not speech_text:
                logger.warning("TTS skipped — could not format speech text from invoice")
                return

            # Execute TTS tool
            tts_result = tool_speak_invoice(text=speech_text)
            if not tts_result.get("success"):
                logger.warning("TTS invocation failed: %s", tts_result.get("message"))

        except Exception as exc:
            # TTS failure must never break the invoice pipeline
            logger.error("speak_invoice auto-invocation failed (non-fatal): %s", exc, exc_info=True)


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
