"""MCP Manager / Router — dispatches messages to domain-specific MCPs.

Supports three domains:
    - **pcb**    → PCBInterpreter / PCBExecutor   (agents.pcb_mcp)
    - **retail** → RetailInterpreter / RetailExecutor (agents.retail_mcp)
    - **general** → MCPInterpreter / MCPExecutor  (agents.mcp)

Routing logic
~~~~~~~~~~~~~
1. If the caller provides an explicit ``domain`` string, use it.
2. Otherwise, scan the user message for domain keywords and pick the best
   match.
3. If no domain keywords match, fall through to the **general** MCP.

The manager exposes a thin façade that mirrors the interpreter / executor
interfaces the WebSocket chat and REST API already depend on.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from agents.mcp import (
    AgentState,
    MCPToolCallProposal,
    MCPToolRegistry,
    get_agent_state_machine,
    get_audit_log,
    get_mcp_executor,
    get_mcp_interpreter,
    get_tool_registry,
    AuditEventType,
    AuditLogEntry,
)
from core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# DOMAIN ENUM
# =============================================================================

DOMAIN_PCB = "pcb"
DOMAIN_RETAIL = "retail"
DOMAIN_GENERAL = "general"

VALID_DOMAINS = frozenset([DOMAIN_PCB, DOMAIN_RETAIL, DOMAIN_GENERAL])


# =============================================================================
# KEYWORD SETS FOR AUTO-DETECTION
# =============================================================================

_PCB_KEYWORDS: frozenset = frozenset([
    # Board / component
    "pcb", "circuit board", "printed circuit",
    "solder", "soldering", "solder bridge",
    "trace", "traces", "via", "vias",
    "capacitor", "resistor", "ic chip",
    # Board names
    "arduino", "raspberry pi", "esp32", "esp8266",
    "stm32", "teensy", "nodemcu", "jetson",
    # Defects
    "defect", "defective", "defects",
    "short circuit", "cold solder", "dry joint",
    "tombstone", "bridging", "whisker",
    "missing component", "misaligned",
    # Inspection verbs
    "inspect pcb", "inspect board",
    "quality check", "quality inspection",
    "board inspection",
])

_RETAIL_KEYWORDS: frozenset = frozenset([
    # Items / products
    "item", "items", "product", "products",
    "goods", "merchandise",
    "tray", "shelf", "counter",
    # Billing / pricing
    "bill", "billing", "invoice",
    "price", "prices", "pricing",
    "cost", "total", "subtotal",
    "receipt", "checkout",
    # Actions
    "ring up", "scan tray", "scan items",
    "count items", "count them",
    "create a bill", "generate invoice",
    "send invoice", "email invoice",
    "catalog", "catalogue",
    "retail", "packaging",
    "sku", "barcode",
])


# =============================================================================
# MCP MANAGER
# =============================================================================

class MCPManager:
    """Central router that holds all domain MCPs and dispatches by domain.

    Usage::

        mgr = get_mcp_manager()
        domain, interpreter, executor, registry = mgr.route(user_message)
        proposal = interpreter.interpret(user_message, agent_state, session_id)
        result  = executor.submit_proposal(proposal)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialised = False

        # Lazy references — populated on first access
        self._pcb_interpreter = None
        self._pcb_executor = None
        self._pcb_registry = None

        self._retail_interpreter = None
        self._retail_executor = None
        self._retail_registry = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_init(self) -> None:
        """Lazily import and instantiate domain MCPs."""
        if self._initialised:
            return
        with self._lock:
            if self._initialised:
                return
            try:
                from agents.pcb_mcp import (
                    get_pcb_interpreter,
                    get_pcb_executor,
                    get_pcb_registry,
                )
                self._pcb_interpreter = get_pcb_interpreter()
                self._pcb_executor = get_pcb_executor()
                self._pcb_registry = get_pcb_registry()
                logger.info("PCB MCP loaded successfully")
            except Exception as exc:
                logger.error("Failed to load PCB MCP: %s", exc, exc_info=True)

            try:
                from agents.retail_mcp import (
                    get_retail_interpreter,
                    get_retail_executor,
                    get_retail_registry,
                )
                self._retail_interpreter = get_retail_interpreter()
                self._retail_executor = get_retail_executor()
                self._retail_registry = get_retail_registry()
                logger.info("Retail MCP loaded successfully")
            except Exception as exc:
                logger.error("Failed to load Retail MCP: %s", exc, exc_info=True)

            self._initialised = True

    # ------------------------------------------------------------------
    # Domain detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_domain(message: str) -> str:
        """Return the best-matching domain for *message*.

        Uses the LLM intent classifier first; falls back to keyword
        scoring when the inference backend is unreachable.
        """
        try:
            from agents.llm_classifier import get_classifier
            result = get_classifier().classify(message)
            if result.domain in VALID_DOMAINS and result.confidence >= 0.3:
                logger.info(
                    "LLM classifier domain=%s confidence=%.2f source=%s",
                    result.domain, result.confidence, result.source,
                )
                return result.domain
        except Exception as exc:
            logger.warning("LLM classifier unavailable, using keyword fallback: %s", exc)

        # Keyword fallback
        msg = message.lower()
        pcb_score = sum(1 for kw in _PCB_KEYWORDS if kw in msg)
        retail_score = sum(1 for kw in _RETAIL_KEYWORDS if kw in msg)

        if pcb_score == 0 and retail_score == 0:
            return DOMAIN_GENERAL
        if pcb_score > retail_score:
            return DOMAIN_PCB
        if retail_score > pcb_score:
            return DOMAIN_RETAIL
        return DOMAIN_GENERAL

    # ------------------------------------------------------------------
    # Public routing API
    # ------------------------------------------------------------------

    def route(
        self,
        message: str,
        domain: Optional[str] = None,
    ) -> Tuple[str, Any, Any, Any]:
        """Select the correct interpreter / executor / registry triple.

        Parameters
        ----------
        message : str
            The raw user message (used for auto-detection if *domain* is None).
        domain : str, optional
            Explicit domain override (``"pcb"``, ``"retail"``, ``"general"``).

        Returns
        -------
        (domain, interpreter, executor, registry)
        """
        self._ensure_init()

        resolved_domain = domain if domain in VALID_DOMAINS else self.detect_domain(message)

        if resolved_domain == DOMAIN_PCB and self._pcb_interpreter:
            logger.info("Routing to PCB MCP")
            return (
                DOMAIN_PCB,
                self._pcb_interpreter,
                self._pcb_executor,
                self._pcb_registry,
            )

        if resolved_domain == DOMAIN_RETAIL and self._retail_interpreter:
            logger.info("Routing to Retail MCP")
            return (
                DOMAIN_RETAIL,
                self._retail_interpreter,
                self._retail_executor,
                self._retail_registry,
            )

        # Fall through to generic MCP
        logger.info("Routing to General MCP")
        return (
            DOMAIN_GENERAL,
            get_mcp_interpreter(),
            get_mcp_executor(),
            get_tool_registry(),
        )

    # ------------------------------------------------------------------
    # Convenience: interpret + submit in one call
    # ------------------------------------------------------------------

    def interpret(
        self,
        message: str,
        agent_state: Optional[AgentState] = None,
        session_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Tuple[str, Optional[MCPToolCallProposal]]:
        """Interpret a message through the correct domain interpreter.

        Returns ``(resolved_domain, proposal_or_none)``.
        """
        state_machine = get_agent_state_machine()
        state = agent_state or state_machine.state

        resolved, interpreter, _executor, _registry = self.route(message, domain)

        proposal = interpreter.interpret(
            user_message=message,
            agent_state=state,
            session_id=session_id,
        )

        # Tag the proposal with the domain for downstream routing
        if proposal and not proposal.is_rejected:
            proposal.arguments.setdefault("__domain", resolved)

        return resolved, proposal

    def submit(
        self,
        proposal: MCPToolCallProposal,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit a proposal to the correct domain executor.

        The domain can be supplied explicitly, read from the proposal's
        ``__domain`` argument tag, or re-detected from the proposal source.
        """
        self._ensure_init()

        resolved = (
            domain
            or proposal.arguments.pop("__domain", None)
            or DOMAIN_GENERAL
        )

        _, _interpreter, executor, _registry = self.route("", resolved)
        return executor.submit_proposal(proposal)

    def approve_proposal(
        self,
        proposal_id: str,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Approve a pending proposal by checking all domain executors."""
        self._ensure_init()

        # If caller knows the domain, check that executor first
        if domain == DOMAIN_PCB and self._pcb_executor:
            result = self._pcb_executor.approve_proposal(proposal_id)
            if result.get("status") != "error":
                return result

        if domain == DOMAIN_RETAIL and self._retail_executor:
            result = self._retail_executor.approve_proposal(proposal_id)
            if result.get("status") != "error":
                return result

        if domain == DOMAIN_GENERAL:
            return get_mcp_executor().approve_proposal(proposal_id)

        # No explicit domain — search all executors
        for executor in self._all_executors():
            result = executor.approve_proposal(proposal_id)
            if result.get("status") != "error":
                return result

        return {"status": "error", "reason": f"Proposal {proposal_id} not found in any domain"}

    def reject_proposal(
        self,
        proposal_id: str,
        reason: str = "User rejected",
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reject a pending proposal across all domain executors."""
        self._ensure_init()

        if domain == DOMAIN_PCB and self._pcb_executor:
            result = self._pcb_executor.reject_proposal(proposal_id, reason)
            if result.get("status") != "error":
                return result

        if domain == DOMAIN_RETAIL and self._retail_executor:
            result = self._retail_executor.reject_proposal(proposal_id, reason)
            if result.get("status") != "error":
                return result

        if domain == DOMAIN_GENERAL:
            return get_mcp_executor().reject_proposal(proposal_id, reason)

        for executor in self._all_executors():
            result = executor.reject_proposal(proposal_id, reason)
            if result.get("status") != "error":
                return result

        return {"status": "error", "reason": f"Proposal {proposal_id} not found"}

    def get_pending_proposals(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Collect pending proposals, optionally filtered by domain."""
        self._ensure_init()
        proposals: List[Dict[str, Any]] = []

        if domain is None or domain == DOMAIN_PCB:
            if self._pcb_executor:
                for p in self._pcb_executor.get_pending_proposals():
                    p["domain"] = DOMAIN_PCB
                    proposals.append(p)

        if domain is None or domain == DOMAIN_RETAIL:
            if self._retail_executor:
                for p in self._retail_executor.get_pending_proposals():
                    p["domain"] = DOMAIN_RETAIL
                    proposals.append(p)

        if domain is None or domain == DOMAIN_GENERAL:
            for p in get_mcp_executor().get_pending_proposals():
                p["domain"] = DOMAIN_GENERAL
                proposals.append(p)

        return proposals

    def get_all_tools(self, agent_state: Optional[AgentState] = None) -> Dict[str, List]:
        """Return all tools grouped by domain."""
        self._ensure_init()
        state_machine = get_agent_state_machine()
        state = agent_state or state_machine.state

        result: Dict[str, List] = {}

        if self._pcb_registry:
            result[DOMAIN_PCB] = self._pcb_registry.get_display_list(state)
        if self._retail_registry:
            result[DOMAIN_RETAIL] = self._retail_registry.get_display_list(state)
        result[DOMAIN_GENERAL] = get_tool_registry().get_display_list(state)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _all_executors(self):
        """Yield all available executors (domain-specific first)."""
        if self._pcb_executor:
            yield self._pcb_executor
        if self._retail_executor:
            yield self._retail_executor
        yield get_mcp_executor()


# =============================================================================
# MODULE SINGLETON
# =============================================================================

_mcp_manager: Optional[MCPManager] = None
_manager_lock = threading.Lock()


def get_mcp_manager() -> MCPManager:
    """Return the global ``MCPManager`` singleton."""
    global _mcp_manager
    if _mcp_manager is None:
        with _manager_lock:
            if _mcp_manager is None:
                _mcp_manager = MCPManager()
                logger.info("MCPManager singleton created")
    return _mcp_manager
