"""General-domain MCP interpreter — classifies user intent into tool proposals."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from core.logging import get_logger

from .audit import AuditEventType, AuditLogEntry
from .lifecycle import MCPToolCallProposal
from .registry import MCPToolRegistry
from .state_machine import AgentState

logger = get_logger(__name__)


class MCPInterpreter:
    """Interprets user intent and produces tool call proposals.

    This is the INTERPRETATION PHASE. No tools are executed here.
    Uses the LLM intent classifier to determine which tool to call.
    """

    _VALID_TOOLS: frozenset = frozenset([
        "start_monitoring_session",
        "end_session",
        "go_idle",
        "get_agent_status",
        "analyze_current_frame",
        "query_history",
        "get_session_summary",
        "set_detection_task",
        "send_alert_email",
        "save_evidence",
        "log_event",
        "shutdown_agent",
        "acknowledge_error",
    ])

    def __init__(self, registry: MCPToolRegistry, audit_log: "MCPAuditLog") -> None:
        self.registry = registry
        self.audit_log = audit_log

    def interpret(
        self,
        user_message: str,
        agent_state: AgentState,
        session_id: Optional[str] = None,
    ) -> Optional[MCPToolCallProposal]:
        """Interpret user message and produce a tool call proposal.

        Returns None if no tool call is needed (conversational messages).
        """
        start_time = time.time()

        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.INTENT_DETECTED,
            details={"message": user_message[:500], "agent_state": agent_state.value},
            session_id=session_id,
        ))

        try:
            from agents.classifiers.llm_classifier import get_classifier

            result = get_classifier().classify(user_message)
            logger.info(
                "General interpreter: LLM classified domain=%s tool=%s confidence=%.2f source=%s",
                result.domain, result.tool, result.confidence, result.source,
            )
        except Exception as exc:
            logger.warning("LLM classifier failed in general interpreter: %s", exc)
            return None

        if not result.tool or result.tool not in self._VALID_TOOLS:
            return None

        tool_def = self.registry.get(result.tool)
        if not tool_def:
            return None

        # NOTE: No state-based filtering — the LLM decides all actions.

        arguments: Dict[str, Any] = {}
        if result.params:
            arguments.update(result.params)

        if result.tool == "analyze_current_frame" and "query" not in arguments:
            arguments["query"] = user_message

        proposal = MCPToolCallProposal.create(
            tool_name=result.tool,
            arguments=arguments,
            rationale=result.rationale or f"LLM classified as {result.tool}",
            confidence=result.confidence,
            requires_confirmation=tool_def.requires_confirmation,
            session_id=session_id,
        )

        return self._finalize_proposal(proposal, start_time, session_id)

    def _create_state_violation_proposal(
        self,
        tool_name: str,
        current_state: AgentState,
        allowed_states: tuple,
        session_id: Optional[str],
    ) -> MCPToolCallProposal:
        proposal = MCPToolCallProposal.create(
            tool_name=tool_name,
            arguments={},
            rationale=f"Tool '{tool_name}' not available in state '{current_state.value}'",
            confidence=0.0,
            requires_confirmation=False,
            session_id=session_id,
        )
        proposal.reject(
            f"Cannot execute '{tool_name}' in state '{current_state.value}'. "
            f"Allowed states: {[s.value for s in allowed_states]}"
        )
        return proposal

    def _finalize_proposal(
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
                "rationale": proposal.rationale,
                "confidence": proposal.confidence,
                "requires_confirmation": proposal.requires_confirmation,
            },
            session_id=session_id,
            latency_ms=latency_ms,
        ))

        return proposal
