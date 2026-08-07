"""General-domain MCP interpreter — classifies user intent into tool proposals."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from core.logging import get_logger

from agents.mcp.audit import AuditEventType, AuditLogEntry, MCPAuditLog
from agents.mcp.lifecycle import MCPToolCallProposal
from agents.mcp.registry import MCPToolRegistry
from agents.mcp.state_machine import AgentState

from .tool_defs import TOOL_PARAM_ALLOWLIST

logger = get_logger(__name__)


class MCPInterpreter:
    """Interprets user intent and produces tool call proposals.

    This is the INTERPRETATION PHASE. No tools are executed here.
    Uses the LLM intent classifier to determine which tool to call.
    """

    # All tools registered in the registry are available to the agent.
    # No static whitelist — the registry is the single source of truth.

    def __init__(self, registry: MCPToolRegistry, audit_log: MCPAuditLog) -> None:
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

        if not result.tool:
            return None

        # Remap deprecated tool names to their current equivalents
        _DEPRECATED_TOOL_MAP = {
            "start_defect_monitoring": "start_monitoring_session",
            "stop_defect_monitoring": "end_session",
        }
        tool_name = _DEPRECATED_TOOL_MAP.get(result.tool, result.tool)
        if tool_name != result.tool:
            logger.info(
                "General interpreter: remapped deprecated tool '%s' -> '%s'",
                result.tool, tool_name,
            )
            result.tool = tool_name

        tool_def = self.registry.get(result.tool)
        if not tool_def:
            return None

        # Runtime state policy is enforced by the executor.

        arguments: Dict[str, Any] = {}
        allowed = TOOL_PARAM_ALLOWLIST.get(result.tool, frozenset())
        for key, value in (result.params or {}).items():
            if key in allowed and value is not None:
                arguments[key] = value

        if result.tool == "analyze_current_frame" and "query" not in arguments:
            arguments["query"] = user_message

        if result.tool == "send_alert_email":
            from agents.tools._email_utils import resolve_recipients
            recipients = resolve_recipients(arguments, user_message)
            if recipients:
                arguments["recipients"] = recipients

            if not isinstance(arguments.get("subject"), str) or not arguments.get("subject", "").strip():
                arguments["subject"] = "PCB Defect Alert"

            if not isinstance(arguments.get("body"), str) or not arguments.get("body", "").strip():
                arguments["body"] = user_message.strip() or "Automated alert requested by user."

        proposal = MCPToolCallProposal.create(
            tool_name=result.tool,
            arguments=arguments,
            rationale=result.rationale or f"LLM classified as {result.tool}",
            confidence=result.confidence,
            requires_confirmation=tool_def.requires_confirmation,
            session_id=session_id,
        )

        return self._finalize_proposal(proposal, start_time, session_id)

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
