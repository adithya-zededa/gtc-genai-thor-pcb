"""PCB Inspection domain — intent interpreter.

Classifies user intent into PCB tool call proposals using the
LLM classifier with deterministic fallbacks for time-window inference.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

from agents.mcp.audit import AuditEventType, AuditLogEntry
from agents.mcp.globals import get_audit_log
from agents.mcp.lifecycle import MCPToolCallProposal
from agents.mcp.state_machine import AgentState
from agents.tools._email_utils import resolve_recipients

from .tool_defs import PCBToolRegistry, TOOL_PARAM_ALLOWLIST, HOURS_AWARE_TOOLS

from core.logging import get_logger

logger = get_logger(__name__)


def _infer_hours_from_message(user_message: str) -> Optional[float]:
    """Infer a time window in hours from natural-language temporal phrases.

    This is a deterministic fallback used when the LLM classifier does not
    provide ``hours`` for analytics tools that support time windows.
    """
    if not user_message:
        return None

    text = user_message.strip().lower()
    if not text:
        return None

    # Common natural-language shorthands.
    if "today" in text:
        return 24.0
    if "last hour" in text or "past hour" in text:
        return 1.0
    if "last day" in text or "past day" in text or "daily" in text:
        return 24.0
    if "last week" in text or "past week" in text or "weekly" in text:
        return 168.0

    # Numeric windows: "last 6 hours", "past 2 days", etc.
    match = re.search(r"(?:last|past)\s+(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs|day|days|week|weeks)", text)
    if not match:
        return None

    quantity = float(match.group(1))
    unit = match.group(2)

    if unit in ("hour", "hours", "hr", "hrs"):
        return quantity
    if unit in ("day", "days"):
        return quantity * 24.0
    if unit in ("week", "weeks"):
        return quantity * 168.0
    return None


class PCBInterpreter:
    """Intent interpreter for PCB inspection — LLM-first with keyword fallback."""

    # All tools registered in the PCB registry are available to the agent.
    # No static whitelist — the registry is the single source of truth.

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

        if result.domain != "pcb" or not result.tool:
            return None

        # Deprecated PCB monitoring tools belong to the general domain —
        # return None so the manager falls back to the general interpreter
        # which will remap them to the correct tool names.
        if result.tool in ("start_defect_monitoring", "stop_defect_monitoring"):
            logger.info(
                "PCB interpreter: deprecated tool '%s' — deferring to general interpreter",
                result.tool,
            )
            return None

        tool_def = self.registry.get(result.tool)
        if not tool_def:
            return None

        # Runtime state policy is enforced by the executor.

        arguments: Dict[str, Any] = {}
        allowed = TOOL_PARAM_ALLOWLIST.get(result.tool, frozenset())
        for key, value in result.params.items():
            if key in allowed and value is not None:
                arguments[key] = value

        if result.tool in HOURS_AWARE_TOOLS:
            raw_hours = arguments.get("hours")
            parsed_hours: Optional[float] = None
            if raw_hours is not None:
                try:
                    parsed_hours = float(raw_hours)
                except (TypeError, ValueError):
                    parsed_hours = None

            if parsed_hours is not None and parsed_hours > 0:
                arguments["hours"] = parsed_hours
            else:
                inferred_hours = _infer_hours_from_message(user_message)
                if inferred_hours is not None and inferred_hours > 0:
                    arguments["hours"] = inferred_hours
                else:
                    arguments.pop("hours", None)

        if result.tool == "inspect_pcb" and "query" not in arguments:
            arguments["query"] = user_message

        if result.tool in ("send_defect_alert", "start_defect_monitoring"):
            recipients = resolve_recipients(arguments, user_message)
            if recipients:
                arguments["recipients"] = recipients

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
