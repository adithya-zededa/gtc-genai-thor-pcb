"""Base domain executor — shared submit/approve/reject/dedup/timeout logic.

Both ``PCBExecutor`` and the general ``MCPExecutor``
previously contained ~200 lines of near-identical proposal lifecycle code.
This base class factors that out; subclasses only implement ``_invoke()``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Dict, List, Optional

from core.logging import get_logger

from .audit import AuditEventType, AuditLogEntry, MCPAuditLog
from .lifecycle import MCPToolCallProposal, MCPToolResult, ToolLifecycleState
from .registry import MCPToolDefinition, MCPToolRegistry
from .session import MCPSession
from .state_machine import AgentStateMachine

logger = get_logger(__name__)

_DEFAULT_INVOKE_TIMEOUT = 60
_DEDUP_WINDOW = 5.0


class BaseDomainExecutor:
    """Shared executor skeleton for any MCP domain.

    Subclasses must override:
    - ``_invoke(tool_name, arguments, **kw) -> Dict[str, Any]``

    Optionally override:
    - ``_domain_label`` (str) for audit/log messages
    """

    _domain_label: str = "general"

    def __init__(
        self,
        registry: MCPToolRegistry,
        state_machine: AgentStateMachine,
        audit_log: MCPAuditLog,
        context: Optional[Dict[str, Any]] = None,
        invoke_timeout: int = _DEFAULT_INVOKE_TIMEOUT,
        max_workers: int = 4,
    ) -> None:
        self.registry = registry
        self.state_machine = state_machine
        self.audit_log = audit_log
        self.context = context or {}
        self._invoke_timeout = invoke_timeout

        self._current_session: Optional[MCPSession] = None
        self._sessions: List[MCPSession] = []
        self._session_lock = threading.Lock()

        self._pending_proposals: Dict[str, MCPToolCallProposal] = {}
        self._proposals_lock = threading.Lock()

        self._recent_hashes: Dict[str, float] = {}
        self._dedup_lock = threading.Lock()

        self._context_lock = threading.Lock()

        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"{self._domain_label}-exec",
        )

    # ── context helpers ───────────────────────────────────────────────────

    @property
    def current_session(self) -> Optional[MCPSession]:
        with self._session_lock:
            return self._current_session

    # ── dedup ─────────────────────────────────────────────────────────────

    def _is_duplicate(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        sig = hashlib.sha256(
            json.dumps({"t": tool_name, "a": arguments}, sort_keys=True, default=str).encode()
        ).hexdigest()
        now = time.time()
        with self._dedup_lock:
            expired = [h for h, ts in self._recent_hashes.items() if now - ts > _DEDUP_WINDOW]
            for h in expired:
                del self._recent_hashes[h]
            if sig in self._recent_hashes:
                return True
            self._recent_hashes[sig] = now
        return False

    # ── public API ────────────────────────────────────────────────────────

    def submit_proposal(self, proposal: MCPToolCallProposal) -> Dict[str, Any]:
        tool = self.registry.get(proposal.tool_name)
        if not tool:
            proposal.reject(f"Unknown {self._domain_label} tool: {proposal.tool_name}")
            self._log_rejection(proposal)
            return {"status": "rejected", "reason": proposal.rejection_reason, "proposal": proposal.to_dict()}

        is_valid, error = tool.validate_input(proposal.arguments)
        if not is_valid:
            proposal.reject(f"Validation error: {error}")
            self._log_rejection(proposal)
            return {"status": "rejected", "reason": error, "proposal": proposal.to_dict()}

        current_state = self.state_machine.state
        if current_state not in tool.allowed_in_states:
            allowed = [state.value for state in tool.allowed_in_states]
            reason = (
                f"Tool '{proposal.tool_name}' is not allowed in state "
                f"'{current_state.value}'. Allowed states: {allowed}"
            )
            proposal.reject(reason)
            self._log_rejection(proposal)
            self.audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.VALIDATION_ERROR,
                details={
                    "proposal_id": proposal.id,
                    "tool_name": proposal.tool_name,
                    "current_state": current_state.value,
                    "allowed_states": allowed,
                    "domain": self._domain_label,
                },
                session_id=proposal.session_id,
            ))
            return {"status": "rejected", "reason": reason, "proposal": proposal.to_dict()}

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
                details={
                    "proposal_id": proposal.id,
                    "tool_name": proposal.tool_name,
                    "domain": self._domain_label,
                },
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
            return {"status": "error", "reason": f"No pending {self._domain_label} proposal: {proposal_id}"}
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
            return {"status": "error", "reason": f"No pending {self._domain_label} proposal: {proposal_id}"}
        proposal.reject(reason)
        self._log_rejection(proposal)
        return {"status": "rejected", "proposal_id": proposal_id, "reason": reason}

    def get_pending_proposals(self) -> List[Dict[str, Any]]:
        with self._proposals_lock:
            return [p.to_dict() for p in self._pending_proposals.values()]

    def submit_agentic_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context_updates: Optional[Dict[str, Any]] = None,
        rationale: str = "",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit a VLM-chosen tool call through the normal proposal pipeline.

        Agentic tool calls are picked by the VLM from image/prompt content, which
        can include untrusted text (prompt injection). Routing them through
        ``submit_proposal`` (instead of calling ``_invoke`` directly) ensures the
        same validation, state-machine gating, dedup, and ``requires_confirmation``
        checks apply as for every other tool call. ``context_updates`` is applied
        under ``_context_lock`` so concurrent callers can't race on shared state
        such as ``image_data``.
        """
        tool = self.registry.get(tool_name)
        if not tool:
            return {
                "success": False,
                "status": "rejected",
                "error": f"Unknown {self._domain_label} tool: {tool_name}",
            }

        with self._context_lock:
            self.context.update(context_updates or {})
            proposal = MCPToolCallProposal.create(
                tool_name=tool_name,
                arguments=arguments,
                rationale=rationale,
                confidence=1.0,
                requires_confirmation=tool.requires_confirmation,
                session_id=session_id,
            )
            result = self.submit_proposal(proposal)

        return self._adapt_agentic_result(result)

    @staticmethod
    def _adapt_agentic_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the dict returned to the VLM tool loop always has ``success``.

        ``UnifiedVLMClient.analyze_with_tools`` defaults ``result.get("success", True)``
        when the key is absent, so every status branch here must set it explicitly.
        """
        adapted = dict(result)
        status = result.get("status")
        if status == "executed":
            output = result.get("result", {}).get("output")
            adapted["success"] = bool(output.get("success", True)) if isinstance(output, dict) else True
        elif status == "pending_approval":
            adapted["success"] = False
            adapted["error"] = result.get("confirmation_message") or "Awaiting user approval"
        elif status == "rejected":
            adapted["success"] = False
            adapted["error"] = result.get("reason", "Tool call rejected")
        else:
            adapted["success"] = False
            adapted["error"] = result.get("error", "Tool execution failed")
        return adapted

    # ── execution ─────────────────────────────────────────────────────────

    def _execute_proposal(self, proposal: MCPToolCallProposal, tool: MCPToolDefinition) -> Dict[str, Any]:
        start = time.time()
        proposal.start_execution()
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_EXECUTING,
            details={"proposal_id": proposal.id, "tool_name": proposal.tool_name, "domain": self._domain_label},
            session_id=proposal.session_id,
        ))

        try:
            future = self._pool.submit(self._invoke, proposal.tool_name, proposal.arguments)
            result = future.result(timeout=self._invoke_timeout)
            duration = (time.time() - start) * 1000
            proposal.complete(success=True)
            tr = MCPToolResult(
                proposal_id=proposal.id, tool_name=proposal.tool_name,
                success=True, output=result, duration_ms=duration,
            )
            self.audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.TOOL_SUCCEEDED,
                details={
                    "proposal_id": proposal.id,
                    "tool_name": proposal.tool_name,
                    "output_preview": str(result)[:500],
                },
                session_id=proposal.session_id,
                latency_ms=duration,
            ))
            if self._current_session:
                self._current_session.tool_executions += 1
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
            logger.error("%s tool execution failed: %s", self._domain_label, e, exc_info=True)
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

    def _invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch to the concrete tool handler. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement _invoke()")

    # ── helpers ───────────────────────────────────────────────────────────

    def _format_confirmation(self, tool: MCPToolDefinition, proposal: MCPToolCallProposal) -> str:
        if not tool.confirmation_message:
            return f"Execute {self._domain_label} tool '{tool.name}'?"
        msg = tool.confirmation_message
        for k, v in proposal.arguments.items():
            msg = msg.replace(f"{{{k}}}", str(v))
        return msg

    def _log_approval(self, proposal: MCPToolCallProposal) -> None:
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_APPROVED,
            details={
                "proposal_id": proposal.id,
                "tool_name": proposal.tool_name,
                "approved_by": proposal.approved_by,
                "domain": self._domain_label,
            },
            session_id=proposal.session_id,
        ))

    def _log_rejection(self, proposal: MCPToolCallProposal) -> None:
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_REJECTED,
            details={
                "proposal_id": proposal.id,
                "tool_name": proposal.tool_name,
                "reason": proposal.rejection_reason,
                "domain": self._domain_label,
            },
            session_id=proposal.session_id,
        ))
        if self._current_session:
            self._current_session.tool_rejections += 1
