"""Tool execution lifecycle types for MCP proposals and results."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class ToolLifecycleState(str, Enum):
    """Explicit execution lifecycle states for tool calls."""
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MCPToolCallProposal:
    """A proposed tool call awaiting approval.

    Represents the INTERPRETATION phase output.
    The tool is NOT executed until explicitly approved.
    """
    id: str
    tool_name: str
    arguments: Dict[str, Any]
    rationale: str
    confidence: float
    requires_confirmation: bool
    state: ToolLifecycleState = ToolLifecycleState.PROPOSED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: Optional[str] = None

    # Lifecycle tracking
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None
    rejected_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    executed_at: Optional[str] = None
    completed_at: Optional[str] = None

    @classmethod
    def create(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        rationale: str,
        confidence: float,
        requires_confirmation: bool,
        session_id: Optional[str] = None,
    ) -> "MCPToolCallProposal":
        return cls(
            id=f"proposal_{uuid.uuid4().hex[:12]}",
            tool_name=tool_name,
            arguments=arguments,
            rationale=rationale,
            confidence=confidence,
            requires_confirmation=requires_confirmation,
            session_id=session_id,
        )

    @property
    def is_rejected(self) -> bool:
        return self.state == ToolLifecycleState.REJECTED

    def approve(self, approved_by: str = "user") -> None:
        """Mark proposal as approved."""
        if self.state not in (ToolLifecycleState.PROPOSED, ToolLifecycleState.PENDING_APPROVAL):
            raise ValueError(f"Cannot approve proposal in state: {self.state}")
        self.state = ToolLifecycleState.APPROVED
        self.approved_at = datetime.now().isoformat()
        self.approved_by = approved_by

    def reject(self, reason: str = "") -> None:
        """Mark proposal as rejected."""
        if self.state not in (ToolLifecycleState.PROPOSED, ToolLifecycleState.PENDING_APPROVAL):
            raise ValueError(f"Cannot reject proposal in state: {self.state}")
        self.state = ToolLifecycleState.REJECTED
        self.rejected_at = datetime.now().isoformat()
        self.rejection_reason = reason

    def start_execution(self) -> None:
        """Mark proposal as executing."""
        if self.state != ToolLifecycleState.APPROVED:
            raise ValueError(f"Cannot execute proposal in state: {self.state}")
        self.state = ToolLifecycleState.EXECUTING
        self.executed_at = datetime.now().isoformat()

    def complete(self, success: bool, error: Optional[str] = None) -> None:
        """Mark execution as complete."""
        if self.state != ToolLifecycleState.EXECUTING:
            raise ValueError(f"Cannot complete proposal in state: {self.state}")
        self.state = ToolLifecycleState.SUCCEEDED if success else ToolLifecycleState.FAILED
        self.completed_at = datetime.now().isoformat()
        if error:
            self.rejection_reason = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "requires_confirmation": self.requires_confirmation,
            "state": self.state.value,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "approved_at": self.approved_at,
            "approved_by": self.approved_by,
            "rejected_at": self.rejected_at,
            "rejection_reason": self.rejection_reason,
            "executed_at": self.executed_at,
            "completed_at": self.completed_at,
        }


@dataclass
class MCPToolResult:
    """Result from executing a tool."""
    proposal_id: str
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: float = 0.0
    executed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "executed_at": self.executed_at,
        }
