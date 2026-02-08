"""Session management for MCP agent interactions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class SessionType(str, Enum):
    """Types of agent sessions."""
    MONITORING = "monitoring"
    INVESTIGATION = "investigation"
    CONFIGURATION = "configuration"
    GENERAL = "general"


@dataclass
class MCPSession:
    """Represents a bounded conversation session."""
    id: str
    type: SessionType
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: Optional[str] = None
    summary: Optional[str] = None

    # Session metrics
    message_count: int = 0
    tool_proposals: int = 0
    tool_executions: int = 0
    tool_rejections: int = 0

    # Context
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, session_type: SessionType, metadata: Optional[Dict[str, Any]] = None) -> "MCPSession":
        return cls(
            id=f"session_{uuid.uuid4().hex[:12]}",
            type=session_type,
            metadata=metadata or {},
        )

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    def end(self, summary: Optional[str] = None) -> None:
        """End the session."""
        self.ended_at = datetime.now().isoformat()
        self.summary = summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "summary": self.summary,
            "is_active": self.is_active,
            "message_count": self.message_count,
            "tool_proposals": self.tool_proposals,
            "tool_executions": self.tool_executions,
            "tool_rejections": self.tool_rejections,
            "metadata": self.metadata,
        }
