"""The chat message value type."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class ChatMessage:
    """A single chat message."""

    def __init__(
        self,
        role: str,  # "user", "assistant", "system", "tool"
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
        id: Optional[str] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.role = role
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(
            role=str(data.get("role", "assistant")),
            content=str(data.get("content", "")),
            metadata=data.get("metadata") or {},
            timestamp=data.get("timestamp"),
            id=data.get("id"),
        )

    @classmethod
    def user(cls, content: str, **kwargs) -> "ChatMessage":
        return cls(role="user", content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str, **kwargs) -> "ChatMessage":
        return cls(role="assistant", content=content, **kwargs)

    @classmethod
    def system(cls, content: str, **kwargs) -> "ChatMessage":
        return cls(role="system", content=content, **kwargs)

    @classmethod
    def tool(
        cls, content: str, tool_name: str, success: bool, **kwargs
    ) -> "ChatMessage":
        return cls(
            role="tool",
            content=content,
            metadata={
                "tool_name": tool_name,
                "success": success,
                **kwargs.get("metadata", {}),
            },
            **{k: v for k, v in kwargs.items() if k != "metadata"},
        )
