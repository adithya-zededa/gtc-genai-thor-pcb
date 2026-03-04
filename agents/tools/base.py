"""Tool calling data types for the agentic camera monitoring agent.

Lightweight DTOs used by the VLM client to parse tool calls from
model responses and wrap execution results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ToolCall:
    """Represents a tool call request parsed from a VLM response."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str = ""

    def __post_init__(self):
        if not self.call_id:
            self.call_id = f"call_{datetime.now().strftime('%H%M%S%f')}"


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_name: str
    call_id: str
    success: bool
    result: Any
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
        }
