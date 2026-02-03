"""Agent modules for AI-powered analysis."""

from .camera_agent import StreamlinedAgent, CircuitBreaker
from .state import AgentMemory, AgentState, DetectionEvent
from .alerting import AlertManager
from .tools import (
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolExecutor,
    TOOL_REGISTRY,
)

__all__ = [
    # Camera agent
    "StreamlinedAgent",
    "CircuitBreaker",
    # State management
    "AgentMemory",
    "AgentState",
    "DetectionEvent",
    # Alerting
    "AlertManager",
    # Tools
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ToolExecutor",
    "TOOL_REGISTRY",
]
