"""Core agent components - camera agent, state management, and alerting."""

from .camera_agent import StreamlinedAgent, CircuitBreaker
from .state import AgentMemory, AgentSnapshot, DetectionEvent

# Backward-compat alias
AgentState = AgentSnapshot
from .alerting import AlertManager

__all__ = [
    "StreamlinedAgent",
    "CircuitBreaker",
    "AgentMemory",
    "AgentSnapshot",
    "AgentState",  # backward-compat alias
    "DetectionEvent",
    "AlertManager",
]
