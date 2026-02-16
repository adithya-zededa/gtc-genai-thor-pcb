"""Core agent components - detection, state management, and alerting."""

from .detection_agent import StreamlinedAgent
from .resilience import CircuitBreaker
from .state import AgentMemory, AgentSnapshot, DetectionEvent
from .alerting import AlertManager

# Backward-compat alias
AgentState = AgentSnapshot

__all__ = [
    "StreamlinedAgent",
    "CircuitBreaker",
    "AgentMemory",
    "AgentSnapshot",
    "AgentState",  # backward-compat alias
    "DetectionEvent",
    "AlertManager",
]
