"""Core agent components — detection, state, and the monitoring loop.

Separation of concerns
~~~~~~~~~~~~~~~~~~~~~~
- **state.py**            – Data models (DetectionEvent, AgentMemory, …)
- **monitoring_loop.py**  – The ONLY deterministic piece: observe → detect
                            state change → delegate to the LLM.
- **detection_agent.py**  – VLM analysis coordinator.

Everything else (inspection decisions, alerting, classification) is
decided by the LLM via MCP tool calls.
"""

from .detection_agent import StreamlinedAgent
from .monitoring_loop import MonitoringLoop
from .state import AgentMemory, AgentSnapshot, DetectionEvent
from core.resilience import CircuitBreaker

# Backward-compat aliases
AgentState = AgentSnapshot
ProactiveMonitoringAgent = MonitoringLoop  # old name

__all__ = [
    "StreamlinedAgent",
    "MonitoringLoop",
    "ProactiveMonitoringAgent",  # backward-compat alias
    "CircuitBreaker",
    "AgentMemory",
    "AgentSnapshot",
    "AgentState",
    "DetectionEvent",
]
