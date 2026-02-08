"""Agent state machine with validated transitions and audit logging."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)


class AgentState(str, Enum):
    """First-class agent operational states."""
    OFF = "off"
    IDLE = "idle"
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    ALERTING = "alerting"
    ERROR = "error"


# Valid state transitions
VALID_STATE_TRANSITIONS: Dict[AgentState, tuple] = {
    AgentState.OFF: (AgentState.IDLE,),
    AgentState.IDLE: (AgentState.OFF, AgentState.MONITORING, AgentState.ANALYZING),
    AgentState.MONITORING: (AgentState.OFF, AgentState.IDLE, AgentState.ANALYZING, AgentState.ALERTING, AgentState.ERROR),
    AgentState.ANALYZING: (AgentState.IDLE, AgentState.MONITORING, AgentState.ALERTING, AgentState.ERROR),
    AgentState.ALERTING: (AgentState.MONITORING, AgentState.IDLE, AgentState.ERROR),
    AgentState.ERROR: (AgentState.OFF, AgentState.IDLE),
}


@dataclass
class AgentStateTransition:
    """Records a state transition for audit purposes."""
    id: str
    from_state: AgentState
    to_state: AgentState
    trigger: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "trigger": self.trigger,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }


class AgentStateMachine:
    """Manages agent state with validated transitions and audit logging."""

    def __init__(self, initial_state: AgentState = AgentState.OFF):
        self._state = initial_state
        self._lock = threading.RLock()
        self._transitions: List[AgentStateTransition] = []
        self._state_entered_at = datetime.now().isoformat()
        self._listeners: List[Callable[[AgentState, AgentState, str], None]] = []
        self._current_session_id: Optional[str] = None

    @property
    def state(self) -> AgentState:
        """Get current state (thread-safe)."""
        with self._lock:
            return self._state

    @property
    def state_duration_seconds(self) -> float:
        """Get how long we've been in the current state."""
        entered = datetime.fromisoformat(self._state_entered_at)
        return (datetime.now() - entered).total_seconds()

    @property
    def state_history(self) -> List[Dict[str, Any]]:
        """Get state transition history."""
        with self._lock:
            return [t.to_dict() for t in self._transitions]

    def can_transition_to(self, target: AgentState) -> bool:
        """Check if transition to target state is valid."""
        with self._lock:
            valid_targets = VALID_STATE_TRANSITIONS.get(self._state, ())
            return target in valid_targets

    def transition_to(
        self,
        target: AgentState,
        trigger: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentStateTransition:
        """Attempt to transition to a new state."""
        with self._lock:
            if not self.can_transition_to(target):
                valid = [s.value for s in VALID_STATE_TRANSITIONS.get(self._state, ())]
                raise ValueError(
                    f"Invalid state transition: {self._state.value} -> {target.value}. "
                    f"Valid transitions from {self._state.value}: {valid}"
                )

            transition = AgentStateTransition(
                id=f"trans_{uuid.uuid4().hex[:12]}",
                from_state=self._state,
                to_state=target,
                trigger=trigger,
                session_id=self._current_session_id,
                metadata=metadata or {},
            )
            self._transitions.append(transition)

            logger.info(
                "Agent state transition: %s -> %s (trigger: %s)",
                self._state.value,
                target.value,
                trigger,
            )

            old_state = self._state
            self._state = target
            self._state_entered_at = datetime.now().isoformat()

            for listener in self._listeners:
                try:
                    listener(old_state, target, trigger)
                except Exception as e:
                    logger.error("State listener error: %s", e)

            return transition

    def add_listener(self, callback: Callable[[AgentState, AgentState, str], None]) -> None:
        """Add a state change listener."""
        self._listeners.append(callback)

    def set_session(self, session_id: Optional[str]) -> None:
        """Set the current session ID for transition logging."""
        self._current_session_id = session_id

    def get_transitions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent state transitions."""
        with self._lock:
            recent = self._transitions[-limit:] if len(self._transitions) > limit else self._transitions
            return [t.to_dict() for t in reversed(recent)]

    def get_status(self) -> Dict[str, Any]:
        """Get current state machine status."""
        with self._lock:
            return {
                "state": self._state.value,
                "state_entered_at": self._state_entered_at,
                "state_duration_seconds": self.state_duration_seconds,
                "session_id": self._current_session_id,
                "total_transitions": len(self._transitions),
            }
