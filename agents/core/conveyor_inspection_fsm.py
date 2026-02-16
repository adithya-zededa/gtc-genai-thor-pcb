"""Deterministic conveyor inspection FSM with hard runtime guarantees."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

from agents.core.state import DetectionEvent


class ConveyorState(str, Enum):
    """Hard conveyor runtime states."""

    MOVING = "MOVING"
    STOPPED_PENDING_INSPECTION = "STOPPED_PENDING_INSPECTION"
    INSPECTING = "INSPECTING"
    INSPECTED_COMPLETE = "INSPECTED_COMPLETE"


class ConveyorDecision(str, Enum):
    """Final decisions emitted by FSM."""

    DEFECT_FOUND = "DEFECT_FOUND"
    NO_DEFECT = "NO_DEFECT"


@dataclass(frozen=True)
class ConveyorDecisionPayload:
    """Decision data emitted by the FSM."""

    decision: ConveyorDecision
    board_signature: str
    reason: str
    event: Optional[DetectionEvent]


class MonotonicInspectionTimer:
    """Inspection deadline timer backed by monotonic clock and timer thread."""

    def __init__(
        self,
        duration_seconds: float,
        on_expire: Callable[[], None],
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._duration_seconds = float(duration_seconds)
        self._on_expire = on_expire
        self._monotonic = monotonic_fn
        self._lock = threading.RLock()
        self._deadline: Optional[float] = None
        self._timer: Optional[threading.Timer] = None
        self._generation = 0

    def start(self) -> None:
        """Start the inspection timer."""
        with self._lock:
            self.cancel()
            self._generation += 1
            generation = self._generation
            self._deadline = self._monotonic() + self._duration_seconds
            self._timer = threading.Timer(self._duration_seconds, self._expire, args=(generation,))
            self._timer.daemon = True
            self._timer.start()

    def cancel(self) -> None:
        """Cancel the inspection timer."""
        with self._lock:
            timer = self._timer
            self._timer = None
            self._deadline = None
            if timer:
                timer.cancel()

    def remaining_seconds(self) -> Optional[float]:
        """Get remaining seconds until deadline."""
        with self._lock:
            if self._deadline is None:
                return None
            return max(0.0, self._deadline - self._monotonic())

    def _expire(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation or self._deadline is None:
                return
            self._timer = None
            self._deadline = None
        self._on_expire()


class ConveyorInspectionFSM:  # pylint: disable=too-many-instance-attributes
    """Conveyor-aware FSM that enforces runtime gating, timing, and emission."""

    def __init__(
        self,
        *,
        inspection_window_seconds: float,
        on_decision: Callable[[ConveyorDecisionPayload], None],
    ) -> None:
        self._lock = threading.RLock()
        self._on_decision = on_decision
        self.state = ConveyorState.MOVING
        self.inspected_boards: Dict[str, Dict[str, Any]] = {}
        self.active_board_signature: Optional[str] = None
        self._decision_emitted = False
        self._last_motion_moving = True
        self._last_board_in_zone = False
        self._timer = MonotonicInspectionTimer(
            duration_seconds=inspection_window_seconds,
            on_expire=self._on_deadline_expired,
        )

    def update_inputs(
        self,
        *,
        motion_moving: bool,
        board_in_zone: bool,
        board_signature: Optional[str],
    ) -> None:
        """Update FSM from deterministic sensor/observation inputs."""
        with self._lock:
            if motion_moving or not board_in_zone:
                self._transition_to_moving_locked("motion_resumed_or_board_exited")
                self._last_motion_moving = motion_moving
                self._last_board_in_zone = board_in_zone
                return

            moving_to_stopped = self._last_motion_moving and not motion_moving
            if self.state == ConveyorState.MOVING and moving_to_stopped and board_in_zone:
                if board_signature and board_signature in self.inspected_boards:
                    self.state = ConveyorState.INSPECTED_COMPLETE
                    self.active_board_signature = board_signature
                    self._decision_emitted = True
                else:
                    self.state = ConveyorState.STOPPED_PENDING_INSPECTION
                    self.active_board_signature = board_signature
                    self._decision_emitted = False
                    self._timer.start()

            self._last_motion_moving = motion_moving
            self._last_board_in_zone = board_in_zone

    def begin_inspection_on_first_frame(self) -> bool:
        """Transition pending stop to active inspection on first processed frame."""
        with self._lock:
            if self.state != ConveyorState.STOPPED_PENDING_INSPECTION:
                return False
            self.state = ConveyorState.INSPECTING
            return True

    def submit_analysis(self, event: Optional[DetectionEvent]) -> None:
        """Accept analysis proposal while INSPECTING; FSM emits final event."""
        payload: Optional[ConveyorDecisionPayload] = None
        with self._lock:
            if self.state != ConveyorState.INSPECTING or self._decision_emitted:
                return
            if event and bool(event.detected):
                payload = self._complete_locked(
                    decision=ConveyorDecision.DEFECT_FOUND,
                    event=event,
                    reason="defect_detected",
                )
        if payload:
            self._on_decision(payload)

    def snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current FSM state."""
        with self._lock:
            return {
                "state": self.state.value,
                "active_board_signature": self.active_board_signature,
                "decision_emitted": self._decision_emitted,
                "inspection_window_remaining_seconds": self._timer.remaining_seconds(),
                "inspected_boards_count": len(self.inspected_boards),
            }

    def _on_deadline_expired(self) -> None:
        payload: Optional[ConveyorDecisionPayload] = None
        with self._lock:
            if self._decision_emitted:
                return
            if self.state not in {
                ConveyorState.STOPPED_PENDING_INSPECTION,
                ConveyorState.INSPECTING,
            }:
                return
            if self.state == ConveyorState.STOPPED_PENDING_INSPECTION:
                self.state = ConveyorState.INSPECTING
            payload = self._complete_locked(
                decision=ConveyorDecision.NO_DEFECT,
                event=None,
                reason="inspection_deadline_expired",
            )
        if payload:
            self._on_decision(payload)

    def _transition_to_moving_locked(
        self, reason: str  # pylint: disable=unused-argument
    ) -> None:
        """Transition the FSM to MOVING state."""
        if self.state == ConveyorState.MOVING:
            return
        self._timer.cancel()
        self.state = ConveyorState.MOVING
        self.active_board_signature = None
        self._decision_emitted = False

    def _complete_locked(
        self,
        *,
        decision: ConveyorDecision,
        event: Optional[DetectionEvent],
        reason: str,  # pylint: disable=unused-argument
    ) -> Optional[ConveyorDecisionPayload]:
        """Complete the inspection and emit a decision."""
        signature = self.active_board_signature
        if not signature:
            return None
        if self._decision_emitted:
            return None

        self._decision_emitted = True
        self._timer.cancel()
        self.inspected_boards[signature] = {
            "decision": decision.value,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        self.state = ConveyorState.INSPECTED_COMPLETE

        if not self._decision_emitted:
            raise AssertionError("FSM completion requires emitted decision")

        return ConveyorDecisionPayload(
            decision=decision,
            board_signature=signature,
            reason=reason,
            event=event,
        )
