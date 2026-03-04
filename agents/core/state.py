"""State management for the camera monitoring agent."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import deque


@dataclass(slots=True)
class DetectionEvent:
    """Represents a PCB inspection event."""  # pylint: disable=too-many-instance-attributes

    timestamp: str
    confidence: float
    primary_label: str
    full_response: str
    image_path: Optional[str] = None
    vision_description: str = ""
    decision_trace: Dict[str, Any] = field(default_factory=dict)
    detected: bool = False
    pcb_stable: Optional[bool] = None
    should_alert: bool = True
    tools_used: List[str] = field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "primary_label": self.primary_label,
            "full_response": self.full_response,
            "image_path": self.image_path,
            "vision_description": self.vision_description,
            "decision_trace": self.decision_trace,
            "detected": self.detected,
            "pcb_stable": self.pcb_stable,
            "should_alert": self.should_alert,
            "tools_used": self.tools_used,
            "tool_trace": self.tool_trace,
            "token_usage": self.token_usage,
        }


@dataclass(slots=True)
class AgentSnapshot:
    """Snapshot of high-level runtime state for exposure to UIs.

    Renamed from ``AgentState`` to avoid collision with the
    operational ``AgentState`` enum in ``agents.mcp.state_machine``.
    """

    last_event: Optional[Dict[str, Any]] = None
    counts: Dict[str, int] = field(default_factory=dict)
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AgentMemory:
    """Thread-safe ring buffer with aggregated counters and summaries."""

    def __init__(self, max_events: int = 50, summary_window: int = 10) -> None:
        """Initialize the agent memory."""
        self._max_events = max(1, int(max_events or 1))
        self._summary_window = max(1, min(self._max_events, int(summary_window or 1)))
        self._events: deque[Dict[str, Any]] = deque(maxlen=self._max_events)
        self._lock = threading.RLock()

    def resize(self, max_events: int, summary_window: int) -> None:
        """Resize the memory buffer."""
        new_max = max(1, int(max_events) if max_events else 1)
        new_window = max(1, min(new_max, int(summary_window) if summary_window else 1))
        with self._lock:
            preserved = list(self._events)[-new_max:]
            self._events = deque(preserved, maxlen=new_max)
            self._max_events = new_max
            self._summary_window = new_window

    def append(self, event: Dict[str, Any]) -> None:
        """Add an event to memory (stores a shallow copy)."""
        with self._lock:
            self._events.append(dict(event))

    add_event = append  # Alias for backward compatibility

    def list_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List events from memory."""
        with self._lock:
            events = list(self._events)
        if limit is not None:
            return events[-limit:]
        return events

    def snapshot(self, limit: Optional[int] = None) -> AgentSnapshot:
        """Get a state snapshot."""  # pylint: disable=too-many-locals
        actual_limit = (
            limit if isinstance(limit, int) and limit > 0
            else self._summary_window
        )
        events = self.list_events(actual_limit)
        counts = self._calc_counts(events)
        last_event = events[-1] if events else None
        return AgentSnapshot(last_event=last_event, counts=counts)

    def summarise(  # pylint: disable=too-many-locals
        self, limit: Optional[int] = None
    ) -> str:
        """Get a text summary of recent activity."""
        snapshot = self.snapshot(limit)
        counts = snapshot.counts
        events = self.list_events(limit if limit is not None else self._summary_window)
        if not events:
            return "No recent events processed yet."

        total = counts["total"]
        detections = counts["detections"]
        stable = counts["stable"]
        unstable = counts["unstable"]
        alerts = counts["alerts"]
        reused = counts["reused"]

        summary_parts: List[str] = []
        summary_parts.append(f"Processed {total} event{'s' if total != 1 else ''}.")

        detection_clause = f"Detections: {detections}"
        breakdown_bits: List[str] = []
        if stable:
            breakdown_bits.append(f"{stable} stable")
        if unstable:
            breakdown_bits.append(f"{unstable} unstable")
        if breakdown_bits:
            detection_clause += f" ({', '.join(breakdown_bits)})"
        summary_parts.append(detection_clause + ".")

        summary_parts.append(f"Alerts raised: {alerts}.")
        if reused:
            summary_parts.append(f"Reused decisions: {reused}.")

        last_event = snapshot.last_event or {}
        timestamp = last_event.get("timestamp")
        descriptor = self._describe_event(last_event)
        if timestamp and descriptor:
            summary_parts.append(f"Last event at {timestamp}: {descriptor}.")

        return " ".join(part for part in summary_parts if part)

    @staticmethod
    def _calc_counts(events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate event counts."""
        detections = sum(1 for event in events if event.get("detected"))
        stable = sum(
            1 for event in events
            if event.get("detected") and event.get("pcb_stable") is True
        )
        unstable = sum(
            1 for event in events
            if event.get("detected") and event.get("pcb_stable") is False
        )
        alerts = sum(1 for event in events if event.get("should_alert"))
        reused = sum(1 for event in events if event.get("source") == "agent_ssim_guard")
        no_detections = sum(1 for event in events if not event.get("detected"))
        return {
            "total": len(events),
            "detections": detections,
            "stable": stable,
            "unstable": unstable,
            "alerts": alerts,
            "reused": reused,
            "no_detections": no_detections,
        }

    @staticmethod
    def _describe_event(event: Dict[str, Any]) -> str:
        """Generate a human-readable event description."""
        if not event:
            return ""

        if not event.get("detected"):
            return "no PCB requiring inspection"

        pcb_stable = event.get("pcb_stable")
        if pcb_stable is True:
            descriptor = "PCB present and stable under camera"
        elif pcb_stable is False:
            descriptor = "PCB present but not yet stable"
        else:
            descriptor = "PCB detected with unknown stability"

        if event.get("should_alert"):
            descriptor += " (alert raised)"

        return descriptor
