"""State management helpers for the camera monitoring agent."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional
from collections import deque


@dataclass(slots=True)
class DetectionEvent:
    """Represents a packaging box detection event."""

    timestamp: str
    confidence: float
    primary_label: str
    full_response: str
    image_path: Optional[str] = None
    vision_description: str = ""
    decision_trace: Dict[str, Any] = field(default_factory=dict)
    detected: bool = False
    shipping_label_present: Optional[bool] = None
    should_alert: bool = True
    tools_used: List[str] = field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class AgentState:
    """Snapshot of high-level runtime state for exposure to UIs."""

    last_event: Optional[Dict[str, Any]] = None
    counts: Dict[str, int] = field(default_factory=dict)
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AgentMemory:
    """Thread-safe ring buffer with aggregated counters and summaries.
    
    Stores detection events in a fixed-size circular buffer, providing
    thread-safe access and summary statistics.
    
    Attributes:
        _max_events: Maximum number of events to store.
        _summary_window: Number of recent events to consider for summaries.
    """

    def __init__(self, max_events: int = 50, summary_window: int = 10) -> None:
        """Initialize the agent memory.
        
        Args:
            max_events: Maximum number of events to store (min 1).
            summary_window: Number of events to use for summaries (min 1, max max_events).
        """
        self._max_events = max(1, int(max_events or 1))
        self._summary_window = max(1, min(self._max_events, int(summary_window or 1)))
        self._events: Deque[Dict[str, Any]] = deque(maxlen=self._max_events)
        self._lock = threading.RLock()

    def resize(self, max_events: int, summary_window: int) -> None:
        """Resize the memory buffer.
        
        Args:
            max_events: New maximum number of events (min 1).
            summary_window: New summary window size (min 1, max max_events).
        """
        new_max = max(1, int(max_events) if max_events else 1)
        new_window = max(1, min(new_max, int(summary_window) if summary_window else 1))
        with self._lock:
            preserved = list(self._events)[-new_max:]
            self._events = deque(preserved, maxlen=new_max)
            self._max_events = new_max
            self._summary_window = new_window

    def append(self, event: Dict[str, Any]) -> None:
        sanitized = {key: event.get(key) for key in event.keys()}
        with self._lock:
            self._events.append(sanitized)

    # Alias for backward compatibility
    add_event = append

    def list_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            events = list(self._events)
        if limit is not None:
            return events[-limit:]
        return events

    def snapshot(self, limit: Optional[int] = None) -> AgentState:
        actual_limit = limit if isinstance(limit, int) and limit > 0 else self._summary_window
        events = self.list_events(actual_limit)
        counts = self._calc_counts(events)
        last_event = events[-1] if events else None
        return AgentState(last_event=last_event, counts=counts)

    def summarise(self, limit: Optional[int] = None) -> str:
        snapshot = self.snapshot(limit)
        counts = snapshot.counts
        events = self.list_events(limit if limit is not None else self._summary_window)
        if not events:
            return "No recent events processed yet."

        total = counts["total"]
        detections = counts["detections"]
        unlabeled = counts["unlabeled"]
        labeled = counts["labeled"]
        alerts = counts["alerts"]
        reused = counts["reused"]

        summary_parts: List[str] = []
        summary_parts.append(f"Processed {total} event{'s' if total != 1 else ''}.")

        detection_clause = f"Detections: {detections}"
        breakdown_bits: List[str] = []
        if unlabeled:
            breakdown_bits.append(f"{unlabeled} unlabeled")
        if labeled:
            breakdown_bits.append(f"{labeled} labeled")
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
        detections = sum(1 for event in events if event.get("detected"))
        unlabeled = sum(1 for event in events if event.get("detected") and event.get("shipping_label_present") is False)
        labeled = sum(1 for event in events if event.get("detected") and event.get("shipping_label_present") is True)
        alerts = sum(1 for event in events if event.get("should_alert"))
        reused = sum(1 for event in events if event.get("source") == "agent_ssim_guard")
        no_detections = sum(1 for event in events if not event.get("detected"))
        return {
            "total": len(events),
            "detections": detections,
            "unlabeled": unlabeled,
            "labeled": labeled,
            "alerts": alerts,
            "reused": reused,
            "no_detections": no_detections,
        }

    @staticmethod
    def _describe_event(event: Dict[str, Any]) -> str:
        if not event:
            return ""

        if not event.get("detected"):
            return "no packaging boxes required action"

        shipping_label_present = event.get("shipping_label_present")
        if shipping_label_present is True:
            descriptor = "packaging box with visible label"
        elif shipping_label_present is False:
            descriptor = "unlabeled packaging box"
        else:
            descriptor = "packaging box with unknown label status"

        if event.get("should_alert"):
            descriptor += " (alert raised)"

        return descriptor
