"""Centralized audit logging for all MCP decisions."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    INTENT_DETECTED = "intent_detected"
    TOOL_PROPOSED = "tool_proposed"
    TOOL_PENDING_APPROVAL = "tool_pending_approval"
    TOOL_APPROVED = "tool_approved"
    TOOL_REJECTED = "tool_rejected"
    TOOL_EXECUTING = "tool_executing"
    TOOL_SUCCEEDED = "tool_succeeded"
    TOOL_FAILED = "tool_failed"
    STATE_TRANSITION = "state_transition"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    USER_MESSAGE = "user_message"
    AGENT_RESPONSE = "agent_response"
    VALIDATION_ERROR = "validation_error"
    DETECTION_EVENT = "detection_event"


@dataclass
class AuditLogEntry:
    """An audit log entry for MCP decisions."""
    id: str
    event_type: AuditEventType
    timestamp: str
    session_id: Optional[str]
    details: Dict[str, Any]
    latency_ms: Optional[float] = None

    @classmethod
    def create(
        cls,
        event_type: AuditEventType,
        details: Dict[str, Any],
        session_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> "AuditLogEntry":
        return cls(
            id=f"audit_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            details=details,
            latency_ms=latency_ms,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "details": self.details,
            "latency_ms": self.latency_ms,
        }


class MCPAuditLog:
    """Centralized audit log for all MCP decisions."""

    def __init__(self, max_entries: int = 10000):
        self._entries: List[AuditLogEntry] = []
        self._lock = threading.Lock()
        self._max_entries = max_entries

        self._metrics = {
            "total_proposals": 0,
            "total_approvals": 0,
            "total_rejections": 0,
            "total_executions": 0,
            "total_failures": 0,
            "total_latency_ms": 0.0,
            "latency_samples": 0,
        }

    def log(self, entry: AuditLogEntry) -> None:
        """Add an entry to the audit log."""
        with self._lock:
            self._entries.append(entry)

            if entry.event_type == AuditEventType.TOOL_PROPOSED:
                self._metrics["total_proposals"] += 1
            elif entry.event_type == AuditEventType.TOOL_APPROVED:
                self._metrics["total_approvals"] += 1
            elif entry.event_type == AuditEventType.TOOL_REJECTED:
                self._metrics["total_rejections"] += 1
            elif entry.event_type == AuditEventType.TOOL_SUCCEEDED:
                self._metrics["total_executions"] += 1
            elif entry.event_type == AuditEventType.TOOL_FAILED:
                self._metrics["total_failures"] += 1

            if entry.latency_ms is not None:
                self._metrics["total_latency_ms"] += entry.latency_ms
                self._metrics["latency_samples"] += 1

            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]

        logger.info(
            "AUDIT [%s] %s: %s",
            entry.event_type.value,
            entry.session_id or "no-session",
            json.dumps(entry.details, default=str)[:500],
        )

    def get_entries(
        self,
        limit: int = 100,
        event_type: Optional[AuditEventType] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit entries with optional filtering."""
        with self._lock:
            entries = self._entries

            if event_type:
                entries = [e for e in entries if e.event_type == event_type]

            if session_id:
                entries = [e for e in entries if e.session_id == session_id]

            recent = entries[-limit:] if len(entries) > limit else entries
            return [e.to_dict() for e in reversed(recent)]

    def get_metrics(self) -> Dict[str, Any]:
        """Get audit metrics."""
        with self._lock:
            metrics = dict(self._metrics)

            if metrics["total_proposals"] > 0:
                metrics["approval_rate"] = metrics["total_approvals"] / metrics["total_proposals"]
                metrics["rejection_rate"] = metrics["total_rejections"] / metrics["total_proposals"]
                metrics["execution_rate"] = metrics["total_executions"] / metrics["total_proposals"]
            else:
                metrics["approval_rate"] = 0.0
                metrics["rejection_rate"] = 0.0
                metrics["execution_rate"] = 0.0

            if metrics["latency_samples"] > 0:
                metrics["avg_latency_ms"] = metrics["total_latency_ms"] / metrics["latency_samples"]
            else:
                metrics["avg_latency_ms"] = 0.0

            metrics["total_entries"] = len(self._entries)

            return metrics
