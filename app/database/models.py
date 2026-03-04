"""Database model dataclasses representing table structures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class User:
    """User model representing a system user."""

    id: Optional[int] = None
    email: str = ""
    name: str = ""
    role: str = "user"
    active: bool = True
    created_at: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> "User":
        """Create User from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            email=row["email"],
            name=row["name"],
            role=row["role"],
            active=bool(row["active"]),
            created_at=row["created_at"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "active": self.active,
            "created_at": self.created_at,
        }


@dataclass
class DetectionLog:  # pylint: disable=too-many-instance-attributes
    """Detection log model representing a detection event."""

    id: Optional[int] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None
    response: str = ""
    image_path: str = ""
    frame_number: Optional[int] = None
    reason: str = ""
    vision_description: str = ""
    decision_details: Dict[str, Any] = field(default_factory=dict)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_row(cls, row) -> "DetectionLog":
        """Create DetectionLog from database row."""
        if row is None:
            return None

        decision_details = {}
        if row["decision_details"]:
            try:
                decision_details = json.loads(row["decision_details"])
            except json.JSONDecodeError:
                decision_details = {"raw": row["decision_details"]}

        tool_trace = []
        try:
            tool_trace_raw = row["tool_trace"] if "tool_trace" in row.keys() else None
            if tool_trace_raw:
                tool_trace = json.loads(tool_trace_raw)
        except (json.JSONDecodeError, KeyError):
            tool_trace = []

        return cls(
            id=row["id"],
            timestamp=row["timestamp"],
            confidence=row["confidence"],
            response=row["response"],
            image_path=row["image_path"],
            frame_number=row["frame_number"],
            reason=row["reason"],
            vision_description=row["vision_description"],
            decision_details=decision_details,
            tool_trace=tool_trace,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        is_agentic = self.decision_details.get("classification") == "AGENTIC_ANALYSIS"

        # Determine detected status
        if "detected" in self.decision_details:
            detected = self.decision_details.get("detected", False)
        else:
            detected = self.confidence is not None and self.confidence > 0

        # Extract token usage and tools_used from decision_details
        token_usage = self.decision_details.get("token_usage", {})
        tools_used = self.decision_details.get("tools_used", [])

        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "response": self.response,
            "image_path": self.image_path,
            "frame_number": self.frame_number,
            "reason": self.reason,
            "vision_description": self.vision_description,
            "decision_details": self.decision_details,
            "tool_trace": self.tool_trace,
            "agentic_mode": is_agentic,
            "detected": detected,
            "token_usage": token_usage,
            "tools_used": tools_used,
        }


@dataclass
class ConfigHistory:
    """Configuration history model for tracking config changes."""

    id: Optional[int] = None
    timestamp: Optional[str] = None
    config_type: str = ""
    changes: str = ""
    user_email: str = ""

    @classmethod
    def from_row(cls, row) -> "ConfigHistory":
        """Create ConfigHistory from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            timestamp=row["timestamp"],
            config_type=row["config_type"],
            changes=row["changes"],
            user_email=row["user_email"],
        )


@dataclass
class LogSettings:
    """Log settings model for logging configuration."""

    log_level: str = "INFO"
    log_retention: int = 30
    max_log_size: int = 100
    log_to_file: bool = True
    log_to_console: bool = True
    log_database: bool = False

    @classmethod
    def from_row(cls, row) -> "LogSettings":
        """Create LogSettings from database row."""
        if row is None:
            return cls()  # Return defaults
        return cls(
            log_level=(row["log_level"] or "INFO").upper(),
            log_retention=(
                int(row["log_retention"]) if row["log_retention"] is not None else 30
            ),
            max_log_size=(
                int(row["max_log_size"]) if row["max_log_size"] is not None else 100
            ),
            log_to_file=bool(row["log_to_file"]),
            log_to_console=bool(row["log_to_console"]),
            log_database=bool(row["log_database"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "log_level": self.log_level,
            "log_retention": self.log_retention,
            "max_log_size": self.max_log_size,
            "log_to_file": self.log_to_file,
            "log_to_console": self.log_to_console,
            "log_database": self.log_database,
        }


@dataclass
class PCBDefect:  # pylint: disable=too-many-instance-attributes
    """PCB defect record from inspection analysis."""

    id: Optional[int] = None
    timestamp: Optional[str] = None
    board_type: str = "unknown"
    defect_type: str = ""
    severity: str = "low"
    confidence: float = 0.0
    image_path: str = ""
    description: str = ""
    created_at: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> Optional["PCBDefect"]:
        """Create PCBDefect from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            timestamp=row["timestamp"],
            board_type=row["board_type"],
            defect_type=row["defect_type"],
            severity=row["severity"],
            confidence=float(row["confidence"]),
            image_path=row["image_path"],
            description=row["description"],
            created_at=row["created_at"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "board_type": self.board_type,
            "defect_type": self.defect_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "image_path": self.image_path,
            "description": self.description,
            "created_at": self.created_at,
        }


@dataclass
class PCBFrameStore:
    """A stored camera frame captured when a PCB was detected with low motion."""

    id: Optional[int] = None
    timestamp: Optional[str] = None
    image_path: str = ""
    motion_score: float = 0.0
    board_signature: str = ""
    frame_number: Optional[int] = None
    quality_score: float = 0.0
    consumed: bool = False
    created_at: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> Optional["PCBFrameStore"]:
        """Create PCBFrameStore from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            timestamp=row["timestamp"],
            image_path=row["image_path"],
            motion_score=float(row["motion_score"]),
            board_signature=row["board_signature"],
            frame_number=row["frame_number"],
            quality_score=float(row["quality_score"]),
            consumed=bool(row["consumed"]),
            created_at=row["created_at"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "motion_score": self.motion_score,
            "board_signature": self.board_signature,
            "frame_number": self.frame_number,
            "quality_score": self.quality_score,
            "consumed": self.consumed,
            "created_at": self.created_at,
        }


@dataclass
class PCBInspection:  # pylint: disable=too-many-instance-attributes
    """PCB inspection outcome record with explicit PASS/FAIL result."""

    id: Optional[int] = None
    timestamp: Optional[str] = None
    board_signature: str = ""
    result: str = "PASS"
    confidence: float = 0.0
    reason: str = ""
    defect_type: str = ""
    description: str = ""
    image_path: str = ""
    decision_trace: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> Optional["PCBInspection"]:
        """Create PCBInspection from database row."""
        if row is None:
            return None

        decision_trace = {}
        if row["decision_trace"]:
            try:
                decision_trace = json.loads(row["decision_trace"])
            except json.JSONDecodeError:
                decision_trace = {"raw": row["decision_trace"]}

        return cls(
            id=row["id"],
            timestamp=row["timestamp"],
            board_signature=row["board_signature"],
            result=row["result"],
            confidence=float(row["confidence"]),
            reason=row["reason"],
            defect_type=row["defect_type"],
            description=row["description"],
            image_path=row["image_path"],
            decision_trace=decision_trace,
            created_at=row["created_at"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "board_signature": self.board_signature,
            "result": self.result,
            "confidence": self.confidence,
            "reason": self.reason,
            "defect_type": self.defect_type,
            "description": self.description,
            "image_path": self.image_path,
            "decision_trace": self.decision_trace,
            "created_at": self.created_at,
        }
