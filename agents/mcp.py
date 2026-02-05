"""Model Context Protocol (MCP) implementation for explicit tool calling.

This module provides a unified MCP implementation with:
- JSON Schema validation for all inputs/outputs
- Explicit lifecycle states for tool execution
- Separation between interpretation and execution phases
- First-class agent state machine
- Session management
- Comprehensive audit logging

Core Principles:
- Language is the ONLY control surface
- Tools are PROPOSED, then APPROVED, then EXECUTED
- No automated/heuristic tool invocation
- All actions require explicit approval or policy-based consent
- Every decision is logged and auditable
"""

from __future__ import annotations

import json
import time
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# JSON SCHEMA DEFINITIONS
# =============================================================================

class MCPSchemaType(str, Enum):
    """Supported JSON Schema types."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


@dataclass(frozen=True)
class MCPParameterSchema:
    """JSON Schema definition for a tool parameter."""
    name: str
    type: MCPSchemaType
    description: str
    required: bool = False
    enum: Optional[tuple] = None
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    items_type: Optional[MCPSchemaType] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = list(self.enum)
        if self.default is not None:
            schema["default"] = self.default
        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.items_type and self.type == MCPSchemaType.ARRAY:
            schema["items"] = {"type": self.items_type.value}
        return schema


@dataclass(frozen=True)
class MCPOutputSchema:
    """JSON Schema definition for tool output."""
    type: MCPSchemaType
    description: str
    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required_properties: tuple = field(default_factory=tuple)

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.properties:
            schema["properties"] = self.properties
        if self.required_properties:
            schema["required"] = list(self.required_properties)
        return schema


# =============================================================================
# TOOL EXECUTION LIFECYCLE
# =============================================================================

class ToolLifecycleState(str, Enum):
    """Explicit execution lifecycle states for tool calls."""
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MCPToolCallProposal:
    """A proposed tool call awaiting approval.
    
    This represents the INTERPRETATION phase output.
    The tool is NOT executed until explicitly approved.
    """
    id: str
    tool_name: str
    arguments: Dict[str, Any]
    rationale: str
    confidence: float
    requires_confirmation: bool
    state: ToolLifecycleState = ToolLifecycleState.PROPOSED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: Optional[str] = None
    
    # Lifecycle tracking
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None
    rejected_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    executed_at: Optional[str] = None
    completed_at: Optional[str] = None

    @classmethod
    def create(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        rationale: str,
        confidence: float,
        requires_confirmation: bool,
        session_id: Optional[str] = None,
    ) -> "MCPToolCallProposal":
        return cls(
            id=f"proposal_{uuid.uuid4().hex[:12]}",
            tool_name=tool_name,
            arguments=arguments,
            rationale=rationale,
            confidence=confidence,
            requires_confirmation=requires_confirmation,
            session_id=session_id,
        )

    @property
    def is_rejected(self) -> bool:
        return self.state == ToolLifecycleState.REJECTED

    def approve(self, approved_by: str = "user") -> None:
        """Mark proposal as approved."""
        if self.state not in (ToolLifecycleState.PROPOSED, ToolLifecycleState.PENDING_APPROVAL):
            raise ValueError(f"Cannot approve proposal in state: {self.state}")
        self.state = ToolLifecycleState.APPROVED
        self.approved_at = datetime.now().isoformat()
        self.approved_by = approved_by

    def reject(self, reason: str = "") -> None:
        """Mark proposal as rejected."""
        if self.state not in (ToolLifecycleState.PROPOSED, ToolLifecycleState.PENDING_APPROVAL):
            raise ValueError(f"Cannot reject proposal in state: {self.state}")
        self.state = ToolLifecycleState.REJECTED
        self.rejected_at = datetime.now().isoformat()
        self.rejection_reason = reason

    def start_execution(self) -> None:
        """Mark proposal as executing."""
        if self.state != ToolLifecycleState.APPROVED:
            raise ValueError(f"Cannot execute proposal in state: {self.state}")
        self.state = ToolLifecycleState.EXECUTING
        self.executed_at = datetime.now().isoformat()

    def complete(self, success: bool, error: Optional[str] = None) -> None:
        """Mark execution as complete."""
        if self.state != ToolLifecycleState.EXECUTING:
            raise ValueError(f"Cannot complete proposal in state: {self.state}")
        self.state = ToolLifecycleState.SUCCEEDED if success else ToolLifecycleState.FAILED
        self.completed_at = datetime.now().isoformat()
        if error:
            self.rejection_reason = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "requires_confirmation": self.requires_confirmation,
            "state": self.state.value,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "approved_at": self.approved_at,
            "approved_by": self.approved_by,
            "rejected_at": self.rejected_at,
            "rejection_reason": self.rejection_reason,
            "executed_at": self.executed_at,
            "completed_at": self.completed_at,
        }


@dataclass
class MCPToolResult:
    """Result from executing a tool."""
    proposal_id: str
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: float = 0.0
    executed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "executed_at": self.executed_at,
        }


# =============================================================================
# AGENT STATE MACHINE
# =============================================================================

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


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class SessionType(str, Enum):
    """Types of agent sessions."""
    MONITORING = "monitoring"
    INVESTIGATION = "investigation"
    CONFIGURATION = "configuration"
    GENERAL = "general"


@dataclass
class MCPSession:
    """Represents a bounded conversation session."""
    id: str
    type: SessionType
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: Optional[str] = None
    summary: Optional[str] = None
    
    # Session metrics
    message_count: int = 0
    tool_proposals: int = 0
    tool_executions: int = 0
    tool_rejections: int = 0
    
    # Context
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, session_type: SessionType, metadata: Optional[Dict[str, Any]] = None) -> "MCPSession":
        return cls(
            id=f"session_{uuid.uuid4().hex[:12]}",
            type=session_type,
            metadata=metadata or {},
        )

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    def end(self, summary: Optional[str] = None) -> None:
        """End the session."""
        self.ended_at = datetime.now().isoformat()
        self.summary = summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "summary": self.summary,
            "is_active": self.is_active,
            "message_count": self.message_count,
            "tool_proposals": self.tool_proposals,
            "tool_executions": self.tool_executions,
            "tool_rejections": self.tool_rejections,
            "metadata": self.metadata,
        }


# =============================================================================
# MCP TOOL DEFINITION
# =============================================================================

@dataclass
class MCPToolDefinition:
    """Complete tool definition with schemas and policies."""
    name: str
    description: str
    category: str
    input_schema: List[MCPParameterSchema]
    output_schema: MCPOutputSchema
    requires_confirmation: bool = False
    confirmation_message: Optional[str] = None
    allowed_in_states: tuple = (AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING)
    auto_approve_policy: Optional[Callable[[Dict[str, Any]], bool]] = None
    handler: Optional[Callable[..., Dict[str, Any]]] = None

    def validate_input(self, arguments: Dict[str, Any]) -> tuple:
        """Validate input arguments against schema."""
        for param in self.input_schema:
            value = arguments.get(param.name)
            
            if param.required and value is None:
                return False, f"Missing required parameter: {param.name}"
            
            if value is None:
                continue
            
            if param.type == MCPSchemaType.STRING:
                if not isinstance(value, str):
                    return False, f"Parameter {param.name} must be a string"
                if param.min_length and len(value) < param.min_length:
                    return False, f"Parameter {param.name} must be at least {param.min_length} characters"
                if param.max_length and len(value) > param.max_length:
                    return False, f"Parameter {param.name} must be at most {param.max_length} characters"
            
            elif param.type == MCPSchemaType.NUMBER:
                if not isinstance(value, (int, float)):
                    return False, f"Parameter {param.name} must be a number"
                if param.min_value is not None and value < param.min_value:
                    return False, f"Parameter {param.name} must be >= {param.min_value}"
                if param.max_value is not None and value > param.max_value:
                    return False, f"Parameter {param.name} must be <= {param.max_value}"
            
            elif param.type == MCPSchemaType.INTEGER:
                if not isinstance(value, int) or isinstance(value, bool):
                    return False, f"Parameter {param.name} must be an integer"
            
            elif param.type == MCPSchemaType.BOOLEAN:
                if not isinstance(value, bool):
                    return False, f"Parameter {param.name} must be a boolean"
            
            elif param.type == MCPSchemaType.ARRAY:
                if not isinstance(value, list):
                    return False, f"Parameter {param.name} must be an array"
            
            if param.enum and value not in param.enum:
                return False, f"Parameter {param.name} must be one of: {param.enum}"
        
        return True, None

    def can_auto_approve(self, arguments: Dict[str, Any]) -> bool:
        """Check if this tool call can be auto-approved by policy."""
        if self.requires_confirmation:
            return False
        if self.auto_approve_policy:
            return self.auto_approve_policy(arguments)
        return not self.requires_confirmation

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to complete JSON Schema format."""
        properties = {}
        required = []
        
        for param in self.input_schema:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                "returns": self.output_schema.to_json_schema(),
            },
            "metadata": {
                "category": self.category,
                "requires_confirmation": self.requires_confirmation,
                "confirmation_message": self.confirmation_message,
                "allowed_in_states": [s.value for s in self.allowed_in_states],
            }
        }

    def to_display(self) -> Dict[str, Any]:
        """Convert to display format for UI."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "requires_confirmation": self.requires_confirmation,
            "confirmation_message": self.confirmation_message,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.value,
                    "description": p.description,
                    "required": p.required,
                    "enum": list(p.enum) if p.enum else None,
                    "default": p.default,
                }
                for p in self.input_schema
            ],
        }


# =============================================================================
# AUDIT LOG
# =============================================================================

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


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

def _create_standard_output_schema() -> MCPOutputSchema:
    """Standard output schema for most tools."""
    return MCPOutputSchema(
        type=MCPSchemaType.OBJECT,
        description="Tool execution result",
        properties={
            "success": {"type": "boolean", "description": "Whether the operation succeeded"},
            "message": {"type": "string", "description": "Human-readable result message"},
            "data": {"type": "object", "description": "Additional result data"},
        },
        required_properties=("success",),
    )


# Tool definitions
TOOL_START_MONITORING_SESSION = MCPToolDefinition(
    name="start_monitoring_session",
    description="Start a new monitoring session. The agent will begin actively watching the camera feed.",
    category="session",
    input_schema=[
        MCPParameterSchema(
            name="description",
            type=MCPSchemaType.STRING,
            description="Optional description of what to monitor for",
            required=False,
            max_length=500,
        ),
    ],
    output_schema=_create_standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.OFF, AgentState.IDLE),
)

TOOL_END_SESSION = MCPToolDefinition(
    name="end_session",
    description="End the current session. Monitoring will stop and a summary will be provided.",
    category="session",
    input_schema=[],
    output_schema=_create_standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING),
)

TOOL_GET_SESSION_SUMMARY = MCPToolDefinition(
    name="get_session_summary",
    description="Get a summary of the current or previous session including events and statistics.",
    category="session",
    input_schema=[
        MCPParameterSchema(
            name="session_id",
            type=MCPSchemaType.STRING,
            description="Optional session ID. If not provided, summarizes the current/most recent session.",
            required=False,
        ),
    ],
    output_schema=_create_standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)

TOOL_GET_AGENT_STATUS = MCPToolDefinition(
    name="get_agent_status",
    description="Get the current status of the monitoring agent including state, session info, and statistics.",
    category="status",
    input_schema=[],
    output_schema=MCPOutputSchema(
        type=MCPSchemaType.OBJECT,
        description="Agent status information",
        properties={
            "state": {"type": "string", "description": "Current agent state"},
            "session": {"type": "object", "description": "Current session info"},
            "stats": {"type": "object", "description": "Monitoring statistics"},
        },
        required_properties=("state",),
    ),
    requires_confirmation=False,
    allowed_in_states=(AgentState.OFF, AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING, AgentState.ALERTING, AgentState.ERROR),
)

TOOL_ANALYZE_FRAME = MCPToolDefinition(
    name="analyze_current_frame",
    description="Analyze the current camera frame and describe what is visible. Can include a specific question.",
    category="analysis",
    input_schema=[
        MCPParameterSchema(
            name="query",
            type=MCPSchemaType.STRING,
            description="Optional specific question about the frame",
            required=False,
            max_length=500,
        ),
    ],
    output_schema=MCPOutputSchema(
        type=MCPSchemaType.OBJECT,
        description="Frame analysis result",
        properties={
            "detected": {"type": "boolean", "description": "Whether objects of interest were detected"},
            "confidence": {"type": "number", "description": "Detection confidence 0-1"},
            "description": {"type": "string", "description": "Description of what was observed"},
            "should_alert": {"type": "boolean", "description": "Whether this warrants an alert"},
        },
        required_properties=("detected", "description"),
    ),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING),
)

TOOL_SEND_ALERT_EMAIL = MCPToolDefinition(
    name="send_alert_email",
    description="Send an alert email to specified recipients with optional image attachment.",
    category="alerts",
    input_schema=[
        MCPParameterSchema(
            name="recipients",
            type=MCPSchemaType.ARRAY,
            description="List of email addresses to send the alert to",
            required=True,
            items_type=MCPSchemaType.STRING,
        ),
        MCPParameterSchema(
            name="subject",
            type=MCPSchemaType.STRING,
            description="Email subject line",
            required=True,
            min_length=1,
            max_length=200,
        ),
        MCPParameterSchema(
            name="body",
            type=MCPSchemaType.STRING,
            description="Email body content",
            required=True,
            min_length=1,
            max_length=5000,
        ),
        MCPParameterSchema(
            name="include_image",
            type=MCPSchemaType.BOOLEAN,
            description="Whether to attach the current camera frame",
            required=False,
            default=True,
        ),
        MCPParameterSchema(
            name="priority",
            type=MCPSchemaType.STRING,
            description="Email priority level",
            required=False,
            enum=("low", "normal", "high"),
            default="normal",
        ),
    ],
    output_schema=_create_standard_output_schema(),
    requires_confirmation=True,
    confirmation_message="This will send an email to {recipients}. Do you want to proceed?",
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ALERTING),
)

TOOL_SAVE_EVIDENCE = MCPToolDefinition(
    name="save_evidence",
    description="Save the current frame as evidence for later review.",
    category="evidence",
    input_schema=[
        MCPParameterSchema(
            name="label",
            type=MCPSchemaType.STRING,
            description="Label to identify this evidence",
            required=True,
            min_length=1,
            max_length=100,
        ),
        MCPParameterSchema(
            name="notes",
            type=MCPSchemaType.STRING,
            description="Additional notes about the evidence",
            required=False,
            max_length=1000,
        ),
    ],
    output_schema=MCPOutputSchema(
        type=MCPSchemaType.OBJECT,
        description="Evidence save result",
        properties={
            "success": {"type": "boolean"},
            "filepath": {"type": "string", "description": "Path to saved evidence"},
            "timestamp": {"type": "string", "description": "When the evidence was saved"},
        },
        required_properties=("success",),
    ),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ALERTING),
)

TOOL_LOG_EVENT = MCPToolDefinition(
    name="log_event",
    description="Log an event to the persistent audit log.",
    category="logging",
    input_schema=[
        MCPParameterSchema(
            name="event_type",
            type=MCPSchemaType.STRING,
            description="Type of event",
            required=True,
            enum=("observation", "detection", "alert", "system", "user_action"),
        ),
        MCPParameterSchema(
            name="description",
            type=MCPSchemaType.STRING,
            description="Description of the event",
            required=True,
            min_length=1,
            max_length=1000,
        ),
        MCPParameterSchema(
            name="severity",
            type=MCPSchemaType.STRING,
            description="Event severity level",
            required=False,
            enum=("info", "warning", "error"),
            default="info",
        ),
    ],
    output_schema=_create_standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING, AgentState.ALERTING),
)

TOOL_QUERY_HISTORY = MCPToolDefinition(
    name="query_history",
    description="Query recent detection history and events.",
    category="history",
    input_schema=[
        MCPParameterSchema(
            name="limit",
            type=MCPSchemaType.INTEGER,
            description="Maximum number of events to return",
            required=False,
            default=10,
            min_value=1,
            max_value=100,
        ),
        MCPParameterSchema(
            name="event_type",
            type=MCPSchemaType.STRING,
            description="Filter by event type",
            required=False,
            enum=("detection", "alert", "all"),
        ),
    ],
    output_schema=MCPOutputSchema(
        type=MCPSchemaType.OBJECT,
        description="Query results",
        properties={
            "events": {"type": "array", "description": "List of matching events"},
            "total": {"type": "integer", "description": "Total matching events"},
        },
        required_properties=("events", "total"),
    ),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING),
)

TOOL_SET_DETECTION_TASK = MCPToolDefinition(
    name="set_detection_task",
    description="Configure what the agent should look for during monitoring.",
    category="configuration",
    input_schema=[
        MCPParameterSchema(
            name="task_type",
            type=MCPSchemaType.STRING,
            description="The type of detection task",
            required=True,
            enum=("package_detection", "ppe_detection", "person_counting", "scene_description", "custom"),
        ),
        MCPParameterSchema(
            name="custom_instructions",
            type=MCPSchemaType.STRING,
            description="Custom instructions for the detection task (required for custom task type)",
            required=False,
            max_length=2000,
        ),
    ],
    output_schema=_create_standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING),
)

TOOL_GO_IDLE = MCPToolDefinition(
    name="go_idle",
    description="Pause active monitoring and enter idle state. Camera monitoring will stop.",
    category="control",
    input_schema=[],
    output_schema=_create_standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.MONITORING, AgentState.ALERTING, AgentState.ERROR),
)

TOOL_SHUTDOWN_AGENT = MCPToolDefinition(
    name="shutdown_agent",
    description="Completely shut down the agent. All monitoring will stop.",
    category="control",
    input_schema=[],
    output_schema=_create_standard_output_schema(),
    requires_confirmation=True,
    confirmation_message="This will completely shut down the agent. Are you sure?",
    allowed_in_states=(AgentState.IDLE, AgentState.MONITORING, AgentState.ERROR),
)

TOOL_ACKNOWLEDGE_ERROR = MCPToolDefinition(
    name="acknowledge_error",
    description="Acknowledge an error state and attempt to recover to idle.",
    category="control",
    input_schema=[],
    output_schema=_create_standard_output_schema(),
    requires_confirmation=False,
    allowed_in_states=(AgentState.ERROR,),
)


# =============================================================================
# TOOL REGISTRY
# =============================================================================

class MCPToolRegistry:
    """Registry of available MCP tools with schema validation."""

    def __init__(self):
        self._tools: Dict[str, MCPToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        default_tools = [
            TOOL_START_MONITORING_SESSION,
            TOOL_END_SESSION,
            TOOL_GET_SESSION_SUMMARY,
            TOOL_GET_AGENT_STATUS,
            TOOL_ANALYZE_FRAME,
            TOOL_SEND_ALERT_EMAIL,
            TOOL_SAVE_EVIDENCE,
            TOOL_LOG_EVENT,
            TOOL_QUERY_HISTORY,
            TOOL_SET_DETECTION_TASK,
            TOOL_GO_IDLE,
            TOOL_SHUTDOWN_AGENT,
            TOOL_ACKNOWLEDGE_ERROR,
        ]
        for tool in default_tools:
            self.register(tool)

    def register(self, tool: MCPToolDefinition) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug("Registered MCP tool: %s", tool.name)

    def get(self, name: str) -> Optional[MCPToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[MCPToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_tools_for_state(self, state: AgentState) -> List[MCPToolDefinition]:
        """Get tools available in a given agent state."""
        return [t for t in self._tools.values() if state in t.allowed_in_states]

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for all tools."""
        return [tool.to_json_schema() for tool in self._tools.values()]

    def get_display_list(self, state: Optional[AgentState] = None) -> List[Dict[str, Any]]:
        """Get display information for tools, optionally filtered by state."""
        tools = self.get_tools_for_state(state) if state else self.list_tools()
        return [tool.to_display() for tool in tools]


# =============================================================================
# MCP INTERPRETER
# =============================================================================

class MCPInterpreter:
    """Interprets user intent and produces tool call proposals.
    
    This is the INTERPRETATION PHASE. No tools are executed here.
    """

    START_SESSION_PHRASES = frozenset([
        "start monitoring",
        "begin monitoring",
        "turn the agent on",
        "turn on the agent",
        "enable the agent",
        "start the agent",
        "activate monitoring",
        "start a monitoring session",
        "begin a new session",
    ])

    END_SESSION_PHRASES = frozenset([
        "stop monitoring",
        "end monitoring",
        "turn the agent off",
        "turn off the agent",
        "disable the agent",
        "stop the agent",
        "deactivate monitoring",
        "end the session",
        "end session",
    ])

    IDLE_PHRASES = frozenset([
        "pause monitoring",
        "go idle",
        "take a break",
        "pause the agent",
    ])

    STATUS_PHRASES = frozenset([
        "status",
        "what is your status",
        "what's your status",
        "are you running",
        "is the agent on",
        "is monitoring active",
        "agent status",
        "current state",
    ])

    ANALYZE_PHRASES = frozenset([
        "analyze",
        "look at",
        "what do you see",
        "describe",
        "check the camera",
        "what's in the frame",
        "what is in view",
        "examine",
    ])

    HISTORY_PHRASES = frozenset([
        "history",
        "recent detections",
        "what happened",
        "past events",
        "show me events",
        "recent events",
    ])

    SUMMARY_PHRASES = frozenset([
        "summarize",
        "summary",
        "what happened in the session",
        "session summary",
        "recap",
    ])

    def __init__(self, registry: Optional[MCPToolRegistry] = None):
        self.registry = registry or get_tool_registry()
        self.audit_log = get_audit_log()

    def interpret(
        self,
        user_message: str,
        agent_state: AgentState,
        session_id: Optional[str] = None,
    ) -> Optional[MCPToolCallProposal]:
        """Interpret user message and produce a tool call proposal.
        
        Returns None if no tool call is needed.
        """
        start_time = time.time()
        message_lower = user_message.lower().strip()
        
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.INTENT_DETECTED,
            details={"message": user_message[:500], "agent_state": agent_state.value},
            session_id=session_id,
        ))

        proposal = self._check_session_intents(message_lower, agent_state, session_id)
        if proposal:
            return self._finalize_proposal(proposal, start_time, session_id)

        proposal = self._check_status_intents(message_lower, agent_state, session_id)
        if proposal:
            return self._finalize_proposal(proposal, start_time, session_id)

        proposal = self._check_analysis_intents(user_message, message_lower, agent_state, session_id)
        if proposal:
            return self._finalize_proposal(proposal, start_time, session_id)

        proposal = self._check_history_intents(message_lower, agent_state, session_id)
        if proposal:
            return self._finalize_proposal(proposal, start_time, session_id)

        return None

    def _check_session_intents(
        self,
        message_lower: str,
        agent_state: AgentState,
        session_id: Optional[str],
    ) -> Optional[MCPToolCallProposal]:
        """Check for session control intents."""
        
        for phrase in self.START_SESSION_PHRASES:
            if phrase in message_lower:
                tool = self.registry.get("start_monitoring_session")
                if tool and agent_state in tool.allowed_in_states:
                    return MCPToolCallProposal.create(
                        tool_name="start_monitoring_session",
                        arguments={},
                        rationale=f"User requested to start monitoring: '{message_lower}'",
                        confidence=0.95,
                        requires_confirmation=tool.requires_confirmation,
                        session_id=session_id,
                    )
                elif tool:
                    return self._create_state_violation_proposal(
                        "start_monitoring_session",
                        agent_state,
                        tool.allowed_in_states,
                        session_id,
                    )

        for phrase in self.END_SESSION_PHRASES:
            if phrase in message_lower:
                tool = self.registry.get("end_session")
                if tool and agent_state in tool.allowed_in_states:
                    return MCPToolCallProposal.create(
                        tool_name="end_session",
                        arguments={},
                        rationale=f"User requested to end session: '{message_lower}'",
                        confidence=0.95,
                        requires_confirmation=tool.requires_confirmation,
                        session_id=session_id,
                    )

        for phrase in self.IDLE_PHRASES:
            if phrase in message_lower:
                tool = self.registry.get("go_idle")
                if tool and agent_state in tool.allowed_in_states:
                    return MCPToolCallProposal.create(
                        tool_name="go_idle",
                        arguments={},
                        rationale=f"User requested to pause: '{message_lower}'",
                        confidence=0.90,
                        requires_confirmation=tool.requires_confirmation,
                        session_id=session_id,
                    )

        return None

    def _check_status_intents(
        self,
        message_lower: str,
        agent_state: AgentState,
        session_id: Optional[str],
    ) -> Optional[MCPToolCallProposal]:
        """Check for status query intents."""
        for phrase in self.STATUS_PHRASES:
            if phrase in message_lower:
                return MCPToolCallProposal.create(
                    tool_name="get_agent_status",
                    arguments={},
                    rationale=f"User asked about status: '{message_lower}'",
                    confidence=0.90,
                    requires_confirmation=False,
                    session_id=session_id,
                )
        return None

    def _check_analysis_intents(
        self,
        user_message: str,
        message_lower: str,
        agent_state: AgentState,
        session_id: Optional[str],
    ) -> Optional[MCPToolCallProposal]:
        """Check for frame analysis intents."""
        for phrase in self.ANALYZE_PHRASES:
            if phrase in message_lower:
                tool = self.registry.get("analyze_current_frame")
                if tool and agent_state in tool.allowed_in_states:
                    return MCPToolCallProposal.create(
                        tool_name="analyze_current_frame",
                        arguments={"query": user_message},
                        rationale=f"User requested frame analysis: '{message_lower}'",
                        confidence=0.85,
                        requires_confirmation=tool.requires_confirmation,
                        session_id=session_id,
                    )
        return None

    def _check_history_intents(
        self,
        message_lower: str,
        agent_state: AgentState,
        session_id: Optional[str],
    ) -> Optional[MCPToolCallProposal]:
        """Check for history/summary intents."""
        
        for phrase in self.SUMMARY_PHRASES:
            if phrase in message_lower:
                return MCPToolCallProposal.create(
                    tool_name="get_session_summary",
                    arguments={},
                    rationale=f"User requested summary: '{message_lower}'",
                    confidence=0.85,
                    requires_confirmation=False,
                    session_id=session_id,
                )

        for phrase in self.HISTORY_PHRASES:
            if phrase in message_lower:
                tool = self.registry.get("query_history")
                if tool and agent_state in tool.allowed_in_states:
                    return MCPToolCallProposal.create(
                        tool_name="query_history",
                        arguments={"limit": 10},
                        rationale=f"User requested history: '{message_lower}'",
                        confidence=0.85,
                        requires_confirmation=False,
                        session_id=session_id,
                    )
        return None

    def _create_state_violation_proposal(
        self,
        tool_name: str,
        current_state: AgentState,
        allowed_states: tuple,
        session_id: Optional[str],
    ) -> MCPToolCallProposal:
        """Create a proposal that will be rejected due to state violation."""
        proposal = MCPToolCallProposal.create(
            tool_name=tool_name,
            arguments={},
            rationale=f"Tool '{tool_name}' not available in state '{current_state.value}'",
            confidence=0.0,
            requires_confirmation=False,
            session_id=session_id,
        )
        proposal.reject(
            f"Cannot execute '{tool_name}' in state '{current_state.value}'. "
            f"Allowed states: {[s.value for s in allowed_states]}"
        )
        return proposal

    def _finalize_proposal(
        self,
        proposal: MCPToolCallProposal,
        start_time: float,
        session_id: Optional[str],
    ) -> MCPToolCallProposal:
        """Finalize and log a proposal."""
        latency_ms = (time.time() - start_time) * 1000
        
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_PROPOSED,
            details={
                "proposal_id": proposal.id,
                "tool_name": proposal.tool_name,
                "arguments": proposal.arguments,
                "rationale": proposal.rationale,
                "confidence": proposal.confidence,
                "requires_confirmation": proposal.requires_confirmation,
            },
            session_id=session_id,
            latency_ms=latency_ms,
        ))
        
        return proposal


# =============================================================================
# MCP EXECUTOR
# =============================================================================

class MCPExecutor:
    """Executes approved tool call proposals.
    
    This is the EXECUTION PHASE. Tools are only executed after:
    1. Explicit user approval, OR
    2. Policy-based auto-approval (for non-confirmation tools)
    """

    def __init__(
        self,
        registry: Optional[MCPToolRegistry] = None,
        state_machine: Optional[AgentStateMachine] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.registry = registry or get_tool_registry()
        self.state_machine = state_machine or get_agent_state_machine()
        self.audit_log = get_audit_log()
        self.context = context or {}
        
        self._current_session: Optional[MCPSession] = None
        self._sessions: List[MCPSession] = []
        self._session_lock = threading.Lock()
        
        self._pending_proposals: Dict[str, MCPToolCallProposal] = {}
        self._proposals_lock = threading.Lock()

    def update_context(self, **kwargs) -> None:
        """Update the execution context."""
        self.context.update(kwargs)

    @property
    def current_session(self) -> Optional[MCPSession]:
        """Get the current active session."""
        with self._session_lock:
            return self._current_session

    def submit_proposal(self, proposal: MCPToolCallProposal) -> Dict[str, Any]:
        """Submit a proposal for approval and possible execution."""
        tool = self.registry.get(proposal.tool_name)
        if not tool:
            proposal.reject(f"Unknown tool: {proposal.tool_name}")
            self._log_rejection(proposal)
            return {
                "status": "rejected",
                "reason": f"Unknown tool: {proposal.tool_name}",
                "proposal": proposal.to_dict(),
            }

        is_valid, error = tool.validate_input(proposal.arguments)
        if not is_valid:
            proposal.reject(f"Validation error: {error}")
            self._log_rejection(proposal)
            return {
                "status": "rejected",
                "reason": error,
                "proposal": proposal.to_dict(),
            }

        current_state = self.state_machine.state
        if current_state not in tool.allowed_in_states:
            proposal.reject(
                f"Tool not allowed in state '{current_state.value}'. "
                f"Allowed: {[s.value for s in tool.allowed_in_states]}"
            )
            self._log_rejection(proposal)
            return {
                "status": "rejected",
                "reason": proposal.rejection_reason,
                "proposal": proposal.to_dict(),
            }

        if tool.requires_confirmation and not tool.can_auto_approve(proposal.arguments):
            with self._proposals_lock:
                self._pending_proposals[proposal.id] = proposal
            
            proposal.state = ToolLifecycleState.PENDING_APPROVAL
            
            self.audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.TOOL_PENDING_APPROVAL,
                details={
                    "proposal_id": proposal.id,
                    "tool_name": proposal.tool_name,
                    "confirmation_message": tool.confirmation_message,
                },
                session_id=proposal.session_id,
            ))
            
            return {
                "status": "pending_approval",
                "proposal_id": proposal.id,
                "tool_name": proposal.tool_name,
                "confirmation_message": self._format_confirmation_message(tool, proposal),
                "proposal": proposal.to_dict(),
            }

        proposal.approve(approved_by="policy")
        self._log_approval(proposal)
        return self._execute_proposal(proposal, tool)

    def approve_proposal(self, proposal_id: str) -> Dict[str, Any]:
        """Approve a pending proposal for execution."""
        with self._proposals_lock:
            proposal = self._pending_proposals.pop(proposal_id, None)
        
        if not proposal:
            return {
                "status": "error",
                "reason": f"No pending proposal with ID: {proposal_id}",
            }

        tool = self.registry.get(proposal.tool_name)
        if not tool:
            return {
                "status": "error",
                "reason": f"Tool no longer available: {proposal.tool_name}",
            }

        proposal.approve(approved_by="user")
        self._log_approval(proposal)
        return self._execute_proposal(proposal, tool)

    def reject_proposal(self, proposal_id: str, reason: str = "User rejected") -> Dict[str, Any]:
        """Reject a pending proposal."""
        with self._proposals_lock:
            proposal = self._pending_proposals.pop(proposal_id, None)
        
        if not proposal:
            return {
                "status": "error",
                "reason": f"No pending proposal with ID: {proposal_id}",
            }

        proposal.reject(reason)
        self._log_rejection(proposal)
        
        return {
            "status": "rejected",
            "proposal_id": proposal_id,
            "reason": reason,
        }

    def get_pending_proposals(self) -> List[Dict[str, Any]]:
        """Get all pending proposals awaiting approval."""
        with self._proposals_lock:
            return [p.to_dict() for p in self._pending_proposals.values()]

    def _execute_proposal(
        self,
        proposal: MCPToolCallProposal,
        tool: MCPToolDefinition,
    ) -> Dict[str, Any]:
        """Execute an approved proposal."""
        start_time = time.time()
        
        proposal.start_execution()
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_EXECUTING,
            details={"proposal_id": proposal.id, "tool_name": proposal.tool_name},
            session_id=proposal.session_id,
        ))

        try:
            result = self._invoke_tool_handler(proposal.tool_name, proposal.arguments)
            duration_ms = (time.time() - start_time) * 1000
            
            proposal.complete(success=True)
            
            tool_result = MCPToolResult(
                proposal_id=proposal.id,
                tool_name=proposal.tool_name,
                success=True,
                output=result,
                duration_ms=duration_ms,
            )
            
            self.audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.TOOL_SUCCEEDED,
                details={
                    "proposal_id": proposal.id,
                    "tool_name": proposal.tool_name,
                    "output_preview": str(result)[:500],
                },
                session_id=proposal.session_id,
                latency_ms=duration_ms,
            ))
            
            if self._current_session:
                self._current_session.tool_executions += 1
            
            return {
                "status": "executed",
                "result": tool_result.to_dict(),
                "proposal": proposal.to_dict(),
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            proposal.complete(success=False, error=error_msg)
            
            tool_result = MCPToolResult(
                proposal_id=proposal.id,
                tool_name=proposal.tool_name,
                success=False,
                output=None,
                error=error_msg,
                duration_ms=duration_ms,
            )
            
            self.audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.TOOL_FAILED,
                details={
                    "proposal_id": proposal.id,
                    "tool_name": proposal.tool_name,
                    "error": error_msg,
                },
                session_id=proposal.session_id,
                latency_ms=duration_ms,
            ))
            
            return {
                "status": "failed",
                "error": error_msg,
                "result": tool_result.to_dict(),
                "proposal": proposal.to_dict(),
            }

    def _invoke_tool_handler(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the appropriate handler for a tool."""
        
        if tool_name == "start_monitoring_session":
            return self._handle_start_monitoring_session(arguments)
        elif tool_name == "end_session":
            return self._handle_end_session(arguments)
        elif tool_name == "get_session_summary":
            return self._handle_get_session_summary(arguments)
        elif tool_name == "get_agent_status":
            return self._handle_get_agent_status(arguments)
        elif tool_name == "analyze_current_frame":
            return self._handle_analyze_frame(arguments)
        elif tool_name == "go_idle":
            return self._handle_go_idle(arguments)
        elif tool_name == "shutdown_agent":
            return self._handle_shutdown_agent(arguments)
        elif tool_name == "acknowledge_error":
            return self._handle_acknowledge_error(arguments)
        elif tool_name == "send_alert_email":
            return self._handle_send_alert_email(arguments)
        elif tool_name == "save_evidence":
            return self._handle_save_evidence(arguments)
        elif tool_name == "log_event":
            return self._handle_log_event(arguments)
        elif tool_name == "query_history":
            return self._handle_query_history(arguments)
        elif tool_name == "set_detection_task":
            return self._handle_set_detection_task(arguments)
        
        raise ValueError(f"No handler for tool: {tool_name}")

    # --- Tool Handlers ---

    def _handle_start_monitoring_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Start a monitoring session."""
        from services.monitoring_service import get_monitoring_service
        
        with self._session_lock:
            if self._current_session and self._current_session.is_active:
                self._current_session.end("Replaced by new session")
                self._sessions.append(self._current_session)
            
            self._current_session = MCPSession.create(
                SessionType.MONITORING,
                metadata={"description": args.get("description", "")},
            )
            self.state_machine.set_session(self._current_session.id)
        
        if self.state_machine.state == AgentState.OFF:
            self.state_machine.transition_to(AgentState.IDLE, "start_monitoring_session")
        
        self.state_machine.transition_to(AgentState.MONITORING, "start_monitoring_session")
        
        service = get_monitoring_service()
        if service:
            if not service.agent:
                service.initialize()
            service.start_monitoring()
        
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.SESSION_STARTED,
            details={
                "session_id": self._current_session.id,
                "type": self._current_session.type.value,
            },
            session_id=self._current_session.id,
        ))
        
        return {
            "success": True,
            "message": "Monitoring session started",
            "data": {"session_id": self._current_session.id},
        }

    def _handle_end_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """End the current session."""
        from services.monitoring_service import get_monitoring_service
        
        with self._session_lock:
            if not self._current_session:
                return {"success": False, "message": "No active session"}
            
            service = get_monitoring_service()
            if service and service.is_monitoring:
                service.stop_monitoring()
            
            if self.state_machine.state == AgentState.MONITORING:
                self.state_machine.transition_to(AgentState.IDLE, "end_session")
            
            session_id = self._current_session.id
            self._current_session.end()
            self._sessions.append(self._current_session)
            
            self.audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.SESSION_ENDED,
                details={"session_id": session_id},
                session_id=session_id,
            ))
            
            summary = self._generate_session_summary(self._current_session)
            self._current_session = None
            self.state_machine.set_session(None)
        
        return {
            "success": True,
            "message": "Session ended",
            "data": {"summary": summary},
        }

    def _handle_get_session_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get session summary."""
        session_id = args.get("session_id")
        
        with self._session_lock:
            if session_id:
                session = next((s for s in self._sessions if s.id == session_id), None)
            elif self._current_session:
                session = self._current_session
            elif self._sessions:
                session = self._sessions[-1]
            else:
                return {"success": False, "message": "No sessions available"}
        
        if not session:
            return {"success": False, "message": f"Session not found: {session_id}"}
        
        summary = self._generate_session_summary(session)
        
        return {
            "success": True,
            "message": "Session summary retrieved",
            "data": {"session": session.to_dict(), "summary": summary},
        }

    def _handle_get_agent_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get agent status."""
        from services.monitoring_service import get_monitoring_service
        
        service = get_monitoring_service()
        
        status = {
            "state": self.state_machine.state.value,
            "state_machine": self.state_machine.get_status(),
            "session": self._current_session.to_dict() if self._current_session else None,
            "monitoring_active": service.is_monitoring if service else False,
            "stats": service._serialize_stats() if service and hasattr(service, '_serialize_stats') else {},
            "metrics": self.audit_log.get_metrics(),
        }
        
        return {
            "success": True,
            "message": f"Agent is {self.state_machine.state.value}",
            "data": status,
        }

    def _handle_analyze_frame(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current frame."""
        from services.monitoring_service import get_monitoring_service
        
        old_state = self.state_machine.state
        if self.state_machine.can_transition_to(AgentState.ANALYZING):
            self.state_machine.transition_to(AgentState.ANALYZING, "analyze_current_frame")
        
        try:
            service = get_monitoring_service()
            if not service:
                return {"success": False, "message": "Monitoring service not available"}
            
            query = args.get("query")
            event = service.analyze_single_frame(custom_prompt=query)
            
            if event:
                return {
                    "success": True,
                    "message": "Frame analyzed",
                    "data": {
                        "detected": event.detected,
                        "confidence": event.confidence,
                        "description": event.vision_description,
                        "should_alert": event.should_alert,
                    },
                }
            return {"success": False, "message": service.last_error or "Analysis failed"}
        finally:
            if self.state_machine.state == AgentState.ANALYZING:
                if self.state_machine.can_transition_to(old_state):
                    self.state_machine.transition_to(old_state, "analysis_complete")
                elif self.state_machine.can_transition_to(AgentState.IDLE):
                    self.state_machine.transition_to(AgentState.IDLE, "analysis_complete")

    def _handle_go_idle(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Transition to idle state."""
        from services.monitoring_service import get_monitoring_service
        
        service = get_monitoring_service()
        if service and service.is_monitoring:
            service.stop_monitoring()
        
        self.state_machine.transition_to(AgentState.IDLE, "go_idle")
        
        return {"success": True, "message": "Agent is now idle"}

    def _handle_shutdown_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Shutdown the agent."""
        from services.monitoring_service import get_monitoring_service
        
        if self._current_session:
            self._handle_end_session({})
        
        service = get_monitoring_service()
        if service:
            service.stop_monitoring()
        
        if self.state_machine.state != AgentState.IDLE:
            if self.state_machine.can_transition_to(AgentState.IDLE):
                self.state_machine.transition_to(AgentState.IDLE, "shutdown_agent")
        
        self.state_machine.transition_to(AgentState.OFF, "shutdown_agent")
        
        return {"success": True, "message": "Agent shut down"}

    def _handle_acknowledge_error(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Acknowledge error and recover."""
        self.state_machine.transition_to(AgentState.IDLE, "acknowledge_error")
        return {"success": True, "message": "Error acknowledged, agent is now idle"}

    def _handle_send_alert_email(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Send alert email."""
        from agents.tools import _tool_send_alert_email
        return _tool_send_alert_email(
            recipients=args.get("recipients", []),
            subject=args.get("subject", "Alert"),
            body=args.get("body", ""),
            priority=args.get("priority", "normal"),
            include_image=args.get("include_image", True),
            image_data=self.context.get("image_data"),
        )

    def _handle_save_evidence(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Save evidence."""
        from agents.tools import _tool_save_evidence
        return _tool_save_evidence(
            image_data=self.context.get("image_data", b""),
            label=args.get("label", "evidence"),
            metadata={"notes": args.get("notes", "")},
        )

    def _handle_log_event(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Log event."""
        from agents.tools import _tool_log_event
        return _tool_log_event(
            event_type=args.get("event_type", "general"),
            description=args.get("description", ""),
            severity=args.get("severity", "info"),
        )

    def _handle_query_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query history."""
        from agents.tools import _tool_query_history
        return _tool_query_history(
            limit=args.get("limit", 10),
            detected_only=args.get("event_type") == "detection",
        )

    def _handle_set_detection_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set detection task."""
        from services.monitoring_service import get_monitoring_service
        from agents.vlm.task_types import TaskType
        
        service = get_monitoring_service()
        if not service:
            return {"success": False, "message": "Service not available"}
        
        task_map = {
            "package_detection": TaskType.PACKAGE_DETECTION,
            "ppe_detection": TaskType.PPE_DETECTION,
            "person_counting": TaskType.PERSON_COUNTING,
            "scene_description": TaskType.SCENE_DESCRIPTION,
            "custom": TaskType.CUSTOM,
        }
        
        task_type = task_map.get(args.get("task_type"))
        if not task_type:
            return {"success": False, "message": "Invalid task type"}
        
        service.set_active_prompt(
            task_type=task_type,
            custom_prompt=args.get("custom_instructions", ""),
            alerts_enabled=True,
            agentic_mode=True,
        )
        
        return {"success": True, "message": f"Detection task set to {args.get('task_type')}"}

    # --- Helper Methods ---

    def _format_confirmation_message(
        self,
        tool: MCPToolDefinition,
        proposal: MCPToolCallProposal,
    ) -> str:
        """Format confirmation message with argument substitution."""
        if not tool.confirmation_message:
            return f"Execute {tool.name}?"
        
        message = tool.confirmation_message
        for key, value in proposal.arguments.items():
            message = message.replace(f"{{{key}}}", str(value))
        return message

    def _log_approval(self, proposal: MCPToolCallProposal) -> None:
        """Log tool approval."""
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_APPROVED,
            details={
                "proposal_id": proposal.id,
                "tool_name": proposal.tool_name,
                "approved_by": proposal.approved_by,
            },
            session_id=proposal.session_id,
        ))

    def _log_rejection(self, proposal: MCPToolCallProposal) -> None:
        """Log tool rejection."""
        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.TOOL_REJECTED,
            details={
                "proposal_id": proposal.id,
                "tool_name": proposal.tool_name,
                "reason": proposal.rejection_reason,
            },
            session_id=proposal.session_id,
        ))
        
        if self._current_session:
            self._current_session.tool_rejections += 1

    def _generate_session_summary(self, session: MCPSession) -> str:
        """Generate a human-readable session summary."""
        duration = ""
        if session.ended_at:
            start = datetime.fromisoformat(session.started_at)
            end = datetime.fromisoformat(session.ended_at)
            minutes = int((end - start).total_seconds() / 60)
            duration = f"{minutes} minutes"
        else:
            start = datetime.fromisoformat(session.started_at)
            minutes = int((datetime.now() - start).total_seconds() / 60)
            duration = f"{minutes} minutes (ongoing)"
        
        return (
            f"Session '{session.type.value}' ({session.id}):\n"
            f"- Duration: {duration}\n"
            f"- Messages: {session.message_count}\n"
            f"- Tool proposals: {session.tool_proposals}\n"
            f"- Executed: {session.tool_executions}\n"
            f"- Rejected: {session.tool_rejections}"
        )


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_agent_state_machine: Optional[AgentStateMachine] = None
_audit_log: Optional[MCPAuditLog] = None
_tool_registry: Optional[MCPToolRegistry] = None
_mcp_executor: Optional[MCPExecutor] = None
_mcp_interpreter: Optional[MCPInterpreter] = None
_global_lock = threading.RLock()  # Use RLock to allow re-entrant acquisition


def get_agent_state_machine() -> AgentStateMachine:
    """Get the global agent state machine instance."""
    global _agent_state_machine
    with _global_lock:
        if _agent_state_machine is None:
            _agent_state_machine = AgentStateMachine()
        return _agent_state_machine


def get_audit_log() -> MCPAuditLog:
    """Get the global audit log instance."""
    global _audit_log
    with _global_lock:
        if _audit_log is None:
            _audit_log = MCPAuditLog()
        return _audit_log


def get_tool_registry() -> MCPToolRegistry:
    """Get the global tool registry instance."""
    global _tool_registry
    with _global_lock:
        if _tool_registry is None:
            _tool_registry = MCPToolRegistry()
        return _tool_registry


def get_mcp_executor() -> MCPExecutor:
    """Get the global MCP executor instance."""
    global _mcp_executor
    with _global_lock:
        if _mcp_executor is None:
            _mcp_executor = MCPExecutor()
        return _mcp_executor


def get_mcp_interpreter() -> MCPInterpreter:
    """Get the global MCP interpreter instance."""
    global _mcp_interpreter
    with _global_lock:
        if _mcp_interpreter is None:
            _mcp_interpreter = MCPInterpreter()
        return _mcp_interpreter
