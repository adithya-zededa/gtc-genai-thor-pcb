"""Tool calling system for the agentic camera monitoring agent.

This module provides a tool registry and execution framework that allows
the LLM to dynamically invoke tools based on its analysis.

Tools available:
- send_alert_email: Send email notifications with optional image attachments
- save_evidence: Save detection images to disk for later review
- log_event: Record events to the persistent log database
- notify_desktop: Show desktop notification (when available)
- query_history: Query recent detection history for context
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by the LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str] = field(default_factory=list)
    handler: Optional[Callable[..., Dict[str, Any]]] = None

    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format for LLM prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params,
            }
        }


@dataclass
class ToolCall:
    """Represents a tool call request from the LLM."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str = ""

    def __post_init__(self):
        if not self.call_id:
            self.call_id = f"call_{datetime.now().strftime('%H%M%S%f')}"


@dataclass  
class ToolResult:
    """Result from executing a tool."""
    tool_name: str
    call_id: str
    success: bool
    result: Any
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
        }


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

def _tool_send_alert_email(
    recipients: List[str],
    subject: str,
    body: str,
    priority: str = "normal",
    include_image: bool = False,
    image_data: Optional[bytes] = None,
    **kwargs
) -> Dict[str, Any]:
    """Send an alert email to specified recipients.
    
    Args:
        recipients: List of email addresses to send to
        subject: Email subject line
        body: Email body content
        priority: Email priority (low, normal, high)
        include_image: Whether to attach the detection image
        image_data: Raw image bytes to attach (if include_image is True)
    
    Returns:
        Dictionary with success status and details
    """
    from agent_runtime.email_tools import send_email
    
    if not recipients:
        return {"success": False, "error": "No recipients specified"}
    
    payload: Dict[str, Any] = {
        "to": recipients,
        "subject": subject,
        "body": body,
    }
    
    if include_image and image_data:
        payload["image_data"] = image_data
        payload["image_filename"] = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    
    try:
        result = send_email(payload)
        return {
            "success": True,
            "message": result,
            "recipients_count": len(recipients),
        }
    except Exception as e:
        logger.error("Failed to send alert email: %s", e)
        return {"success": False, "error": str(e)}


def _tool_save_evidence(
    image_data: bytes,
    label: str = "detection",
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Save detection image as evidence.
    
    Args:
        image_data: Raw image bytes (JPEG encoded)
        label: Label for the evidence (used in filename)
        metadata: Additional metadata to store alongside
    
    Returns:
        Dictionary with file path and status
    """
    evidence_dir = Path(os.getenv("DETECTED_IMAGES_DIR", "detected_images"))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
    filename = f"{safe_label}_{timestamp}.jpg"
    filepath = evidence_dir / filename
    
    try:
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        # Save metadata if provided
        if metadata:
            meta_path = filepath.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "label": label,
                    **metadata
                }, f, indent=2)
        
        return {
            "success": True,
            "filepath": str(filepath),
            "filename": filename,
        }
    except Exception as e:
        logger.error("Failed to save evidence: %s", e)
        return {"success": False, "error": str(e)}


def _tool_log_event(
    event_type: str,
    description: str,
    severity: str = "info",
    details: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Log an event to the persistent database.
    
    Args:
        event_type: Type of event (detection, alert, system, etc.)
        description: Human-readable description
        severity: Event severity (debug, info, warning, error, critical)
        details: Additional structured details
    
    Returns:
        Dictionary with log entry ID and status
    """
    db_path = Path(os.getenv("CAMERA_AGENT_DB", "camera_agent.db"))
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                details TEXT
            )
        """)
        
        cursor.execute(
            """INSERT INTO tool_events (timestamp, event_type, severity, description, details)
               VALUES (?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                event_type,
                severity,
                description,
                json.dumps(details) if details else None,
            )
        )
        
        log_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "log_id": log_id,
            "event_type": event_type,
        }
    except Exception as e:
        logger.error("Failed to log event: %s", e)
        return {"success": False, "error": str(e)}


def _tool_notify_desktop(
    title: str,
    message: str,
    timeout: int = 10,
    urgency: str = "normal",
    **kwargs
) -> Dict[str, Any]:
    """Show a desktop notification.
    
    Args:
        title: Notification title
        message: Notification message body
        timeout: How long to show (seconds)
        urgency: Notification urgency (low, normal, critical)
    
    Returns:
        Dictionary with status
    """
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            timeout=timeout,
            app_name="Camera Agent",
        )
        return {"success": True, "message": "Desktop notification shown"}
    except ImportError:
        return {"success": False, "error": "Desktop notifications not available (plyer not installed)"}
    except Exception as e:
        logger.error("Failed to show desktop notification: %s", e)
        return {"success": False, "error": str(e)}


def _tool_query_history(
    limit: int = 10,
    event_type: Optional[str] = None,
    since_minutes: int = 60,
    **kwargs
) -> Dict[str, Any]:
    """Query recent detection history for context.
    
    Args:
        limit: Maximum number of events to return
        event_type: Filter by event type (optional)
        since_minutes: Only return events from last N minutes
    
    Returns:
        Dictionary with recent events
    """
    db_path = Path(os.getenv("CAMERA_AGENT_DB", "camera_agent.db"))
    
    if not db_path.exists():
        return {"success": True, "events": [], "count": 0}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Try to read from logs table
        query = """
            SELECT timestamp, event_type, details 
            FROM logs 
            WHERE timestamp >= datetime('now', ?)
        """
        params: List[Any] = [f"-{since_minutes} minutes"]
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            rows = []
        
        conn.close()
        
        events = [
            {
                "timestamp": row[0],
                "event_type": row[1],
                "details": json.loads(row[2]) if row[2] else {},
            }
            for row in rows
        ]
        
        return {
            "success": True,
            "events": events,
            "count": len(events),
        }
    except Exception as e:
        logger.error("Failed to query history: %s", e)
        return {"success": False, "error": str(e), "events": []}


def _tool_analyze_trend(
    metric: str = "detections",
    window_minutes: int = 30,
    **kwargs
) -> Dict[str, Any]:
    """Analyze detection trends over time.
    
    Args:
        metric: What to analyze (detections, alerts, etc.)
        window_minutes: Time window to analyze
    
    Returns:
        Dictionary with trend analysis
    """
    history = _tool_query_history(limit=100, since_minutes=window_minutes)
    
    if not history.get("success"):
        return history
    
    events = history.get("events", [])
    
    detection_count = sum(1 for e in events if e.get("event_type") == "detection")
    alert_count = sum(1 for e in events if e.get("event_type") == "alert")
    
    return {
        "success": True,
        "window_minutes": window_minutes,
        "total_events": len(events),
        "detection_count": detection_count,
        "alert_count": alert_count,
        "detections_per_minute": round(detection_count / max(window_minutes, 1), 2),
        "alerts_per_minute": round(alert_count / max(window_minutes, 1), 2),
    }


# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOL_REGISTRY: Dict[str, ToolDefinition] = {
    "send_alert_email": ToolDefinition(
        name="send_alert_email",
        description="Send an alert email to notify operators about a detection. Use when something important is detected that requires human attention.",
        parameters={
            "recipients": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of email addresses to send the alert to"
            },
            "subject": {
                "type": "string",
                "description": "Email subject line describing the alert"
            },
            "body": {
                "type": "string",
                "description": "Detailed email body explaining what was detected and why it matters"
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "Alert priority level"
            },
            "include_image": {
                "type": "boolean",
                "description": "Whether to attach the detection image"
            }
        },
        required_params=["recipients", "subject", "body"],
        handler=_tool_send_alert_email,
    ),
    
    "save_evidence": ToolDefinition(
        name="save_evidence",
        description="Save the current frame as evidence for later review. Use when you detect something that should be documented.",
        parameters={
            "label": {
                "type": "string",
                "description": "Label describing what was detected (e.g., 'unlabeled_box', 'ppe_violation')"
            },
            "metadata": {
                "type": "object",
                "description": "Additional metadata to store with the evidence"
            }
        },
        required_params=["label"],
        handler=_tool_save_evidence,
    ),
    
    "log_event": ToolDefinition(
        name="log_event",
        description="Log an event to the system for tracking and auditing. Use for any notable observations.",
        parameters={
            "event_type": {
                "type": "string",
                "enum": ["detection", "alert", "observation", "system"],
                "description": "Type of event being logged"
            },
            "description": {
                "type": "string",
                "description": "Human-readable description of the event"
            },
            "severity": {
                "type": "string",
                "enum": ["debug", "info", "warning", "error", "critical"],
                "description": "Severity level of the event"
            },
            "details": {
                "type": "object",
                "description": "Structured details about the event"
            }
        },
        required_params=["event_type", "description"],
        handler=_tool_log_event,
    ),
    
    "notify_desktop": ToolDefinition(
        name="notify_desktop",
        description="Show a desktop notification to alert the local operator immediately.",
        parameters={
            "title": {
                "type": "string",
                "description": "Notification title"
            },
            "message": {
                "type": "string",
                "description": "Notification message"
            },
            "urgency": {
                "type": "string",
                "enum": ["low", "normal", "critical"],
                "description": "Notification urgency level"
            }
        },
        required_params=["title", "message"],
        handler=_tool_notify_desktop,
    ),
    
    "query_history": ToolDefinition(
        name="query_history",
        description="Query recent detection history to understand patterns or check if similar events occurred recently.",
        parameters={
            "limit": {
                "type": "integer",
                "description": "Maximum number of events to retrieve"
            },
            "event_type": {
                "type": "string",
                "description": "Filter by event type"
            },
            "since_minutes": {
                "type": "integer",
                "description": "Only return events from the last N minutes"
            }
        },
        required_params=[],
        handler=_tool_query_history,
    ),
    
    "analyze_trend": ToolDefinition(
        name="analyze_trend",
        description="Analyze detection trends over a time window to understand patterns.",
        parameters={
            "metric": {
                "type": "string",
                "description": "What metric to analyze"
            },
            "window_minutes": {
                "type": "integer",
                "description": "Time window in minutes to analyze"
            }
        },
        required_params=[],
        handler=_tool_analyze_trend,
    ),
}


# =============================================================================
# TOOL EXECUTOR
# =============================================================================

class ToolExecutor:
    """Executes tool calls from the LLM.
    
    This class manages the tool registry and handles execution of
    tool calls, including validation and error handling.
    """
    
    def __init__(
        self,
        tools: Optional[Dict[str, ToolDefinition]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the tool executor.
        
        Args:
            tools: Custom tool registry (uses default if not provided)
            context: Shared context passed to all tool calls (e.g., current image)
        """
        self.tools = tools or TOOL_REGISTRY.copy()
        self.context = context or {}
        self.call_history: List[ToolResult] = []
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get JSON schema for all available tools."""
        return [tool.to_schema() for tool in self.tools.values()]
    
    def get_tools_prompt(self) -> str:
        """Generate a prompt describing available tools."""
        lines = ["You have access to the following tools:\n"]
        
        for tool in self.tools.values():
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}\n")
            lines.append("Parameters:")
            for param_name, param_info in tool.parameters.items():
                required = "(required)" if param_name in tool.required_params else "(optional)"
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                lines.append(f"  - {param_name} ({param_type}) {required}: {param_desc}")
            lines.append("")
        
        lines.append("""
To use a tool, include a JSON block in your response with this format:
```tool_call
{
  "tool": "tool_name",
  "arguments": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

You can call multiple tools by including multiple tool_call blocks.
After analyzing the image, decide which tools (if any) should be called based on what you observe.
""")
        
        return "\n".join(lines)
    
    def update_context(self, **kwargs) -> None:
        """Update the shared context with new values."""
        self.context.update(kwargs)
    
    def parse_tool_calls(self, response: str) -> List[ToolCall]:
        """Parse tool calls from LLM response.
        
        Args:
            response: Raw LLM response text
        
        Returns:
            List of parsed tool calls
        """
        import re
        
        tool_calls: List[ToolCall] = []
        
        # Find all tool_call blocks
        pattern = r"```tool_call\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                tool_name = data.get("tool", "")
                arguments = data.get("arguments", {})
                
                if tool_name and tool_name in self.tools:
                    tool_calls.append(ToolCall(
                        tool_name=tool_name,
                        arguments=arguments,
                    ))
                else:
                    logger.warning("Unknown tool requested: %s", tool_name)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse tool call: %s", e)
        
        return tool_calls
    
    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call.
        
        Args:
            tool_call: The tool call to execute
        
        Returns:
            Result of the tool execution
        """
        tool = self.tools.get(tool_call.tool_name)
        
        if not tool:
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_call.tool_name}",
            )
        
        if not tool.handler:
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=False,
                result=None,
                error=f"Tool has no handler: {tool_call.tool_name}",
            )
        
        # Validate required parameters
        for param in tool.required_params:
            if param not in tool_call.arguments:
                return ToolResult(
                    tool_name=tool_call.tool_name,
                    call_id=tool_call.call_id,
                    success=False,
                    result=None,
                    error=f"Missing required parameter: {param}",
                )
        
        # Merge context into arguments
        merged_args = {**self.context, **tool_call.arguments}
        
        try:
            result = tool.handler(**merged_args)
            tool_result = ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=result.get("success", True),
                result=result,
                error=result.get("error"),
            )
        except Exception as e:
            logger.exception("Tool execution failed: %s", tool_call.tool_name)
            tool_result = ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=False,
                result=None,
                error=str(e),
            )
        
        self.call_history.append(tool_result)
        return tool_result
    
    def execute_all(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls.
        
        Args:
            tool_calls: List of tool calls to execute
        
        Returns:
            List of results from all executions
        """
        return [self.execute(call) for call in tool_calls]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the tool call history."""
        return [r.to_dict() for r in self.call_history]
    
    def clear_history(self) -> None:
        """Clear the tool call history."""
        self.call_history.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ToolExecutor",
    "TOOL_REGISTRY",
    "AgenticResult",
]


# Import AgenticResult from unified_vlm to avoid circular imports
# This is re-exported here for convenience
def _get_agentic_result():
    """Lazy import to avoid circular dependency."""
    from agent_runtime.unified_vlm import AgenticResult
    return AgenticResult

# Make AgenticResult available at module level
try:
    from agent_runtime.unified_vlm import AgenticResult
except ImportError:
    AgenticResult = None  # type: ignore
