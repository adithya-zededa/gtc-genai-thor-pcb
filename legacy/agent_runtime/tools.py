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
    include_image: bool = True,
    image_data: Optional[bytes] = None,
    **kwargs
) -> Dict[str, Any]:
    """Send an alert email to specified recipients.
    
    Args:
        recipients: List of email addresses to send to
        subject: Email subject line
        body: Email body content
        priority: Email priority (low, normal, high)
        include_image: Whether to attach the detection image (default True)
        image_data: Raw image bytes to attach
    
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
    
    # Always attach image when available (unless explicitly disabled)
    if include_image is not False and image_data:
        payload["image_data"] = image_data
        payload["image_filename"] = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        logger.info("Attaching image to email (%d bytes)", len(image_data))
    
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
    
    Queries both detection_logs (camera detections) and tool_events (agent actions)
    to provide the LLM with full context of prior events.
    
    Args:
        limit: Maximum number of events to return
        event_type: Filter by event type (optional). Use 'detection' for camera
                   detections, 'tool' for agent tool invocations, or specific
                   event types like 'alert', 'system', etc.
        since_minutes: Only return events from last N minutes
    
    Returns:
        Dictionary with recent events from both tables
    """
    db_path = Path(os.getenv("CAMERA_AGENT_DB", "camera_agent.db"))
    
    if not db_path.exists():
        return {"success": True, "events": [], "count": 0}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        events = []
        
        # Query detection_logs table (camera detections)
        if event_type is None or event_type == "detection":
            try:
                detection_query = """
                    SELECT timestamp, 'detection' as event_type, 
                           confidence, response, reason, vision_description
                    FROM detection_logs 
                    WHERE timestamp >= datetime('now', ?)
                    ORDER BY timestamp DESC LIMIT ?
                """
                cursor.execute(detection_query, [f"-{since_minutes} minutes", limit])
                for row in cursor.fetchall():
                    events.append({
                        "timestamp": row[0],
                        "event_type": "detection",
                        "details": {
                            "confidence": row[2],
                            "response": row[3],
                            "reason": row[4],
                            "vision_description": row[5],
                        },
                    })
            except sqlite3.OperationalError:
                # Table doesn't exist yet
                pass
        
        # Query tool_events table (agent tool invocations)
        if event_type is None or event_type == "tool" or (event_type and event_type not in ["detection"]):
            try:
                tool_query = """
                    SELECT timestamp, event_type, severity, description, details
                    FROM tool_events 
                    WHERE timestamp >= datetime('now', ?)
                """
                params: List[Any] = [f"-{since_minutes} minutes"]
                
                if event_type and event_type != "tool":
                    tool_query += " AND event_type = ?"
                    params.append(event_type)
                
                tool_query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(tool_query, params)
                for row in cursor.fetchall():
                    events.append({
                        "timestamp": row[0],
                        "event_type": row[1],
                        "details": {
                            "severity": row[2],
                            "description": row[3],
                            **(json.loads(row[4]) if row[4] else {}),
                        },
                    })
            except sqlite3.OperationalError:
                # Table doesn't exist yet - create it for future use
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
                conn.commit()
        
        conn.close()
        
        # Sort combined events by timestamp descending and limit
        events.sort(key=lambda x: x["timestamp"] or "", reverse=True)
        events = events[:limit]
        
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
        description="Send an alert email to notify operators about a detection. Use when something important or safety-critical is detected that requires immediate human attention (e.g., unauthorized person, safety violation, equipment malfunction).",
        parameters={
            "recipients": {
                "type": "array",
                "items": {"type": "string", "format": "email"},
                "description": "List of email addresses to send the alert to.",
                "example": ["operator@example.com", "supervisor@example.com"]
            },
            "subject": {
                "type": "string",
                "description": "Concise email subject line describing the alert.",
                "example": "[ALERT] Unauthorized Person Detected in Warehouse Zone A"
            },
            "body": {
                "type": "string",
                "description": "Detailed email body explaining: (1) what was detected, (2) where/when, (3) why it matters, (4) recommended action.",
                "example": "An unauthorized person was detected in Warehouse Zone A at 14:32:15. The individual is not wearing required PPE and appears to be near restricted equipment. Immediate verification recommended."
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "default": "normal",
                "description": "Alert priority: 'low' for informational, 'normal' for standard alerts, 'high' for safety-critical issues requiring immediate attention."
            },
            "include_image": {
                "type": "boolean",
                "default": True,
                "description": "Whether to attach the detection image. Defaults to true - images are always attached unless explicitly set to false."
            }
        },
        required_params=["recipients", "subject", "body"],
        handler=_tool_send_alert_email,
    ),
    
    "save_evidence": ToolDefinition(
        name="save_evidence",
        description="Save the current camera frame as timestamped evidence for audit trails and later review. Use whenever you detect something noteworthy that should be documented, even if no immediate alert is needed.",
        parameters={
            "label": {
                "type": "string",
                "description": "Short descriptive label for the evidence. Use snake_case. This becomes part of the filename.",
                "example": "unlabeled_box",
                "enum_suggestion": ["unlabeled_box", "ppe_violation", "unauthorized_access", "equipment_anomaly", "spill_detected", "safety_hazard"]
            },
            "metadata": {
                "type": "object",
                "description": "Structured metadata to store alongside the image for searchability and context.",
                "properties": {
                    "location": {"type": "string", "description": "Camera location or zone"},
                    "confidence": {"type": "number", "description": "Detection confidence 0.0-1.0"},
                    "objects_detected": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string", "description": "Additional observations"}
                },
                "example": {"location": "Zone A", "confidence": 0.87, "objects_detected": ["person", "forklift"], "notes": "Person not wearing hard hat near active forklift"}
            }
        },
        required_params=["label"],
        handler=_tool_save_evidence,
    ),
    
    "log_event": ToolDefinition(
        name="log_event",
        description="Log an event to the persistent database for tracking, auditing, and trend analysis. Use for any notable observation even if no action is taken. Good logging enables pattern detection over time.",
        parameters={
            "event_type": {
                "type": "string",
                "enum": ["detection", "alert", "observation", "system"],
                "description": "Event category: 'detection' for objects/people found, 'alert' when notification sent, 'observation' for notable but non-actionable items, 'system' for operational events."
            },
            "description": {
                "type": "string",
                "description": "Clear, human-readable description of what happened. Be specific about what, where, and relevant context.",
                "example": "Forklift operator detected without safety vest in loading dock area"
            },
            "severity": {
                "type": "string",
                "enum": ["debug", "info", "warning", "error", "critical"],
                "default": "info",
                "description": "Severity level: 'debug' for verbose details, 'info' for normal events, 'warning' for potential issues, 'error' for problems, 'critical' for immediate safety concerns."
            },
            "details": {
                "type": "object",
                "description": "Structured key-value data for filtering and analysis.",
                "properties": {
                    "camera_id": {"type": "string"},
                    "zone": {"type": "string"},
                    "object_count": {"type": "integer"},
                    "action_taken": {"type": "string"},
                    "related_alert_id": {"type": "string"}
                },
                "example": {"camera_id": "cam-01", "zone": "loading_dock", "object_count": 2, "action_taken": "email_sent"}
            }
        },
        required_params=["event_type", "description"],
        handler=_tool_log_event,
    ),
    
    "notify_desktop": ToolDefinition(
        name="notify_desktop",
        description="Show a desktop notification popup to alert the local operator immediately. Use for time-sensitive alerts when someone is monitoring the system locally. Note: Only works when desktop environment is available.",
        parameters={
            "title": {
                "type": "string",
                "description": "Short, attention-grabbing title (max ~50 chars). Should convey urgency level.",
                "example": "⚠️ Safety Violation Detected"
            },
            "message": {
                "type": "string",
                "description": "Brief message with key details. Keep concise as notification space is limited (~150 chars).",
                "example": "Person without PPE detected in Zone B. Check camera feed."
            },
            "urgency": {
                "type": "string",
                "enum": ["low", "normal", "critical"],
                "default": "normal",
                "description": "Notification urgency: 'low' may not interrupt user, 'normal' standard popup, 'critical' may override do-not-disturb."
            }
        },
        required_params=["title", "message"],
        handler=_tool_notify_desktop,
    ),
    
    "query_history": ToolDefinition(
        name="query_history",
        description="Query recent detection history from the database. Retrieves events from both detection_logs (camera detections) and tool_events (agent actions). Use to: (1) check if similar events occurred recently to avoid duplicate alerts, (2) understand patterns before deciding on action, (3) provide context in alert messages.",
        parameters={
            "limit": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
                "description": "Maximum number of events to retrieve. Use smaller values for quick checks, larger for pattern analysis."
            },
            "event_type": {
                "type": "string",
                "enum": ["detection", "tool", "alert", "observation", "system"],
                "description": "Filter to only return events of this type. 'detection' for camera detections, 'tool' for agent tool invocations. Omit to get all types."
            },
            "since_minutes": {
                "type": "integer",
                "default": 60,
                "minimum": 1,
                "maximum": 1440,
                "description": "Time window in minutes. Only events from the last N minutes are returned. Max 1440 (24 hours)."
            }
        },
        required_params=[],
        handler=_tool_query_history,
    ),
    
    "analyze_trend": ToolDefinition(
        name="analyze_trend",
        description="Analyze detection trends over a time window. Use to understand if current conditions are unusual compared to recent history. Helps decide if an alert is warranted or if this is normal activity.",
        parameters={
            "metric": {
                "type": "string",
                "enum": ["detections", "alerts"],
                "default": "detections",
                "description": "What to analyze: 'detections' for all detected events, 'alerts' for only events that triggered notifications."
            },
            "window_minutes": {
                "type": "integer",
                "default": 30,
                "minimum": 5,
                "maximum": 1440,
                "description": "Time window in minutes to analyze. Shorter windows (5-30) for recent activity, longer (60-1440) for daily patterns."
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
        lines = ["## Available Tools\n"]
        lines.append("You have access to the following tools to take actions based on your observations:\n")
        
        for tool in self.tools.values():
            lines.append(f"### `{tool.name}`")
            lines.append(f"**Purpose:** {tool.description}\n")
            
            # Separate required and optional params for clarity
            required_params = [(k, v) for k, v in tool.parameters.items() if k in tool.required_params]
            optional_params = [(k, v) for k, v in tool.parameters.items() if k not in tool.required_params]
            
            if required_params:
                lines.append("**Required Parameters:**")
                for param_name, param_info in required_params:
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    enum_vals = param_info.get("enum", [])
                    example = param_info.get("example", "")
                    
                    line = f"- `{param_name}` ({param_type}): {param_desc}"
                    if enum_vals:
                        line += f" Allowed values: {enum_vals}"
                    if example:
                        line += f" Example: `{json.dumps(example) if isinstance(example, (dict, list)) else example}`"
                    lines.append(line)
                lines.append("")
            
            if optional_params:
                lines.append("**Optional Parameters:**")
                for param_name, param_info in optional_params:
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    default = param_info.get("default", "")
                    enum_vals = param_info.get("enum", [])
                    
                    line = f"- `{param_name}` ({param_type}): {param_desc}"
                    if enum_vals:
                        line += f" Allowed: {enum_vals}"
                    if default != "":
                        line += f" Default: `{default}`"
                    lines.append(line)
                lines.append("")
        
        lines.append("""---
## How to Call Tools

To invoke a tool, include a JSON code block with the `tool_call` language tag:

```tool_call
{
  "tool": "tool_name",
  "arguments": {
    "required_param": "value",
    "optional_param": "value"
  }
}
```

### Rules:
1. You may call **multiple tools** by including multiple `tool_call` blocks
2. Only include parameters you need - omit optional params to use defaults
3. Ensure all **required parameters** are provided
4. Use the exact parameter names and types specified

### Decision Guidelines:
- **save_evidence**: Always save when you detect something noteworthy
- **log_event**: Log all significant observations for audit trails
- **query_history**: Check history before alerting to avoid duplicates
- **send_alert_email**: Use for safety-critical or policy violations requiring human action
- **notify_desktop**: Use for immediate local operator attention
- **analyze_trend**: Use when you need context about whether activity is unusual

### Example:
If you detect a person without safety equipment, you might:
```tool_call
{
  "tool": "save_evidence",
  "arguments": {
    "label": "ppe_violation",
    "metadata": {"zone": "loading_dock", "violation_type": "missing_hard_hat"}
  }
}
```

```tool_call
{
  "tool": "log_event",
  "arguments": {
    "event_type": "detection",
    "description": "Person detected without hard hat in loading dock area",
    "severity": "warning"
  }
}
```
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
