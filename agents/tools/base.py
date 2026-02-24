"""Tool calling system for the agentic camera monitoring agent."""

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

from core.logging import get_logger

logger = get_logger(__name__)


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


# Tool implementations
def _tool_send_alert_email(
    recipients: List[str],
    subject: str,
    body: str,
    priority: str = "normal",
    include_image: bool = True,
    image_data: Optional[bytes] = None,
    **kwargs
) -> Dict[str, Any]:
    """Send an alert email to specified recipients."""
    from agents.tools.email import send_email
    
    if not recipients:
        return {"success": False, "error": "No recipients specified"}
    
    payload: Dict[str, Any] = {
        "to": recipients,
        "subject": subject,
        "body": body,
    }
    
    if include_image is not False and image_data:
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
    """Save detection image as evidence."""
    evidence_dir = Path(os.getenv("DETECTED_IMAGES_DIR", "detected_images"))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
    filename = f"{safe_label}_{timestamp}.jpg"
    filepath = evidence_dir / filename
    
    try:
        with open(filepath, "wb") as f:
            f.write(image_data)
        
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

    Records the event in the ``detection_logs`` table and also logs it via
    the application logger so it appears both in the database and in the
    console / log file for operational visibility.
    """
    try:
        from app.database import DetectionLogRepository
        
        log_id = DetectionLogRepository.create(
            timestamp=datetime.now().isoformat(),
            confidence=0.0,
            response=description,
            reason=event_type,
            vision_description=description,
            decision_details=details or {},
        )
        
        logger.info(
            "Event logged: id=%d type=%s severity=%s desc=%s",
            log_id, event_type, severity, description[:120],
        )
        
        return {
            "success": True,
            "log_id": log_id,
            "event_type": event_type,
        }
    except Exception as e:
        logger.error("Failed to log event (type=%s): %s", event_type, e)
        return {"success": False, "error": str(e)}


def _tool_query_history(
    limit: int = 10,
    detected_only: bool = False,
    include_defects: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Query recent detection history with defect information for context.

    Retrieves detection logs from the ``detection_logs`` table and, when
    ``include_defects`` is True (the default), enriches the response with
    recent entries from the ``pcb_defects`` table so the caller gets a
    unified view of what has been detected and logged.
    """
    try:
        from app.database import DetectionLogRepository
        
        logs, total = DetectionLogRepository.get_paginated(
            page=1,
            per_page=limit,
            detected_only=detected_only,
        )
        log_entries = [log.to_dict() for log in logs]

        result: Dict[str, Any] = {
            "success": True,
            "count": len(log_entries),
            "total": total,
            "events": log_entries,
        }

        # Enrich with defect data from pcb_defects table
        if include_defects:
            try:
                from services.domains.pcb.defect_store import (
                    count_defects,
                    get_defects_in_range,
                    get_latest_defect,
                )

                recent_defects = get_defects_in_range(limit=limit)
                total_defects = count_defects()
                latest_defect = get_latest_defect()
                defects_24h = count_defects(hours=24)

                result["defects"] = {
                    "recent": recent_defects,
                    "total_defects": total_defects,
                    "defects_last_24h": defects_24h,
                    "latest_defect": latest_defect,
                }
                logger.info(
                    "query_history: %d detection log(s), %d defect(s) total, %d in last 24h",
                    len(log_entries), total_defects, defects_24h,
                )
            except Exception as defect_exc:
                logger.warning("Failed to enrich history with defect data: %s", defect_exc)
                result["defects"] = {"error": str(defect_exc)}
        
        return result
    except Exception as e:
        logger.error("Failed to query history: %s", e)
        return {"success": False, "error": str(e)}


# Tool registry
TOOL_REGISTRY: Dict[str, ToolDefinition] = {
    "send_alert_email": ToolDefinition(
        name="send_alert_email",
        description="Send an email alert to specified recipients with optional image attachment",
        parameters={
            "recipients": {"type": "array", "items": {"type": "string"}, "description": "Email addresses"},
            "subject": {"type": "string", "description": "Email subject line"},
            "body": {"type": "string", "description": "Email body content"},
            "priority": {"type": "string", "enum": ["low", "normal", "high"], "description": "Email priority"},
            "include_image": {"type": "boolean", "description": "Whether to attach detection image"},
        },
        required_params=["recipients", "subject", "body"],
        handler=_tool_send_alert_email,
    ),
    "save_evidence": ToolDefinition(
        name="save_evidence",
        description="Save a detection image as evidence for later review",
        parameters={
            "label": {"type": "string", "description": "Label for the evidence"},
            "metadata": {"type": "object", "description": "Additional metadata to store"},
        },
        required_params=[],
        handler=_tool_save_evidence,
    ),
    "log_event": ToolDefinition(
        name="log_event",
        description="Log an event to the persistent audit log",
        parameters={
            "event_type": {"type": "string", "description": "Type of event"},
            "description": {"type": "string", "description": "Event description"},
            "severity": {"type": "string", "enum": ["info", "warning", "error"], "description": "Severity level"},
            "details": {"type": "object", "description": "Additional event details"},
        },
        required_params=["event_type", "description"],
        handler=_tool_log_event,
    ),
    "query_history": ToolDefinition(
        name="query_history",
        description="Query recent detection history for context",
        parameters={
            "limit": {"type": "integer", "description": "Maximum number of events to return"},
            "detected_only": {"type": "boolean", "description": "Only return events with detections"},
        },
        required_params=[],
        handler=_tool_query_history,
    ),
}


class ToolExecutor:
    """Executes tool calls requested by the LLM."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """Initialize with optional context (e.g., current image data)."""
        self.context = context or {}
    
    def update_context(self, **kwargs) -> None:
        """Update the execution context with new values."""
        self.context.update(kwargs)
    
    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        tool_def = TOOL_REGISTRY.get(tool_call.tool_name)
        
        if not tool_def:
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_call.tool_name}",
            )
        
        if not tool_def.handler:
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=False,
                result=None,
                error=f"Tool has no handler: {tool_call.tool_name}",
            )
        
        try:
            # Merge context with arguments
            args = {**self.context, **tool_call.arguments}
            result = tool_def.handler(**args)
            
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=result.get("success", True),
                result=result,
                error=result.get("error"),
            )
        except Exception as e:
            logger.error("Tool execution failed: %s - %s", tool_call.tool_name, e)
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=False,
                result=None,
                error=str(e),
            )
    
    def execute_many(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls."""
        return [self.execute(call) for call in tool_calls]
