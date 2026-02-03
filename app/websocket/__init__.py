"""WebSocket event handlers for real-time communication."""

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING

from flask import request
from flask_socketio import emit

from core.logging import get_logger

if TYPE_CHECKING:
    from flask_socketio import SocketIO

logger = get_logger(__name__)


def register_handlers(socketio: "SocketIO") -> None:
    """Register all WebSocket event handlers."""
    
    @socketio.on("connect")
    def handle_connect():
        """Handle client connection."""
        logger.info("Client connected: %s", request.sid)
        emit("connected", {"status": "ok", "message": "Connected to camera agent"})

    @socketio.on("disconnect")
    def handle_disconnect():
        """Handle client disconnection."""
        logger.info("Client disconnected: %s", request.sid)

    @socketio.on("ping")
    def handle_ping():
        """Handle keepalive ping."""
        emit("pong", {"timestamp": time.time()})

    @socketio.on("subscribe_monitoring")
    def handle_subscribe_monitoring(data=None):
        """Subscribe to monitoring updates."""
        logger.info("Client %s subscribed to monitoring", request.sid)
        emit("subscribed", {"channel": "monitoring"})

    @socketio.on("unsubscribe_monitoring")
    def handle_unsubscribe_monitoring(data=None):
        """Unsubscribe from monitoring updates."""
        logger.info("Client %s unsubscribed from monitoring", request.sid)
        emit("unsubscribed", {"channel": "monitoring"})


def emit_detection_event(socketio: "SocketIO", event_data: dict) -> None:
    """Emit a detection event to all connected clients."""
    socketio.emit("detection_event", event_data)


def emit_frame_update(socketio: "SocketIO", frame_bytes: bytes, metadata: dict = None) -> None:
    """Emit a frame update to connected clients."""
    encoded = base64.b64encode(frame_bytes).decode("utf-8")
    socketio.emit("frame_update", {
        "image": encoded,
        "timestamp": time.time(),
        "metadata": metadata or {},
    })


def emit_monitoring_status(socketio: "SocketIO", status: dict) -> None:
    """Emit monitoring status update."""
    socketio.emit("monitoring_status", status)


def emit_alert(socketio: "SocketIO", alert_data: dict) -> None:
    """Emit an alert notification."""
    socketio.emit("alert", alert_data)
