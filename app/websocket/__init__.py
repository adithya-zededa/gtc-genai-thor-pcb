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
    
    # Register chat handlers
    from app.websocket.chat import register_chat_handlers, initialize_chat_for_client
    
    register_chat_handlers(socketio)
    
    @socketio.on("connect")
    def handle_connect():
        """Handle client connection."""
        logger.info("Client connected: %s", request.sid)
        # Auto-initialize chat session on connect
        try:
            initialize_chat_for_client(request.sid)
        except Exception as e:
            logger.error("Failed to initialize chat for client: %s", e, exc_info=True)
            emit("chat_error", {"error": str(e)})
    
    @socketio.on("test_event")
    def handle_test(data=None):
        """Test event handler."""
        logger.info(">>> TEST EVENT received from %s with data: %s", request.sid, data)
        emit("test_response", {"message": "Test received!"})

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
    
    # Emit to chat interface
    from app.websocket.chat import emit_detection_to_chat
    
    emit_detection_to_chat(socketio, event_data)


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
