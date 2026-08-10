"""Socket.IO binding for the conversation layer.

This module is transport only. The turn itself — interpretation, domain
routing, proposal submission, response generation — lives in
``agents.conversation`` and is shared with any other surface that wants
it. What remains here is:

* :class:`SocketIOEventSink`, which turns the orchestrator's progress and
  approval notifications into socket events;
* the ``@socketio.on`` handlers;
* session bootstrap on connect.

Key principles (unchanged):
1. Language is the ONLY control surface
2. Tools are PROPOSED, then APPROVED, then EXECUTED
3. All actions are logged and auditable
4. UI suggests but never directly executes
"""

# pylint: disable=import-outside-toplevel,missing-function-docstring,broad-exception-caught,unused-argument

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from flask import request
from flask_socketio import emit, join_room, leave_room

from agents.conversation import (
    ChatMessage,
    ChatSession,
    ConversationEventSink,
    ConversationOrchestrator,
    friendly_tool_name,
    generate_welcome_message,
    get_chat_session,
    get_or_bind_chat_session,
    get_session_manager,
    issue_session_token,
)
from agents.mcp.base import (
    AgentState,
    AuditEventType,
    AuditLogEntry,
    MCPExecutor,
    get_agent_state_machine,
    get_audit_log,
    get_mcp_executor,
    get_tool_registry,
)
from agents.mcp.manager import get_mcp_manager
from core.logging import get_logger

if TYPE_CHECKING:
    from flask_socketio import SocketIO

logger = get_logger(__name__)


# =============================================================================
# EVENT SINK
# =============================================================================


class SocketIOEventSink(ConversationEventSink):
    """Delivers a turn's side-channel events to one connected client.

    Every emit is scoped to the originating socket's room. A turn belongs
    to the client that started it; broadcasting its progress to everyone
    would leak one operator's activity into another's chat pane. The one
    exception is ``state_changed``, which is genuinely global — the agent
    state machine is process-wide.
    """

    def __init__(self, socketio: "SocketIO", session_id: str) -> None:
        self._socketio = socketio
        self._session_id = session_id

    def _room(self, chat_session: Optional[ChatSession]) -> str:
        return chat_session.socket_session_id if chat_session else self._session_id

    def activity(
        self,
        *,
        status: str,
        title: str,
        detail: str,
        tool_name: Optional[str] = None,
        chat_session: Optional[ChatSession] = None,
    ) -> None:
        payload = {
            "status": status,
            "title": title,
            "detail": detail,
            "tool_name": tool_name,
            "display_name": friendly_tool_name(tool_name) if tool_name else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if chat_session is not None:
            # Activity lines are part of the transcript so a restored
            # session still shows what the agent did, not just what it said.
            chat_session.add_message(
                ChatMessage.system(
                    detail,
                    metadata={"ui": "activity", "activity": payload},
                )
            )

        self._socketio.emit("agent_activity", payload, room=self._room(chat_session))

    def pending_proposal(
        self,
        result: Dict[str, Any],
        *,
        chat_session: Optional[ChatSession] = None,
    ) -> None:
        self._socketio.emit(
            "tool_confirmation_required",
            {
                "proposal_id": result["proposal_id"],
                "tool_name": result["tool_name"],
                "confirmation_message": result["confirmation_message"],
                "proposal": result["proposal"],
            },
            room=self._room(chat_session),
        )

    def state_changed(self) -> None:
        _broadcast_state_update(self._socketio, get_mcp_executor())

    def message(
        self,
        message: ChatMessage,
        *,
        chat_session: Optional[ChatSession] = None,
    ) -> None:
        self._socketio.emit(
            "chat_message",
            {
                "message": message.to_dict(),
                "agent_state": get_agent_state_machine().state.value,
            },
            room=self._room(chat_session),
        )


def _orchestrator_for(socketio: "SocketIO", session_id: str) -> ConversationOrchestrator:
    """Build an orchestrator wired to one client's socket."""
    return ConversationOrchestrator(SocketIOEventSink(socketio, session_id))


# =============================================================================
# SESSION BOOTSTRAP
# =============================================================================


def _initialize_session(
    session_id: str,
    *,
    room: str | None = None,
    client_session_id: Optional[str] = None,
    session_token: Optional[str] = None,
) -> None:
    """Core chat-session bootstrap shared by all connection paths.

    Args:
        session_id: The SocketIO ``request.sid``.
        room: Explicit room for ``emit``.  When *None* the default
              SocketIO behaviour (emit to current request) is used.
    """
    audit_log = get_audit_log()
    state_machine = get_agent_state_machine()

    chat_session = get_or_bind_chat_session(
        session_id, client_session_id, session_token
    )
    issued_token = (
        issue_session_token(chat_session.client_session_id)
        if chat_session.client_session_id
        else None
    )
    join_room(session_id)
    history = chat_session.get_history(limit=0)
    is_restored_session = len(history) > 0

    # Log connection
    audit_log.log(
        AuditLogEntry.create(
            event_type=AuditEventType.SESSION_STARTED,
            details={
                "socket_session_id": session_id,
                "chat_session_id": chat_session.id,
                "client_session_id": chat_session.client_session_id,
            },
        )
    )

    # Transition from OFF → IDLE on first connection
    current_state = state_machine.state
    if current_state == AgentState.OFF:
        try:
            state_machine.transition_to(
                AgentState.IDLE,
                trigger="user_connected",
                metadata={"session_id": session_id},
            )
            current_state = state_machine.state
            logger.info("Agent transitioned to IDLE on connection")
        except ValueError as e:
            logger.warning("Could not transition to IDLE: %s", e)

    executor = get_mcp_executor()
    mcp_session = executor.current_session

    registry = get_tool_registry()
    available_tools = registry.get_display_list(current_state)

    from services.core.monitoring import get_monitoring_service

    monitoring_service = get_monitoring_service()
    monitoring_active = (
        monitoring_service.is_monitoring if monitoring_service else False
    )

    emit_kwargs = {"room": room} if room else {}

    emit(
        "chat_connected",
        {
            "chat_session_id": chat_session.id,
            "client_session_id": chat_session.client_session_id,
            "session_token": issued_token,
            "agent_state": current_state.value,
            "available_tools": available_tools,
            "pending_proposals": executor.get_pending_proposals(),
            "mcp_session": mcp_session.to_dict() if mcp_session else None,
            "metrics": audit_log.get_metrics(),
            "monitoring_active": monitoring_active,
            "history": history,
            "restored": is_restored_session,
        },
        **emit_kwargs,
    )

    # Send welcome message only for brand-new sessions
    if not is_restored_session:
        welcome = ChatMessage.assistant(generate_welcome_message(current_state))
        chat_session.add_message(welcome)

        emit(
            "chat_message",
            {
                "message": welcome.to_dict(),
                "agent_state": current_state.value,
            },
            **emit_kwargs,
        )


def initialize_chat_for_client(
    session_id: str,
    *,
    client_session_id: Optional[str] = None,
    session_token: Optional[str] = None,
) -> None:
    """Initialize a chat session for a newly connected client."""
    _initialize_session(
        session_id,
        room=session_id,
        client_session_id=client_session_id,
        session_token=session_token,
    )


# =============================================================================
# SOCKET.IO HANDLERS
# =============================================================================


def register_chat_handlers(socketio: "SocketIO") -> None:
    """Register all chat-related Socket.IO handlers."""
    audit_log = get_audit_log()
    state_machine = get_agent_state_machine()

    logger.info(
        "Chat handlers initialized with state machine in state: %s", state_machine.state
    )

    @socketio.on("chat_connect")
    def handle_chat_connect(data=None):
        """Handle explicit chat connection (fallback for clients that
        emit ``chat_connect`` instead of relying on auto-init)."""
        logger.info(">>> chat_connect event received!")
        try:
            client_session_id = (
                data.get("client_session_id")
                if isinstance(data, dict)
                else None
            )
            session_token = (
                data.get("session_token") if isinstance(data, dict) else None
            )
            _initialize_session(
                request.sid,
                client_session_id=client_session_id,
                session_token=session_token,
            )
        except Exception as e:
            logger.error("Error in chat_connect handler: %s", e, exc_info=True)
            emit("chat_error", {"error": str(e)})

    @socketio.on("chat_disconnect")
    def handle_chat_disconnect():
        """Handle chat disconnection."""
        session_id = request.sid
        leave_room(session_id)

        # Release the socket-keyed entry. Without this every connection
        # ever made stayed resident, each holding its own message list.
        get_session_manager().remove_session(session_id)

        audit_log.log(
            AuditLogEntry.create(
                event_type=AuditEventType.SESSION_ENDED,
                details={"socket_session_id": session_id},
            )
        )

        logger.info("Chat disconnected: %s", session_id)

    @socketio.on("chat_message")
    def handle_chat_message(data):
        """Handle an incoming chat message.

        The turn itself belongs to the orchestrator; this handler echoes
        the user message, audits it, and lets the event sink stream the
        results back.
        """
        session_id = request.sid
        chat_session = get_chat_session(session_id)

        user_text = data.get("content", "").strip()
        if not user_text:
            return

        # Create and store user message
        user_msg = ChatMessage.user(user_text)
        chat_session.add_message(user_msg)

        # Echo user message back
        emit(
            "chat_message",
            {
                "message": user_msg.to_dict(),
                "agent_state": state_machine.state.value,
            },
            room=chat_session.socket_session_id,
        )

        # Log user message
        executor = get_mcp_executor()
        mcp_session = executor.current_session
        audit_log.log(
            AuditLogEntry.create(
                event_type=AuditEventType.USER_MESSAGE,
                details={"content": user_text[:500]},
                session_id=mcp_session.id if mcp_session else None,
            )
        )

        # The sink emits each message as it lands, so nothing needs
        # emitting here — handle_turn also persists to the session.
        _orchestrator_for(socketio, session_id).handle_turn(chat_session, user_text)

    @socketio.on("approve_proposal")
    def handle_approve_proposal(data):
        """Handle approval of a pending tool call proposal.

        This is the EXECUTION PHASE trigger for confirmed tools.
        Searches all domain executors for the proposal.
        """
        session_id = request.sid
        chat_session = get_chat_session(session_id)

        proposal_id = data.get("proposal_id")
        if not proposal_id:
            emit("proposal_error", {"error": "No proposal ID provided"})
            return

        orchestrator = _orchestrator_for(socketio, session_id)
        response, result = orchestrator.approve(
            proposal_id, domain=data.get("domain")
        )

        chat_session.add_message(response)

        emit(
            "chat_message",
            {
                "message": response.to_dict(),
                "agent_state": state_machine.state.value,
            },
        )

        emit(
            "proposal_result",
            {
                "proposal_id": proposal_id,
                "result": result,
            },
        )

    @socketio.on("reject_proposal")
    def handle_reject_proposal(data):
        """Handle rejection of a pending tool call proposal."""
        session_id = request.sid
        chat_session = get_chat_session(session_id)

        proposal_id = data.get("proposal_id")
        if not proposal_id:
            emit("proposal_error", {"error": "No proposal ID provided"})
            return

        orchestrator = _orchestrator_for(socketio, session_id)
        response, result = orchestrator.reject(
            proposal_id,
            reason=data.get("reason", "User rejected"),
            domain=data.get("domain"),
        )

        chat_session.add_message(response)

        emit(
            "chat_message",
            {
                "message": response.to_dict(),
                "agent_state": state_machine.state.value,
            },
        )

        emit(
            "proposal_result",
            {
                "proposal_id": proposal_id,
                "result": result,
            },
        )

    @socketio.on("get_conversation_history")
    def handle_get_history(data=None):
        """Get conversation history."""
        session_id = request.sid
        chat_session = get_chat_session(session_id)

        limit = data.get("limit", 50) if data else 50

        emit(
            "conversation_history",
            {
                "chat_session_id": chat_session.id,
                "messages": chat_session.get_history(limit),
                "agent_state": state_machine.state.value,
            },
        )

    @socketio.on("clear_conversation")
    def handle_clear_conversation(data=None):
        """Clear conversation history."""
        session_id = request.sid
        chat_session = get_chat_session(session_id)
        chat_session.clear()

        emit(
            "conversation_cleared",
            {
                "chat_session_id": chat_session.id,
            },
        )

    @socketio.on("get_agent_state")
    def handle_get_agent_state(data=None):
        """Get current agent state and available tools."""
        executor = get_mcp_executor()
        mcp_session = executor.current_session
        current_state = state_machine.state

        emit(
            "agent_state",
            {
                "state": current_state.value,
                "available_tools": get_tool_registry().get_display_list(current_state),
                "mcp_session": mcp_session.to_dict() if mcp_session else None,
                "metrics": audit_log.get_metrics(),
                "pending_proposals": executor.get_pending_proposals(),
            },
        )

    @socketio.on("get_available_tools")
    def handle_get_available_tools(data=None):
        """Get tools available in current state."""
        registry = get_tool_registry()
        current_state = state_machine.state

        emit(
            "available_tools",
            {
                "tools": registry.get_display_list(current_state),
                "agent_state": current_state.value,
            },
        )

    @socketio.on("get_tool_schemas")
    def handle_get_tool_schemas(data=None):
        """Get JSON schemas for all tools."""
        registry = get_tool_registry()

        emit(
            "tool_schemas",
            {
                "schemas": registry.get_schemas(),
            },
        )

    @socketio.on("get_pending_proposals")
    def handle_get_pending_proposals(data=None):
        """Get all proposals pending approval across all domains."""
        manager = get_mcp_manager()

        emit(
            "pending_proposals",
            {
                "proposals": manager.get_pending_proposals(),
            },
        )

    @socketio.on("get_audit_metrics")
    def handle_get_audit_metrics(data=None):
        """Get audit log metrics."""
        emit(
            "audit_metrics",
            {
                "metrics": audit_log.get_metrics(),
            },
        )


# =============================================================================
# BROADCAST HELPERS
# =============================================================================


def _broadcast_state_update(socketio: "SocketIO", executor: MCPExecutor) -> None:
    """Broadcast agent state update to all clients."""
    from services.core.monitoring import get_monitoring_service

    state_machine = get_agent_state_machine()
    audit_log = get_audit_log()
    mcp_session = executor.current_session
    monitoring_service = get_monitoring_service()

    socketio.emit(
        "agent_state_changed",
        {
            "state": state_machine.state.value,
            "mcp_session": mcp_session.to_dict() if mcp_session else None,
            "metrics": audit_log.get_metrics(),
            "available_tools": get_tool_registry().get_display_list(
                state_machine.state
            ),
            "monitoring_active": (
                monitoring_service.is_monitoring if monitoring_service else False
            ),
        },
    )


# =============================================================================
# DETECTION EVENT HANDLER
# =============================================================================


def emit_detection_to_chat(socketio: "SocketIO", event_data: Dict[str, Any]) -> None:
    """Emit a detection event to the chat interface.

    This is called by the monitoring service when a detection occurs.
    It creates a chat message but does NOT trigger any automatic actions.
    """
    detected = event_data.get("detected", False)
    should_alert = event_data.get("should_alert", False)
    confidence = event_data.get("confidence", 0)
    description = event_data.get("vision_description", "")

    if detected:
        icon = "🚨" if should_alert else "📦"
        alert_text = " **[ALERT TRIGGERED]**" if should_alert else ""

        message = ChatMessage.assistant(
            f"{icon}{alert_text} **Detection Event**\n\n"
            f"**Confidence:** {confidence:.1%}\n\n"
            f"{description}",
            metadata={"event_type": "detection", "event_data": event_data},
        )

        # Log detection
        audit_log = get_audit_log()
        audit_log.log(
            AuditLogEntry.create(
                event_type=AuditEventType.DETECTION_EVENT,
                details={
                    "detected": detected,
                    "confidence": confidence,
                    "should_alert": should_alert,
                },
            )
        )

        socketio.emit(
            "chat_detection",
            {
                "message": message.to_dict(),
                "event_data": event_data,
            },
        )
