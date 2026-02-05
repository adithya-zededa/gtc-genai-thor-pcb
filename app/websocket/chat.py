"""Chat-based WebSocket handlers for MCP agent communication.

This module provides real-time chat communication between the UI and the
MCP-based agent using the formalized protocol with interpretation/execution
separation and proposal/approval workflow.

Key principles:
1. Language is the ONLY control surface
2. Tools are PROPOSED, then APPROVED, then EXECUTED
3. All actions are logged and auditable
4. UI suggests but never directly executes
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from flask import request
from flask_socketio import emit, join_room, leave_room

from agents.mcp import (
    AgentState,
    AuditEventType,
    AuditLogEntry,
    MCPExecutor,
    MCPInterpreter,
    get_agent_state_machine,
    get_audit_log,
    get_mcp_executor,
    get_mcp_interpreter,
    get_tool_registry,
)
from agents.mcp_manager import get_mcp_manager, DOMAIN_PCB, DOMAIN_RETAIL, DOMAIN_GENERAL
from core.logging import get_logger

if TYPE_CHECKING:
    from flask_socketio import SocketIO

logger = get_logger(__name__)


# =============================================================================
# CONVERSATION MANAGEMENT
# =============================================================================

class ChatMessage:
    """A single chat message."""
    
    def __init__(
        self,
        role: str,  # "user", "assistant", "system", "tool"
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
        id: Optional[str] = None,
    ):
        import uuid
        self.id = id or str(uuid.uuid4())
        self.role = role
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def user(cls, content: str, **kwargs) -> "ChatMessage":
        return cls(role="user", content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str, **kwargs) -> "ChatMessage":
        return cls(role="assistant", content=content, **kwargs)

    @classmethod
    def system(cls, content: str, **kwargs) -> "ChatMessage":
        return cls(role="system", content=content, **kwargs)

    @classmethod
    def tool(cls, content: str, tool_name: str, success: bool, **kwargs) -> "ChatMessage":
        return cls(
            role="tool",
            content=content,
            metadata={"tool_name": tool_name, "success": success, **kwargs.get("metadata", {})},
            **{k: v for k, v in kwargs.items() if k != "metadata"},
        )


class ChatSession:
    """A chat session with message history."""
    
    def __init__(self, session_id: str):
        import uuid
        self.id = str(uuid.uuid4())
        self.socket_session_id = session_id
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now().isoformat()
        self._lock = threading.Lock()

    def add_message(self, message: ChatMessage) -> None:
        with self._lock:
            self.messages.append(message)

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            messages = self.messages[-limit:] if limit else self.messages
            return [m.to_dict() for m in messages]

    def clear(self) -> None:
        with self._lock:
            self.messages = []


# Global session storage
_chat_sessions: Dict[str, ChatSession] = {}
_sessions_lock = threading.Lock()


def get_chat_session(session_id: str) -> ChatSession:
    """Get or create a chat session."""
    with _sessions_lock:
        if session_id not in _chat_sessions:
            _chat_sessions[session_id] = ChatSession(session_id)
        return _chat_sessions[session_id]


def initialize_chat_for_client(session_id: str) -> None:
    """Initialize chat session for a client on connect.
    
    This is called from the main connect handler to auto-initialize
    the chat session without requiring a separate chat_connect event.
    """
    from flask_socketio import emit, join_room
    from flask import request
    
    # Use the current request's session ID to ensure we emit to the right client
    actual_sid = getattr(request, 'sid', session_id)
    logger.info("Auto-initializing chat for client: %s (passed: %s)", actual_sid, session_id)
    
    audit_log = get_audit_log()
    state_machine = get_agent_state_machine()
    
    chat_session = get_chat_session(actual_sid)
    join_room(actual_sid)
    
    # Log connection
    audit_log.log(AuditLogEntry.create(
        event_type=AuditEventType.SESSION_STARTED,
        details={"socket_session_id": actual_sid, "chat_session_id": chat_session.id},
    ))
    
    # Transition from OFF to IDLE if agent is off
    current_state = state_machine.state
    if current_state == AgentState.OFF:
        try:
            state_machine.transition_to(
                AgentState.IDLE,
                trigger="user_connected",
                metadata={"session_id": actual_sid},
            )
            current_state = state_machine.state
            logger.info("Agent transitioned to IDLE on connection")
        except ValueError as e:
            logger.warning("Could not transition to IDLE: %s", e)
    
    executor = get_mcp_executor()
    mcp_session = executor.current_session
    
    # Get available tools for current state
    registry = get_tool_registry()
    available_tools = registry.get_display_list(current_state)
    
    # Get monitoring status
    from services.monitoring_service import get_monitoring_service
    monitoring_service = get_monitoring_service()
    monitoring_active = monitoring_service.is_monitoring if monitoring_service else False
    
    logger.info("Sending chat_connected to client %s with state %s", actual_sid, current_state.value)
    
    # Send initial state - explicitly specify room to ensure delivery
    emit("chat_connected", {
        "chat_session_id": chat_session.id,
        "agent_state": current_state.value,
        "available_tools": available_tools,
        "pending_proposals": executor.get_pending_proposals(),
        "mcp_session": mcp_session.to_dict() if mcp_session else None,
        "metrics": audit_log.get_metrics(),
        "monitoring_active": monitoring_active,
    }, room=actual_sid)
    
    # Send welcome message
    welcome_content = _generate_welcome_message(current_state)
    welcome = ChatMessage.assistant(welcome_content)
    chat_session.add_message(welcome)
    
    emit("chat_message", {
        "message": welcome.to_dict(),
        "agent_state": current_state.value,
    }, room=actual_sid)
    
    logger.info("Chat initialization complete for client: %s", actual_sid)


# =============================================================================
# WEBSOCKET HANDLERS
# =============================================================================

def register_chat_handlers(socketio: "SocketIO") -> None:
    """Register chat-related WebSocket event handlers."""
    
    logger.info("Registering chat WebSocket handlers...")
    
    audit_log = get_audit_log()
    state_machine = get_agent_state_machine()
    
    logger.info("Chat handlers initialized with state machine in state: %s", state_machine.state)

    @socketio.on("chat_connect")
    def handle_chat_connect(data=None):
        """Handle chat connection."""
        logger.info(">>> chat_connect event received!")
        try:
            session_id = request.sid
            logger.info("Chat connect received from: %s", session_id)
            
            chat_session = get_chat_session(session_id)
            join_room(session_id)
            
            # Log connection
            audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.SESSION_STARTED,
                details={"socket_session_id": session_id, "chat_session_id": chat_session.id},
            ))
            
            logger.info("Chat connected: %s", session_id)
            
            # Transition from OFF to IDLE if agent is off
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
            
            # Get available tools for current state
            registry = get_tool_registry()
            available_tools = registry.get_display_list(current_state)
            
            # Get monitoring status
            from services.monitoring_service import get_monitoring_service
            monitoring_service = get_monitoring_service()
            monitoring_active = monitoring_service.is_monitoring if monitoring_service else False
            
            # Send initial state
            emit("chat_connected", {
                "chat_session_id": chat_session.id,
                "agent_state": current_state.value,
                "available_tools": available_tools,
                "pending_proposals": executor.get_pending_proposals(),
                "mcp_session": mcp_session.to_dict() if mcp_session else None,
                "metrics": audit_log.get_metrics(),
                "monitoring_active": monitoring_active,
            })
            
            # Send welcome message
            welcome_content = _generate_welcome_message(current_state)
            welcome = ChatMessage.assistant(welcome_content)
            chat_session.add_message(welcome)
            
            emit("chat_message", {
                "message": welcome.to_dict(),
                "agent_state": current_state.value,
            })
        except Exception as e:
            logger.error("Error in chat_connect handler: %s", e, exc_info=True)
            emit("chat_error", {"error": str(e)})

    @socketio.on("chat_disconnect")
    def handle_chat_disconnect():
        """Handle chat disconnection."""
        session_id = request.sid
        leave_room(session_id)
        
        audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.SESSION_ENDED,
            details={"socket_session_id": session_id},
        ))
        
        logger.info("Chat disconnected: %s", session_id)

    @socketio.on("chat_message")
    def handle_chat_message(data):
        """Handle incoming chat message from user.
        
        This is the INTERPRETATION PHASE entry point.
        User messages are interpreted to detect intent and
        produce tool call proposals when appropriate.
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
        emit("chat_message", {
            "message": user_msg.to_dict(),
            "agent_state": state_machine.state.value,
        })
        
        # Log user message
        executor = get_mcp_executor()
        mcp_session = executor.current_session
        audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.USER_MESSAGE,
            details={"content": user_text[:500]},
            session_id=mcp_session.id if mcp_session else None,
        ))
        
        # Process the message
        try:
            response = _process_user_message(
                chat_session=chat_session,
                user_text=user_text,
                socketio=socketio,
                session_id=session_id,
            )
            
            chat_session.add_message(response)
            
            emit("chat_message", {
                "message": response.to_dict(),
                "agent_state": state_machine.state.value,
            })
            
        except Exception as e:
            logger.error("Error processing chat message: %s", e, exc_info=True)
            error_msg = ChatMessage.assistant(
                f"I apologize, but I encountered an error: {str(e)}. Please try again."
            )
            chat_session.add_message(error_msg)
            emit("chat_message", {
                "message": error_msg.to_dict(),
                "agent_state": state_machine.state.value,
            })

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
        
        domain = data.get("domain")  # optional hint
        manager = get_mcp_manager()
        result = manager.approve_proposal(proposal_id, domain=domain)
        
        # Create response message based on result
        if result["status"] == "executed":
            tool_result = result["result"]
            output = tool_result.get("output", {})
            message_text = output.get("message", "Tool executed successfully")
            
            response = ChatMessage.tool(
                content=f"✅ **{tool_result['tool_name']}** executed successfully.\n\n{message_text}",
                tool_name=tool_result["tool_name"],
                success=True,
                metadata={"result": tool_result},
            )
        elif result["status"] == "failed":
            response = ChatMessage.tool(
                content=f"❌ Tool execution failed: {result['error']}",
                tool_name=result.get("result", {}).get("tool_name", "unknown"),
                success=False,
                metadata={"error": result["error"]},
            )
        else:
            response = ChatMessage.assistant(
                f"⚠️ Unexpected result: {result.get('reason', 'Unknown error')}"
            )
        
        chat_session.add_message(response)
        
        emit("chat_message", {
            "message": response.to_dict(),
            "agent_state": state_machine.state.value,
        })
        
        emit("proposal_result", {
            "proposal_id": proposal_id,
            "result": result,
        })
        
        # Broadcast state change if applicable
        _broadcast_state_update(socketio, get_mcp_executor())

    @socketio.on("reject_proposal")
    def handle_reject_proposal(data):
        """Handle rejection of a pending tool call proposal."""
        session_id = request.sid
        chat_session = get_chat_session(session_id)
        
        proposal_id = data.get("proposal_id")
        reason = data.get("reason", "User rejected")
        domain = data.get("domain")  # optional hint
        
        if not proposal_id:
            emit("proposal_error", {"error": "No proposal ID provided"})
            return
        
        manager = get_mcp_manager()
        result = manager.reject_proposal(proposal_id, reason, domain=domain)
        
        response = ChatMessage.assistant(
            f"🚫 Proposal rejected: {reason}"
        )
        chat_session.add_message(response)
        
        emit("chat_message", {
            "message": response.to_dict(),
            "agent_state": state_machine.state.value,
        })
        
        emit("proposal_result", {
            "proposal_id": proposal_id,
            "result": result,
        })

    @socketio.on("get_conversation_history")
    def handle_get_history(data=None):
        """Get conversation history."""
        session_id = request.sid
        chat_session = get_chat_session(session_id)
        
        limit = data.get("limit", 50) if data else 50
        
        emit("conversation_history", {
            "chat_session_id": chat_session.id,
            "messages": chat_session.get_history(limit),
            "agent_state": state_machine.state.value,
        })

    @socketio.on("clear_conversation")
    def handle_clear_conversation(data=None):
        """Clear conversation history."""
        session_id = request.sid
        chat_session = get_chat_session(session_id)
        chat_session.clear()
        
        emit("conversation_cleared", {
            "chat_session_id": chat_session.id,
        })

    @socketio.on("get_agent_state")
    def handle_get_agent_state(data=None):
        """Get current agent state and metrics."""
        executor = get_mcp_executor()
        mcp_session = executor.current_session
        
        emit("agent_state", {
            "state": state_machine.state.value,
            "state_history": state_machine.state_history[-10:],
            "available_tools": get_tool_registry().get_display_list(state_machine.state),
            "pending_proposals": executor.get_pending_proposals(),
            "mcp_session": mcp_session.to_dict() if mcp_session else None,
            "metrics": audit_log.get_metrics(),
        })

    @socketio.on("get_available_tools")
    def handle_get_available_tools(data=None):
        """Get tools available in current state."""
        registry = get_tool_registry()
        current_state = state_machine.state
        
        emit("available_tools", {
            "tools": registry.get_display_list(current_state),
            "agent_state": current_state.value,
        })

    @socketio.on("get_tool_schemas")
    def handle_get_tool_schemas(data=None):
        """Get JSON schemas for all tools."""
        registry = get_tool_registry()
        
        emit("tool_schemas", {
            "schemas": registry.get_schemas(),
        })

    @socketio.on("get_pending_proposals")
    def handle_get_pending_proposals(data=None):
        """Get all proposals pending approval across all domains."""
        manager = get_mcp_manager()
        
        emit("pending_proposals", {
            "proposals": manager.get_pending_proposals(),
        })

    @socketio.on("get_audit_metrics")
    def handle_get_audit_metrics(data=None):
        """Get audit log metrics."""
        emit("audit_metrics", {
            "metrics": audit_log.get_metrics(),
        })


# =============================================================================
# MESSAGE PROCESSING
# =============================================================================

def _process_user_message(
    chat_session: ChatSession,
    user_text: str,
    socketio: "SocketIO",
    session_id: str,
) -> ChatMessage:
    """Process a user message through the interpretation phase.
    
    This function:
    1. Routes to the correct domain MCP (pcb / retail / general)
    2. Interprets user intent
    3. Produces tool call proposals if appropriate
    4. Submits proposals for approval/auto-execution
    5. Returns a response message
    
    NO TOOLS ARE DIRECTLY EXECUTED HERE.
    Tools are either auto-approved (policy-based) or require explicit approval.
    """
    manager = get_mcp_manager()
    state_machine = get_agent_state_machine()
    executor = get_mcp_executor()
    
    current_state = state_machine.state
    mcp_session = executor.current_session
    session_id_str = mcp_session.id if mcp_session else None
    
    # INTERPRETATION PHASE: Route to the correct domain and detect intent
    resolved_domain, proposal = manager.interpret(
        message=user_text,
        agent_state=current_state,
        session_id=session_id_str,
    )
    
    # If no tool call needed, generate conversational response
    if proposal is None:
        return _generate_conversational_response(user_text, current_state)
    
    # If proposal was rejected during interpretation (e.g., state violation)
    if proposal.is_rejected:
        return ChatMessage.assistant(
            f"⚠️ I understood your request, but I can't do that right now.\n\n"
            f"**Reason:** {proposal.rejection_reason}\n\n"
            f"The agent is currently in **{current_state.value}** state."
        )
    
    # EXECUTION PHASE: Submit proposal through the correct domain executor
    result = manager.submit(proposal, domain=resolved_domain)
    
    if result["status"] == "pending_approval":
        # Tool requires user confirmation
        _emit_pending_proposal(socketio, session_id, result)
        
        domain_label = ""
        if resolved_domain in (DOMAIN_PCB, DOMAIN_RETAIL):
            domain_label = f" [{resolved_domain.upper()}]"
        
        return ChatMessage.assistant(
            f"🔔 **Confirmation Required**{domain_label}\n\n"
            f"I'd like to execute **{proposal.tool_name}**.\n\n"
            f"_{result['confirmation_message']}_\n\n"
            f"Please approve or reject this action using the buttons above.",
            metadata={
                "proposal_id": proposal.id,
                "requires_approval": True,
                "domain": resolved_domain,
            },
        )
    
    elif result["status"] == "executed":
        # Tool was auto-approved and executed
        tool_result = result["result"]
        output = tool_result.get("output", {})
        success = tool_result.get("success", False)
        
        # Broadcast state update
        _broadcast_state_update(socketio, executor)
        
        if success:
            message_text = output.get("message", "Done")
            data = output.get("data", {})
            
            response_content = f"✅ **{proposal.tool_name}** completed.\n\n{message_text}"
            
            # Add additional data if present
            if data:
                if "session_id" in data:
                    response_content += f"\n\n**Session ID:** `{data['session_id']}`"
                if "summary" in data:
                    response_content += f"\n\n**Summary:**\n{data['summary']}"
                if "description" in data and data["description"]:
                    response_content += f"\n\n{data['description']}"
            
            return ChatMessage.tool(
                content=response_content,
                tool_name=proposal.tool_name,
                success=True,
                metadata={"result": tool_result, "proposal_id": proposal.id},
            )
        else:
            return ChatMessage.tool(
                content=f"❌ **{proposal.tool_name}** failed: {tool_result.get('error', 'Unknown error')}",
                tool_name=proposal.tool_name,
                success=False,
                metadata={"result": tool_result, "proposal_id": proposal.id},
            )
    
    elif result["status"] == "rejected":
        # Proposal was rejected (validation or state error)
        return ChatMessage.assistant(
            f"⚠️ I couldn't execute that request.\n\n"
            f"**Reason:** {result['reason']}"
        )
    
    elif result["status"] == "failed":
        # Execution failed
        return ChatMessage.tool(
            content=f"❌ Execution failed: {result['error']}",
            tool_name=proposal.tool_name,
            success=False,
            metadata={"error": result["error"], "proposal_id": proposal.id},
        )
    
    # Unexpected status
    return ChatMessage.assistant(
        f"⚠️ Unexpected result: {result}"
    )


# =============================================================================
# RESPONSE GENERATION
# =============================================================================

def _generate_welcome_message(state: AgentState) -> str:
    """Generate a welcome message based on current state."""
    state_descriptions = {
        AgentState.OFF: "completely **off**",
        AgentState.IDLE: "**idle** and ready",
        AgentState.MONITORING: "**actively monitoring** the camera feed",
        AgentState.ANALYZING: "**analyzing** a frame",
        AgentState.ALERTING: "in **alerting** mode",
        AgentState.ERROR: "in an **error** state",
    }
    
    state_desc = state_descriptions.get(state, f"in **{state.value}** state")
    
    message = (
        f"# 👋 Hello!\n\n"
        f"I'm your AI monitoring assistant. The agent is currently {state_desc}.\n\n"
        f"## What You Can Say\n\n"
        f"### 🔌 PCB Inspection\n"
        f"- **\"Inspect the PCB\"** - Analyze a board for defects\n"
        f"- **\"Send an alert if you see a defective Arduino\"** - Defect alert\n"
        f"- **\"Generate a defect report\"** - View defect summary\n\n"
        f"### 🛒 Retail Billing\n"
        f"- **\"Scan the tray and count the items\"** - Identify items\n"
        f"- **\"Create a bill and generate an invoice\"** - Billing workflow\n"
        f"- **\"Send the invoice to email@example.com\"** - Email invoice\n\n"
        f"### 📷 Camera Monitoring\n"
        f"- **\"Start monitoring\"** - Begin a new monitoring session\n"
        f"- **\"End session\"** or **\"Stop monitoring\"** - End the current session\n"
        f"- **\"What do you see?\"** - Analyze the current frame\n\n"
        f"---\n"
        f"💡 *All actions are controlled through conversation. "
        f"I'll ask for confirmation before doing anything sensitive.*"
    )
    
    return message


def _generate_conversational_response(user_text: str, state: AgentState) -> ChatMessage:
    """Generate a conversational response for general queries."""
    text_lower = user_text.lower()
    
    # Help request
    if any(word in text_lower for word in ["help", "what can you do", "commands", "how to"]):
        return ChatMessage.assistant(_generate_help_text(state))
    
    # Greeting
    if any(word in text_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
        return ChatMessage.assistant(
            f"Hello! 👋 I'm here to help with camera monitoring.\n\n"
            f"The agent is currently **{state.value}**. What would you like to do?"
        )
    
    # Thank you
    if any(word in text_lower for word in ["thank", "thanks", "appreciate"]):
        return ChatMessage.assistant(
            "You're welcome! Let me know if there's anything else I can help with. 😊"
        )
    
    # Catch-all
    return ChatMessage.assistant(
        "I'm not sure I understood that. Here are some things you can try:\n\n"
        "- **\"Start monitoring\"** or **\"Stop monitoring\"** to control the agent\n"
        "- **\"Analyze the frame\"** to see what's in view\n"
        "- **\"Show history\"** to see past detections\n"
        "- **\"Help\"** to see all commands\n\n"
        "Feel free to ask in your own words!"
    )


def _generate_help_text(state: AgentState) -> str:
    """Generate help text based on current state."""
    help_text = (
        "## 🤖 Available Commands\n\n"
        "### 🔌 PCB Inspection\n"
        "- **\"Inspect the PCB\"** / **\"Check for defects\"** - Analyze board\n"
        "- **\"Classify board\"** / **\"What board is this?\"** - Identify board type\n"
        "- **\"Send defect alert\"** - Alert about detected defects\n"
        "- **\"Log defect\"** - Record a defect to history\n"
        "- **\"Defect report\"** / **\"Show defects\"** - Generate report\n\n"
        "### 🛒 Retail Billing\n"
        "- **\"Scan the tray\"** / **\"Count items\"** - Identify items\n"
        "- **\"Look up price\"** - Check item price in catalog\n"
        "- **\"Create a bill\"** - Calculate totals\n"
        "- **\"Generate invoice\"** - Render HTML invoice\n"
        "- **\"Send invoice\"** - Email the invoice\n\n"
        "### 📷 Session Control\n"
        "- **\"Start monitoring\"** - Begin a new monitoring session\n"
        "- **\"Stop monitoring\"** or **\"End session\"** - Stop and get summary\n"
        "- **\"Go idle\"** or **\"Pause\"** - Pause without ending session\n\n"
        "### ℹ️ Analysis & Information\n"
        "- **\"What do you see?\"** / **\"Analyze\"** - Analyze current frame\n"
        "- **\"What's your status?\"** - Get agent status\n"
        "- **\"Show history\"** / **\"Recent events\"** - View detection history\n\n"
        "---\n"
        f"🔵 **Current State:** {state.value}\n\n"
        "*Commands are auto-routed to the correct agent (PCB / Retail / General).*"
    )
    return help_text


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _emit_pending_proposal(socketio: "SocketIO", session_id: str, result: Dict[str, Any]) -> None:
    """Emit a pending proposal notification."""
    socketio.emit(
        "tool_confirmation_required",
        {
            "proposal_id": result["proposal_id"],
            "tool_name": result["tool_name"],
            "confirmation_message": result["confirmation_message"],
            "proposal": result["proposal"],
        },
        room=session_id,
    )


def _broadcast_state_update(socketio: "SocketIO", executor: MCPExecutor) -> None:
    """Broadcast agent state update to all clients."""
    from services.monitoring_service import get_monitoring_service
    
    state_machine = get_agent_state_machine()
    audit_log = get_audit_log()
    mcp_session = executor.current_session
    monitoring_service = get_monitoring_service()
    
    socketio.emit("agent_state_changed", {
        "state": state_machine.state.value,
        "mcp_session": mcp_session.to_dict() if mcp_session else None,
        "metrics": audit_log.get_metrics(),
        "available_tools": get_tool_registry().get_display_list(state_machine.state),
        "monitoring_active": monitoring_service.is_monitoring if monitoring_service else False,
    })


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
        audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.DETECTION_EVENT,
            details={
                "detected": detected,
                "confidence": confidence,
                "should_alert": should_alert,
            },
        ))
        
        socketio.emit("chat_detection", {
            "message": message.to_dict(),
            "event_data": event_data,
        })
