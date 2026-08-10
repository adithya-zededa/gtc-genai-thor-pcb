"""Transport-independent conversation layer.

The chat turn — interpret the user's message, route it to a domain MCP,
submit the resulting proposal, summarise what happened — is the same work
whether it arrives over Socket.IO or over ``/api/mcp/*``. It used to live
inside the Socket.IO handler module, which meant the two surfaces ran
visibly different code over the same MCP primitives and the loop could
not be tested without a socket.

It lives here instead. :class:`ConversationOrchestrator` performs the turn
and *returns* messages; anything the UI needs to see mid-turn (progress
activity, an approval prompt, a state change) goes to a
:class:`ConversationEventSink` that the caller supplies. The Socket.IO
binding in ``app/websocket/chat.py`` is now just a sink plus handlers.
"""

from .events import ConversationEventSink, NullEventSink
from .messages import ChatMessage
from .naming import TOOL_DISPLAY_NAMES, friendly_tool_name
from .orchestrator import ConversationOrchestrator
from .responses import (
    generate_conversational_response,
    generate_tool_followup_response,
    generate_welcome_message,
)
from .session import (
    ChatSession,
    ChatSessionManager,
    get_chat_session,
    get_or_bind_chat_session,
    get_session_manager,
    issue_session_token,
)

__all__ = [
    "ChatMessage",
    "ChatSession",
    "ChatSessionManager",
    "ConversationEventSink",
    "ConversationOrchestrator",
    "NullEventSink",
    "TOOL_DISPLAY_NAMES",
    "friendly_tool_name",
    "generate_conversational_response",
    "generate_tool_followup_response",
    "generate_welcome_message",
    "get_chat_session",
    "get_or_bind_chat_session",
    "get_session_manager",
    "issue_session_token",
]
