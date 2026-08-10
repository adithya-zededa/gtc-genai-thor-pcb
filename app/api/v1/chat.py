"""Conversational REST surface.

``/api/mcp/*`` speaks the raw protocol: interpret a message, submit a
proposal, get the executor's result back. That is the right shape for a
tool-level client, but it is not a *conversation* — it produces no
assistant prose, no tool-result summary, and no persisted transcript.

Those lived only inside the Socket.IO handler until the turn moved into
``agents.conversation``. This endpoint runs the same
:class:`ConversationOrchestrator` the websocket does, so a REST caller
and a chat client now get identical behaviour from identical code.
"""

# pylint: disable=broad-exception-caught

from __future__ import annotations

import uuid
from typing import Any, Dict

from flask import jsonify, request

from agents.conversation import (
    ChatMessage,
    ConversationOrchestrator,
    get_chat_session,
    get_or_bind_chat_session,
    issue_session_token,
)
from agents.mcp.base import (
    AuditEventType,
    AuditLogEntry,
    get_agent_state_machine,
    get_audit_log,
    get_mcp_executor,
)
from core.logging import get_logger

from . import api_bp

logger = get_logger(__name__)


def _resolve_session(payload: Dict[str, Any]):
    """Bind to the caller's chat session, or mint an ephemeral one.

    Reusing an existing transcript requires the same signed token the
    websocket path requires — this endpoint is not a way around that
    check. Callers that pass nothing get a fresh throwaway session.
    """
    client_session_id = payload.get("client_session_id")
    if not client_session_id:
        return get_chat_session(f"rest-{uuid.uuid4().hex[:12]}")

    return get_or_bind_chat_session(
        f"rest-{uuid.uuid4().hex[:12]}",
        client_session_id,
        payload.get("session_token"),
    )


@api_bp.route("/chat/message", methods=["POST"])
def chat_message():
    """Run one conversation turn and return every message it produced.

    Request body::

        {
            "message": "how many defective boards today?",
            "client_session_id": "...",   // optional, to continue a session
            "session_token": "..."        // required with client_session_id
        }
    """
    payload = request.get_json() or {}
    user_text = str(payload.get("message", "")).strip()

    if not user_text:
        return jsonify({"success": False, "error": "No message provided"}), 400

    chat_session = _resolve_session(payload)

    user_msg = ChatMessage.user(user_text)
    chat_session.add_message(user_msg)

    executor = get_mcp_executor()
    mcp_session = executor.current_session
    get_audit_log().log(
        AuditLogEntry.create(
            event_type=AuditEventType.USER_MESSAGE,
            details={"content": user_text[:500], "transport": "rest"},
            session_id=mcp_session.id if mcp_session else None,
        )
    )

    # No event sink: a request/response caller has no mid-turn channel, so
    # progress notifications are dropped and only the messages come back.
    messages = ConversationOrchestrator().handle_turn(chat_session, user_text)

    return jsonify(
        {
            "success": True,
            "agent_state": get_agent_state_machine().state.value,
            "chat_session_id": chat_session.id,
            "client_session_id": chat_session.client_session_id,
            "session_token": (
                issue_session_token(chat_session.client_session_id)
                if chat_session.client_session_id
                else None
            ),
            "messages": [m.to_dict() for m in messages],
        }
    )


@api_bp.route("/chat/history", methods=["GET"])
def chat_history():
    """Return a client session's transcript.

    Requires the signed token for that session, same as binding to it
    over the websocket.
    """
    client_session_id = request.args.get("client_session_id", "").strip()
    if not client_session_id:
        return jsonify({"success": False, "error": "client_session_id required"}), 400

    chat_session = get_or_bind_chat_session(
        f"rest-{uuid.uuid4().hex[:12]}",
        client_session_id,
        request.args.get("session_token"),
    )

    # A bad/absent token mints a *new* id rather than granting access, so
    # a mismatch here means the caller was not entitled to that history.
    if chat_session.client_session_id != client_session_id:
        return jsonify({"success": False, "error": "Invalid session token"}), 403

    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50

    return jsonify(
        {
            "success": True,
            "chat_session_id": chat_session.id,
            "client_session_id": chat_session.client_session_id,
            "messages": chat_session.get_history(limit),
        }
    )
