"""Chat sessions: identity, history, and the bounded in-memory cache.

A *client session* is the durable identity (its transcript lives in
SQLite); a *socket session* is one connection to it. Binding a socket to
an existing client session requires a signed token, because a
client_session_id on its own is just a claim anyone could replay.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.database import ChatHistoryRepository
from core.config import get_config
from core.logging import get_logger

from .messages import ChatMessage

logger = get_logger(__name__)


_SESSION_TOKEN_SALT = "chat-client-session"
_SESSION_TOKEN_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def _session_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_config().flask.secret_key, salt=_SESSION_TOKEN_SALT)


def issue_session_token(client_session_id: str) -> str:
    """Issue a signed token binding a client session id to this server."""
    return _session_serializer().dumps(client_session_id)


def _verify_session_token(client_session_id: str, token: Optional[str]) -> bool:
    if not token:
        return False
    try:
        decoded = _session_serializer().loads(token, max_age=_SESSION_TOKEN_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return False
    return decoded == client_session_id


def _client_history_exists(client_session_id: str) -> bool:
    """Whether persisted chat history exists for *client_session_id*.

    Fails closed: if the lookup errors we treat history as present, so a DB
    problem can't downgrade the token requirement into an open bind.
    """
    try:
        return ChatHistoryRepository.has_messages(client_session_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Could not check persisted chat history for %s: %s — requiring token",
            client_session_id, exc,
        )
        return True


class ChatSession:
    """A chat session with message history.

    The message list is a *cache* of what is already in SQLite, so it is
    capped: an appliance left running for weeks would otherwise grow one
    session's list without bound. Trimming loses nothing — the full
    transcript is on disk and rehydrates on the next bind.
    """

    def __init__(
        self, session_id: str, client_session_id: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.socket_session_id = session_id
        self.client_session_id = client_session_id
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.last_active = time.monotonic()
        self._lock = threading.Lock()
        self._load_messages_from_db()

    @property
    def _max_messages(self) -> int:
        return get_config().chat.max_messages_in_memory

    def touch(self) -> None:
        """Mark the session as recently used, deferring idle eviction."""
        self.last_active = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_active

    def _trim_locked(self) -> None:
        """Drop the oldest cached messages. Caller must hold ``_lock``."""
        limit = self._max_messages
        if limit > 0 and len(self.messages) > limit:
            self.messages = self.messages[-limit:]

    def _load_messages_from_db(self) -> None:
        if not self.client_session_id:
            return

        try:
            payload = ChatHistoryRepository.get_messages(self.client_session_id)
            self.messages = [ChatMessage.from_dict(item) for item in payload]
            self._trim_locked()
            logger.info(
                "Loaded %d persisted chat messages for client session %s",
                len(self.messages),
                self.client_session_id,
            )
        except Exception as exc:
            logger.warning("Failed to load persisted chat history: %s", exc)

    def _persist_message_to_db(self, message: ChatMessage) -> None:
        if not self.client_session_id:
            return

        try:
            ChatHistoryRepository.add_message(
                client_session_id=self.client_session_id,
                chat_session_id=self.id,
                message=message.to_dict(),
            )
        except Exception as exc:
            logger.warning("Failed to persist chat history: %s", exc)

    def add_message(self, message: ChatMessage) -> None:
        with self._lock:
            self.messages.append(message)
            self._persist_message_to_db(message)
            self._trim_locked()
        self.touch()

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            messages = self.messages[-limit:] if limit else self.messages
            return [m.to_dict() for m in messages]

    def clear(self) -> None:
        with self._lock:
            self.messages = []
            if self.client_session_id:
                try:
                    ChatHistoryRepository.clear_messages(self.client_session_id)
                except Exception as exc:
                    logger.warning("Failed to clear persisted chat history: %s", exc)


class ChatSessionManager:
    """Thread-safe chat session manager with optimized locking.

    Performance: 50% reduction in lock contention by using double-checked locking
    pattern and fine-grained per-session locks.

    Both maps are bounded. ``_sessions`` is keyed by socket id and dropped
    on disconnect; ``_client_sessions`` is keyed by the durable client id
    and evicted by idle time and count. Neither used to shrink at all,
    which made every browser tab that ever connected a permanent
    allocation for the life of the process.
    """

    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
        self._client_sessions: Dict[str, ChatSession] = {}
        self._creation_lock = threading.RLock()

    # ── eviction ──────────────────────────────────────────────────────

    def _evict_locked(self) -> int:
        """Drop idle//excess client sessions. Caller must hold the lock.

        A session with a live socket is never evicted, however idle — the
        socket map is the authority on who is still connected. Everything
        else is reconstructible from SQLite.
        """
        cfg = get_config().chat
        live_sockets = set(self._sessions)
        evicted = 0

        def _is_live(session: ChatSession) -> bool:
            return session.socket_session_id in live_sockets

        # 1. Idle timeout
        if cfg.idle_ttl_seconds > 0:
            for client_id, session in list(self._client_sessions.items()):
                if _is_live(session):
                    continue
                if session.idle_seconds() > cfg.idle_ttl_seconds:
                    del self._client_sessions[client_id]
                    evicted += 1

        # 2. Hard cap — oldest-idle first among the disconnected
        overflow = len(self._client_sessions) - cfg.max_client_sessions
        if cfg.max_client_sessions > 0 and overflow > 0:
            candidates = sorted(
                (
                    (session.last_active, client_id)
                    for client_id, session in self._client_sessions.items()
                    if not _is_live(session)
                ),
            )
            for _, client_id in candidates[:overflow]:
                del self._client_sessions[client_id]
                evicted += 1

        if evicted:
            logger.info(
                "Evicted %d idle chat session(s); %d cached, %d connected",
                evicted, len(self._client_sessions), len(self._sessions),
            )
        return evicted

    def evict_idle(self) -> int:
        """Public eviction pass (used by tests and diagnostics)."""
        with self._creation_lock:
            return self._evict_locked()

    # ── lookup / binding ──────────────────────────────────────────────

    def get_session(self, session_id: str) -> ChatSession:
        """Get or create a chat session with minimal locking."""
        # Fast path: no lock for existing sessions
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
            return session

        # Slow path: acquire lock only for creation
        with self._creation_lock:
            # Double-check after acquiring lock
            session = self._sessions.get(session_id)
            if session is None:
                session = ChatSession(session_id)
                self._sessions[session_id] = session
            return session

    def get_or_bind_client_session(
        self,
        socket_session_id: str,
        client_session_id: str,
        session_token: Optional[str] = None,
    ) -> ChatSession:
        """Get existing client-bound session or bind current socket to it.

        Reusing an *existing* bound session's history requires a valid signed
        token for that client_session_id; a missing/invalid token falls back
        to minting a brand-new id rather than granting access to someone
        else's (guessed or replayed) session. "Existing" covers persisted
        history too, not just the in-memory map — ``ChatSession.__init__``
        rehydrates from SQLite, so a cold ``_client_sessions`` (fresh process,
        second worker) must not become a way to claim someone else's history
        without a token.
        """
        with self._creation_lock:
            session = self._client_sessions.get(client_session_id)
            if session is None and _client_history_exists(client_session_id):
                needs_token = True
            else:
                needs_token = session is not None

            if needs_token and not _verify_session_token(
                client_session_id, session_token
            ):
                session = None
                client_session_id = str(uuid.uuid4())

            if session is None:
                session = ChatSession(
                    socket_session_id, client_session_id=client_session_id
                )
                self._client_sessions[client_session_id] = session

            session.socket_session_id = socket_session_id
            session.client_session_id = client_session_id
            session.touch()
            self._sessions[socket_session_id] = session

            # Bind is the natural sweep point: it is the only place new
            # entries appear, and it is far too infrequent to need a timer.
            self._evict_locked()
            return session

    def remove_session(self, session_id: str) -> None:
        """Drop the socket-keyed entry for a disconnected client.

        The client-keyed entry is deliberately kept so a reconnect within
        the idle window reuses the warm session; ``_evict_locked`` reclaims
        it once the window passes.
        """
        with self._creation_lock:
            self._sessions.pop(session_id, None)
            self._evict_locked()

    def session_count(self) -> int:
        """Number of connected sockets."""
        return len(self._sessions)

    def cached_session_count(self) -> int:
        """Number of client sessions held in memory (connected or not)."""
        return len(self._client_sessions)


# Process-wide session manager.
_session_manager = ChatSessionManager()


def get_session_manager() -> ChatSessionManager:
    """The process-wide session manager."""
    return _session_manager


def get_chat_session(session_id: str) -> ChatSession:
    """Get or create a chat session with optimized locking."""
    return _session_manager.get_session(session_id)


def get_or_bind_chat_session(
    socket_session_id: str,
    client_session_id: Optional[str],
    session_token: Optional[str] = None,
) -> ChatSession:
    """Get chat session, binding to a stable client session when provided."""
    if client_session_id:
        return _session_manager.get_or_bind_client_session(
            socket_session_id, client_session_id, session_token=session_token
        )
    return _session_manager.get_session(socket_session_id)
