"""Chat session maps must not grow without bound.

Before this, ``handle_chat_disconnect`` never released the socket-keyed
entry and ``_client_sessions`` was never pruned at all, so a long-running
appliance accumulated one ``ChatSession`` (with its message list) per
connection ever made.
"""

from __future__ import annotations

import pytest

from core import config as core_config


def _token_for(client_session_id: str) -> str:
    from agents.conversation import issue_session_token

    return issue_session_token(client_session_id)


@pytest.fixture
def sessions(monkeypatch, tmp_path):
    """A fresh manager with no database behind it."""
    monkeypatch.setenv("CAMERA_AGENT_DATA_DIR", str(tmp_path))
    core_config.reset_config()

    from agents.conversation import session as chat_mod

    # Sessions are pure cache here — keep SQLite out of the unit test.
    monkeypatch.setattr(chat_mod.ChatHistoryRepository, "get_messages", lambda _id: [])
    monkeypatch.setattr(
        chat_mod.ChatHistoryRepository, "add_message", lambda **_kw: None
    )
    monkeypatch.setattr(chat_mod, "_client_history_exists", lambda _id: False)

    yield chat_mod.ChatSessionManager()
    core_config.reset_config()


def test_remove_session_releases_the_socket_entry(sessions):
    sessions.get_or_bind_client_session("socket-1", "client-1")
    assert sessions.session_count() == 1

    sessions.remove_session("socket-1")

    assert sessions.session_count() == 0


def test_disconnected_session_survives_a_quick_reconnect(sessions):
    """Within the idle window the warm session is reused, not rebuilt."""
    first = sessions.get_or_bind_client_session("socket-1", "client-1")
    token = _token_for("client-1")
    sessions.remove_session("socket-1")

    second = sessions.get_or_bind_client_session(
        "socket-2", "client-1", session_token=token
    )

    assert second is first
    assert second.socket_session_id == "socket-2"


def test_idle_client_sessions_are_evicted(sessions):
    sessions.get_or_bind_client_session("socket-1", "client-1")
    sessions.remove_session("socket-1")
    assert sessions.cached_session_count() == 1

    # Age the session past the TTL without sleeping through it.
    cached = list(sessions._client_sessions.values())[0]  # pylint: disable=protected-access
    cached.last_active -= core_config.get_config().chat.idle_ttl_seconds + 1

    assert sessions.evict_idle() == 1
    assert sessions.cached_session_count() == 0


def test_connected_sessions_are_never_evicted(sessions):
    """A live socket outranks the idle clock."""
    session = sessions.get_or_bind_client_session("socket-1", "client-1")
    session.last_active -= core_config.get_config().chat.idle_ttl_seconds * 10

    assert sessions.evict_idle() == 0
    assert sessions.cached_session_count() == 1


def test_client_session_map_respects_the_hard_cap(sessions, monkeypatch):
    monkeypatch.setenv("CHAT_MAX_CLIENT_SESSIONS", "3")
    core_config.reset_config()

    for i in range(10):
        sessions.get_or_bind_client_session(f"socket-{i}", f"client-{i}")
        sessions.remove_session(f"socket-{i}")

    assert sessions.cached_session_count() <= 3


def test_message_list_is_capped(monkeypatch, tmp_path):
    monkeypatch.setenv("CAMERA_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CHAT_MAX_MESSAGES_IN_MEMORY", "5")
    core_config.reset_config()

    from agents.conversation import session as chat_mod

    monkeypatch.setattr(chat_mod.ChatHistoryRepository, "get_messages", lambda _id: [])
    monkeypatch.setattr(
        chat_mod.ChatHistoryRepository, "add_message", lambda **_kw: None
    )

    from agents.conversation import ChatMessage

    session = chat_mod.ChatSession("socket-1", client_session_id="client-1")
    for i in range(20):
        session.add_message(ChatMessage.user(f"message {i}"))

    assert len(session.messages) == 5
    # The tail is what survives — the newest turns are the useful ones.
    assert session.messages[-1].content == "message 19"
    core_config.reset_config()
