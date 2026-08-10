"""The conversation turn runs without a transport attached.

That is the point of the extraction: the same orchestrator drives
Socket.IO and REST, and a test needs neither.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.conversation import (
    ChatMessage,
    ConversationEventSink,
    ConversationOrchestrator,
)


class RecordingSink(ConversationEventSink):
    """Captures everything a transport would have emitted."""

    def __init__(self) -> None:
        self.activities = []
        self.proposals = []
        self.state_changes = 0
        self.messages = []

    def activity(self, *, status, title, detail, tool_name=None, chat_session=None):
        self.activities.append(
            {"status": status, "title": title, "tool_name": tool_name}
        )

    def pending_proposal(self, result, *, chat_session=None):
        self.proposals.append(result)

    def state_changed(self):
        self.state_changes += 1

    def message(self, message, *, chat_session=None):
        self.messages.append(message)


class FakeSession:
    """Minimal ChatSession stand-in — no database, no socket."""

    def __init__(self) -> None:
        self.socket_session_id = "test-socket"
        self.client_session_id = "test-client"
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def get_history(self, limit=50):
        return [m.to_dict() for m in self.messages[-limit:]]


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def sink():
    return RecordingSink()


def _patch_manager(monkeypatch, *, interpret, submit=None):
    """Point the orchestrator's MCP manager at a stub."""
    from agents.conversation import orchestrator as orch_mod

    manager = SimpleNamespace(
        interpret=interpret,
        submit=submit or (lambda proposal, domain=None: {}),
    )
    monkeypatch.setattr(orch_mod, "get_mcp_manager", lambda: manager)
    return manager


def test_no_proposal_falls_through_to_a_conversational_reply(
    monkeypatch, session, sink
):
    _patch_manager(monkeypatch, interpret=lambda **_kw: ("general", None))

    from agents.conversation import orchestrator as orch_mod

    monkeypatch.setattr(
        orch_mod,
        "generate_conversational_response",
        lambda user_text, state, chat_session: ChatMessage.assistant("hello there"),
    )

    result = ConversationOrchestrator(sink).process_message(session, "hi")

    assert result.content == "hello there"
    # The UI is told the turn started and finished, even with no tool run.
    assert [a["status"] for a in sink.activities] == ["in_progress", "completed"]


def test_rejected_proposal_explains_itself_without_executing(
    monkeypatch, session, sink
):
    proposal = SimpleNamespace(
        id="p1",
        tool_name="inspect_pcb",
        is_rejected=True,
        rejection_reason="Agent is OFF",
    )
    submitted = []
    _patch_manager(
        monkeypatch,
        interpret=lambda **_kw: ("pcb", proposal),
        submit=lambda proposal, domain=None: submitted.append(proposal) or {},
    )

    result = ConversationOrchestrator(sink).process_message(session, "inspect")

    assert "Agent is OFF" in result.content
    assert submitted == [], "a rejected proposal must never reach the executor"
    assert sink.activities[-1]["status"] == "failed"


def test_pending_approval_reaches_the_sink(monkeypatch, session, sink):
    proposal = SimpleNamespace(
        id="p2", tool_name="send_alert_email", is_rejected=False, rejection_reason=None
    )
    _patch_manager(
        monkeypatch,
        interpret=lambda **_kw: ("general", proposal),
        submit=lambda proposal, domain=None: {
            "status": "pending_approval",
            "proposal_id": "p2",
            "tool_name": "send_alert_email",
            "confirmation_message": "Send an alert to 2 recipients?",
            "proposal": {"id": "p2"},
        },
    )

    result = ConversationOrchestrator(sink).process_message(session, "email the team")

    assert result.metadata["requires_approval"] is True
    assert len(sink.proposals) == 1
    assert sink.proposals[0]["proposal_id"] == "p2"
    assert sink.activities[-1]["status"] == "requires_approval"


def test_successful_tool_broadcasts_state_and_returns_a_tool_message(
    monkeypatch, session, sink
):
    proposal = SimpleNamespace(
        id="p3", tool_name="get_agent_status", is_rejected=False, rejection_reason=None
    )
    _patch_manager(
        monkeypatch,
        interpret=lambda **_kw: ("general", proposal),
        submit=lambda proposal, domain=None: {
            "status": "executed",
            "result": {
                "success": True,
                "tool_name": "get_agent_status",
                "output": {"message": "Agent is idle", "data": {}},
            },
        },
    )

    result = ConversationOrchestrator(sink).process_message(session, "status?")

    assert result.role == "tool"
    assert result.metadata["success"] is True
    assert sink.state_changes == 1


def test_handle_turn_appends_the_followup_summary(monkeypatch, session, sink):
    proposal = SimpleNamespace(
        id="p4", tool_name="count_defective_pcbs", is_rejected=False, rejection_reason=None
    )
    _patch_manager(
        monkeypatch,
        interpret=lambda **_kw: ("pcb", proposal),
        submit=lambda proposal, domain=None: {
            "status": "executed",
            "result": {
                "success": True,
                "tool_name": "count_defective_pcbs",
                "output": {"message": "3 defects", "data": {"total": 3}},
            },
        },
    )

    from agents.conversation import orchestrator as orch_mod

    monkeypatch.setattr(
        orch_mod,
        "generate_tool_followup_response",
        lambda **_kw: ChatMessage.assistant("You have 3 defective boards."),
    )

    messages = ConversationOrchestrator(sink).handle_turn(session, "how many defects?")

    assert [m.role for m in messages] == ["tool", "assistant"]
    assert messages[-1].content == "You have 3 defective boards."
    # Both were persisted and streamed, not just returned.
    assert session.messages[-2:] == messages
    assert sink.messages == messages


def test_handle_turn_reports_an_error_instead_of_raising(monkeypatch, session, sink):
    def _explode(**_kw):
        raise RuntimeError("interpreter down")

    _patch_manager(monkeypatch, interpret=_explode)

    messages = ConversationOrchestrator(sink).handle_turn(session, "anything")

    assert len(messages) == 1
    assert "interpreter down" in messages[0].content


def test_orchestrator_works_with_no_sink_at_all(monkeypatch, session):
    """A REST caller passes nothing and still gets its messages."""
    _patch_manager(monkeypatch, interpret=lambda **_kw: ("general", None))

    from agents.conversation import orchestrator as orch_mod

    monkeypatch.setattr(
        orch_mod,
        "generate_conversational_response",
        lambda user_text, state, chat_session: ChatMessage.assistant("ok"),
    )

    messages = ConversationOrchestrator().handle_turn(session, "hi")

    assert [m.content for m in messages] == ["ok"]
