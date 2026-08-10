"""Side-channel events a conversation turn produces.

A turn returns messages, but the UI also wants to watch it happen:
progress ticks while a tool runs, an approval prompt, a state change.
Those are *notifications*, not results, so they go through a sink the
caller supplies rather than being emitted from inside the orchestrator.

That seam is the whole reason the orchestrator is transport-independent.
Socket.IO passes a sink that emits; a REST caller passes
:class:`NullEventSink` and just takes the returned messages.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .messages import ChatMessage
from .session import ChatSession


class ConversationEventSink:
    """No-op sink. Subclass and override the parts a transport cares about.

    Defaulting every method to a no-op (rather than raising) is
    deliberate: a caller that only wants the final messages should not
    have to implement three methods to get them.
    """

    def activity(
        self,
        *,
        status: str,
        title: str,
        detail: str,
        tool_name: Optional[str] = None,
        chat_session: Optional[ChatSession] = None,
    ) -> None:
        """Progress update for the turn currently being processed."""

    def pending_proposal(
        self,
        result: Dict[str, Any],
        *,
        chat_session: Optional[ChatSession] = None,
    ) -> None:
        """A tool needs explicit user approval before it can run."""

    def state_changed(self) -> None:
        """The agent state machine moved; refresh any state-derived UI."""

    def message(
        self,
        message: ChatMessage,
        *,
        chat_session: Optional[ChatSession] = None,
    ) -> None:
        """A message became available mid-turn.

        Turns can produce more than one message (a tool result, then a
        natural-language follow-up). Transports that stream deliver each
        as it lands; the returned list carries the same messages for
        transports that do not.
        """


class NullEventSink(ConversationEventSink):
    """Explicitly-named do-nothing sink, for callers with no UI to update."""
