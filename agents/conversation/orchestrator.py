"""The conversation turn, independent of how it was delivered.

``ConversationOrchestrator`` owns the sequence that used to live inside
the Socket.IO handler: interpret → route to a domain MCP → submit a
proposal → turn the outcome into messages. It emits nothing; progress and
approval prompts go to the :class:`ConversationEventSink` supplied at
construction, and the messages are returned.

No tools are executed directly here. A proposal is either auto-approved
by policy or surfaced for explicit approval, which is what makes
``approve``/``reject`` separate entry points rather than a continuation
of ``process_message``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.mcp.base import (
    get_agent_state_machine,
    get_mcp_executor,
)
from agents.mcp.manager import DOMAIN_PCB, get_mcp_manager
from core.logging import get_logger

from .events import ConversationEventSink, NullEventSink
from .messages import ChatMessage
from .naming import friendly_tool_name
from .responses import (
    generate_conversational_response,
    generate_tool_followup_response,
)
from .session import ChatSession

logger = get_logger(__name__)


class ConversationOrchestrator:
    """Runs chat turns against the MCP layer.

    Args:
        events: Where progress/approval notifications go. Defaults to a
            sink that discards them, which is what a caller wanting only
            the returned messages wants.
    """

    def __init__(self, events: Optional[ConversationEventSink] = None) -> None:
        self._events = events or NullEventSink()

    @property
    def events(self) -> ConversationEventSink:
        return self._events

    # ── full turn ─────────────────────────────────────────────────────

    def handle_turn(
        self,
        chat_session: ChatSession,
        user_text: str,
    ) -> List[ChatMessage]:
        """Run a complete turn and return every message it produced.

        This is the entry point both transports should use: it covers the
        tool result *and* the natural-language follow-up that summarises
        it, which a caller reproducing the sequence by hand would have to
        remember to ask for.

        Messages are appended to *chat_session* (and so persisted) and
        handed to ``events.message`` as they land, so a streaming
        transport can deliver them without waiting for the turn to end.
        """
        produced: List[ChatMessage] = []

        def _record(message: ChatMessage) -> None:
            chat_session.add_message(message)
            produced.append(message)
            self._events.message(message, chat_session=chat_session)

        try:
            response = self.process_message(chat_session, user_text)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Error processing chat message: %s", exc)
            _record(
                ChatMessage.assistant(
                    f"I apologize, but I encountered an error: {exc}. "
                    "Please try again."
                )
            )
            return produced

        _record(response)

        followup = self._followup_for(chat_session, user_text, response)
        if followup is not None:
            _record(followup)

        return produced

    def _followup_for(
        self,
        chat_session: ChatSession,
        user_text: str,
        response: ChatMessage,
    ) -> Optional[ChatMessage]:
        """Natural-language summary of a *successful* tool result, if any."""
        if response.role != "tool":
            return None

        metadata = response.metadata or {}
        tool_result = metadata.get("result") if isinstance(metadata, dict) else None
        if not isinstance(tool_result, dict) or not tool_result.get("success"):
            return None

        return generate_tool_followup_response(
            user_text=user_text,
            tool_result=tool_result,
            state=get_agent_state_machine().state,
            chat_session=chat_session,
        )

    # ── approval lifecycle ────────────────────────────────────────────

    def approve(
        self,
        proposal_id: str,
        *,
        domain: Optional[str] = None,
    ) -> Tuple[ChatMessage, Dict[str, Any]]:
        """Approve a pending proposal. Returns (message, raw result)."""
        result = get_mcp_manager().approve_proposal(proposal_id, domain=domain)

        if result["status"] == "executed":
            tool_result = result["result"]
            output = tool_result.get("output", {})
            message_text = output.get("message", "Tool executed successfully")
            display_name = friendly_tool_name(tool_result["tool_name"])
            message = ChatMessage.tool(
                content=(
                    f"✅ **{display_name}** executed successfully.\n\n{message_text}"
                ),
                tool_name=tool_result["tool_name"],
                success=True,
                metadata={"result": tool_result},
            )
        elif result["status"] == "failed":
            message = ChatMessage.tool(
                content=f"❌ Tool execution failed: {result['error']}",
                tool_name=result.get("result", {}).get("tool_name", "unknown"),
                success=False,
                metadata={"error": result["error"]},
            )
        else:
            message = ChatMessage.assistant(
                f"⚠️ Unexpected result: {result.get('reason', 'Unknown error')}"
            )

        self._events.state_changed()
        return message, result

    def reject(
        self,
        proposal_id: str,
        *,
        reason: str = "User rejected",
        domain: Optional[str] = None,
    ) -> Tuple[ChatMessage, Dict[str, Any]]:
        """Reject a pending proposal. Returns (message, raw result)."""
        result = get_mcp_manager().reject_proposal(
            proposal_id, reason=reason, domain=domain
        )
        message = ChatMessage.assistant(f"🚫 Proposal rejected: {reason}")
        return message, result

    # ── single message ────────────────────────────────────────────────

    def process_message(
        self,
        chat_session: ChatSession,
        user_text: str,
    ) -> ChatMessage:
        """Process a user message through the interpretation phase.

        This function:
        1. Routes to the correct domain MCP (pcb / general)
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

        self._events.activity(
            status="in_progress",
            title="Understanding Request",
            detail="Reading your message and selecting the best next action.",
            chat_session=chat_session,
        )

        # INTERPRETATION PHASE: Route to the correct domain and detect intent
        resolved_domain, proposal = manager.interpret(
            message=user_text,
            agent_state=current_state,
            session_id=session_id_str,
        )

        # If no tool call needed, generate conversational response
        if proposal is None:
            self._events.activity(
                status="completed",
                title="Response Ready",
                detail="Prepared a direct response without running any actions.",
                chat_session=chat_session,
            )
            return generate_conversational_response(user_text, current_state, chat_session)

        # If proposal was rejected during interpretation (e.g., state violation)
        if proposal.is_rejected:
            self._events.activity(
                status="failed",
                title="Action Unavailable",
                detail=proposal.rejection_reason or "The requested action is not available in the current state.",
                tool_name=proposal.tool_name,
                chat_session=chat_session,
            )
            return ChatMessage.assistant(
                f"⚠️ I understood your request, but I can't do that right now.\n\n"
                f"**Reason:** {proposal.rejection_reason}\n\n"
                f"The agent is currently in **{current_state.value}** state."
            )

        # EXECUTION PHASE: Submit proposal through the correct domain executor
        friendly_name = friendly_tool_name(proposal.tool_name)
        self._events.activity(
            status="in_progress",
            title="Executing Action",
            detail=f"Running: {friendly_name}.",
            tool_name=proposal.tool_name,
            chat_session=chat_session,
        )

        result = manager.submit(proposal, domain=resolved_domain)

        if result["status"] == "pending_approval":
            # Tool requires user confirmation
            self._events.pending_proposal(result, chat_session=chat_session)
            self._events.activity(
                status="requires_approval",
                title="Approval Needed",
                detail=f"{friendly_name} needs your confirmation before it can continue.",
                tool_name=proposal.tool_name,
                chat_session=chat_session,
            )

            domain_label = ""
            if resolved_domain == DOMAIN_PCB:
                domain_label = f" [{resolved_domain.upper()}]"

            return ChatMessage.assistant(
                f"🔔 **Confirmation Required**{domain_label}\n\n"
                f"I'd like to execute **{friendly_name}**.\n\n"
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
            self._events.state_changed()

            if success:
                message_text = output.get("message", "Done")
                data = output.get("data", {})

                self._events.activity(
                    status="completed",
                    title="Action Completed",
                    detail=f"{friendly_name} finished successfully.",
                    tool_name=proposal.tool_name,
                    chat_session=chat_session,
                )

                response_content = (
                    f"✅ **{friendly_name}** completed.\n\n{message_text}"
                )

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
                    metadata={
                        "result": tool_result,
                        "proposal_id": proposal.id,
                        "display_name": friendly_name,
                    },
                )
            else:
                self._events.activity(
                    status="failed",
                    title="Action Failed",
                    detail=f"{friendly_name} could not be completed.",
                    tool_name=proposal.tool_name,
                    chat_session=chat_session,
                )
                return ChatMessage.tool(
                    content=f"❌ **{friendly_name}** failed: {tool_result.get('error', 'Unknown error')}",
                    tool_name=proposal.tool_name,
                    success=False,
                    metadata={
                        "result": tool_result,
                        "proposal_id": proposal.id,
                        "display_name": friendly_name,
                    },
                )

        elif result["status"] == "rejected":
            # Proposal was rejected (validation or state error)
            self._events.activity(
                status="failed",
                title="Action Rejected",
                detail=result["reason"],
                tool_name=proposal.tool_name,
                chat_session=chat_session,
            )
            return ChatMessage.assistant(
                f"⚠️ I couldn't execute that request.\n\n" f"**Reason:** {result['reason']}"
            )

        elif result["status"] == "failed":
            # Execution failed
            self._events.activity(
                status="failed",
                title="Execution Error",
                detail=result["error"],
                tool_name=proposal.tool_name,
                chat_session=chat_session,
            )
            return ChatMessage.tool(
                content=f"❌ Execution failed: {result['error']}",
                tool_name=proposal.tool_name,
                success=False,
                metadata={"error": result["error"], "proposal_id": proposal.id},
            )

        # Unexpected status
        return ChatMessage.assistant(f"⚠️ Unexpected result: {result}")
