"""Natural-language response generation for the chat turn.

Everything here runs on the *agent* model — it reasons over the vision
model's structured output and over tool results, and never sees pixels.
Deterministic formatters exist only as a fallback for when the agent
endpoint is unreachable, so the UI degrades to plain summaries instead of
going silent.
"""

from __future__ import annotations

import json as _json
from typing import Any, Dict, Optional

from agents.mcp.base import AgentState
from core.logging import get_logger

from .messages import ChatMessage
from .session import ChatSession

logger = get_logger(__name__)


def get_chat_router():
    """Get the LLM router for chat responses, or None if not configured.

    Chat is the agent model's job — it reasons over the vision model's
    structured output and drives tools, and never needs to see pixels.
    """
    try:
        from router import get_agent_router

        return get_agent_router()
    except Exception as exc:
        logger.debug("LLM router not available for chat: %s", exc)
    return None


def generate_llm_response(
    user_text: str,
    state: AgentState,
    chat_session: ChatSession,
) -> Optional[ChatMessage]:
    """Generate a response via the vLLM router.

    Returns None if the router is not available or the call fails,
    so callers can fall back to the hardcoded response.
    """
    router = get_chat_router()
    if router is None:
        return None

    try:
        # Build messages with system prompt + recent history for context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant embedded in an industrial camera "
                    "monitoring system. You help users with PCB inspection and "
                    "camera monitoring tasks.\n\n"
                    f"The agent is currently in **{state.value}** state.\n\n"
                    "Available capabilities:\n"
                    "- PCB inspection: inspect boards, classify boards, detect defects, send alerts\n"
                    "- Camera monitoring: start/stop monitoring, analyze frames\n\n"
                    "Keep responses concise and helpful. Use markdown formatting.\n"
                    "Respond directly without internal reasoning or analysis preamble."
                ),
            },
        ]

        # Add recent chat history for context (last 6 messages)
        for msg in chat_session.get_history(limit=6):
            role = msg.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": msg.get("content", "")})

        # Add the current user message if not already at the end
        if not messages or messages[-1].get("content") != user_text:
            messages.append({"role": "user", "content": user_text})

        response = router.chat(
            messages=messages,
            # Disable Qwen3 thinking for faster conversational responses
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if response and response.content:
            return ChatMessage.assistant(
                response.content,
                metadata={
                    "llm_provider": response.provider,
                    "llm_model": response.model,
                    "generated": True,
                },
            )
    except Exception as exc:
        logger.warning("LLM router chat failed: %s", exc)

    return None


def deterministic_tool_summary(tool_name: str, output_message: str, output_data: Any) -> Optional[str]:
    """Return a plain-text summary of a tool result without using the LLM.

    Used as a fallback when the LLM router is unavailable.
    """
    if tool_name == "count_defective_pcbs" and isinstance(output_data, dict):
        return (
            f"Based on the current records, {output_data.get('total', 0)} defective PCB(s) "
            f"were detected ({output_data.get('time_window', 'all time')}): "
            f"{output_data.get('high', 0)} high, {output_data.get('medium', 0)} medium, "
            f"{output_data.get('low', 0)} low severity."
        )
    if tool_name == "query_pcb_inspections" and isinstance(output_data, dict):
        time_window = output_data.get("time_window", "all time")
        fail_types = output_data.get("fail_defect_types", {})
        fail_detail = ""
        if fail_types:
            parts = [f"{dt}: {cnt}" for dt, cnt in fail_types.items()]
            fail_detail = f" Failure breakdown: {'; '.join(parts)}."
        return (
            f"I found {output_data.get('total', 0)} inspection(s) ({time_window}), "
            f"showing {output_data.get('showing', 0)} recent record(s): "
            f"{output_data.get('pass_count', 0)} pass and {output_data.get('fail_count', 0)} fail."
            f"{fail_detail}"
        )
    if tool_name == "get_defect_type_breakdown" and isinstance(output_data, dict):
        defect_types = output_data.get("types", [])
        if defect_types:
            top = defect_types[0]
            return (
                f"The top recorded defect type is {top.get('defect_type', 'unknown')} "
                f"with {top.get('count', 0)} occurrence(s) ({top.get('percentage', 0)}%)."
            )
        return "No defect types are recorded yet."
    if tool_name == "get_latest_defect":
        if output_data and isinstance(output_data, dict):
            return (
                f"The last detection was {output_data.get('defect_type', 'unknown')} "
                f"(severity: {output_data.get('severity', 'unknown')}) "
                f"recorded at {output_data.get('timestamp', 'unknown')}."
            )
        return output_message or "No defects have been recorded yet."
    if tool_name == "get_most_severe_defect":
        if output_data and isinstance(output_data, dict):
            return (
                f"The most severe defect on record is {output_data.get('defect_type', 'unknown')} "
                f"(severity: {output_data.get('severity', 'unknown')}) "
                f"detected at {output_data.get('timestamp', 'unknown')}."
            )
        return output_message or "No defects have been recorded yet."
    if tool_name == "get_defect_summary" and isinstance(output_data, dict):
        sev = output_data.get("severity_breakdown", {})
        return (
            f"Defect summary ({output_data.get('time_window', 'all time')}): "
            f"{output_data.get('total', 0)} total — "
            f"{sev.get('high', 0)} high, {sev.get('medium', 0)} medium, "
            f"{sev.get('low', 0)} low severity."
        )
    if tool_name == "get_defect_trend" and isinstance(output_data, dict):
        return (
            f"Defect trend over the last {output_data.get('window_hours', '?')}h: "
            f"{output_data.get('trend', 'unknown').upper()}. "
            f"First half rate: {output_data.get('first_half_rate', '?')}/period, "
            f"second half rate: {output_data.get('second_half_rate', '?')}/period. "
            f"Total: {output_data.get('total_defects', 0)} defect(s)."
        )
    if tool_name == "get_top_defect_sources" and isinstance(output_data, dict):
        sources = output_data.get("sources", [])
        if sources:
            parts = [f"{s.get('board_type', '?')}: {s.get('count', 0)}" for s in sources[:5]]
            return f"Top defect sources ({output_data.get('time_window', 'all time')}): {'; '.join(parts)}."
        return "No defect sources recorded yet."
    if tool_name == "generate_summary_report" and isinstance(output_data, dict):
        return (
            f"{output_data.get('period', 'Summary').capitalize()} report: "
            f"{output_data.get('total_defects', 0)} defect(s), "
            f"trend: {(output_data.get('trend') or {}).get('trend', 'unknown')}."
        )
    if tool_name == "check_threshold_alerts" and isinstance(output_data, dict):
        if output_data.get("threshold_exceeded"):
            alerts = output_data.get("alerts", [])
            msgs = [a.get("message", "") for a in alerts]
            return "THRESHOLD EXCEEDED: " + "; ".join(msgs)
        return (
            f"All thresholds OK. {output_data.get('total_defects', 0)} defect(s) "
            f"in the last {output_data.get('window_hours', '?')}h."
        )
    if tool_name == "get_defect_insights" and isinstance(output_data, dict):
        recommendations = output_data.get("recommendations", [])
        risk = output_data.get("risk_level", "unknown")
        trend = output_data.get("trend_direction", "unknown")
        rec_str = " ".join(recommendations[:3]) if recommendations else "No recommendations."
        return f"Risk level: {risk.upper()}. Trend: {trend}. {rec_str}"
    if tool_name == "get_monitoring_status" and isinstance(output_data, dict):
        active = "active" if output_data.get("monitoring_active") else "inactive"
        return (
            f"Monitoring is {active}. "
            f"{output_data.get('defects_last_24h', 0)} defect(s) in the last 24h, "
            f"{output_data.get('total_defects', 0)} total."
        )
    if tool_name in ("inspect_pcb", "inspect_pcb_frame") and isinstance(output_data, dict):
        detected = bool(output_data.get("detected", False))
        confidence = output_data.get("confidence", 0)
        description = output_data.get("description", "")
        return (
            f"Visual inspection result: {'defect indicated' if detected else 'no defect indicated'} "
            f"(confidence {float(confidence):.2f}). "
            f"{description} "
            "Note: image analysis cannot confirm non-visual faults such as electrical performance issues."
        ).strip()
    return None


def generate_tool_followup_response(
    *,
    user_text: str,
    tool_result: Dict[str, Any],
    state: AgentState,
    chat_session: ChatSession,
) -> Optional[ChatMessage]:
    """Generate a natural-language follow-up after tool execution.

    The LLM always summarises the response, receiving the full tool output as
    grounded context so it cannot hallucinate.  Deterministic formatters are
    used only as a fallback when the LLM router is unavailable.
    """
    output = tool_result.get("output", {}) if isinstance(tool_result, dict) else {}
    tool_name = str(tool_result.get("tool_name", "action"))
    output_message = str(output.get("message", "")) if isinstance(output, dict) else ""
    output_data = output.get("data", {}) if isinstance(output, dict) else {}

    # Fallback: if output has no "message" key, check for "error"
    if not output_message and isinstance(output, dict):
        output_message = str(output.get("error", ""))

    # Fallback: if "data" is missing/empty/None, collect all extra keys from
    # output so that tools not following the message/data convention still have
    # their return values visible to the LLM.
    _STANDARD_KEYS = {"success", "message", "data", "error"}
    if not output_data and isinstance(output, dict):
        extra = {k: v for k, v in output.items() if k not in _STANDARD_KEYS}
        if extra:
            output_data = extra

    router = get_chat_router()
    if router is not None:
        try:
            # Build recent conversation history (user/assistant turns only, skip activity/system)
            history_messages = []
            for msg in chat_session.get_history(limit=12):
                role = msg.get("role", "")
                ui_type = (msg.get("metadata") or {}).get("ui")
                if role in ("user", "assistant") and ui_type != "activity":
                    history_messages.append({"role": role, "content": msg.get("content", "")})

            # Remove the last user turn from history — we'll add it explicitly below
            # so the tool result sits between the question and the final answer request
            if history_messages and history_messages[-1].get("role") == "user":
                history_messages = history_messages[:-1]

            # Format output_data as readable plain text instead of JSON to
            # avoid escaped newlines that the LLM may misread.
            if isinstance(output_data, dict):
                formatted_parts = []
                for key, value in output_data.items():
                    if isinstance(value, str):
                        formatted_parts.append(f"**{key.replace('_', ' ').title()}:**\n{value}")
                    else:
                        formatted_parts.append(
                            f"**{key.replace('_', ' ').title()}:** {_json.dumps(value, default=str)}"
                        )
                formatted_data = "\n\n".join(formatted_parts) if formatted_parts else "No additional data."
            elif isinstance(output_data, str):
                formatted_data = output_data
            else:
                formatted_data = _json.dumps(output_data, default=str)[:3000]

            messages = [
                # 1. System prompt: persona + instructions only, no data
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant embedded in an industrial PCB camera monitoring system. "
                        "Answer questions accurately and concisely. Use markdown for emphasis where helpful. "
                        "Ground your answers in the tool results provided in the conversation. "
                        "Do not invent dates, counts, defect names, severities, or any values not present "
                        "in the tool result. If a value is missing, say it is not available. "
                        "IMPORTANT: Always quote exact numbers, durations, counts, and identifiers "
                        "directly from the tool result. Never approximate or round values. "
                        f"The agent is currently in **{state.value}** state."
                    ),
                },
                # 2. Recent conversation memory
                *history_messages[-6:],
                # 3. The user's original question
                {"role": "user", "content": user_text},
                # 4. Simulated assistant turn: "I ran the tool and got this result"
                {
                    "role": "assistant",
                    "content": (
                        f"I ran the `{tool_name}` tool. Here is the result:\n\n"
                        f"**Message:** {output_message}\n\n"
                        f"{formatted_data}"
                    ),
                },
                # 5. User asks for the summary — forces the model to synthesise from the result above
                {
                    "role": "user",
                    "content": "Based on that result, please give me a clear, concise answer to my question.",
                },
            ]

            response = router.chat(
                messages=messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            if response and response.content and response.content.strip():
                return ChatMessage.assistant(
                    response.content.strip(),
                    metadata={
                        "llm_provider": response.provider,
                        "llm_model": response.model,
                        "generated_from_tool": True,
                        "grounded": True,
                    },
                )
        except Exception as exc:
            logger.warning("LLM tool follow-up failed, falling back to deterministic: %s", exc)

    # LLM unavailable — use deterministic formatter as fallback
    try:
        fallback = deterministic_tool_summary(tool_name, output_message, output_data)
        if fallback:
            return ChatMessage.assistant(
                fallback,
                metadata={"generated_from_tool": True, "grounded": True},
            )
    except Exception as exc:
        logger.warning("Deterministic tool follow-up failed: %s", exc)

    return ChatMessage.assistant(output_message or "The action completed successfully.")


def generate_welcome_message(state: AgentState) -> str:
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
        f'- **"Inspect the PCB"** - Analyze a board for defects\n'
        f'- **"Send an alert if you see a defective Arduino"** - Defect alert\n'
        f'- **"Generate a defect report"** - View defect summary\n\n'
        f"### 📷 Camera Monitoring\n"
        f'- **"Start monitoring"** - Begin a new monitoring session\n'
        f'- **"End session"** or **"Stop monitoring"** - End the current session\n'
        f'- **"What do you see?"** - Analyze the current frame\n\n'
        f"---\n"
        f"💡 *All actions are controlled through conversation. "
        f"I'll ask for confirmation before doing anything sensitive.*"
    )

    return message


def generate_conversational_response(
    user_text: str,
    state: AgentState,
    chat_session: Optional[ChatSession] = None,
) -> ChatMessage:
    """Generate a conversational response for general queries.

    Uses the LLM router for intelligent responses.  Returns a minimal
    fallback only when no LLM provider is reachable.
    """
    # Try LLM-powered response via router
    if chat_session is not None:
        llm_response = generate_llm_response(user_text, state, chat_session)
        if llm_response is not None:
            return llm_response

    # Minimal degraded-mode fallback (no keyword matching)
    return ChatMessage.assistant(
        f"I'm currently unable to reach the language model, so I can't fully "
        f"process your request right now.\n\n"
        f"The agent is **{state.value}**. You can still use direct commands "
        f'like **"start monitoring"**, **"analyze the frame"**, or '
        f'**"show history"**.\n\n'
        f"Please check the LLM provider settings if this persists."
    )
