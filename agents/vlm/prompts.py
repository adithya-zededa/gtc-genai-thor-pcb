"""Prompt templates and builders for VLM analysis tasks.

**Single source of truth** for every prompt sent to the VLM.  No other
module should compose raw prompt strings — callers use the builder
functions exported here.

Layout
~~~~~~
* Generic analysis template (``CUSTOM_QUERY_TEMPLATE``)
* PCB defect inspection prompt (``DEFAULT_MONITORING_DEFECT_PROMPT``)
* ``build_prompt()``  — the main prompt builder used by ``UnifiedVLMClient``
* ``build_agentic_prompt()`` / ``build_tools_prompt()`` — agentic tool loop
* Proactive monitoring prompt builders (observation / decision / quick-check)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .task_types import TaskType


# =====================================================================
# Generic analysis template
# =====================================================================

CUSTOM_QUERY_TEMPLATE = """Analyze this image based on the following instructions:

{user_query}

Respond with ONLY valid JSON (no other text):
{{
  "detected": boolean (true ONLY if the condition in the instructions is actually met),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Your analysis and findings",
  "should_alert": boolean (true if user should be alerted — must match detected),
  "details": {{any additional structured data}}
}}

IMPORTANT: Set "detected" based on your actual findings, not the mere mention of keywords. If the condition is NOT met, "detected" must be false.
"""

# Legacy aliases — all resolve to the same template.
DEFAULT_DETECTION_PROMPT = CUSTOM_QUERY_TEMPLATE
TASK_PROMPTS: Dict[TaskType, str] = {}  # empty; build_prompt falls through to CUSTOM_QUERY_TEMPLATE


# =====================================================================
# PCB defect inspection prompt (canonical location)
# =====================================================================

DEFAULT_MONITORING_DEFECT_PROMPT = """You are a PCB quality inspector. Look at this image and answer the questions below. Output ONLY valid JSON.

This is an Arduino Uno R4 Minima PCB on a green surface.

Answer each question by describing ONLY what you see in the image. Do NOT repeat my instructions back. Do NOT guess. If you cannot see a location clearly, say "uncertain".

Q1: Look at the LEFT EDGE of the board, upper area. Is there a tall black cylindrical barrel jack connector there? Describe the shape, color, and height of whatever object is at that location. If you only see flat copper pads or bare solder points with no tall connector, it is missing.

Q2: Look at the TOP EDGE. Is there a USB port? Describe it.

Q3: Look at the header pins along the LEFT EDGE (below the power area) and the BOTTOM EDGE. Do the header pins have small black plastic caps on them? Check both sides.

Q4: Any other defects? (solder bridges, missing parts, cold joints, trace damage, mechanical damage)

Fill in this JSON. Start by filling in "details", then derive "detected" from the details.
{
    "details": {
        "power_jack_status": "present_intact" or "missing" or "damaged" or "uncertain",
        "usb_port_status": "present_intact" or "missing" or "damaged" or "uncertain",
        "uart_cap_status": "present_both_sides" or "missing_one_side" or "missing_both_sides" or "uncertain",
        "defects": []
    },
    "detected": <DEFECT_BOOLEAN — see decision rule below>,
    "confidence": <float 0-1>,
    "reasoning": "<your observations from Q1-Q4>",
    "should_alert": <MUST be the same value as detected>
}

DECISION RULE for "detected" (apply AFTER filling in details):
  Look at the details you just wrote.
  - If power_jack_status is "missing" or "damaged" → detected = true
  - If usb_port_status is "missing" or "damaged" → detected = true
  - If uart_cap_status contains "missing" → detected = true
  - If defects list is non-empty → detected = true
  - Otherwise → detected = false
  "should_alert" must always equal "detected".
"""


# =====================================================================
# Prompt builder helpers
# =====================================================================

# Compiled pattern to detect whether a user query already embeds a
# JSON response schema (e.g. DEFAULT_MONITORING_DEFECT_PROMPT).
# If it does, wrapping it inside CUSTOM_QUERY_TEMPLATE would produce
# two competing schemas and confuse the model.
_HAS_JSON_SCHEMA_RE = re.compile(r'"detected"\s*:', re.IGNORECASE)


def get_prompt(task_type: TaskType, custom_query: str = "") -> str:
    """Get the prompt for a given task type.

    All task types now use the custom query template.
    """
    effective_query = custom_query or "Describe what you see in this image in detail."
    return CUSTOM_QUERY_TEMPLATE.format(user_query=effective_query)


def build_prompt(
    task_type: TaskType,
    cv_context: Optional[Dict[str, Any]] = None,
    user_query: Optional[str] = None,
) -> str:
    """Build the final prompt string sent to the VLM.

    * If *user_query* already contains its own JSON response schema it
      is sent verbatim (avoids the double-schema bug).
    * Otherwise it is wrapped in ``CUSTOM_QUERY_TEMPLATE``.
    * Optional *cv_context* hints from upstream detectors are appended.
    """
    effective_query = user_query or "Describe what you see in this image in detail."

    # If the caller already embedded a full JSON schema in the query,
    # send it directly so the VLM sees exactly one schema.
    if _HAS_JSON_SCHEMA_RE.search(effective_query):
        prompt = effective_query
    else:
        prompt = CUSTOM_QUERY_TEMPLATE.format(user_query=effective_query)

    if cv_context:
        context_parts = []

        pcb_count = cv_context.get("pcb_count", 0)
        rfdet_hint = cv_context.get("rfdet_hint", "")
        if pcb_count > 0:
            context_parts.append(
                f"Object detector found approximately {pcb_count} potential PCB(s)."
            )
        if rfdet_hint:
            context_parts.append(rfdet_hint)

        person_count = cv_context.get("person_count", 0)
        if person_count > 0:
            context_parts.append(f"Person detector found {person_count} person(s).")

        helmet_count = cv_context.get("helmet_count", 0)
        no_helmet_count = cv_context.get("no_helmet_count", 0)
        if helmet_count > 0 or no_helmet_count > 0:
            context_parts.append(
                f"PPE detector found {helmet_count} with helmet, "
                f"{no_helmet_count} without."
            )

        ml_confidence = cv_context.get("ml_confidence")
        if ml_confidence is not None:
            context_parts.append(f"ML confidence: {ml_confidence:.2f}")

        if context_parts:
            prompt = prompt + "\n\nNote: " + " ".join(context_parts)

    return prompt


# =====================================================================
# Agentic (tool-calling) prompt builders
# =====================================================================

def build_agentic_prompt(
    user_query: Optional[str] = None,
    tools_prompt: str = "",
) -> str:
    """Build the first-round prompt for an agentic analysis with tools.

    The returned prompt instructs the VLM to:
    1. Provide structured JSON analysis.
    2. Call tools if the user's instructions require an action.
    """
    effective_query = user_query or "Describe what you see in this image in detail."

    agentic_base = f"""Analyze this image based on the following instructions:

{effective_query}

First, provide your analysis as JSON:
{{
  "detected": boolean (true if the condition in the instructions is met),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Your analysis and findings",
  "should_alert": boolean (true if action should be taken),
  "details": {{any additional structured data}}
}}

Then, if the user's instructions require an action (like sending an email), you MUST use the tools below to complete that action."""

    return f"""{agentic_base}

---

{tools_prompt}

IMPORTANT: Follow the user's instructions exactly. If they ask you to send an email to a specific address, use THAT EXACT address in the tool call.

After your JSON analysis, if the condition in the user's instructions is met, call the appropriate tools to complete the action.
"""


def build_tool_continuation_prompt(results_text: str) -> str:
    """Build the follow-up prompt after tool calls complete."""
    return f"""Previous tool calls completed:

{results_text}

Based on these results, do you need to take any additional actions?
If yes, make more tool calls. If no, summarize what was done.
"""


def build_tools_prompt(tools: List[Dict[str, Any]]) -> str:
    """Format the available-tools block for the VLM prompt.

    *tools* is a list of dicts, each with ``name``, ``description``,
    and ``parameters`` (a JSON Schema object).
    """
    tools_desc = []
    for tool in tools:
        params_desc = json.dumps(tool.get("parameters", {}), indent=2)
        tools_desc.append(
            f"Tool: {tool['name']}\n"
            f"Description: {tool['description']}\n"
            f"Parameters: {params_desc}"
        )

    return (
        "You have access to the following tools:\n\n"
        + "\n\n".join(tools_desc)
        + "\n\nTo call a tool, use this format:\n"
        "```tool_call\n"
        '{"tool": "tool_name", "arguments": {"param1": "value1"}}\n'
        "```\n"
    )


# =====================================================================
# Proactive monitoring prompt builders
# =====================================================================

def _pretty_json_blob(data: Dict[str, Any]) -> str:
    """Format context dictionaries for inclusion in prompts."""
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def build_proactive_observation_prompt(
    instruction: str,
    last_observation: Optional[Dict[str, Any]],
    context_hint: Dict[str, Any],
  ) -> str:
    """Generate the PCB-only observation-stage prompt for proactive monitoring."""
    last_summary = (last_observation or {}).get("scene_summary", "unknown")
    last_signature = (last_observation or {}).get("scene_signature", "none")
    last_target_state = (last_observation or {}).get("target_state", "unknown")
    context_blob = _pretty_json_blob(context_hint)

    return f"""You are an INTELLIGENT OBSERVATION AGENT for PCB manufacturing quality control.

  Your role: Perceive and describe PCB manufacturing scenes in real-time, tracking PCB state.

USER'S MONITORING OBJECTIVE:
{instruction.strip() or 'Monitor the scene for anomalies'}

TEMPORAL CONTEXT:
- Previous scene: {last_summary}
- Previous signature: {last_signature}
- Previous target state: {last_target_state}
- Monitoring context: {context_blob}

YOUR TASK:
Analyze the current frame and provide a structured observation. Think about:

1. WHAT DO YOU SEE? Describe the scene naturally.
2. HAS THE SCENE CHANGED? Compare to the previous observation.
3. IS A PCB PRESENT? Identify whether the board under inspection is in view.
4. WHAT IS THE PCB STATE? Is it entering, moving, stopped, stable, or leaving?
5. IS IT READY FOR DEFECT INSPECTION? Would this be a good moment to run detailed PCB defect analysis?

IMPORTANT PRINCIPLES:
- Scope constraint: If the user objective is not about PCB manufacturing defects, output a refusal in "notes"
- and set "target_present" and "target_ready" to false.
- Be contextually aware - understand the difference between a new object entering vs. the same object in a different state
- Maintain scene_signature consistency - same object/scene = same signature, even across multiple frames
- scene_changed should reflect MEANINGFUL changes (new objects, significant movement, scene transitions)
- target_ready means the PCB is present, visible, stable, and positioned appropriately for defect inspection
- Use target_state to track temporal progression: entering → moving → stopped → stable

OUTPUT SCHEMA (JSON only, no other text):
{{
  "scene_summary": "Natural 1-2 sentence description of what you observe",
  "primary_objects": ["key", "objects", "in", "scene"],
  "target_present": true | false,
  "target_state": "entering" | "moving" | "stopped" | "stable" | "gone" | "unknown",
  "scene_changed": true | false,
  "scene_signature": "consistent-identifier-for-this-scene-or-object",
  "target_ready": true | false,
  "notes": "Additional temporal or contextual observations",
  "confidence": 0.0 - 1.0
}}

Remember: You are observing, NOT deciding. Your observations will inform the decision agent.
  """


def build_proactive_decision_prompt(
    instruction: str,
    observation_payload: Dict[str, Any],
    context_payload: Dict[str, Any],
) -> str:
    """Generate the PCB-only decision-stage prompt for proactive monitoring."""
    observation_blob = _pretty_json_blob(observation_payload)
    context_blob = _pretty_json_blob(context_payload)

    return f"""You are an INTELLIGENT DECISION AGENT for PCB manufacturing defect monitoring.

YOUR CORE RESPONSIBILITY:
Make strategic decisions about when to analyze frames based on context, temporal awareness, and user intent.

USER'S MONITORING OBJECTIVE:
{instruction.strip() or 'Monitor the scene.'}

CURRENT OBSERVATION:
{observation_blob}

HISTORICAL & TEMPORAL CONTEXT:
{context_blob}

AVAILABLE ACTIONS:
1. "wait" - Continue passive monitoring
  Use when: Scene is empty, PCB is moving, already inspected recently, or nothing actionable

2. "quick_check" - Lightweight verification pass (cheap, fast)
  Use when: Want to confirm PCB state, verify readiness, or gather more info before full analysis

3. "full_inspection" - Comprehensive domain-specific analysis (expensive, detailed)
  Use when: PCB is ready, conditions are optimal, and defect inspection is warranted

YOUR DECISION-MAKING PROCESS:
Think through these questions:

1. WHAT IS THE CURRENT SITUATION?
  - Is a PCB present and in a good state for defect analysis?
   - Has the scene changed significantly?

2. WHAT IS THE TEMPORAL CONTEXT?
   - How long has the scene been stable?
   - Was this already inspected recently?
   - How long since the last action?

3. WHAT DOES THE USER WANT?
  - Is the objective specifically PCB manufacturing defect monitoring?
  - If not PCB-specific, explicitly refuse and choose "wait".

4. WHAT IS THE OPTIMAL ACTION?
   - Is this the right moment for full inspection?
   - Do we need more information (quick_check)?
   - Should we wait for better conditions?

RUNTIME ENFORCEMENT (NON-NEGOTIABLE):
- Workflow control is enforced by deterministic conveyor FSM code, not this prompt
- You cannot start inspections, control timing, or emit events
- 10-second deadlines, one-board-one-decision, and motion/zone gating are hard runtime constraints
- Any action output here is advisory and may be ignored if FSM state disallows it

SCOPE CONSTRAINT (MANDATORY):
- If the instruction is not about PCB manufacturing defects, you MUST refuse.
- Refusal format: set action="wait", should_emit_event=false, and explain refusal in "reasoning".

THINK STRATEGICALLY:
- Balance thoroughness with resource efficiency
- Consider the user's intent and urgency
- Use temporal awareness to avoid redundant work
- Adapt to the specific monitoring context

OUTPUT SCHEMA (JSON only, no other text):
{{
  "action": "wait" | "quick_check" | "full_inspection",
  "confidence": 0.0 - 1.0,
  "reasoning": "Clear explanation of WHY this action is optimal right now, grounded in observation and context",
  "analysis_plan": {{
    "task": "custom",
    "custom_prompt": "Natural language instructions describing exactly what to analyze or inspect",
    "notes": "Additional execution details or focus areas"
  }},
  "scene_signature": "Repeat the scene_signature from observation",
  "should_emit_event": true | false
}}

CRITICAL: Provide analysis guidance only. Runtime state transitions and final event emission are enforced in code.
  """


def build_quick_check_prompt(
    instruction: str,
    observation_payload: Dict[str, Any],
    context_payload: Dict[str, Any],
) -> str:
    """Prompt used for lightweight PCB quick-check confirmations."""
    observation_blob = _pretty_json_blob(observation_payload)
    context_blob = _pretty_json_blob(context_payload)

    return f"""You are performing a QUICK VERIFICATION CHECK to confirm PCB state and readiness.

USER'S MONITORING OBJECTIVE:
{instruction.strip() or 'Monitor the scene.'}

CURRENT OBSERVATION:
{observation_blob}

CONTEXT:
{context_blob}

YOUR QUICK CHECK TASK:
This is a lightweight, fast confirmation to answer two key questions:

1. PCB CONFIRMATION: Is the PCB actually present and identifiable as expected?
2. READINESS ASSESSMENT: Is it in an optimal state for full defect inspection?

Scope constraint:
- If the instruction is non-PCB, set both booleans to false and explain refusal in "notes".

Consider:
- Is the target clearly visible and well-positioned?
- Is it stable enough (not moving or blurry)?
- Are lighting and frame conditions suitable for detailed analysis?
- Is this the right moment, or should we wait a bit longer?

OUTPUT SCHEMA (JSON only, no other text):
{{
  "target_confirmed": true | false,
  "ready_for_full_inspection": true | false,
  "confidence": 0.0 - 1.0,
  "notes": "Brief observation about PCB state and readiness"
}}

Keep it fast and focused. This is a quick sanity check, not a full analysis.
  """
