"""Prompt templates for VLM analysis tasks.

All task-specific prompts have been removed.  Every analysis now goes
through the single CUSTOM_QUERY_TEMPLATE — the user (or the system)
supplies a natural-language instruction and the VLM responds with
structured JSON.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .task_types import TaskType


# =============================================================================
# SINGLE PROMPT TEMPLATE — used for all analysis tasks
# =============================================================================

CUSTOM_QUERY_TEMPLATE = """Analyze this image based on the following instructions:

{user_query}

Respond with ONLY valid JSON (no other text):
{{
  "detected": boolean (true if the condition in the instructions is met),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Your analysis and findings",
  "should_alert": boolean (true if user should be alerted based on the instructions),
  "details": {{any additional structured data}}
}}
"""

# Legacy aliases — all resolve to the same template.
DEFAULT_DETECTION_PROMPT = CUSTOM_QUERY_TEMPLATE
TASK_PROMPTS: Dict[TaskType, str] = {}  # empty; _build_prompt falls through to CUSTOM_QUERY_TEMPLATE


def get_prompt(task_type: TaskType, custom_query: str = "") -> str:
    """Get the prompt for a given task type.

    All task types now use the custom query template.
    """
    effective_query = custom_query or "Describe what you see in this image in detail."
    return CUSTOM_QUERY_TEMPLATE.format(user_query=effective_query)


# =============================================================================
# Proactive monitoring prompt builders
# =============================================================================

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
