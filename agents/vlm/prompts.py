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
from typing import Any, Dict, List, Optional

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

The board under inspection is an **Arduino Uno R4 Minima**. It is expected to have:
  • A black cylindrical DC barrel-jack power connector on one edge.
  • A USB-C port on an adjacent edge.
  • Two rows of through-hole header pins (one along each long edge).
  • A main microcontroller IC (square QFP package) and supporting SMD components.

RULES — read before answering:
  1. Describe ONLY what you actually see in the image.
  2. The barrel-jack connector is a chunky, black, cylindrical part that sticks up from the board — it is NOT flat pads. If you can see such a 3-D connector AND it is straight, undamaged, and properly seated, mark it "present_intact".
  3. Do NOT confuse viewing angle, shadows, or lighting with missing parts.
  4. If a component area is occluded or unclear, say "uncertain" — do NOT default to "missing".
  5. "present_intact" means the component is BOTH present AND in perfect physical condition. If a component is present but shows ANY physical damage (bent, deformed, tilted, cracked, lifted, displaced), mark it "damaged" — NOT "present_intact".

WHAT "DAMAGED" LOOKS LIKE — check carefully for each connector:
  • Bent or deformed metal tabs, shields, or housings (metal sticking out at an angle).
  • Connector body tilted, lifted, or not flush with the PCB.
  • Cracked or broken plastic housing.
  • Pins or leads visibly bent, splayed, or lifted from pads.
  • Any part of the connector that looks physically stressed, warped, or out of its normal shape.

INSPECTION CHECKLIST:

Q1 — DC BARREL JACK: Locate the barrel-jack connector (black cylinder, ~9 mm tall).
  a) Is it physically present on the board?
  b) Is the metal housing straight and properly seated (not bent, tilted, or deformed)?
  c) Are the metal ground tabs/shields flat and in their normal position (not bent outward or upward)?
  If present but any part is bent, deformed, or physically damaged → "damaged".

Q2 — USB PORT: Locate the USB-C port.
  a) Is it physically present?
  b) Is the metal shield/housing straight and undamaged (no bent tabs, no dents)?
  c) Is it properly seated flush with the board edge?
  If present but physically damaged → "damaged".

Q3 — HEADER PINS: Are the two rows of header pins present? Do they appear straight and properly soldered? Any bent, missing, or crooked pins → "damaged".

Q4 — OTHER DEFECTS: Do you see any of the following?
  • Solder bridges or shorts between adjacent pads/pins.
  • Obviously missing ICs, capacitors, resistors, or other passives.
  • Cold or insufficient solder joints.
  • Cracked, cut, or scratched traces on the PCB.
  • Burn marks, discoloration from overheating, or scorch marks.
  • Mechanical damage: cracked PCB substrate, chips in the board edge, bent or broken components, foreign objects or debris on the board.

Respond with ONLY this JSON:
{
    "details": {
        "power_jack_status": "present_intact" or "missing" or "damaged" or "uncertain",
        "usb_port_status": "present_intact" or "missing" or "damaged" or "uncertain",
        "header_pins_status": "present_intact" or "missing" or "damaged" or "uncertain",
        "defects": []
    },
    "detected": <DEFECT_BOOLEAN — see decision rule below>,
    "confidence": <float 0-1>,
    "reasoning": "<brief observations from Q1-Q4>",
    "should_alert": <same value as detected>
}

DECISION RULE (apply AFTER filling in details):
  detected = true ONLY IF at least one of these is true:
    • power_jack_status is "missing" or "damaged"
    • usb_port_status is "missing" or "damaged"
    • header_pins_status is "missing" or "damaged"
    • defects list is non-empty
  Otherwise detected = false.
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


def build_prompt_parts(
    task_type: TaskType,
    cv_context: Optional[Dict[str, Any]] = None,
    user_query: Optional[str] = None,
) -> tuple[str, str]:
    """Split the VLM prompt into its stable and per-frame halves.

    Returns ``(instructions, per_frame_note)``.

    *instructions* is identical for every frame inspected under the same
    configuration; *per_frame_note* carries the upstream CV hints, which
    change frame to frame. Callers place the note **after** the image so the
    instruction block stays a byte-identical prefix across inspections and
    vLLM's prefix cache (``--enable-prefix-caching``) can reuse its KV.
    Concatenating the two reproduces the single string ``build_prompt``
    has always returned.
    """
    effective_query = user_query or "Describe what you see in this image in detail."

    # If the caller already embedded a full JSON schema in the query,
    # send it directly so the VLM sees exactly one schema.
    if _HAS_JSON_SCHEMA_RE.search(effective_query):
        instructions = effective_query
    else:
        instructions = CUSTOM_QUERY_TEMPLATE.format(user_query=effective_query)

    if not cv_context:
        return instructions, ""

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

    if not context_parts:
        return instructions, ""

    return instructions, "\n\nNote: " + " ".join(context_parts)


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

    Prefer ``build_prompt_parts`` when the caller can place the per-frame
    note after the image; this joined form keeps the varying text inside the
    cacheable prefix.
    """
    instructions, note = build_prompt_parts(task_type, cv_context, user_query)
    return instructions + note


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
