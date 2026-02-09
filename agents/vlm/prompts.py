"""Prompt templates for different VLM analysis tasks."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .task_types import TaskType


# =============================================================================
# PROMPT TEMPLATES FOR DIFFERENT TASK TYPES
# =============================================================================

# Package/Shipping Box Detection
PACKAGE_DETECTION_PROMPT = """Analyze this image for shipping/packaging boxes.

TASK: Look for brown cardboard shipping boxes and check if they have shipping labels.

A SHIPPING LABEL is: White/light paper sticker with printed address, barcode, or tracking info.
NOT a shipping label: Product logos, handwritten text, tape, or markings printed on cardboard.

Respond with ONLY valid JSON (no other text):
{
  "detected": boolean (true or false),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Brief description of what you see",
  "box_count": integer,
  "shipping_label_present": boolean or null
}

RULES:
- detected=true if ANY cardboard shipping box is visible
- shipping_label_present=false if ANY box lacks a proper shipping label
- shipping_label_present=true only if ALL boxes have shipping labels
- If no boxes, set detected=false, box_count=0, shipping_label_present=null
"""

# PPE (Personal Protective Equipment) Detection
PPE_DETECTION_PROMPT = """Analyze this image for Personal Protective Equipment (PPE) compliance.

TASK: Identify all people in the scene and check if they are wearing required safety equipment.

PPE items to detect:
- Hard hat / Safety helmet
- Reflective vest / High-visibility jacket
- Safety glasses / Goggles
- Gloves

Respond with ONLY valid JSON (no other text):
{
  "detected": boolean (true if any people visible),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Brief description of PPE compliance status",
  "person_count": integer,
  "helmet_count": integer (people wearing hard hats),
  "no_helmet_count": integer (people without hard hats),
  "reflective_vest_count": integer,
  "no_reflective_vest_count": integer,
  "compliance_status": "compliant" or "non_compliant" or "no_people"
}

RULES:
- detected=true if ANY person is visible
- compliance_status="non_compliant" if ANY person lacks required PPE
- compliance_status="compliant" only if ALL people have required PPE
- If no people, set detected=false, all counts=0, compliance_status="no_people"
"""

# Person Counting
PERSON_COUNTING_PROMPT = """Count the number of people visible in this image.

TASK: Accurately count all visible people, including partially visible individuals.

Respond with ONLY valid JSON (no other text):
{
  "detected": boolean (true if any people visible),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Brief description of what you see",
  "person_count": integer,
  "fully_visible": integer (people fully in frame),
  "partially_visible": integer (people partially in frame)
}

RULES:
- Count each person only once
- Include people in the background if clearly identifiable as human
- detected=true if person_count > 0
"""

# General Scene Description
SCENE_DESCRIPTION_PROMPT = """Describe what you see in this image.

TASK: Provide a detailed but concise description of the scene.

Respond with ONLY valid JSON (no other text):
{
  "detected": true,
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Detailed description of the scene",
  "scene_type": "indoor" or "outdoor" or "unknown",
  "objects": ["list", "of", "main", "objects"],
  "activity": "Description of any activity or action happening"
}
"""

# Custom query template (user provides the task)
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

# PCB (Printed Circuit Board) Inspection
PCB_INSPECTION_PROMPT = """Analyze this image for PCB (Printed Circuit Board) defects and quality issues.

TASK: Inspect the PCB for manufacturing defects, component placement issues, and board identification.

Defect types to look for:
- Solder bridges (unintended solder connections between pads/traces)
- Missing components (empty pads where components should be placed)
- Cold solder joints (dull, grainy, or cracked solder connections)
- Trace cuts or damage (broken copper traces)
- Component misalignment (rotated or offset components)
- Burn marks or discoloration
- Lifted pads or delamination

Board identification:
- Identify the board type if recognizable (e.g., Arduino Uno, Raspberry Pi, custom PCB)
- Note any visible text, logos, or version markings

Respond with ONLY valid JSON (no other text):
{{
  "detected": boolean (true if a PCB is visible in the image),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Detailed description of the PCB and any defects found",
  "board_type": "identified board type or 'unknown'",
  "board_markings": "any visible text, logos, or version numbers",
  "has_defects": boolean (true if any defects were detected),
  "defects": [
    {{
      "type": "defect type (solder_bridge, missing_component, cold_joint, trace_damage, misalignment, burn_mark, lifted_pad, other)",
      "severity": "low, medium, or high",
      "location": "description of where on the board",
      "description": "detailed description of the defect"
    }}
  ],
  "component_count": integer (estimated number of components visible),
  "overall_quality": "good, acceptable, or defective",
  "should_alert": boolean (true if any medium or high severity defects found)
}}

RULES:
- detected=true if ANY PCB is visible in the image
- has_defects=true if ANY manufacturing defect is found
- should_alert=true if any defect has medium or high severity
- If no PCB visible, set detected=false and empty defects list
- Be thorough but avoid false positives - only flag clear defects
"""

# Retail Billing / Tray Item Detection
RETAIL_BILLING_PROMPT = """Analyze this image of items placed on a tray or counter for retail billing purposes.

TASK: Identify, classify, and count each distinct item visible. This is for generating a retail bill/invoice.

For each item, determine:
- Product name / description
- Category (electronics, food, beverage, household, clothing, stationery, other)
- Quantity (count of identical items)
- Any visible price tags or barcodes
- Brand name if visible
- Size/variant if distinguishable

Respond with ONLY valid JSON (no other text):
{{
  "detected": boolean (true if any items are visible on the tray),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Overview of what is visible on the tray",
  "items": [
    {{
      "name": "product name or best description",
      "category": "electronics|food|beverage|household|clothing|stationery|other",
      "quantity": integer,
      "brand": "brand name or null",
      "variant": "size/color/variant or null",
      "visible_price": "price if visible on tag, or null",
      "barcode_visible": boolean
    }}
  ],
  "total_item_count": integer (total number of individual items),
  "total_unique_items": integer (number of distinct product types),
  "tray_description": "brief description of the tray/surface layout",
  "should_alert": false
}}

RULES:
- detected=true if ANY item is visible on the tray/counter
- Count each identical item separately in the quantity field
- Group identical items into a single entry with quantity > 1
- Be specific with product names when possible
- If a barcode or price tag is visible, note it
- should_alert is always false for retail billing
"""

# Registry mapping task types to their prompts
TASK_PROMPTS: Dict[TaskType, str] = {
    TaskType.PACKAGE_DETECTION: PACKAGE_DETECTION_PROMPT,
    TaskType.PPE_DETECTION: PPE_DETECTION_PROMPT,
    TaskType.PERSON_COUNTING: PERSON_COUNTING_PROMPT,
    TaskType.SCENE_DESCRIPTION: SCENE_DESCRIPTION_PROMPT,
    TaskType.PCB_INSPECTION: PCB_INSPECTION_PROMPT,
    TaskType.RETAIL_BILLING: RETAIL_BILLING_PROMPT,
}

# Legacy alias
DEFAULT_DETECTION_PROMPT = PACKAGE_DETECTION_PROMPT


def get_prompt(task_type: TaskType, custom_query: str = "") -> str:
    """Get the prompt for a given task type.
    
    Args:
        task_type: The task type to get the prompt for.
        custom_query: Custom query text for CUSTOM task type.
        
    Returns:
        The prompt string for the task type.
    """
    if task_type == TaskType.CUSTOM:
        return CUSTOM_QUERY_TEMPLATE.format(user_query=custom_query)
    return TASK_PROMPTS.get(task_type, SCENE_DESCRIPTION_PROMPT)


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
    """Generate the observation-stage prompt for proactive monitoring."""
    last_summary = (last_observation or {}).get("scene_summary", "unknown")
    last_signature = (last_observation or {}).get("scene_signature", "none")
    last_target_state = (last_observation or {}).get("target_state", "unknown")
    context_blob = _pretty_json_blob(context_hint)

    return f"""You are an INTELLIGENT OBSERVATION AGENT continuously monitoring a camera feed.

Your role: Perceive and describe what's happening in real-time, tracking changes and target states.

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
3. IS THE TARGET PRESENT? Based on the user's objective, identify if the relevant target/subject is in view.
4. WHAT IS THE TARGET DOING? Is it entering, moving, stopped, stable, or leaving?
5. IS IT READY FOR ANALYSIS? Would this be a good moment to run a detailed inspection?

IMPORTANT PRINCIPLES:
- Be contextually aware - understand the difference between a new object entering vs. the same object in a different state
- Maintain scene_signature consistency - same object/scene = same signature, even across multiple frames
- scene_changed should reflect MEANINGFUL changes (new objects, significant movement, scene transitions)
- target_ready means the target is present, visible, stable, and positioned appropriately for inspection
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
    """Generate the decision-stage prompt for proactive monitoring."""
    observation_blob = _pretty_json_blob(observation_payload)
    context_blob = _pretty_json_blob(context_payload)

    return f"""You are an INTELLIGENT DECISION AGENT that decides when and how to act on camera observations.

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
   Use when: Scene is empty, target is moving, already inspected recently, or nothing actionable

2. "quick_check" - Lightweight verification pass (cheap, fast)
   Use when: Want to confirm target state, verify readiness, or gather more info before full analysis

3. "full_inspection" - Comprehensive domain-specific analysis (expensive, detailed)
   Use when: Target is ready, conditions are optimal, and inspection is warranted

YOUR DECISION-MAKING PROCESS:
Think through these questions:

1. WHAT IS THE CURRENT SITUATION?
   - Is the target present and in a good state for analysis?
   - Has the scene changed significantly?

2. WHAT IS THE TEMPORAL CONTEXT?
   - How long has the scene been stable?
   - Was this already inspected recently?
   - How long since the last action?

3. WHAT DOES THE USER WANT?
   - What is the monitoring objective?
   - What would provide the most value right now?

4. WHAT IS THE OPTIMAL ACTION?
   - Is this the right moment for full inspection?
   - Do we need more information (quick_check)?
   - Should we wait for better conditions?

DECISION PRINCIPLES (not rigid rules, but intelligent guidelines):
- Empty or unchanged scenes → typically wait
- Target entering or in motion → typically wait or quick_check
- Target stable and ready, not recently inspected → consider full_inspection
- Target stable but already inspected → typically wait
- Scene changes after previous inspection → may warrant re-inspection
- Consider efficiency: avoid redundant analysis, but don't miss important moments

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
    "task": "package_detection" | "pcb_inspection" | "ppe_detection" | "person_counting" | "retail_billing" | "scene_description" | "custom",
    "custom_prompt": "Natural language instructions if task is 'custom', otherwise null",
    "notes": "Additional execution details or focus areas"
  }},
  "scene_signature": "Repeat the scene_signature from observation",
  "should_emit_event": true | false
}}

CRITICAL: Base your decision on reasoning and context, NOT on hardcoded rules or thresholds.
You are an intelligent agent, not a rule-based system.
  """


  def build_quick_check_prompt(
    instruction: str,
    observation_payload: Dict[str, Any],
    context_payload: Dict[str, Any],
  ) -> str:
    """Prompt used for lightweight quick-check confirmations."""
    observation_blob = _pretty_json_blob(observation_payload)
    context_blob = _pretty_json_blob(context_payload)

    return f"""You are performing a QUICK VERIFICATION CHECK to confirm target state and readiness.

USER'S MONITORING OBJECTIVE:
{instruction.strip() or 'Monitor the scene.'}

CURRENT OBSERVATION:
{observation_blob}

CONTEXT:
{context_blob}

YOUR QUICK CHECK TASK:
This is a lightweight, fast confirmation to answer two key questions:

1. TARGET CONFIRMATION: Is the target actually present and identifiable as expected?
2. READINESS ASSESSMENT: Is it in an optimal state for full inspection?

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
  "notes": "Brief observation about target state and readiness"
}}

Keep it fast and focused. This is a quick sanity check, not a full analysis.
  """
