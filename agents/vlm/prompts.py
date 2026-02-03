"""Prompt templates for different VLM analysis tasks."""

from __future__ import annotations

from typing import Dict

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

# Registry mapping task types to their prompts
TASK_PROMPTS: Dict[TaskType, str] = {
    TaskType.PACKAGE_DETECTION: PACKAGE_DETECTION_PROMPT,
    TaskType.PPE_DETECTION: PPE_DETECTION_PROMPT,
    TaskType.PERSON_COUNTING: PERSON_COUNTING_PROMPT,
    TaskType.SCENE_DESCRIPTION: SCENE_DESCRIPTION_PROMPT,
}

# Legacy alias
DEFAULT_DETECTION_PROMPT = PACKAGE_DETECTION_PROMPT
