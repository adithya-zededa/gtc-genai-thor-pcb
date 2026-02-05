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
