"""Task type definitions for VLM analysis."""

from enum import Enum


class TaskType(Enum):
    """Supported analysis task types.

    All analysis now funnels through CUSTOM — the VLM receives the
    user's natural-language instruction directly instead of a hard-coded
    prompt template.  The other members are kept as aliases so that
    callers and the proactive-agent decision prompt can still emit
    recognisable task names, but they all resolve to the same code path.
    """

    CUSTOM = "custom"

    # kept as aliases for backward-compat / readability
    PACKAGE_DETECTION = "package_detection"
    PPE_DETECTION = "ppe_detection"
    PERSON_COUNTING = "person_counting"
    SCENE_DESCRIPTION = "scene_description"
    PCB_INSPECTION = "pcb_inspection"
    RETAIL_BILLING = "retail_billing"
