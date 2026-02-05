"""Task type definitions for VLM analysis."""

from enum import Enum


class TaskType(Enum):
    """Supported analysis task types."""
    PACKAGE_DETECTION = "package_detection"
    PPE_DETECTION = "ppe_detection"
    PERSON_COUNTING = "person_counting"
    SCENE_DESCRIPTION = "scene_description"
    PCB_INSPECTION = "pcb_inspection"
    RETAIL_BILLING = "retail_billing"
    CUSTOM = "custom"
