"""Vision Language Model modules for AI-powered analysis."""

from .task_types import TaskType
from .prompts import (
    TASK_PROMPTS,
    DEFAULT_DETECTION_PROMPT,
    PACKAGE_DETECTION_PROMPT,
    PPE_DETECTION_PROMPT,
    PERSON_COUNTING_PROMPT,
    SCENE_DESCRIPTION_PROMPT,
    CUSTOM_QUERY_TEMPLATE,
)
from .client import (
    VLMBackend,
    UnifiedVLMClient,
    AnalysisResult,
    DetectionResult,
    AgenticResult,
    ALERT_CONDITIONS,
)

__all__ = [
    # Task types
    "TaskType",
    # Prompts
    "TASK_PROMPTS",
    "DEFAULT_DETECTION_PROMPT",
    "PACKAGE_DETECTION_PROMPT",
    "PPE_DETECTION_PROMPT",
    "PERSON_COUNTING_PROMPT",
    "SCENE_DESCRIPTION_PROMPT",
    "CUSTOM_QUERY_TEMPLATE",
    # Client
    "VLMBackend",
    "UnifiedVLMClient",
    "AnalysisResult",
    "DetectionResult",
    "AgenticResult",
    "ALERT_CONDITIONS",
]
