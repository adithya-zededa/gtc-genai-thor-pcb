"""Vision Language Model modules for AI-powered analysis."""

from .task_types import TaskType
from .prompts import (
    TASK_PROMPTS,
    DEFAULT_DETECTION_PROMPT,
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
    "CUSTOM_QUERY_TEMPLATE",
    # Client
    "VLMBackend",
    "UnifiedVLMClient",
    "AnalysisResult",
    "DetectionResult",
    "AgenticResult",
    "ALERT_CONDITIONS",
]
