"""Vision Language Model modules for AI-powered analysis."""

from .task_types import TaskType
from .prompts import (
    TASK_PROMPTS,
    DEFAULT_DETECTION_PROMPT,
    DEFAULT_MONITORING_DEFECT_PROMPT,
    CUSTOM_QUERY_TEMPLATE,
    build_prompt,
    build_agentic_prompt,
    build_tools_prompt,
)

from .client import (
    UnifiedVLMClient,
    AnalysisResult,
    DetectionResult,
    AgenticResult,
)

__all__ = [
    # Task types
    "TaskType",
    # Prompts
    "TASK_PROMPTS",
    "DEFAULT_DETECTION_PROMPT",
    "DEFAULT_MONITORING_DEFECT_PROMPT",
    "CUSTOM_QUERY_TEMPLATE",
    "build_prompt",
    "build_agentic_prompt",
    "build_tools_prompt",

    # Client
    "UnifiedVLMClient",
    "AnalysisResult",
    "DetectionResult",
    "AgenticResult",
]
