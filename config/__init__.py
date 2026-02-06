"""Configuration package with Pydantic-based settings.

Provides type-safe configuration models with environment variable support,
validation, and defaults loaded from YAML.
"""

from .settings import (
    Settings,
    CameraSettings,
    VLLMSettings,
    OllamaSettings,
    InferenceSettings,
    FlaskSettings,
    RouterSettings,
    DatabaseSettings,
    get_settings,
    reset_settings,
)

__all__ = [
    "Settings",
    "CameraSettings",
    "VLLMSettings",
    "OllamaSettings",
    "InferenceSettings",
    "FlaskSettings",
    "RouterSettings",
    "DatabaseSettings",
    "get_settings",
    "reset_settings",
]
