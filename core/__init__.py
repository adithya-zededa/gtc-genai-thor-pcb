"""Core infrastructure modules for the Camera Agent application."""

from .config import Config, get_config
from .logging import setup_logging, get_logger
from .errors import (
    CameraAgentError,
    ConfigurationError,
    DatabaseError,
    VLMConnectionError,
    VLMInferenceError,
    CameraDeviceError,
    MonitoringError,
    ToolExecutionError,
    MCPError,
    BadRequestError,
    NotFoundError,
    ServiceUnavailableError,
)

__all__ = [
    "Config",
    "get_config",
    "setup_logging",
    "get_logger",
    "CameraAgentError",
    "ConfigurationError",
    "DatabaseError",
    "VLMConnectionError",
    "VLMInferenceError",
    "CameraDeviceError",
    "MonitoringError",
    "ToolExecutionError",
    "MCPError",
    "BadRequestError",
    "NotFoundError",
    "ServiceUnavailableError",
]
