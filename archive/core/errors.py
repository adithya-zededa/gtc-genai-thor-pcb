"""Centralized error handling for the Camera Agent application.

Defines a hierarchy of domain-specific exceptions for consistent
error handling across the application.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class CameraAgentError(Exception):
    """Base exception for all Camera Agent errors."""

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        *,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the error for API responses."""
        result: Dict[str, Any] = {
            "error": self.__class__.__name__,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


# ---------------------------------------------------------------------------
# Infrastructure errors
# ---------------------------------------------------------------------------

class ConfigurationError(CameraAgentError):
    """Invalid or missing configuration."""

    def __init__(self, message: str = "Configuration error", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


class DatabaseError(CameraAgentError):
    """Database operation failed."""

    def __init__(self, message: str = "Database error", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


# ---------------------------------------------------------------------------
# Service errors
# ---------------------------------------------------------------------------

class VLMConnectionError(CameraAgentError):
    """VLM inference service is unavailable."""

    def __init__(self, message: str = "VLM service unavailable", **kwargs):
        super().__init__(message, status_code=503, **kwargs)


class VLMInferenceError(CameraAgentError):
    """VLM inference call failed."""

    def __init__(self, message: str = "VLM inference failed", **kwargs):
        super().__init__(message, status_code=502, **kwargs)


class CameraDeviceError(CameraAgentError):
    """Camera device access error."""

    def __init__(self, message: str = "Camera device error", **kwargs):
        super().__init__(message, status_code=503, **kwargs)


class MonitoringError(CameraAgentError):
    """Monitoring pipeline error."""

    def __init__(self, message: str = "Monitoring error", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


# ---------------------------------------------------------------------------
# Agent / MCP errors
# ---------------------------------------------------------------------------

class ToolExecutionError(CameraAgentError):
    """Tool execution failed."""

    def __init__(self, message: str = "Tool execution failed", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


class MCPError(CameraAgentError):
    """MCP protocol error."""

    def __init__(self, message: str = "MCP error", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


class ClassificationError(CameraAgentError):
    """Intent classification failed."""

    def __init__(self, message: str = "Classification failed", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


# ---------------------------------------------------------------------------
# API errors
# ---------------------------------------------------------------------------

class BadRequestError(CameraAgentError):
    """Client sent an invalid request."""

    def __init__(self, message: str = "Bad request", **kwargs):
        super().__init__(message, status_code=400, **kwargs)


class NotFoundError(CameraAgentError):
    """Requested resource not found."""

    def __init__(self, message: str = "Not found", **kwargs):
        super().__init__(message, status_code=404, **kwargs)


class ConflictError(CameraAgentError):
    """Resource conflict (e.g. duplicate)."""

    def __init__(self, message: str = "Conflict", **kwargs):
        super().__init__(message, status_code=409, **kwargs)


class ServiceUnavailableError(CameraAgentError):
    """Required service is not available."""

    def __init__(self, message: str = "Service unavailable", **kwargs):
        super().__init__(message, status_code=503, **kwargs)


# ---------------------------------------------------------------------------
# LLM Router errors
# ---------------------------------------------------------------------------

class LLMRouterError(CameraAgentError):
    """LLM Router error."""

    def __init__(self, message: str = "LLM Router error", **kwargs):
        super().__init__(message, status_code=502, **kwargs)


class NoProviderAvailableError(LLMRouterError):
    """No LLM providers are available."""

    def __init__(self, message: str = "No LLM providers available", **kwargs):
        super().__init__(message, **kwargs)
