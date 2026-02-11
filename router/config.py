"""
Router Configuration - Data classes for single-provider vLLM routing.

This module contains configuration classes used by the router:
- Provider configuration
- Status and response objects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMProviderConfig:
    """
    Configuration for the vLLM provider.
    
    Attributes:
        name: Identifier for this provider instance (default: "vllm")
        url: vLLM server URL
        model: Model name to use
        api_key: API key for authentication (optional)
        max_tokens: Maximum output tokens (default: 4096)
        temperature: Sampling temperature (default: 0.1)
        timeout: Request timeout in seconds (default: 60)
        enabled: Whether this provider is enabled (default: True)
        supports_tools: Whether this provider supports function calling
        supports_vision: Whether this provider supports image inputs
        metadata: Additional custom metadata
    """
    name: str = "vllm"
    url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: int = 60
    enabled: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Normalize URL to ensure it has http:// or https:// scheme
        if self.url:
            self.url = self._normalize_url(self.url)
    
    @staticmethod
    def _normalize_url(url: str) -> str:
        """Ensure URL has proper http:// or https:// scheme."""
        url = url.strip()
        if not url:
            return url
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'
        return url.rstrip('/')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (hides API key)."""
        return {
            "name": self.name,
            "provider_type": "vllm",
            "url": self.url,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enabled": self.enabled,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "has_api_key": bool(self.api_key),
        }


@dataclass
class ProviderStatus:
    """Runtime status of the vLLM provider."""
    name: str
    available: bool
    last_check: float
    latency_ms: float = 0.0
    total_requests: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    models_available: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "last_check": self.last_check,
            "latency_ms": self.latency_ms,
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "models_available": self.models_available,
        }


@dataclass 
class ChatMessage:
    """A chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """Response from an LLM chat request."""
    content: str
    provider: str
    model: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
        }
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.usage:
            result["usage"] = self.usage
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
        return result
