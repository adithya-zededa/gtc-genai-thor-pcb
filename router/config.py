"""
Router Configuration - Data classes and enums for LLM routing.

This module contains all configuration-related classes used by the router:
- Provider types and routing strategies
- Configuration data classes
- Status and response objects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"
    VLLM = "vllm"
    TGI = "tgi"  # Text Generation Inference
    LMSTUDIO = "lmstudio"
    OPENAI_COMPATIBLE = "openai-compatible"  # Generic OpenAI-compatible API


class RoutingStrategy(str, Enum):
    """Routing strategies for provider selection."""
    PRIORITY = "priority"  # Use highest priority available provider
    ROUND_ROBIN = "round_robin"  # Rotate between providers
    FAILOVER = "failover"  # Use primary, failover on error
    LATENCY = "latency"  # Use lowest latency provider
    COST = "cost"  # Use lowest cost provider


# Default model names for each provider
# NOTE: These are used only when no model is specified in config.
# Prefer setting models explicitly via environment variables or UI.
DEFAULT_MODELS: Dict[LLMProviderType, str] = {
    LLMProviderType.ANTHROPIC: "",  # Must be set via ANTHROPIC_MODEL env var
    LLMProviderType.OPENAI: "",  # Must be set via OPENAI_MODEL env var
    LLMProviderType.GOOGLE: "",  # Must be set via GOOGLE_MODEL env var
    LLMProviderType.GROQ: "",  # Must be set via GROQ_MODEL env var
    LLMProviderType.OLLAMA: "",  # Must be set via OLLAMA_MODEL env var
    LLMProviderType.VLLM: "",
    LLMProviderType.TGI: "",
    LLMProviderType.LMSTUDIO: "",
    LLMProviderType.OPENAI_COMPATIBLE: "",
}


@dataclass
class LLMProviderConfig:
    """
    Configuration for an LLM provider.
    
    Attributes:
        name: Unique identifier for this provider instance
        provider_type: Type of LLM provider (anthropic, openai, ollama, etc.)
        url: Server URL (required for self-hosted, optional for cloud APIs)
        model: Model name to use
        api_key: API key for authentication
        priority: Lower = higher priority (default: 10)
        max_tokens: Maximum output tokens (default: 4096)
        temperature: Sampling temperature (default: 0.1)
        timeout: Request timeout in seconds (default: 60)
        enabled: Whether this provider is enabled (default: True)
        supports_tools: Whether this provider supports function calling
        supports_vision: Whether this provider supports image inputs
        metadata: Additional custom metadata
    """
    name: str
    provider_type: LLMProviderType
    url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    priority: int = 10
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: int = 60
    enabled: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Convert string to enum if needed
        if isinstance(self.provider_type, str):
            self.provider_type = LLMProviderType(self.provider_type.lower())
        
        # Leave model as None if not specified - adapters should validate
        # or omit the model field from payloads when None. Do NOT default
        # to empty string as OpenAI-compatible servers reject "model": "".
        
        # Normalize URL to ensure it has http:// or https:// scheme
        if self.url:
            self.url = self._normalize_url(self.url)
    
    @staticmethod
    def _normalize_url(url: str) -> str:
        """Ensure URL has proper http:// or https:// scheme."""
        url = url.strip()
        if not url:
            return url
        # If URL doesn't start with http:// or https://, add http://
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'
        # Remove trailing slashes for consistency
        return url.rstrip('/')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (hides API key)."""
        return {
            "name": self.name,
            "provider_type": self.provider_type.value,
            "url": self.url,
            "model": self.model,
            "priority": self.priority,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enabled": self.enabled,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "has_api_key": bool(self.api_key),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMProviderConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            provider_type=data.get("provider_type", "openai-compatible"),
            url=data.get("url"),
            model=data.get("model"),
            api_key=data.get("api_key"),
            priority=data.get("priority", 10),
            max_tokens=data.get("max_tokens", 4096),
            temperature=data.get("temperature", 0.1),
            timeout=data.get("timeout", 60),
            enabled=data.get("enabled", True),
            supports_tools=data.get("supports_tools", True),
            supports_vision=data.get("supports_vision", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProviderStatus:
    """Runtime status of an LLM provider."""
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
