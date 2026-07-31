"""
Agent LLM Router - Single-provider vLLM router

Routes all LLM requests to the vLLM deployment. No multi-provider
routing, failover, or strategy selection — just a direct, optimised
connection to vLLM.

Usage:
    from router import get_router

    router = get_router()
    response = router.chat(messages=[{"role": "user", "content": "Hello!"}])
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import (
    LLMProviderConfig,
    ProviderStatus,
    ChatResponse,
)
from .adapters import VLLMAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# Token Usage Tracking
# =============================================================================

@dataclass
class TokenUsageStats:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.request_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
        }


class TokenUsageTracker:
    """Tracks token usage for the vLLM provider."""

    def __init__(self):
        self._usage: Dict[str, TokenUsageStats] = {}
        self._lock = threading.Lock()

    def record(self, model: str, usage: Optional[Dict[str, int]]) -> None:
        """Record token usage for a request."""
        if not usage:
            return

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        with self._lock:
            key = f"vllm/{model}"
            if key not in self._usage:
                self._usage[key] = TokenUsageStats()
            self._usage[key].add(prompt_tokens, completion_tokens)

        total = prompt_tokens + completion_tokens
        logger.info(
            f"🔢 Token Usage [vllm/{model}]: "
            f"prompt={prompt_tokens}, completion={completion_tokens}, total={total}"
        )

    def get_usage(self) -> Dict[str, Any]:
        """Get token usage stats."""
        with self._lock:
            return {k: v.to_dict() for k, v in self._usage.items()}

    def get_totals(self) -> Dict[str, int]:
        """Get total token usage."""
        with self._lock:
            totals = TokenUsageStats()
            for stats in self._usage.values():
                totals.prompt_tokens += stats.prompt_tokens
                totals.completion_tokens += stats.completion_tokens
                totals.total_tokens += stats.total_tokens
                totals.request_count += stats.request_count
            return totals.to_dict()

    def reset(self) -> None:
        """Reset all usage stats."""
        with self._lock:
            self._usage.clear()
        logger.info("Token usage stats reset")


# Global token tracker instance
_token_tracker = TokenUsageTracker()


def get_token_usage() -> Dict[str, Any]:
    """Get current token usage stats."""
    return {
        "by_provider": _token_tracker.get_usage(),
        "totals": _token_tracker.get_totals(),
    }


def reset_token_usage() -> None:
    """Reset token usage stats."""
    _token_tracker.reset()


# =============================================================================
# Agent LLM Router (vLLM-only)
# =============================================================================

class AgentLLMRouter:
    """
    Single-provider router for vLLM, with one instance per *role*.

    Two roles are used today:
    - ``"agent"`` (default) — the text-only reasoning model: chat replies,
      intent classification, tool selection. Configured from
      ``AGENT_LLM_URL``/``AGENT_MODEL`` (falling back to ``VLLM_URL``/
      ``VISION_MODEL`` so a single-model deployment keeps working).
    - ``"vision"`` — the vision-language model used for frame analysis.
      Configured from ``VLLM_URL``/``VISION_MODEL`` directly.

    Each role gets its own singleton instance (own connection, own
    availability/status tracking), so a slow or unavailable vision model
    never affects chat/classification and vice versa.

    Thread-Safety:
        All operations are thread-safe.
    """

    _instances: Dict[str, "AgentLLMRouter"] = {}
    _lock = threading.Lock()

    def __new__(cls, role: str = "agent"):
        """Singleton-per-role pattern."""
        if role not in cls._instances:
            with cls._lock:
                if role not in cls._instances:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instances[role] = instance
        return cls._instances[role]

    def __init__(self, role: str = "agent"):
        if getattr(self, '_initialized', False):
            return

        self._role = role
        self._adapter = VLLMAdapter()
        self._config: Optional[LLMProviderConfig] = None
        self._status: Optional[ProviderStatus] = None
        self._config_lock = threading.RLock()

        # Auto-configure from environment
        self._auto_configure()

        self._initialized = True
        logger.info("AgentLLMRouter initialized (role=%s, vLLM-only)", role)

    def _auto_configure(self) -> None:
        """Auto-configure the vLLM provider for this role from environment variables."""
        if self._role == "vision":
            vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
            vllm_model = os.environ.get("VISION_MODEL", "")
            vllm_timeout = int(os.environ.get("VLLM_TIMEOUT", "300"))
            vllm_temperature = float(os.environ.get("VLLM_TEMPERATURE", "0.1"))
            vllm_api_key = os.environ.get("VLLM_API_KEY")
        else:
            vllm_url = os.environ.get("AGENT_LLM_URL") or os.environ.get(
                "VLLM_URL", "http://localhost:8000"
            )
            vllm_model = os.environ.get("AGENT_MODEL", "")
            vllm_timeout = int(
                os.environ.get("AGENT_LLM_TIMEOUT") or os.environ.get("VLLM_TIMEOUT", "300")
            )
            vllm_temperature = float(
                os.environ.get("AGENT_LLM_TEMPERATURE")
                or os.environ.get("VLLM_TEMPERATURE", "0.1")
            )
            vllm_api_key = os.environ.get("AGENT_LLM_API_KEY") or os.environ.get("VLLM_API_KEY")

        self._config = LLMProviderConfig(
            name=f"vllm-{self._role}",
            url=vllm_url,
            model=vllm_model or None,
            api_key=vllm_api_key,
            timeout=vllm_timeout,
            temperature=vllm_temperature,
            supports_tools=True,
            supports_vision=(self._role == "vision"),
        )

        self._status = ProviderStatus(
            name=f"vllm-{self._role}",
            available=False,
            last_check=0,
        )

        # Check availability
        self._check_availability()

        logger.info(
            "vLLM provider configured (role=%s): url=%s model=%s",
            self._role,
            self._config.url,
            self._config.model or "auto",
        )

    # =========================================================================
    # Configuration
    # =========================================================================

    def configure(
        self,
        url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Update vLLM configuration at runtime."""
        with self._config_lock:
            if self._config is None:
                self._config = LLMProviderConfig(name="vllm")
            if url is not None:
                self._config.url = LLMProviderConfig._normalize_url(url)
            if model is not None:
                self._config.model = model
            if api_key is not None:
                self._config.api_key = api_key
            if timeout is not None:
                self._config.timeout = timeout
            if temperature is not None:
                self._config.temperature = temperature

        self._check_availability()
        logger.info("vLLM configuration updated: url=%s model=%s", self._config.url, self._config.model)

    def get_config(self) -> Optional[LLMProviderConfig]:
        """Get the current vLLM configuration."""
        return self._config

    # =========================================================================
    # Availability Checking
    # =========================================================================

    def _check_availability(self) -> bool:
        """Check if the vLLM server is reachable."""
        if self._config is None:
            return False

        available, latency, error = self._adapter.check_availability(self._config)

        with self._config_lock:
            if self._status:
                self._status.available = available
                self._status.last_check = time.time()
                self._status.latency_ms = latency
                self._status.last_error = error
                if not available:
                    self._status.error_count += 1

        if available:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("vLLM available (latency: %.1fms)", latency)
        else:
            logger.warning(f"vLLM unavailable: {error}")

        return available

    def check_health(self) -> Dict[str, bool]:
        """Check vLLM health. Returns dict keyed by role for API compatibility."""
        available = self._check_availability()
        return {self._role: available}

    def is_available(self) -> bool:
        """Check if vLLM is currently available."""
        return self._status.available if self._status else False

    # =========================================================================
    # Provider info (API compatibility)
    # =========================================================================

    def list_providers(self) -> List[Dict[str, Any]]:
        """List the vLLM provider with status. Returns a list for API compat."""
        if self._config is None:
            return []
        return [{
            **self._config.to_dict(),
            "status": self._status.to_dict() if self._status else None,
        }]

    def list_models(self) -> List[str]:
        """List available models from the vLLM server."""
        if self._config is None:
            return []
        return self._adapter.list_models(self._config)

    def get_active_provider(self) -> Optional[Dict[str, Any]]:
        """Get the vLLM provider info."""
        if self._config is None:
            return None
        return {
            **self._config.to_dict(),
            "status": self._status.to_dict() if self._status else None,
        }

    # =========================================================================
    # Chat Interface
    # =========================================================================

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat request to the vLLM provider.

        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas for function calling
            **kwargs: Additional arguments passed to the adapter

        Returns:
            ChatResponse with the LLM's response

        Raises:
            RuntimeError: If vLLM is not configured or the request fails
        """
        if self._config is None:
            raise RuntimeError("vLLM provider not configured")

        response = self._adapter.chat(self._config, messages, tools, **kwargs)

        # Track token usage
        _token_tracker.record(self._config.model or "unknown", response.usage)

        # Update stats
        with self._config_lock:
            if self._status:
                self._status.total_requests += 1

        return response

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Send a streaming chat request to vLLM.

        Yields SSE-style events:
        - {"type": "token", "content": "..."} - Text token
        - {"type": "tool_call", ...} - Tool call data
        - {"type": "done", "response": ChatResponse} - Final response
        - {"type": "error", "error": "..."} - Error occurred
        """
        if self._config is None:
            yield {"type": "error", "error": "vLLM provider not configured"}
            return

        for event in self._adapter.chat_stream(self._config, messages, tools, **kwargs):
            event_type = event.get("type")
            if event_type in ("done", "complete"):
                response = event.get("response") or event.get("full_response")
                if response and hasattr(response, 'usage'):
                    _token_tracker.record(self._config.model or "unknown", response.usage)
                    with self._config_lock:
                        if self._status:
                            self._status.total_requests += 1
            yield event

    def to_dict(self) -> Dict[str, Any]:
        """Export router state as dictionary."""
        return {
            "provider": f"vllm-{self._role}",
            "config": self._config.to_dict() if self._config else None,
            "status": self._status.to_dict() if self._status else None,
            "active_provider": self.get_active_provider(),
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

def get_router(role: str = "agent") -> AgentLLMRouter:
    """Get the global LLM router instance for the given role.

    role="agent" (default): text-only reasoning model (chat, classification).
    role="vision": the vision-language model used for frame analysis.
    """
    return AgentLLMRouter(role)


def chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> ChatResponse:
    """Send a chat request using the global router."""
    return get_router().chat(messages, tools, **kwargs)
