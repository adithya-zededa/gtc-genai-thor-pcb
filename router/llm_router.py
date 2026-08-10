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
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.config import get_config

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

ROLE_VISION = "vision"
ROLE_AGENT = "agent"
ROLES = (ROLE_VISION, ROLE_AGENT)


class AgentLLMRouter:
    """
    Single-provider router for one vLLM deployment.

    The system serves two models from two separate vLLM pods, so there is one
    router instance per *role* rather than one global instance:

    ``vision``
        The fine-tuned VLM that looks at frames. Reads
        ``Config.inference`` (``VLLM_URL`` / ``VISION_MODEL``).
    ``agent``
        The text model that classifies intent, answers chat, and drives tool
        calls. Reads ``Config.agent_inference`` (``AGENT_LLM_URL`` /
        ``AGENT_MODEL``), each falling back to the vision endpoint so a
        single-pod deployment keeps working unchanged.

    All configuration is resolved through ``core.config.get_config()``;
    this module reads no environment variables of its own.

    Instances are cached per role — ``get_router(role)`` is the entry point.
    Each role also gets its own concurrency limiter, so a slow vision
    inference cannot starve the agent model of request slots.

    Thread-Safety:
        All operations are thread-safe.
    """

    _instances: Dict[str, "AgentLLMRouter"] = {}
    _lock = threading.Lock()

    def __init__(self, role: str = ROLE_VISION):
        if getattr(self, '_initialized', False):
            return

        if role not in ROLES:
            raise ValueError(f"Unknown LLM role {role!r}; expected one of {ROLES}")

        self.role = role
        self._adapter = VLLMAdapter()
        self._config: Optional[LLMProviderConfig] = None
        self._status: Optional[ProviderStatus] = None
        self._config_lock = threading.RLock()

        # Auto-configure from environment
        self._auto_configure()

        self._initialized = True
        logger.info("AgentLLMRouter initialized (role=%s)", role)

    def _auto_configure(self) -> None:
        """Configure this role's vLLM provider from the application config.

        Every value comes from ``core.config``, which is the single place
        the environment is parsed. The router used to re-read the same
        variables itself, so the two could disagree — and did, whenever a
        fallback rule was changed in one and not the other.
        """
        app_config = get_config()

        if self.role == ROLE_AGENT:
            # agent_inference_url/_model encode the fallback to the vision
            # endpoint, so a single-pod deployment keeps working.
            url = app_config.agent_inference_url
            model = app_config.agent_inference_model
            timeout = app_config.agent_inference.timeout
            temperature = app_config.agent_inference.temperature
            api_key = app_config.agent_inference_api_key
            supports_vision = False
        else:
            url = app_config.inference.vllm_url
            model = app_config.inference.model
            timeout = app_config.inference.timeout
            temperature = app_config.inference.temperature
            api_key = app_config.inference.api_key
            supports_vision = True

        # The provider name doubles as the concurrency-limiter key, so the two
        # roles must not share it.
        provider_name = f"vllm-{self.role}"

        self._config = LLMProviderConfig(
            name=provider_name,
            url=url,
            model=model or None,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            supports_tools=True,
            supports_vision=supports_vision,
        )

        self._status = ProviderStatus(
            name=provider_name,
            available=False,
            last_check=0,
        )

        # Check availability
        self._check_availability()

        logger.info(
            "vLLM provider configured: role=%s url=%s model=%s",
            self.role,
            self._config.url,
            self._config.model or "auto",
        )

    def shares_endpoint_with(self, other: "AgentLLMRouter") -> bool:
        """Whether this role is served by the same vLLM endpoint as *other*."""
        if not self._config or not other._config:
            return False
        return (self._config.url or "") == (other._config.url or "")

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

    def check_health(self) -> Dict[str, Any]:
        """Check vLLM health. Returns dict with status, url, model, latency."""
        available = self._check_availability()
        return {
            "vllm": available,
            "available": available,
            "url": self._config.url if self._config else None,
            "model": self._config.model if self._config else None,
            "latency_ms": self._status.latency_ms if self._status else None,
        }

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
            "provider": "vllm",
            "config": self._config.to_dict() if self._config else None,
            "status": self._status.to_dict() if self._status else None,
            "active_provider": self.get_active_provider(),
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

def get_router(role: str = ROLE_VISION) -> AgentLLMRouter:
    """Get the cached router for *role* (``"vision"`` or ``"agent"``).

    Defaults to the vision role, which preserves the pre-split behaviour for
    any caller that has not been updated: it is the one configured from
    ``VLLM_URL``/``VISION_MODEL``.
    """
    if role not in ROLES:
        raise ValueError(f"Unknown LLM role {role!r}; expected one of {ROLES}")

    instance = AgentLLMRouter._instances.get(role)
    if instance is None:
        with AgentLLMRouter._lock:
            instance = AgentLLMRouter._instances.get(role)
            if instance is None:
                instance = AgentLLMRouter(role)
                AgentLLMRouter._instances[role] = instance
    return instance


def get_agent_router() -> AgentLLMRouter:
    """Router for the text/agent model — intent, chat, and tool calling."""
    return get_router(ROLE_AGENT)


def get_vision_router() -> AgentLLMRouter:
    """Router for the vision model — frame analysis only."""
    return get_router(ROLE_VISION)


def reset_routers() -> None:
    """Drop cached routers so the next call re-reads ``core.config``.

    ``Config`` is itself a cached singleton, so picking up an environment
    change needs both: ``core.config.reset_config()`` first, then this.
    Runtime edits made through ``AgentLLMRouter.configure()`` do not need
    either — they mutate the live provider config directly.
    """
    with AgentLLMRouter._lock:
        AgentLLMRouter._instances.clear()


def chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    role: str = ROLE_VISION,
    **kwargs
) -> ChatResponse:
    """Send a chat request using the router for *role*."""
    return get_router(role).chat(messages, tools, **kwargs)
