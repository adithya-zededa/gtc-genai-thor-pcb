"""
LLM Router Package - vLLM Service Routing

This package provides a direct router to the vLLM deployment for
high-performance LLM inference.

Usage:
    from router import get_router

    router = get_router()
    response = router.chat(messages=[{"role": "user", "content": "Hello!"}])
"""

from .config import (
    LLMProviderConfig,
    ProviderStatus,
    ChatMessage,
    ChatResponse,
)

from .base import LLMAdapter

from .llm_router import (
    AgentLLMRouter,
    ROLE_AGENT,
    ROLE_VISION,
    ROLES,
    get_router,
    get_agent_router,
    get_vision_router,
    reset_routers,
    chat,
    get_token_usage,
    reset_token_usage,
)

from .rate_limit_config import (
    RateLimitConfig,
    get_rate_limit_config,
    is_rate_limit_error,
    is_retryable_error,
    RETRYABLE_STATUS_CODES,
    NON_RETRYABLE_STATUS_CODES,
)

from .resilience import (
    ResilientLLMClient,
    RequestMetrics,
    ConcurrencyLimiter,
    RequestDeduplicator,
    RateLimitErrorResponse,
    RateLimitException,
    make_resilient_request,
    get_concurrency_limiter,
    get_deduplicator,
    get_resilience_stats,
    estimate_tokens,
    estimate_messages_tokens,
    calculate_backoff,
)

__all__ = [
    # Config
    "LLMProviderConfig",
    "ProviderStatus",
    "ChatMessage",
    "ChatResponse",
    # Base
    "LLMAdapter",
    # Router
    "AgentLLMRouter",
    "ROLE_AGENT",
    "ROLE_VISION",
    "ROLES",
    "get_router",
    "get_agent_router",
    "get_vision_router",
    "reset_routers",
    "chat",
    # Token tracking
    "get_token_usage",
    "reset_token_usage",
    # Rate limit config
    "RateLimitConfig",
    "get_rate_limit_config",
    "is_rate_limit_error",
    "is_retryable_error",
    "RETRYABLE_STATUS_CODES",
    "NON_RETRYABLE_STATUS_CODES",
    # Resilience
    "ResilientLLMClient",
    "RequestMetrics",
    "ConcurrencyLimiter",
    "RequestDeduplicator",
    "RateLimitErrorResponse",
    "RateLimitException",
    "make_resilient_request",
    "get_concurrency_limiter",
    "get_deduplicator",
    "get_resilience_stats",
    "estimate_tokens",
    "estimate_messages_tokens",
    "calculate_backoff",
]
