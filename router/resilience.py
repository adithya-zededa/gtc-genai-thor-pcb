"""Resilience primitives for the vLLM request path.

Three things, all of them used by ``router/adapters/vllm.py`` on every
call:

1. :class:`ConcurrencyLimiter` — a per-endpoint semaphore, so a slow
   vision inference cannot starve the agent model of request slots.
2. :func:`calculate_backoff` — exponential backoff with jitter, honouring
   a server ``Retry-After`` hint as a floor.
3. :class:`RequestMetrics` / :class:`RequestLogger` — structured logging
   of the request lifecycle.

This module used to also carry a generic ``ResilientLLMClient``, a
request deduplicator, prompt-token estimation, and a retry decorator,
none of which the adapter ever called — the retry loop lives in the
adapter itself. They were removed rather than left as a second,
divergent implementation of the same behaviour.

Thread Safety:
    All operations are thread-safe via semaphores and locks.
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .rate_limit_config import RateLimitConfig, get_rate_limit_config

logger = logging.getLogger(__name__)


# ============================================================================
# Request Metrics and Observability
# ============================================================================

@dataclass
class RequestMetrics:
    """Metrics for a single LLM request for observability."""
    request_id: str
    provider: str
    model: str
    start_time: float
    end_time: Optional[float] = None
    token_estimate: Optional[int] = None
    actual_tokens: Optional[Dict[str, int]] = None
    retry_count: int = 0
    backoff_durations: List[float] = field(default_factory=list)
    final_status: str = "pending"  # pending, success, rate_limited, failed
    error_message: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for structured logging."""
        return {
            "request_id": self.request_id,
            "provider": self.provider,
            "model": self.model,
            "duration_ms": self.duration_ms,
            "token_estimate": self.token_estimate,
            "actual_tokens": self.actual_tokens,
            "retry_count": self.retry_count,
            "backoff_durations": self.backoff_durations,
            "total_backoff_seconds": sum(self.backoff_durations),
            "final_status": self.final_status,
            "error_message": self.error_message,
        }


def generate_request_id() -> str:
    """Generate a unique request ID for tracking."""
    import uuid
    return f"llm-{uuid.uuid4().hex[:12]}"


class RequestLogger:
    """Structured logger for LLM request lifecycle events."""

    def __init__(self, logger_instance: logging.Logger):
        self._logger = logger_instance

    def log_request_start(self, metrics: RequestMetrics, prompt_preview: str = ""):
        """Log when a request starts."""
        self._logger.info(
            f"🚀 LLM request start | id={metrics.request_id} | "
            f"provider={metrics.provider} | model={metrics.model} | "
            f"token_estimate={metrics.token_estimate}",
            extra={
                "event": "llm_request_start",
                "request_id": metrics.request_id,
                "provider": metrics.provider,
                "model": metrics.model,
                "token_estimate": metrics.token_estimate,
                "prompt_preview": prompt_preview[:100] if prompt_preview else "",
            }
        )

    def log_retry_attempt(self, metrics: RequestMetrics, attempt: int, backoff: float, error: str):
        """Log a retry attempt."""
        self._logger.warning(
            f"🔄 LLM retry | id={metrics.request_id} | "
            f"attempt={attempt}/{get_rate_limit_config().max_retries} | "
            f"backoff={backoff:.2f}s | error={error[:100]}",
            extra={
                "event": "llm_retry_attempt",
                "request_id": metrics.request_id,
                "attempt": attempt,
                "max_retries": get_rate_limit_config().max_retries,
                "backoff_seconds": backoff,
                "error": error,
            }
        )

    def log_rate_limited(self, metrics: RequestMetrics, retry_after: Optional[float] = None):
        """Log rate limit hit."""
        self._logger.warning(
            f"⏳ LLM rate limited | id={metrics.request_id} | "
            f"provider={metrics.provider} | retry_after={retry_after}s",
            extra={
                "event": "llm_rate_limited",
                "request_id": metrics.request_id,
                "provider": metrics.provider,
                "model": metrics.model,
                "retry_after": retry_after,
            }
        )

    def log_request_success(self, metrics: RequestMetrics):
        """Log successful request completion."""
        self._logger.info(
            f"✅ LLM request success | id={metrics.request_id} | "
            f"duration={metrics.duration_ms:.0f}ms | retries={metrics.retry_count} | "
            f"tokens={metrics.actual_tokens}",
            extra={
                "event": "llm_request_success",
                **metrics.to_log_dict(),
            }
        )

    def log_request_failure(self, metrics: RequestMetrics):
        """Log final request failure."""
        self._logger.error(
            f"❌ LLM request failed | id={metrics.request_id} | "
            f"status={metrics.final_status} | retries={metrics.retry_count} | "
            f"error={metrics.error_message}",
            extra={
                "event": "llm_request_failure",
                **metrics.to_log_dict(),
            }
        )


_request_logger = RequestLogger(logger)


# ============================================================================
# Concurrency Limiter
# ============================================================================

class ConcurrencyLimiter:
    """
    Limits concurrent LLM requests using a semaphore.

    Prevents request storms by allowing at most N concurrent requests.
    Additional requests wait in a queue.
    """

    def __init__(self, max_concurrent: int = 2):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._active_count = 0
        self._waiting_count = 0
        self._lock = threading.Lock()
        self._stats = {
            "total_acquired": 0,
            "total_waited": 0,
            "max_wait_time": 0.0,
        }

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a slot for making a request.

        Args:
            timeout: Maximum time to wait (None for infinite)

        Returns:
            True if acquired, False if timeout
        """
        with self._lock:
            self._waiting_count += 1

        start = time.time()
        acquired = self._semaphore.acquire(timeout=timeout)
        elapsed = time.time() - start

        with self._lock:
            self._waiting_count -= 1
            if acquired:
                self._active_count += 1
                self._stats["total_acquired"] += 1
                if elapsed > 0.01:  # Only count meaningful waits
                    self._stats["total_waited"] += 1
                    self._stats["max_wait_time"] = max(self._stats["max_wait_time"], elapsed)

        if acquired and elapsed > 0.1:
            logger.debug(f"Concurrency slot acquired after {elapsed:.2f}s wait")

        return acquired

    def release(self):
        """Release a slot after request completion."""
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
        self._semaphore.release()

    @property
    def active_requests(self) -> int:
        """Number of currently active requests."""
        with self._lock:
            return self._active_count

    @property
    def waiting_requests(self) -> int:
        """Number of requests waiting for a slot."""
        with self._lock:
            return self._waiting_count

    def get_stats(self) -> Dict[str, Any]:
        """Get concurrency limiter statistics."""
        with self._lock:
            return {
                "max_concurrent": self._max_concurrent,
                "active_requests": self._active_count,
                "waiting_requests": self._waiting_count,
                **self._stats,
            }

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# Concurrency limiters, one per provider/endpoint (lazy-initialized).
# The vision and agent models are served by separate vLLM pods, so they must
# not share a semaphore — otherwise a multi-second vision inference holds
# slots the agent model needs to answer a chat turn, and the two models are
# serialised against each other despite running on independent servers.
_concurrency_limiters: Dict[str, ConcurrencyLimiter] = {}
_limiter_lock = threading.Lock()


def get_concurrency_limiter(name: str = "default") -> ConcurrencyLimiter:
    """Get the concurrency limiter for *name* (a provider identifier)."""
    limiter = _concurrency_limiters.get(name)
    if limiter is None:
        with _limiter_lock:
            limiter = _concurrency_limiters.get(name)
            if limiter is None:
                config = get_rate_limit_config()
                limiter = ConcurrencyLimiter(config.max_concurrency)
                _concurrency_limiters[name] = limiter
                logger.info(
                    "Initialized concurrency limiter '%s' with max_concurrent=%d",
                    name, config.max_concurrency,
                )
    return limiter


def get_concurrency_stats() -> Dict[str, Any]:
    """Per-endpoint limiter statistics, keyed by provider name."""
    with _limiter_lock:
        limiters = dict(_concurrency_limiters)
    return {name: limiter.get_stats() for name, limiter in limiters.items()}


# ============================================================================
# Exponential Backoff Calculator
# ============================================================================

def calculate_backoff(
    attempt: int,
    config: Optional[RateLimitConfig] = None,
    retry_after_hint: Optional[float] = None
) -> float:
    """
    Calculate backoff duration using exponential backoff with jitter.

    Formula: min(base^attempt + jitter, max_backoff)

    If retry_after_hint is provided (from API response), use it as a floor.

    Args:
        attempt: The current retry attempt number (1-indexed)
        config: Rate limit configuration
        retry_after_hint: Optional hint from API response

    Returns:
        Backoff duration in seconds
    """
    if config is None:
        config = get_rate_limit_config()

    # Exponential backoff: 2^attempt
    backoff = config.backoff_base ** attempt

    # Add jitter: random value between 0 and jitter * backoff
    jitter = random.uniform(0, config.backoff_jitter * backoff)
    backoff += jitter

    # Respect retry_after hint if provided
    if retry_after_hint is not None:
        backoff = max(backoff, retry_after_hint)

    # Cap at maximum
    backoff = min(backoff, config.backoff_max)

    return backoff
