"""
Resilient LLM Client - Production-grade rate limit handling and request resilience.

This module provides:
1. Automatic retry with exponential backoff (2^attempt + jitter, max 30s)
2. Concurrency limiting via semaphore
3. Request deduplication for burst prevention
4. Token estimation and prompt protection
5. Structured error responses for rate limits
6. Comprehensive observability logging

Usage:
    from router.resilience import ResilientLLMClient, RequestMetrics
    
    client = ResilientLLMClient(anthropic_client)
    response = client.chat(messages, model="claude-3-opus")

Thread Safety:
    All operations are thread-safe via semaphores and locks.
"""

import asyncio
import hashlib
import logging
import random
import threading
import time
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic
from functools import wraps
import traceback

from .rate_limit_config import (
    get_rate_limit_config,
    is_rate_limit_error,
    is_retryable_error,
    extract_retry_after,
    RateLimitConfig,
    RETRYABLE_STATUS_CODES,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


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
    was_deduplicated: bool = False
    
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
            "was_deduplicated": self.was_deduplicated,
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
    
    def log_prompt_truncated(self, original_tokens: int, truncated_tokens: int, max_tokens: int):
        """Log when prompt is truncated."""
        self._logger.warning(
            f"✂️ Prompt truncated | original={original_tokens} | "
            f"truncated_to={truncated_tokens} | max={max_tokens}",
            extra={
                "event": "prompt_truncated",
                "original_tokens": original_tokens,
                "truncated_tokens": truncated_tokens,
                "max_tokens": max_tokens,
            }
        )
    
    def log_deduplication(self, prompt_hash: str, metrics: RequestMetrics):
        """Log when a duplicate request is detected."""
        self._logger.info(
            f"🔁 Request deduplicated | id={metrics.request_id} | hash={prompt_hash[:16]}",
            extra={
                "event": "request_deduplicated",
                "request_id": metrics.request_id,
                "prompt_hash": prompt_hash,
            }
        )


_request_logger = RequestLogger(logger)


# ============================================================================
# Token Estimation
# ============================================================================

def estimate_tokens(text: str, model: str = "claude") -> int:
    """
    Estimate token count for text.
    
    Uses a simple heuristic: ~4 characters per token for English text.
    This is conservative to avoid underestimating.
    
    For production, consider using tiktoken for OpenAI models
    or anthropic's token counting utilities.
    """
    if not text:
        return 0
    
    # Different models have different tokenization
    # Claude: ~3.5 chars/token
    # GPT-4: ~4 chars/token
    # Be conservative and use 3.5
    chars_per_token = 3.5
    
    return int(len(text) / chars_per_token) + 1


def estimate_messages_tokens(messages: List[Dict[str, Any]], model: str = "claude") -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content, model)
        elif isinstance(content, list):
            # Handle structured content (e.g., tool results)
            for item in content:
                if isinstance(item, dict):
                    total += estimate_tokens(json.dumps(item), model)
                else:
                    total += estimate_tokens(str(item), model)
        # Add overhead for role, structure
        total += 4
    return total


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


# ============================================================================
# Request Deduplication
# ============================================================================

class RequestDeduplicator:
    """
    Prevents duplicate requests within a time window.
    
    Uses LRU cache with TTL to detect and deduplicate identical prompts
    that fire repeatedly (e.g., from rapid user clicks or agent loops).
    """
    
    def __init__(self, window_seconds: float = 5.0, max_size: int = 100):
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._window = window_seconds
        self._max_size = max_size
        self._lock = threading.Lock()
        self._stats = {
            "total_requests": 0,
            "deduplicated": 0,
        }
    
    def _compute_hash(self, messages: List[Dict[str, Any]], model: str) -> str:
        """Compute a hash for request deduplication."""
        # Create a deterministic string representation
        key_data = json.dumps({
            "messages": messages,
            "model": model,
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        now = time.time()
        expired = []
        for key, (timestamp, _) in self._cache.items():
            if now - timestamp > self._window:
                expired.append(key)
            else:
                break  # OrderedDict is ordered by insertion time
        
        for key in expired:
            del self._cache[key]
    
    def check_duplicate(
        self, 
        messages: List[Dict[str, Any]], 
        model: str
    ) -> Tuple[bool, Optional[Any], str]:
        """
        Check if this request is a duplicate.
        
        Returns:
            Tuple of (is_duplicate, cached_response, request_hash)
        """
        request_hash = self._compute_hash(messages, model)
        
        with self._lock:
            self._stats["total_requests"] += 1
            self._cleanup_expired()
            
            if request_hash in self._cache:
                timestamp, response = self._cache[request_hash]
                if time.time() - timestamp <= self._window:
                    self._stats["deduplicated"] += 1
                    # Move to end (most recently used)
                    self._cache.move_to_end(request_hash)
                    return True, response, request_hash
        
        return False, None, request_hash
    
    def cache_response(self, request_hash: str, response: Any):
        """Cache a successful response for deduplication."""
        with self._lock:
            # Enforce max size
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            
            self._cache[request_hash] = (time.time(), response)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "window_seconds": self._window,
                **self._stats,
                "dedup_rate": (
                    self._stats["deduplicated"] / self._stats["total_requests"]
                    if self._stats["total_requests"] > 0 else 0
                ),
            }


# Global deduplicator
_deduplicator: Optional[RequestDeduplicator] = None
_dedup_lock = threading.Lock()


def get_deduplicator() -> RequestDeduplicator:
    """Get the global request deduplicator."""
    global _deduplicator
    if _deduplicator is None:
        with _dedup_lock:
            if _deduplicator is None:
                config = get_rate_limit_config()
                _deduplicator = RequestDeduplicator(config.dedup_window_seconds)
                logger.info(f"Initialized request deduplicator with window={config.dedup_window_seconds}s")
    return _deduplicator


# ============================================================================
# Rate Limit Error Response
# ============================================================================

@dataclass
class RateLimitErrorResponse:
    """Structured error response for rate limit scenarios."""
    error: str = "RATE_LIMITED"
    retry_after: Optional[float] = None
    action: str = "retrying"  # retrying, queued, failed
    provider: Optional[str] = None
    model: Optional[str] = None
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.error,
            "retry_after": self.retry_after,
            "action": self.action,
            "provider": self.provider,
            "model": self.model,
            "message": self.message,
        }


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


# ============================================================================
# Resilient Request Wrapper
# ============================================================================

def with_retry(
    func: Callable[..., T],
    provider: str,
    model: str,
    config: Optional[RateLimitConfig] = None,
) -> Callable[..., T]:
    """
    Decorator/wrapper that adds retry logic to an LLM API call.
    
    Args:
        func: The function to wrap (should make the actual API call)
        provider: Provider name for logging
        model: Model name for logging
        config: Rate limit configuration
        
    Returns:
        Wrapped function with retry logic
    """
    if config is None:
        config = get_rate_limit_config()
    
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        metrics = RequestMetrics(
            request_id=generate_request_id(),
            provider=provider,
            model=model,
            start_time=time.time(),
        )
        
        # Estimate tokens if messages are provided
        messages = kwargs.get('messages', args[0] if args else [])
        if messages:
            metrics.token_estimate = estimate_messages_tokens(messages, model)
        
        _request_logger.log_request_start(metrics)
        
        last_error: Optional[Exception] = None
        
        for attempt in range(1, config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # Success!
                metrics.end_time = time.time()
                metrics.final_status = "success"
                metrics.retry_count = attempt - 1
                
                # Extract token usage if available
                if hasattr(result, 'usage'):
                    metrics.actual_tokens = {
                        "input": getattr(result.usage, 'input_tokens', 0),
                        "output": getattr(result.usage, 'output_tokens', 0),
                    }
                
                _request_logger.log_request_success(metrics)
                return result
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if this error is retryable
                if not is_retryable_error(e):
                    metrics.end_time = time.time()
                    metrics.final_status = "failed"
                    metrics.error_message = error_str
                    _request_logger.log_request_failure(metrics)
                    raise
                
                # Check if rate limited specifically
                if is_rate_limit_error(e):
                    retry_after = extract_retry_after(e)
                    _request_logger.log_rate_limited(metrics, retry_after)
                
                # Check if we have retries left
                if attempt >= config.max_retries:
                    break
                
                # Calculate backoff
                retry_after_hint = extract_retry_after(e)
                backoff = calculate_backoff(attempt, config, retry_after_hint)
                metrics.backoff_durations.append(backoff)
                
                _request_logger.log_retry_attempt(metrics, attempt, backoff, error_str)
                
                # Wait before retry
                time.sleep(backoff)
        
        # All retries exhausted
        metrics.end_time = time.time()
        metrics.retry_count = config.max_retries
        metrics.final_status = "rate_limited" if is_rate_limit_error(last_error) else "failed"
        metrics.error_message = str(last_error) if last_error else "Unknown error"
        
        _request_logger.log_request_failure(metrics)
        
        if last_error:
            raise last_error
        raise RuntimeError("Request failed after all retries")
    
    return wrapper


def make_resilient_request(
    request_func: Callable[..., T],
    messages: List[Dict[str, Any]],
    provider: str,
    model: str,
    config: Optional[RateLimitConfig] = None,
    enable_dedup: bool = True,
    **kwargs
) -> T:
    """
    Execute an LLM request with full resilience features.
    
    Features:
    1. Concurrency limiting (prevents request storms)
    2. Request deduplication (prevents duplicate prompts)
    3. Token estimation and validation
    4. Automatic retry with exponential backoff
    5. Comprehensive logging
    
    Args:
        request_func: Function that makes the actual API call
        messages: Chat messages
        provider: Provider name
        model: Model name
        config: Rate limit configuration
        enable_dedup: Whether to check for duplicates
        **kwargs: Additional arguments passed to request_func
        
    Returns:
        API response
        
    Raises:
        Various API-specific exceptions on non-retryable errors
        RuntimeError if all retries exhausted
    """
    if config is None:
        config = get_rate_limit_config()
    
    request_id = generate_request_id()
    metrics = RequestMetrics(
        request_id=request_id,
        provider=provider,
        model=model,
        start_time=time.time(),
    )
    
    # Token estimation and validation
    token_estimate = estimate_messages_tokens(messages, model)
    metrics.token_estimate = token_estimate
    
    if token_estimate > config.max_prompt_tokens:
        if config.auto_truncate_prompts:
            # Truncate by removing older messages (keep system and recent)
            _request_logger.log_prompt_truncated(
                token_estimate,
                config.max_prompt_tokens,
                config.max_prompt_tokens
            )
            # Simple truncation: keep first (system) and last few messages
            if len(messages) > 3:
                messages = [messages[0]] + messages[-2:]
                token_estimate = estimate_messages_tokens(messages, model)
                metrics.token_estimate = token_estimate
        else:
            raise ValueError(
                f"Prompt exceeds maximum token limit "
                f"({token_estimate} > {config.max_prompt_tokens})"
            )
    
    # Deduplication check
    if enable_dedup and config.enable_deduplication:
        deduplicator = get_deduplicator()
        is_dup, cached_response, request_hash = deduplicator.check_duplicate(messages, model)
        
        if is_dup and cached_response is not None:
            metrics.was_deduplicated = True
            metrics.end_time = time.time()
            metrics.final_status = "deduplicated"
            _request_logger.log_deduplication(request_hash, metrics)
            return cached_response
    else:
        request_hash = None
    
    # Get prompt preview for logging
    prompt_preview = ""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                prompt_preview = content
                break
    
    _request_logger.log_request_start(metrics, prompt_preview)
    
    # Acquire concurrency slot
    limiter = get_concurrency_limiter()
    
    if not limiter.acquire(timeout=config.request_timeout):
        metrics.end_time = time.time()
        metrics.final_status = "timeout"
        metrics.error_message = "Timed out waiting for concurrency slot"
        _request_logger.log_request_failure(metrics)
        raise TimeoutError(
            f"Timed out waiting for concurrency slot after {config.request_timeout}s"
        )
    
    try:
        # Execute with retry logic
        last_error: Optional[Exception] = None
        
        for attempt in range(1, config.max_retries + 1):
            try:
                result = request_func(messages=messages, **kwargs)
                
                # Success!
                metrics.end_time = time.time()
                metrics.final_status = "success"
                metrics.retry_count = attempt - 1
                
                # Extract token usage if available
                if hasattr(result, 'usage'):
                    metrics.actual_tokens = {
                        "input": getattr(result.usage, 'input_tokens', 
                                        getattr(result.usage, 'prompt_tokens', 0)),
                        "output": getattr(result.usage, 'output_tokens',
                                         getattr(result.usage, 'completion_tokens', 0)),
                    }
                
                # Cache for deduplication
                if request_hash and config.enable_deduplication:
                    get_deduplicator().cache_response(request_hash, result)
                
                _request_logger.log_request_success(metrics)
                return result
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if this error is retryable
                if not is_retryable_error(e):
                    metrics.end_time = time.time()
                    metrics.final_status = "failed"
                    metrics.error_message = error_str
                    _request_logger.log_request_failure(metrics)
                    raise
                
                # Check if rate limited
                if is_rate_limit_error(e):
                    retry_after = extract_retry_after(e)
                    _request_logger.log_rate_limited(metrics, retry_after)
                
                # Check if we have retries left
                if attempt >= config.max_retries:
                    break
                
                # Calculate backoff
                retry_after_hint = extract_retry_after(e)
                backoff = calculate_backoff(attempt, config, retry_after_hint)
                metrics.backoff_durations.append(backoff)
                
                _request_logger.log_retry_attempt(metrics, attempt, backoff, error_str)
                
                # Wait before retry
                time.sleep(backoff)
        
        # All retries exhausted
        metrics.end_time = time.time()
        metrics.retry_count = config.max_retries
        
        if last_error and is_rate_limit_error(last_error):
            metrics.final_status = "rate_limited"
            retry_after = extract_retry_after(last_error)
            metrics.error_message = str(last_error)
            _request_logger.log_request_failure(metrics)
            
            # Return structured error for rate limits
            raise RateLimitException(
                RateLimitErrorResponse(
                    error="RATE_LIMITED",
                    retry_after=retry_after,
                    action="failed",
                    provider=provider,
                    model=model,
                    message=str(last_error),
                )
            )
        else:
            metrics.final_status = "failed"
            metrics.error_message = str(last_error) if last_error else "Unknown error"
            _request_logger.log_request_failure(metrics)
            
            if last_error:
                raise last_error
            raise RuntimeError("Request failed after all retries")
    
    finally:
        limiter.release()


class RateLimitException(Exception):
    """Exception raised when rate limits are exhausted."""
    
    def __init__(self, error_response: RateLimitErrorResponse):
        self.error_response = error_response
        super().__init__(error_response.message or "Rate limit exceeded")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.error_response.to_dict()


# ============================================================================
# Statistics and Health
# ============================================================================

def get_resilience_stats() -> Dict[str, Any]:
    """Get statistics about rate limit handling and resilience."""
    return {
        # One entry per provider/endpoint — the vision and agent models have
        # independent limiters.
        "concurrency": {
            name: limiter.get_stats()
            for name, limiter in sorted(_concurrency_limiters.items())
        },
        "deduplication": get_deduplicator().get_stats(),
        "config": get_rate_limit_config().to_dict(),
    }


def reset_resilience_stats():
    """Reset all resilience statistics (for testing)."""
    global _deduplicator
    with _limiter_lock:
        _concurrency_limiters.clear()
    with _dedup_lock:
        _deduplicator = None


# ============================================================================
# Resilient LLM Client Wrapper
# ============================================================================

class ResilientLLMClient:
    """
    A wrapper class that adds resilience features to any LLM client.
    
    This class wraps an existing LLM client (Anthropic, OpenAI, etc.) and adds:
    - Automatic retry with exponential backoff
    - Concurrency limiting
    - Request deduplication
    - Token estimation and protection
    - Structured logging
    
    Usage:
        import anthropic
        from router.resilience import ResilientLLMClient
        
        raw_client = anthropic.Anthropic(api_key="...")
        client = ResilientLLMClient(
            raw_client,
            provider="anthropic",
            model="claude-3-opus"
        )
        
        # Now use with automatic resilience
        response = client.messages_create(
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=1024
        )
    """
    
    def __init__(
        self,
        client: Any,
        provider: str,
        model: str,
        config: Optional[RateLimitConfig] = None,
        enable_dedup: bool = True,
    ):
        """
        Initialize a resilient LLM client wrapper.
        
        Args:
            client: The underlying LLM client (anthropic.Anthropic, openai.OpenAI, etc.)
            provider: Provider name for logging
            model: Default model name
            config: Rate limit configuration (uses global if None)
            enable_dedup: Whether to enable request deduplication
        """
        self._client = client
        self._provider = provider
        self._model = model
        self._config = config or get_rate_limit_config()
        self._enable_dedup = enable_dedup
        self._request_count = 0
        self._error_count = 0
        self._lock = threading.Lock()
    
    @property
    def provider(self) -> str:
        return self._provider
    
    @property
    def model(self) -> str:
        return self._model
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        with self._lock:
            return {
                "provider": self._provider,
                "model": self._model,
                "request_count": self._request_count,
                "error_count": self._error_count,
            }
    
    def messages_create(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Create a chat completion with full resilience features.
        
        This is the main entry point for Anthropic-style APIs.
        
        Args:
            messages: List of chat messages
            model: Model override (uses default if None)
            max_tokens: Max output tokens (uses config default if None)
            **kwargs: Additional arguments passed to the client
            
        Returns:
            API response
            
        Raises:
            RateLimitException: If rate limits exhausted
            Various client-specific exceptions for non-retryable errors
        """
        model = model or self._model
        max_tokens = max_tokens or self._config.max_output_tokens
        
        with self._lock:
            self._request_count += 1
        
        try:
            return make_resilient_request(
                request_func=self._make_anthropic_request,
                messages=messages,
                provider=self._provider,
                model=model,
                config=self._config,
                enable_dedup=self._enable_dedup,
                model_param=model,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            with self._lock:
                self._error_count += 1
            raise
    
    def _make_anthropic_request(
        self,
        messages: List[Dict[str, Any]],
        model_param: str,
        max_tokens: int,
        **kwargs
    ) -> Any:
        """Make the actual Anthropic API call."""
        return self._client.messages.create(
            model=model_param,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs
        )
    
    def chat_completions_create(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Create a chat completion with full resilience features.
        
        This is the main entry point for OpenAI-style APIs.
        
        Args:
            messages: List of chat messages
            model: Model override (uses default if None)
            max_tokens: Max output tokens (uses config default if None)
            **kwargs: Additional arguments passed to the client
            
        Returns:
            API response
        """
        model = model or self._model
        max_tokens = max_tokens or self._config.max_output_tokens
        
        with self._lock:
            self._request_count += 1
        
        try:
            return make_resilient_request(
                request_func=self._make_openai_request,
                messages=messages,
                provider=self._provider,
                model=model,
                config=self._config,
                enable_dedup=self._enable_dedup,
                model_param=model,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            with self._lock:
                self._error_count += 1
            raise
    
    def _make_openai_request(
        self,
        messages: List[Dict[str, Any]],
        model_param: str,
        max_tokens: int,
        **kwargs
    ) -> Any:
        """Make the actual OpenAI API call."""
        return self._client.chat.completions.create(
            model=model_param,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs
        )
