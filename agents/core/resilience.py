"""Resilience primitives for core agents."""

from __future__ import annotations

import threading
import time
from enum import Enum, auto
from typing import Any, Dict

from core.logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """States for the circuit breaker."""

    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:  # pylint: disable=too-many-instance-attributes
    """Circuit breaker pattern for resilience."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 120.0):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0.0
        self.total_calls = 0
        self.successful_calls = 0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        """Check if circuit is currently open."""
        with self._lock:
            return self.state == CircuitState.OPEN

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "state": self.state.name,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "success_rate": (
                    round(self.successful_calls / self.total_calls * 100, 2)
                    if self.total_calls > 0
                    else 0.0
                ),
            }

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            logger.info("Circuit breaker manually reset to CLOSED")

    def call(self, func, *args, **kwargs):
        """Execute a function with circuit breaker protection."""
        with self._lock:
            self.total_calls += 1
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF-OPEN state")
                else:
                    raise RuntimeError("Circuit is OPEN")

        try:
            result = func(*args, **kwargs)
            with self._lock:
                self.successful_calls += 1
                if self.state != CircuitState.CLOSED:
                    logger.info("Circuit breaker recovering to CLOSED state")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            return result
        except Exception as error:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                should_trip = (
                    self.state == CircuitState.HALF_OPEN
                    or self.failure_count >= self.failure_threshold
                )
                if should_trip:
                    self.state = CircuitState.OPEN
                    logger.warning(
                        "Circuit breaker tripped to OPEN (failures: %s). Error: %s",
                        self.failure_count,
                        error,
                    )
            raise
