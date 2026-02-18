"""Backward-compat re-export — canonical location is core.resilience."""

from core.resilience import CircuitBreaker, CircuitState  # noqa: F401

__all__ = ["CircuitBreaker", "CircuitState"]
