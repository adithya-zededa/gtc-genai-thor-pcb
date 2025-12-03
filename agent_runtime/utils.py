"""Shared utility helpers for configuration and runtime safety checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List


def coerce_bool(value: Any, default: bool | None = False) -> bool | None:
    """Convert common truthy/falsey representations into booleans."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def clamp_float(
    value: Any,
    minimum: float = 0.0,
    maximum: float = 1.0,
    default: float | None = None,
) -> float:
    """Clamp numeric values to the provided range while handling conversion errors."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        if default is not None:
            numeric = default
        else:
            numeric = minimum
    return max(minimum, min(maximum, numeric))


def safe_positive_int(value: Any, default: int) -> int:
    """Return a positive integer or a fallback default."""
    try:
        candidate = int(value)
        return candidate if candidate > 0 else default
    except (TypeError, ValueError):
        return default


def safe_positive_float(value: Any, default: float) -> float:
    """Return a positive float or a fallback default."""
    try:
        candidate = float(value)
        return candidate if candidate > 0 else default
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float) -> float:
    """Return a float value or a fallback default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ensure_directory(path: Path) -> Path:
    """Create a directory path if it does not already exist and return it."""
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def dedupe_strings(values: Iterable[str]) -> List[str]:
    """Return a list with duplicate strings removed while preserving order."""
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


__all__ = [
    "clamp_float",
    "coerce_bool",
    "dedupe_strings",
    "ensure_directory",
    "safe_float",
    "safe_positive_float",
    "safe_positive_int",
]
