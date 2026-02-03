"""Shared utility functions for the Camera Agent application.

This module consolidates utility functions used across the application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional


def coerce_bool(value: Any, default: Optional[bool] = False) -> Optional[bool]:
    """Convert common truthy/falsey representations into booleans.
    
    Args:
        value: Value to convert.
        default: Default value if conversion fails.
        
    Returns:
        Boolean value or default.
    """
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


def ensure_directory(path: Path) -> Path:
    """Create a directory path if it does not already exist and return it.
    
    Args:
        path: Directory path to create.
        
    Returns:
        The created or existing path.
    """
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def dedupe_strings(values: Iterable[str]) -> List[str]:
    """Return a list with duplicate strings removed while preserving order.
    
    Args:
        values: Iterable of strings to deduplicate.
        
    Returns:
        Deduplicated list preserving original order.
    """
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


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to integer.
    
    Args:
        value: Value to convert.
        default: Default value if conversion fails.
        
    Returns:
        Integer value or default.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float.
    
    Args:
        value: Value to convert.
        default: Default value if conversion fails.
        
    Returns:
        Float value or default.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "coerce_bool",
    "dedupe_strings",
    "ensure_directory",
    "safe_int",
    "safe_float",
]
