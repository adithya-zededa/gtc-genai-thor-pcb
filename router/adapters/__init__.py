"""
LLM Adapters Package

Adapter implementations for LLM backends.  Each adapter subclasses
:class:`router.base.LLMAdapter` and exposes the same interface for
availability checks, model listing, and chat completions.

Currently supported backends:
- **vllm** – vLLM / any OpenAI-compatible server
"""

from __future__ import annotations

from typing import Optional

from ..base import LLMAdapter
from .vllm import VLLMAdapter

# ---------------------------------------------------------------------------
# Adapter registry – maps provider type strings to adapter classes.
# When new backends are added, register them here.
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, type[LLMAdapter]] = {
    "vllm": VLLMAdapter,
    "openai": VLLMAdapter,  # OpenAI-compatible API, same wire format
}


def get_adapter(provider_type: Optional[str] = None) -> LLMAdapter:
    """Return an adapter instance for *provider_type*.

    Parameters
    ----------
    provider_type:
        Key into the adapter registry (e.g. ``"vllm"``).
        ``None`` or an empty string falls back to the default (vLLM).

    Raises
    ------
    ValueError
        If *provider_type* is not recognised.
    """
    key = (provider_type or "vllm").lower().strip()
    cls = _ADAPTER_REGISTRY.get(key)
    if cls is None:
        supported = ", ".join(sorted(_ADAPTER_REGISTRY))
        raise ValueError(
            f"Unknown provider type {key!r}. Supported: {supported}"
        )
    return cls()


__all__ = [
    "LLMAdapter",
    "VLLMAdapter",
    "get_adapter",
]
