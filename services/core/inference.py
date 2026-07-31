"""Inference backend service for VLM availability checks."""

from __future__ import annotations

import requests

from core.config import get_config
from core.logging import get_logger

logger = get_logger(__name__)


def _check_url(url: str, timeout: float) -> bool:
    """Return True if *url* responds with a successful status."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def check_vllm_availability() -> bool:
    """Check if the vision model's vLLM server is available."""
    config = get_config()
    return _check_url(
        f"{config.inference.vllm_url}/v1/models",
        timeout=config.http_timeout,
    )


def check_agent_llm_availability() -> bool:
    """Check if the agent (text-only reasoning) model's vLLM server is available."""
    config = get_config()
    return _check_url(
        f"{config.router.url}/v1/models",
        timeout=config.http_timeout,
    )


def check_inference_backend_availability() -> bool:
    """Check if the vision model's vLLM inference backend is available.

    Kept as the vision-specific name for backward compatibility with
    existing callers; prefer ``check_vllm_availability`` for new code.
    """
    return check_vllm_availability()
