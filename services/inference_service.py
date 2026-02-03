"""Inference backend service for VLM availability checks."""

from __future__ import annotations

import requests

from core.config import get_config
from core.logging import get_logger

logger = get_logger(__name__)


def check_ollama_availability() -> bool:
    """Check if Ollama is available."""
    config = get_config()
    try:
        response = requests.get(
            f"{config.inference.ollama_url}/api/version",
            timeout=config.http_timeout
        )
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def check_vllm_availability() -> bool:
    """Check if vLLM server is available."""
    config = get_config()
    try:
        response = requests.get(
            f"{config.inference.vllm_url}/v1/models",
            timeout=config.http_timeout
        )
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def check_inference_backend_availability() -> bool:
    """Check if the configured inference backend is available."""
    config = get_config()
    backend = config.inference.backend.lower()
    
    if backend == "vllm":
        return check_vllm_availability()
    return check_ollama_availability()
