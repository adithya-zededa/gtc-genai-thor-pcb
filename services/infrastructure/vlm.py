"""VLM client factory service."""

from __future__ import annotations

import os
from typing import Any, Dict

from agents.vlm.client import UnifiedVLMClient, VLMBackend
from core.config import get_config
from core.logging import get_logger

logger = get_logger(__name__)


def create_vlm_client_from_config(cfg: Dict[str, Any]) -> UnifiedVLMClient:
    """Create a VLM client from configuration.
    
    Supports both Ollama and vLLM backends based on config and environment variables.
    Environment variables take precedence over config file settings.
    """
    app_config = get_config()
    
    # Check environment variable for backend selection
    env_backend = os.getenv("INFERENCE_BACKEND", "").lower()
    
    # Determine backend: env var > config > default (vllm)
    if env_backend == "vllm" or (not env_backend and cfg.get("vllm")):
        # Use vLLM backend
        vllm_cfg = cfg.get("vllm", {})
        vllm_url = os.getenv("VLLM_URL") or str(vllm_cfg.get("url", app_config.inference.vllm_url)).rstrip("/")
        default_model = os.getenv("VISION_MODEL", app_config.inference.model)
        vision_model = str(vllm_cfg.get("model", default_model))
        timeout = int(os.getenv("VLLM_TIMEOUT", vllm_cfg.get("timeout", app_config.inference.timeout)))
        temperature = float(os.getenv("VLLM_TEMPERATURE", vllm_cfg.get("temperature", app_config.inference.temperature)))
        
        logger.info("Creating vLLM client: url=%s, model=%s", vllm_url, vision_model)
        return UnifiedVLMClient(
            base_url=vllm_url,
            model=vision_model,
            timeout=timeout,
            backend=VLMBackend.VLLM,
            temperature=temperature,
        )
    
    # Fall back to Ollama config (legacy)
    ollama_cfg = cfg.get("ollama", {})
    ollama_url = os.getenv("OLLAMA_URL") or str(ollama_cfg.get("url", app_config.inference.ollama_url)).rstrip("/")
    default_model = os.getenv("VISION_MODEL", "qwen3-vl:8b")
    vision_model = str(ollama_cfg.get("vision_model", default_model))
    timeout = int(ollama_cfg.get("timeout", app_config.inference.timeout))
    temperature = float(ollama_cfg.get("temperature", app_config.inference.temperature))
    
    logger.info("Creating Ollama client (legacy fallback): url=%s, model=%s", ollama_url, vision_model)
    return UnifiedVLMClient(
        base_url=ollama_url,
        model=vision_model,
        timeout=timeout,
        backend=VLMBackend.OLLAMA,
        temperature=temperature,
    )
