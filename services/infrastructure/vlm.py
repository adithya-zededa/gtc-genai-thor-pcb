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
    
    Uses the vLLM backend exclusively.
    Environment variables take precedence over config file settings.
    """
    app_config = get_config()
    
    vllm_cfg = cfg.get("vllm", {})
    vllm_url = os.getenv("VLLM_URL") or str(vllm_cfg.get("url", app_config.inference.vllm_url)).rstrip("/")
    from core.model_detect import detect_model
    default_model = detect_model(backend="vllm", base_url=vllm_url, wait=False)
    yaml_model = str(vllm_cfg.get("model", ""))
    vision_model = yaml_model if yaml_model and yaml_model != "auto" else default_model
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
