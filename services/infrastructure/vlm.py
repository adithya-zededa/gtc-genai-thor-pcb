"""VLM client factory service."""

from __future__ import annotations

from typing import Any, Dict

from agents.vlm.client import UnifiedVLMClient
from core.config import get_config
from core.logging import get_logger

logger = get_logger(__name__)


def create_vlm_client_from_config(cfg: Dict[str, Any]) -> UnifiedVLMClient:
    """Create a VLM client for the vision role.

    All inference is routed through the centralized ``router`` package.
    Precedence is environment > YAML > dataclass default; the environment
    layer is applied once, inside ``core.config``, rather than re-read
    here. Only values the YAML can legitimately override (``vllm.model``,
    and the timeout/temperature when the env is silent) are read from
    *cfg*.
    """
    app_config = get_config()

    vllm_cfg = cfg.get("vllm", {})
    vllm_url = app_config.inference.vllm_url.rstrip("/")
    from core.model_detect import detect_model
    yaml_model = str(vllm_cfg.get("model", ""))
    if yaml_model and yaml_model != "auto":
        vision_model = yaml_model
    else:
        vision_model = app_config.inference.model or detect_model(
            backend="vllm", base_url=vllm_url, wait=False
        )
    timeout = int(vllm_cfg.get("timeout", app_config.inference.timeout))
    temperature = float(vllm_cfg.get("temperature", app_config.inference.temperature))
    
    logger.info("Creating VLM client: model=%s (routed via centralized router)", vision_model)
    return UnifiedVLMClient(
        model=vision_model,
        timeout=timeout,
        temperature=temperature,
    )
