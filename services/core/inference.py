"""Inference backend availability checks.

The deployment serves two models from two vLLM pods — a vision model that
looks at frames and an agent model that drives chat and tool calls — so
availability is reported *per role*. A single-pod deployment leaves
``AGENT_LLM_URL`` unset, both roles resolve to the same URL, and the two
checks simply agree.
"""

from __future__ import annotations

from typing import Dict

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
    """Check whether the vision vLLM server is reachable."""
    config = get_config()
    return _check_url(
        f"{config.inference.vllm_url}/v1/models",
        timeout=config.http_timeout,
    )


def check_agent_llm_availability() -> bool:
    """Check whether the agent (text) vLLM server is reachable."""
    config = get_config()
    return _check_url(
        f"{config.agent_inference_url}/v1/models",
        timeout=config.http_timeout,
    )


def check_inference_roles() -> Dict[str, bool]:
    """Availability of each inference role, keyed ``vision`` / ``agent``.

    When both roles share one endpoint the second check is skipped and the
    first result is reused, so this costs one HTTP call on a single-pod
    deployment and two on a split one.
    """
    config = get_config()
    vision = check_vllm_availability()
    if config.agent_inference_url == config.inference.vllm_url:
        agent = vision
    else:
        agent = check_agent_llm_availability()
    return {"vision": vision, "agent": agent}


def check_inference_backend_availability() -> bool:
    """Whether *every* configured inference role is reachable.

    Chat is as much a part of the product as inspection, so a reachable
    vision pod alone is not "inference available".
    """
    return all(check_inference_roles().values())
