"""Auto-detect the served model from a running vLLM backend.

Instead of hardcoding a model name, the app queries the vLLM inference
server on startup and uses whatever model it finds.  The detected model
is cached so subsequent calls are free.

Usage::

    from core.model_detect import detect_model

    model = detect_model()                    # uses VLLM_URL env
    model = detect_model(base_url="http://vllm:8000")
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import requests
from requests import RequestException

from core.logging import get_logger

logger = get_logger(__name__)

# Module-level cache
_state: dict[str, Optional[str]] = {"detected_model": None}
_detect_lock = threading.Lock()

# How long to wait for server to become available (seconds)
_MAX_WAIT = 120
_POLL_INTERVAL = 5


def _fetch_vllm_model(base_url: str, timeout: int = 10) -> Optional[str]:
    """Query ``GET /v1/models`` and return the first model id."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            model_id = data[0].get("id")
            if model_id:
                return model_id
    except (RequestException, ValueError, TypeError, KeyError, IndexError) as exc:
        logger.debug("vLLM /v1/models probe failed: %s", exc)
    return None


def detect_model(
    *,
    backend: Optional[str] = None,
    base_url: Optional[str] = None,
    wait: bool = True,
    force: bool = False,
) -> str:
    """Return the model served by the vLLM backend.

    Resolution order:
    1. ``VISION_MODEL`` env var (explicit override always wins).
    2. Cached result from a prior detection.
    3. Live query to the vLLM server (``/v1/models``).

    Parameters
    ----------
    backend : str, optional
        Kept for API compatibility. Always uses vLLM.
    base_url : str, optional
        Server URL.  Defaults to ``VLLM_URL`` env.
    wait : bool
        If *True* (default), retry for up to ~2 min while the server
        starts up.  Set to *False* for a single non-blocking probe.
    force : bool
        Bypass the cache and re-query the server.
    """
    if backend and backend.lower() != "vllm":
        logger.debug(
            "Ignoring unsupported backend=%s; using vLLM autodetection for compatibility.",
            backend,
        )

    # 1. Explicit env override — never auto-detect
    env_model = os.getenv("VISION_MODEL")
    if env_model and env_model.strip() and env_model.strip().lower() != "auto":
        return env_model.strip()

    # 2. Cached
    if _state["detected_model"] and not force:
        return _state["detected_model"]

    # 3. Live detection
    url = (base_url or os.getenv("VLLM_URL", "http://localhost:8000")).rstrip("/")
    fetch = _fetch_vllm_model

    model = fetch(url)

    if model is None and wait:
        logger.info(
            "Waiting for vLLM server at %s to report a model (up to %ds)...",
            url, _MAX_WAIT,
        )
        deadline = time.monotonic() + _MAX_WAIT
        while time.monotonic() < deadline:
            time.sleep(_POLL_INTERVAL)
            model = fetch(url)
            if model:
                break

    if model:
        with _detect_lock:
            _state["detected_model"] = model
        logger.info("Auto-detected model from vLLM: %s", model)
        return model

    # Final fallback
    fallback = "auto"
    logger.warning(
        "Could not auto-detect model from vLLM at %s — using fallback '%s'.  "
        "Set VISION_MODEL env var to override.",
        url, fallback,
    )
    return fallback


def reset_cache() -> None:
    """Clear the cached model (for testing)."""
    with _detect_lock:
        _state["detected_model"] = None
