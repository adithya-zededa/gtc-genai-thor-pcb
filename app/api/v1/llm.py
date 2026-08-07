"""LLM API endpoints for provider management.

This module provides REST API endpoints for:
- Checking LLM provider health and status
- Querying token usage statistics
- Sending test chat messages
- Listing available models
- Updating provider configuration

The router layer abstracts the underlying backend (vLLM, OpenAI, etc.),
so these endpoints work with any configured adapter.
"""

# pylint: disable=broad-exception-caught,import-outside-toplevel

from urllib.parse import urlparse

from flask import jsonify, request

from core.logging import get_logger

from . import api_bp

logger = get_logger(__name__)

# Cloud instance-metadata endpoints. These have no auth of their own and are
# the highest-value SSRF target (credential theft) reachable through a
# provider-URL field on an unauthenticated endpoint. This is a partial
# mitigation only — private/loopback hosts are intentionally NOT blocked,
# since the app's own vLLM target is legitimately a private address.
_BLOCKED_HOSTS = frozenset({
    "169.254.169.254",
    "fd00:ec2::254",
    "metadata.google.internal",
})


def _validate_provider_url(url):
    """Reject non-http(s) schemes and known cloud-metadata hosts.

    Returns an error string if *url* is invalid/unsafe, else ``None``.
    Empty/missing URLs are not this function's concern — callers only
    invoke it when a URL was actually supplied.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Unsupported URL scheme: {parsed.scheme or '(none)'!r}"
    hostname = (parsed.hostname or "").lower()
    if hostname in _BLOCKED_HOSTS:
        return "URL host is not allowed"
    return None


def _get_router():
    """Get the LLM router instance."""
    try:
        from router import get_router

        return get_router()
    except Exception as exc:
        logger.error("Failed to get LLM router: %s", exc)
        return None


# =============================================================================
# PROVIDER STATUS
# =============================================================================


@api_bp.route("/llm/providers", methods=["GET"])
def list_llm_providers():
    """List configured LLM providers with status."""
    router = _get_router()
    if router is None:
        return (
            jsonify(
                {
                    "success": True,
                    "enabled": False,
                    "providers": [],
                    "count": 0,
                }
            ),
            200,
        )

    providers = router.list_providers()
    active = router.get_active_provider()

    return jsonify(
        {
            "success": True,
            "enabled": True,
            "providers": providers,
            "active_provider": active,
            "count": len(providers),
        }
    )


# =============================================================================
# HEALTH & STATUS
# =============================================================================


@api_bp.route("/llm/health", methods=["GET"])
def check_llm_health():
    """Check health of every configured LLM role.

    The system serves two models from two vLLM pods, so this reports each
    role separately — a healthy vision model with a down agent model still
    breaks chat and tool calling, and vice versa.
    """
    try:
        from router import ROLES, get_router
    except Exception:
        return jsonify({"success": False, "error": "LLM router not available"}), 200

    roles = {}
    for role in ROLES:
        try:
            router = get_router(role)
            health = router.check_health()
            config = router.get_config()
            roles[role] = {
                "available": bool(health.get("available", False)),
                "url": config.url if config else None,
                "model": config.model if config else None,
                "detail": health,
            }
        except Exception as exc:
            logger.exception("Health check failed for LLM role %s: %s", role, exc)
            roles[role] = {"available": False, "error": "Health check failed"}

    return jsonify(
        {
            "success": True,
            "roles": roles,
            # Retained for existing callers that read a single provider's health.
            "health": roles.get("vision", {}).get("detail", {}),
            "all_healthy": all(r.get("available") for r in roles.values()),
        }
    )


@api_bp.route("/llm/status", methods=["GET"])
def get_llm_status():
    """Get full LLM status including config and token usage."""
    router = _get_router()

    if router is None:
        return jsonify(
            {
                "success": True,
                "enabled": False,
            }
        )

    try:
        from router import get_token_usage

        usage = get_token_usage()
    except Exception:
        usage = {}

    return jsonify(
        {
            "success": True,
            "enabled": True,
            "router": router.to_dict(),
            "token_usage": usage,
        }
    )


# =============================================================================
# CONFIGURATION
# =============================================================================


@api_bp.route("/llm/config", methods=["PUT"])
def update_llm_config():
    """Update LLM provider configuration at runtime.

    Request body::

        {
            "url": "http://llm-service:8000",
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "timeout": 300,
            "temperature": 0.1
        }
    """
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "LLM router not available"}), 400

    data = request.get_json() or {}
    url = data.get("url")
    if url:
        error = _validate_provider_url(url)
        if error:
            return jsonify({"success": False, "error": error}), 400

    try:
        router.configure(
            url=url,
            model=data.get("model"),
            api_key=data.get("api_key"),
            timeout=data.get("timeout"),
            temperature=data.get("temperature"),
        )
        active = router.get_active_provider()
        provider_name = (active or {}).get("name", "llm")
        return jsonify(
            {
                "success": True,
                "message": f"{provider_name} configuration updated",
                "config": (
                    router.get_config().to_dict() if router.get_config() else None
                ),
            }
        )
    except Exception as exc:
        logger.exception("Failed to update LLM config: %s", exc)
        return jsonify({"success": False, "error": "Failed to update configuration"}), 400


# =============================================================================
# MODELS
# =============================================================================


@api_bp.route("/llm/models", methods=["GET"])
def list_llm_models():
    """List models available from the configured LLM provider."""
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "LLM router not available"}), 200

    active = router.get_active_provider()
    provider_name = (active or {}).get("name", "default")

    try:
        models = router.list_models()
        return jsonify(
            {
                "success": True,
                "provider": provider_name,
                "models": models,
            }
        )
    except Exception as exc:
        logger.exception("Failed to list LLM models: %s", exc)
        return jsonify(
            {
                "success": True,
                "provider": provider_name,
                "models": [],
                "error": "Unable to list models",
            }
        )


@api_bp.route("/llm/models/fetch", methods=["POST"])
def fetch_models_for_provider():
    """Fetch available models from an LLM provider.

    Uses the active provider's adapter by default.  Supply an optional
    ``provider_type`` (e.g. ``"vllm"``) to target a specific backend.

    Request body::

        {
            "url": "http://localhost:8000",
            "api_key": null,
            "provider_type": "vllm"
        }
    """
    data = request.get_json() or {}
    url = data.get("url")
    provider_type = data.get("provider_type")

    if url:
        error = _validate_provider_url(url)
        if error:
            return jsonify({"success": False, "error": error}), 400

    try:
        from router.adapters import get_adapter
        from router.config import LLMProviderConfig

        adapter = get_adapter(provider_type)

        temp_config = LLMProviderConfig(
            name="_temp_fetch",
            url=url,
            model="temp",
            api_key=data.get("api_key"),
        )

        available, _latency, error = adapter.check_availability(temp_config)
        if not available:
            return jsonify(
                {
                    "success": True,
                    "models": [],
                    "error": error or "Provider not reachable",
                }
            )

        models = adapter.list_models(temp_config)
        return jsonify({"success": True, "models": models})
    except Exception as exc:
        logger.exception("Model fetch failed: %s", exc)
        return jsonify(
            {"success": True, "models": [], "error": "Unable to fetch models"}
        )


# =============================================================================
# TOKEN USAGE
# =============================================================================


@api_bp.route("/llm/usage", methods=["GET"])
def get_llm_token_usage():
    """Get token usage statistics."""
    try:
        from router import get_token_usage

        usage = get_token_usage()
        return jsonify({"success": True, "usage": usage})
    except Exception as exc:
        logger.exception("Failed to get token usage: %s", exc)
        return jsonify({"success": False, "error": "Unable to retrieve token usage"}), 500


@api_bp.route("/llm/usage", methods=["DELETE"])
def reset_llm_token_usage():
    """Reset token usage statistics."""
    try:
        from router import reset_token_usage

        reset_token_usage()
        return jsonify({"success": True, "message": "Token usage reset"})
    except Exception as exc:
        logger.exception("Failed to reset token usage: %s", exc)
        return jsonify({"success": False, "error": "Unable to reset token usage"}), 500


# =============================================================================
# TEST CHAT
# =============================================================================


@api_bp.route("/llm/chat", methods=["POST"])
def llm_test_chat():
    """Send a test chat message through the configured LLM provider.

    Request body::

        {
            "message": "Hello, what can you do?"
        }
    """
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "LLM router not available"}), 400

    data = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"success": False, "error": "No message provided"}), 400

    try:
        response = router.chat(
            messages=[{"role": "user", "content": message}],
        )
        return jsonify(
            {
                "success": True,
                "response": response.to_dict(),
            }
        )
    except Exception as exc:
        logger.exception("Test chat failed: %s", exc)
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Chat request failed",
                }
            ),
            500,
        )
