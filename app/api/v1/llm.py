"""LLM API endpoints for vLLM provider management.

This module provides REST API endpoints for:
- Checking vLLM provider health and status
- Querying token usage statistics
- Sending test chat messages
- Listing available models
- Updating vLLM configuration
"""

# pylint: disable=broad-exception-caught,import-outside-toplevel

from flask import jsonify, request

from core.logging import get_logger

from . import api_bp

logger = get_logger(__name__)


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
    """List the vLLM provider with status."""
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
            "provider": "vllm",
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
    """Check health of the vLLM provider."""
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "LLM router not available"}), 200

    results = router.check_health()
    return jsonify(
        {
            "success": True,
            "health": results,
            "all_healthy": all(results.values()) if results else False,
        }
    )


@api_bp.route("/llm/status", methods=["GET"])
def get_llm_status():
    """Get full vLLM status including config and token usage."""
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
    """Update vLLM configuration at runtime.

    Request body::

        {
            "url": "http://vllm-service:8000",
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "timeout": 300,
            "temperature": 0.1
        }
    """
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "LLM router not available"}), 400

    data = request.get_json() or {}

    try:
        router.configure(
            url=data.get("url"),
            model=data.get("model"),
            api_key=data.get("api_key"),
            timeout=data.get("timeout"),
            temperature=data.get("temperature"),
        )
        return jsonify(
            {
                "success": True,
                "message": "vLLM configuration updated",
                "config": (
                    router.get_config().to_dict() if router.get_config() else None
                ),
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


# =============================================================================
# MODELS
# =============================================================================


@api_bp.route("/llm/models", methods=["GET"])
def list_llm_models():
    """List models available from the vLLM server."""
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "LLM router not available"}), 200

    try:
        models = router.list_models()
        return jsonify(
            {
                "success": True,
                "models": {"vllm": models},
            }
        )
    except Exception as exc:
        return jsonify(
            {
                "success": True,
                "models": {"vllm": {"error": str(exc)}},
            }
        )


@api_bp.route("/llm/models/fetch", methods=["POST"])
def fetch_models_for_provider():
    """Fetch available models from the vLLM server.

    Request body::

        {
            "url": "http://localhost:8000",
            "api_key": null
        }
    """
    data = request.get_json() or {}
    url = data.get("url")

    try:
        from router.adapters import VLLMAdapter
        from router.config import LLMProviderConfig

        temp_config = LLMProviderConfig(
            name="_temp_fetch",
            url=url,
            model="temp",
            api_key=data.get("api_key"),
        )

        adapter = VLLMAdapter()
        available, _latency, error = adapter.check_availability(temp_config)
        if not available:
            return jsonify(
                {"success": True, "models": [], "error": error or "vLLM not reachable"}
            )

        models = adapter.list_models(temp_config)
        return jsonify({"success": True, "models": models})
    except Exception as exc:
        logger.warning("Model fetch failed: %s", exc)
        return jsonify({"success": True, "models": [], "error": str(exc)})


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
        return jsonify({"success": False, "error": str(exc)}), 500


@api_bp.route("/llm/usage", methods=["DELETE"])
def reset_llm_token_usage():
    """Reset token usage statistics."""
    try:
        from router import reset_token_usage

        reset_token_usage()
        return jsonify({"success": True, "message": "Token usage reset"})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# =============================================================================
# TEST CHAT
# =============================================================================


@api_bp.route("/llm/chat", methods=["POST"])
def llm_test_chat():
    """Send a test chat message through vLLM.

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
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                }
            ),
            500,
        )
