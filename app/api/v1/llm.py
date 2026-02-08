"""LLM Router API endpoints for managing LLM providers.

This module provides REST API endpoints for:
- Listing registered LLM providers and their status
- Adding / removing providers at runtime
- Checking provider health
- Querying token usage statistics
- Sending test chat messages through the router
- Listing available models per provider
- Fetching models from a provider without registering it
- Saving / loading / exporting / importing provider configs
"""

import json
import os
import time
from pathlib import Path

from flask import jsonify, request

from . import api_bp
from core.logging import get_logger

logger = get_logger(__name__)

# Path for persisting LLM provider configurations
_LLM_CONFIG_FILE = str(Path(__file__).resolve().parents[3] / "llm_providers.json")


def _get_router():
    """Get the LLM router instance, auto-enabling if needed."""
    try:
        from router import get_router
        return get_router()
    except Exception as exc:
        logger.error("Failed to get LLM router: %s", exc)
        return None


def _ensure_router_enabled():
    """Enable the router config flag if not already set."""
    from core.config import get_config
    cfg = get_config()
    if not cfg.router.enabled:
        cfg.router.enabled = True
        logger.info("LLM Router auto-enabled via settings UI")


def _serialize_provider(config) -> dict:
    """Serialise a single ``LLMProviderConfig`` to a plain dict."""
    return {
        "name": config.name,
        "provider_type": config.provider_type.value,
        "url": config.url,
        "model": config.model,
        "api_key": config.api_key,
        "priority": config.priority,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "enabled": config.enabled,
        "supports_tools": config.supports_tools,
        "supports_vision": config.supports_vision,
    }


def _collect_provider_configs(router) -> list[dict]:
    """Collect serialised configs for every registered provider."""
    configs = []
    for prov in router.list_providers():
        config = router.get_provider(prov.get("name", ""))
        if config is not None:
            configs.append(_serialize_provider(config))
    return configs


def _save_provider_bundle(router) -> tuple[int, str | None]:
    """Persist provider configs to disk. Returns ``(count, error_msg | None)``."""
    configs = _collect_provider_configs(router)
    bundle = {
        "version": 1,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "routing_strategy": router._routing_strategy.value,
        "providers": configs,
    }
    try:
        with open(_LLM_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        logger.info("Saved %d LLM provider configs to %s", len(configs), _LLM_CONFIG_FILE)
        return len(configs), None
    except Exception as exc:
        return 0, str(exc)


# =============================================================================
# PROVIDER MANAGEMENT
# =============================================================================

@api_bp.route("/llm/providers", methods=["GET"])
def list_llm_providers():
    """List all registered LLM providers with status."""
    router = _get_router()
    if router is None:
        return jsonify({
            "success": True,
            "enabled": False,
            "providers": [],
            "count": 0,
        }), 200

    providers = router.list_providers()
    active = router.get_active_provider()

    return jsonify({
        "success": True,
        "enabled": True,
        "routing_strategy": router._routing_strategy.value,
        "providers": providers,
        "active_provider": active,
        "count": len(providers),
    })


@api_bp.route("/llm/providers", methods=["POST"])
def add_llm_provider():
    """Register a new LLM provider.

    Request body::

        {
            "name": "my-anthropic",
            "provider_type": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-ant-...",
            "priority": 1,
            "enabled": true
        }
    """
    _ensure_router_enabled()
    router = _get_router()
    if router is None:
        return jsonify({
            "success": False,
            "error": "LLM router could not be initialised",
        }), 500

    data = request.get_json() or {}

    required_fields = ["name", "provider_type"]
    for f in required_fields:
        if not data.get(f):
            return jsonify({
                "success": False,
                "error": f"Missing required field: {f}",
            }), 400

    try:
        from router.config import LLMProviderConfig
        config = LLMProviderConfig.from_dict(data)
        router.register_provider(config)

        return jsonify({
            "success": True,
            "message": f"Provider '{config.name}' registered",
            "provider": config.to_dict(),
        })
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 400


@api_bp.route("/llm/providers/<name>", methods=["DELETE"])
def remove_llm_provider(name: str):
    """Remove a registered LLM provider."""
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "LLM router not available"}), 400

    removed = router.unregister_provider(name)
    if removed:
        return jsonify({"success": True, "message": f"Provider '{name}' removed"})
    return jsonify({"success": False, "error": f"Provider '{name}' not found"}), 404


# =============================================================================
# HEALTH & STATUS
# =============================================================================

@api_bp.route("/llm/health", methods=["GET"])
def check_llm_health():
    """Check health of all LLM providers."""
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "LLM router not available"}), 200

    results = router.check_all_providers()
    return jsonify({
        "success": True,
        "health": results,
        "all_healthy": all(results.values()) if results else False,
    })


@api_bp.route("/llm/status", methods=["GET"])
def get_llm_status():
    """Get full router status including config and token usage."""
    router = _get_router()
    from core.config import get_config
    cfg = get_config()

    if router is None:
        return jsonify({
            "success": True,
            "enabled": False,
            "router_config": {
                "enabled": cfg.router.enabled,
                "routing_strategy": cfg.router.routing_strategy,
                "auto_discover": cfg.router.auto_discover,
                "use_for_classification": cfg.router.use_for_classification,
                "use_for_chat": cfg.router.use_for_chat,
            },
        })

    try:
        from router import get_token_usage
        usage = get_token_usage()
    except Exception:
        usage = {}

    return jsonify({
        "success": True,
        "enabled": True,
        "router": router.to_dict(),
        "token_usage": usage,
        "router_config": {
            "enabled": cfg.router.enabled,
            "routing_strategy": cfg.router.routing_strategy,
            "auto_discover": cfg.router.auto_discover,
            "use_for_classification": cfg.router.use_for_classification,
            "use_for_chat": cfg.router.use_for_chat,
        },
    })


# =============================================================================
# MODELS
# =============================================================================

@api_bp.route("/llm/models", methods=["GET"])
def list_llm_models():
    """List models available from a specific provider or all providers."""
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "LLM router not available"}), 200

    provider_name = request.args.get("provider")
    results = {}

    providers = router.list_providers()
    for prov in providers:
        name = prov["name"]
        if provider_name and name != provider_name:
            continue

        try:
            config = router.get_provider(name)
            if config:
                adapter = router._get_adapter(config.provider_type)
                models = adapter.list_models(config)
                results[name] = models
        except Exception as exc:
            results[name] = {"error": str(exc)}

    return jsonify({
        "success": True,
        "models": results,
    })


@api_bp.route("/llm/models/fetch", methods=["POST"])
def fetch_models_for_provider():
    """Fetch available models from a provider type + credentials.

    Used to populate the model dropdown *before* registering. For vLLM
    it queries ``/v1/models`` on the configured URL.

    Request body::

        {
            "provider_type": "vllm",
            "url": "http://localhost:8000",
            "api_key": null
        }
    """
    data = request.get_json() or {}
    provider_type_str = data.get("provider_type", "openai-compatible")
    api_key = data.get("api_key")
    url = data.get("url")

    try:
        from router.config import LLMProviderConfig, LLMProviderType

        try:
            provider_type = LLMProviderType(provider_type_str.lower())
        except ValueError:
            provider_type = LLMProviderType.OPENAI_COMPATIBLE

        temp_config = LLMProviderConfig(
            name="_temp_fetch",
            provider_type=provider_type,
            api_key=api_key,
            url=url,
            model="temp",
        )

        router = _get_router()
        if router is None:
            # Instantiate a temporary router just to get the adapter
            from router.llm_router import ADAPTER_REGISTRY
            adapter_cls = ADAPTER_REGISTRY.get(provider_type)
            if adapter_cls is None:
                return jsonify({"success": True, "models": [], "error": "Unknown provider type"})
            adapter = adapter_cls()
        else:
            adapter = router._get_adapter(provider_type)

        available, _latency, error = adapter.check_availability(temp_config)
        if not available:
            return jsonify({"success": True, "models": [], "error": error or "Provider not reachable"})

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
    """Send a test chat message through the router.

    Request body::

        {
            "message": "Hello, what can you do?",
            "provider": "anthropic"  // optional - use specific provider
        }
    """
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "LLM router not available"}), 400

    data = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"success": False, "error": "No message provided"}), 400

    provider_name = data.get("provider")

    try:
        response = router.chat(
            messages=[{"role": "user", "content": message}],
            provider_name=provider_name,
        )
        return jsonify({
            "success": True,
            "response": response.to_dict(),
        })
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 500


# =============================================================================
# ROUTING STRATEGY
# =============================================================================

@api_bp.route("/llm/strategy", methods=["PUT"])
def set_llm_strategy():
    """Change the routing strategy.

    Request body::

        {
            "strategy": "failover"  // priority, round_robin, failover, latency
        }
    """
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "LLM router not available"}), 400

    data = request.get_json() or {}
    strategy_str = data.get("strategy", "").strip().lower()
    if not strategy_str:
        return jsonify({"success": False, "error": "No strategy provided"}), 400

    try:
        from router.config import RoutingStrategy
        strategy = RoutingStrategy(strategy_str)
        router.set_routing_strategy(strategy)
        return jsonify({
            "success": True,
            "message": f"Routing strategy set to '{strategy.value}'",
        })
    except ValueError:
        valid = [s.value for s in RoutingStrategy]
        return jsonify({
            "success": False,
            "error": f"Invalid strategy. Valid: {valid}",
        }), 400


# =============================================================================
# SAVE / LOAD CONFIG  (persist to llm_providers.json)
# =============================================================================

@api_bp.route("/llm/config/save", methods=["POST"])
def save_llm_config():
    """Persist current LLM provider configs to disk.

    Saves all registered providers (with API keys) so they survive restarts.
    """
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "No router available"}), 400

    saved, error = _save_provider_bundle(router)
    if error:
        return jsonify({"success": False, "error": error}), 500
    return jsonify({"success": True, "saved": saved})


@api_bp.route("/llm/config/load", methods=["POST"])
def load_llm_config():
    """Load LLM provider configs from disk and register them."""
    if not os.path.isfile(_LLM_CONFIG_FILE):
        return jsonify({"success": False, "error": "No saved config found"}), 404

    try:
        with open(_LLM_CONFIG_FILE, "r", encoding="utf-8") as f:
            bundle = json.load(f)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Failed to read config: {exc}"}), 500

    _ensure_router_enabled()
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "Router init failed"}), 500

    from router.config import LLMProviderConfig, RoutingStrategy

    # Apply strategy
    strategy_str = bundle.get("routing_strategy", "failover")
    try:
        router.set_routing_strategy(RoutingStrategy(strategy_str))
    except ValueError:
        pass

    loaded = 0
    errors = []
    for prov in bundle.get("providers", []):
        try:
            config = LLMProviderConfig.from_dict(prov)
            router.register_provider(config)
            loaded += 1
        except Exception as exc:
            errors.append(f"{prov.get('name', '?')}: {exc}")

    return jsonify({
        "success": True,
        "loaded": loaded,
        "errors": errors,
    })


# =============================================================================
# EXPORT / IMPORT  (JSON bundle for sharing)
# =============================================================================

@api_bp.route("/llm/config/export", methods=["GET"])
def export_llm_config():
    """Export LLM provider configs as a JSON bundle (masks API keys)."""
    router = _get_router()

    providers = _collect_provider_configs(router) if router else []

    bundle = {
        "version": 1,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "routing_strategy": router._routing_strategy.value if router else "failover",
        "providers": providers,
    }
    return jsonify({"success": True, "bundle": bundle})


@api_bp.route("/llm/config/import", methods=["POST"])
def import_llm_config():
    """Import LLM provider configs from a JSON bundle.

    Request body::

        {
            "bundle": { ... },
            "overwrite": false
        }
    """
    data = request.get_json() or {}
    bundle = data.get("bundle")
    if not bundle or not isinstance(bundle, dict):
        return jsonify({"success": False, "error": "Invalid bundle"}), 400

    overwrite = data.get("overwrite", False)

    _ensure_router_enabled()
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "Router init failed"}), 500

    from router.config import LLMProviderConfig, RoutingStrategy

    strategy_str = bundle.get("routing_strategy", "failover")
    try:
        router.set_routing_strategy(RoutingStrategy(strategy_str))
    except ValueError:
        pass

    imported = 0
    skipped = 0
    errors = []
    for prov in bundle.get("providers", []):
        name = prov.get("name", "")
        try:
            existing = router.get_provider(name)
            if existing and not overwrite:
                skipped += 1
                continue
            config = LLMProviderConfig.from_dict(prov)
            router.register_provider(config)
            imported += 1
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    return jsonify({
        "success": True,
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
    })


# =============================================================================
# ACTIVATE  (save + check health in one step)
# =============================================================================

@api_bp.route("/llm/activate", methods=["POST"])
def activate_llm_config():
    """Save current config to disk and verify all providers are healthy."""
    router = _get_router()
    if router is None:
        return jsonify({"success": False, "error": "No router available"}), 400

    # Save first
    saved, save_error = _save_provider_bundle(router)
    if save_error:
        logger.warning("Failed to persist LLM config: %s", save_error)

    # Health check
    health = router.check_all_providers()
    healthy = sum(1 for v in health.values() if v)

    return jsonify({
        "success": True,
        "saved": saved,
        "health": health,
        "healthy_count": healthy,
        "total_count": len(health),
    })
