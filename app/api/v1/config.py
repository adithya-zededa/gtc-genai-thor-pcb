"""Configuration management API endpoints."""

# pylint: disable=broad-exception-caught,redefined-outer-name,import-outside-toplevel

import json

from flask import jsonify, request

from app.database import ConfigHistoryRepository
from core.logging import get_logger
from core.utils import coerce_bool
from services.core.monitoring import get_monitoring_service
from services.infrastructure.config import (
    apply_recipients,
    extract_recipients,
    load_camera_config,
    reset_camera_config,
    sanitize_config_payload,
    save_camera_config,
    update_email_recipients,
)

from . import api_bp

logger = get_logger(__name__)


def _apply_config_to_agent(config_data: dict) -> None:
    """Push updated configuration to the running monitoring agent (if any)."""
    service = get_monitoring_service()
    if service and service.agent:
        try:
            service.agent.apply_config(config_data)
            service.refresh_configuration(config_data)
        except Exception as exc:
            logger.error("Failed to apply configuration to agent: %s", exc)


@api_bp.route("/config", methods=["GET", "POST"])
def config():
    """Configuration management API - get or update config."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid configuration payload",
                    }
                ),
                400,
            )

        try:
            persisted = save_camera_config(data)

            # Log configuration change
            try:
                ConfigHistoryRepository.create(
                    config_type="camera_config",
                    changes=json.dumps(persisted),
                )
            except Exception as db_exc:
                logger.error("Failed to record configuration history: %s", db_exc)

            # Refresh running agent configuration if active
            _apply_config_to_agent(persisted)

            notifications_cfg = (persisted or {}).get("notifications", {})
            email_cfg = (
                notifications_cfg.get("email", {})
                if isinstance(notifications_cfg, dict)
                else {}
            )
            if coerce_bool((email_cfg or {}).get("auto_sync_users"), True):
                update_email_recipients()

            return jsonify(
                {
                    "success": True,
                    "message": "Configuration updated",
                    "config": persisted,
                }
            )
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    # GET method
    try:
        config = load_camera_config()
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Failed to load configuration: {exc}",
                }
            ),
            500,
        )

    sanitized = sanitize_config_payload(config)
    return jsonify(sanitized)


@api_bp.route("/config/defaults", methods=["GET"])
def config_defaults():
    """Return the default configuration without persisting it."""
    from agents.core.detection_agent import StreamlinedAgent

    defaults = StreamlinedAgent.default_config()
    sanitized = sanitize_config_payload(defaults)
    return jsonify({"success": True, "config": sanitized})


@api_bp.route("/config/reset", methods=["POST"])
def config_reset():
    """Reset the configuration to defaults and persist the change."""
    defaults = reset_camera_config()
    _apply_config_to_agent(defaults)

    return jsonify(
        {
            "success": True,
            "message": "Configuration reset to defaults",
            "config": defaults,
        }
    )


@api_bp.route("/config/inspection-prompt", methods=["GET", "PUT"])
def inspection_prompt_config():
    """Get or update the PCB defect inspection prompt.

    GET  — returns the current inspection prompt (from config or default).
    PUT  — persists a new inspection prompt to config.yaml.
    """
    if request.method == "GET":
        try:
            config = load_camera_config()
        except Exception as exc:
            return (
                jsonify({"success": False, "error": f"Failed to load config: {exc}"}),
                500,
            )
        detection = config.get("detection") or {}
        custom = (detection.get("inspection_prompt") or "").strip()

        from agents.tools.pcb import DEFAULT_PCB_DEFECT_INSPECTION_PROMPT

        return jsonify({
            "success": True,
            "prompt": custom if custom else DEFAULT_PCB_DEFECT_INSPECTION_PROMPT,
            "is_default": not bool(custom),
        })

    # PUT
    payload = request.get_json(silent=True) or {}
    new_prompt = payload.get("prompt", "")
    if not isinstance(new_prompt, str):
        return jsonify({"success": False, "error": "prompt must be a string"}), 400

    try:
        config = load_camera_config()
    except Exception as exc:
        return (
            jsonify({"success": False, "error": f"Failed to load config: {exc}"}),
            500,
        )

    detection = config.setdefault("detection", {})
    detection["inspection_prompt"] = new_prompt.strip()
    persisted = save_camera_config(config)

    saved_prompt = (persisted.get("detection") or {}).get("inspection_prompt", "")
    return jsonify({
        "success": True,
        "message": "Inspection prompt updated",
        "prompt": saved_prompt,
        "is_default": not bool(saved_prompt.strip()),
    })


@api_bp.route("/notifications/recipients", methods=["GET", "PUT"])
def notification_recipients():
    """Manage notification email recipients."""
    if request.method == "GET":
        try:
            config = load_camera_config()
        except Exception as exc:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Failed to load configuration: {exc}",
                    }
                ),
                500,
            )

        recipients = extract_recipients(config)
        return jsonify({"success": True, "recipients": recipients})

    # PUT method
    payload = request.get_json(silent=True) or {}
    recipients = payload.get("recipients", [])
    if not isinstance(recipients, list):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Recipients must be provided as a list",
                }
            ),
            400,
        )

    sanitized = []
    for email in recipients:
        if not isinstance(email, str):
            continue
        candidate = email.strip()
        if not candidate or "@" not in candidate:
            continue
        sanitized.append(candidate)

    try:
        config = load_camera_config()
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Failed to load configuration: {exc}",
                }
            ),
            500,
        )

    apply_recipients(config, sanitized)
    persisted = save_camera_config(config)
    _apply_config_to_agent(persisted)

    return jsonify({"success": True, "recipients": extract_recipients(persisted)})
