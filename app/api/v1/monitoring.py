"""Monitoring status & utility API endpoints.

Monitoring is now controlled entirely through the chat agent (MCP tool
calls like ``start_monitoring_session`` / ``end_session``).  The legacy
``/api/start_monitoring`` and ``/api/stop_monitoring`` direct-toggle
endpoints have been removed.
"""

from typing import Any, Dict, Optional, Tuple

from flask import jsonify, request

from core.config import get_config
from core.logging import get_logger
from services.core.camera import check_camera_availability
from services.core.inference import check_inference_roles
from services.core.monitoring import get_monitoring_service

from . import api_bp

logger = get_logger(__name__)

PROACTIVE_CONFIG_KEY_MAP = {
    "frame_interval": "frame_interval_seconds",
    "frame_interval_seconds": "frame_interval_seconds",
    "stability_frames": "stability_frame_count",
    "stability_frame_count": "stability_frame_count",
    "observation_temperature": "observation_temperature",
    "fast_observation_mode": "fast_observation_mode",
    "stationary_motion_threshold": "stationary_motion_threshold",
    "zone_crop_top_ratio": "zone_crop_top_ratio",
    "zone_crop_bottom_ratio": "zone_crop_bottom_ratio",
    "zone_presence_threshold": "zone_presence_threshold",
    "selector_min_stable_frames": "selector_min_stable_frames",
    "selector_min_quality_score": "selector_min_quality_score",
    "selector_attempt_cooldown_seconds": "selector_attempt_cooldown_seconds",
    "selector_quality_improvement_delta": "selector_quality_improvement_delta",
    "selector_max_attempts_per_board": "selector_max_attempts_per_board",
    "selector_near_deadline_seconds": "selector_near_deadline_seconds",
    "decision_temperature": "decision_temperature",
    "quick_check_temperature": "quick_check_temperature",
    "inspection_ttl_seconds": "inspection_ttl_seconds",
    "max_idle_seconds": "max_idle_seconds",
}


def _monitoring_service_or_error() -> Tuple[Optional[Any], Optional[Tuple[Any, int]]]:
    """Return monitoring service or a standard API error response tuple."""
    service = get_monitoring_service()
    if service:
        return service, None
    return None, (jsonify({"success": False, "error": "Monitoring service unavailable"}), 500)


def _extract_proactive_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map API payload keys onto proactive agent config keys."""
    return {
        target_key: payload[source_key]
        for source_key, target_key in PROACTIVE_CONFIG_KEY_MAP.items()
        if source_key in payload
    }


@api_bp.route("/status")
def get_status():
    """Get comprehensive system status including circuit breaker state.

    This is a dashboard endpoint, not a probe — it reaches out to the
    inference pods. Kubernetes probes use ``/api/health/live`` and
    ``/api/ready`` instead.
    """
    service = get_monitoring_service()
    stats: Dict[str, Any] = {}
    if service and hasattr(service, "_serialize_stats"):
        stats = service._serialize_stats()  # pylint: disable=protected-access

    config = get_config()
    inference_roles = check_inference_roles()
    status = {
        "monitoring_active": (
            (service.get_active_monitoring_mode() != "idle")
            if service else False
        ),
        "camera_available": check_camera_availability(),
        "video_simulated": bool(config.camera.video_source),
        "inference_backend": "vllm",
        # True only when both the vision and agent models are reachable —
        # a live vision pod alone still leaves chat dead.
        "inference_available": all(inference_roles.values()),
        "inference_roles": inference_roles,
        "stats": stats,
    }

    # Add circuit breaker status if agent is running
    if service and service.agent:
        agent = service.agent
        if hasattr(agent, "circuit_breaker"):
            status["circuit_breaker"] = agent.circuit_breaker.get_stats()

        # Add memory summary
        if hasattr(agent, "_agent_memory"):
            memory_snapshot = agent.get_memory_snapshot(limit=5)
            status["recent_events_count"] = memory_snapshot.get("counts", {}).get(
                "total", 0
            )

    return jsonify(status)


@api_bp.route("/circuit_breaker/reset", methods=["POST"])
def reset_circuit_breaker():
    """Reset the circuit breaker to closed state."""
    service = get_monitoring_service()

    if not service or not service.agent:
        return jsonify({"success": False, "error": "No active monitoring agent"}), 400

    agent = service.agent
    if not hasattr(agent, "circuit_breaker"):
        return (
            jsonify({"success": False, "error": "Circuit breaker not available"}),
            400,
        )

    agent.circuit_breaker.reset()
    return jsonify(
        {
            "success": True,
            "message": "Circuit breaker reset to CLOSED state",
            "stats": agent.circuit_breaker.get_stats(),
        }
    )


@api_bp.route("/agent/memory", methods=["GET", "DELETE"])
def agent_memory():
    """Expose recent agent memory and summarised activity."""
    service = get_monitoring_service()
    agent = getattr(service, "agent", None) if service else None

    if request.method == "DELETE":
        if agent and hasattr(agent, "clear_memory"):
            agent.clear_memory()
            return jsonify({"success": True, "message": "Agent memory cleared"})
        return jsonify({"success": True, "message": "No memory to clear"})

    # GET method
    limit = request.args.get("limit", type=int)

    if not agent:
        empty_counts = {
            "total": 0,
            "detections": 0,
            "stable": 0,
            "unstable": 0,
            "alerts": 0,
            "reused": 0,
            "no_detections": 0,
        }
        return jsonify(
            {
                "success": True,
                "summary": "No agent memory available yet.",
                "memory": {
                    "events": [],
                    "counts": empty_counts,
                    "last_event": None,
                },
            }
        )

    snapshot = agent.get_memory_snapshot(limit=limit)
    summary = agent.summarise_recent_events(limit=limit)
    return jsonify({"success": True, "summary": summary, "memory": snapshot})


@api_bp.route("/monitoring/proactive/status", methods=["GET"])
def proactive_status():
    """Return the latest snapshot from the proactive monitoring agent."""
    service, error = _monitoring_service_or_error()
    if error:
        return error
    return jsonify({"success": True, "status": service.get_proactive_snapshot()})


@api_bp.route("/monitoring/proactive/start", methods=["POST"])
def proactive_start():
    """Start or update the proactive monitoring loop with a new instruction."""
    payload = request.get_json(silent=True) or {}
    instruction = payload.get("instruction", "").strip()
    config = _extract_proactive_config(payload)

    if not instruction:
        return jsonify({"success": False, "error": "Instruction is required"}), 400

    service, error = _monitoring_service_or_error()
    if error:
        return error

    if not service.start_monitoring(
        mode="proactive",
        instruction=instruction,
        config=config,
    ):
        return (
            jsonify(
                {
                    "success": False,
                    "error": service.last_error or "Unable to start proactive mode",
                }
            ),
            400,
        )

    return jsonify(
        {
            "success": True,
            "status": service.get_proactive_snapshot(),
        }
    )


@api_bp.route("/monitoring/proactive/stop", methods=["POST"])
def proactive_stop():
    """Stop the proactive monitoring loop and return its final snapshot."""
    service, error = _monitoring_service_or_error()
    if error:
        return error
    service.stop_proactive_monitoring()
    return jsonify({"success": True, "status": service.get_proactive_snapshot()})
