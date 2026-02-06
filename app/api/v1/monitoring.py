"""Monitoring status & utility API endpoints.

Monitoring is now controlled entirely through the chat agent (MCP tool
calls like ``start_monitoring_session`` / ``end_session``).  The legacy
``/api/start_monitoring`` and ``/api/stop_monitoring`` direct-toggle
endpoints have been removed.
"""

from flask import jsonify, request

from . import api_bp
from services.core.monitoring import get_monitoring_service
from services.core.camera import check_camera_availability
from services.core.inference import check_inference_backend_availability
from core.config import get_config
from core.logging import get_logger

logger = get_logger(__name__)


@api_bp.route("/status")
def get_status():
    """Get comprehensive system status including circuit breaker state."""
    config = get_config()
    service = get_monitoring_service()

    status = {
        "monitoring_active": service.is_monitoring if service else False,
        "camera_available": check_camera_availability(),
        "inference_backend": config.inference.backend,
        "inference_available": check_inference_backend_availability(),
        "stats": service._serialize_stats() if service else {},
    }

    # Add circuit breaker status if agent is running
    if service and service.agent:
        agent = service.agent
        if hasattr(agent, 'circuit_breaker'):
            status["circuit_breaker"] = agent.circuit_breaker.get_stats()

        # Add memory summary
        if hasattr(agent, '_agent_memory'):
            memory_snapshot = agent.get_memory_snapshot(limit=5)
            status["recent_events_count"] = memory_snapshot.get("counts", {}).get("total", 0)

    return jsonify(status)


@api_bp.route("/circuit_breaker/reset", methods=["POST"])
def reset_circuit_breaker():
    """Reset the circuit breaker to closed state."""
    service = get_monitoring_service()

    if not service or not service.agent:
        return jsonify({
            "success": False,
            "error": "No active monitoring agent"
        }), 400

    agent = service.agent
    if not hasattr(agent, 'circuit_breaker'):
        return jsonify({
            "success": False,
            "error": "Circuit breaker not available"
        }), 400

    agent.circuit_breaker.reset()
    return jsonify({
        "success": True,
        "message": "Circuit breaker reset to CLOSED state",
        "stats": agent.circuit_breaker.get_stats()
    })


@api_bp.route("/agent/memory", methods=["GET", "DELETE"])
def agent_memory():
    """Expose recent agent memory and summarised activity."""
    service = get_monitoring_service()
    agent = getattr(service, "agent", None) if service else None

    if request.method == "DELETE":
        if agent and hasattr(agent, 'clear_memory'):
            agent.clear_memory()
            return jsonify({"success": True, "message": "Agent memory cleared"})
        return jsonify({"success": True, "message": "No memory to clear"})

    # GET method
    limit = request.args.get("limit", type=int)

    if not agent:
        empty_counts = {
            "total": 0,
            "detections": 0,
            "unlabeled": 0,
            "labeled": 0,
            "alerts": 0,
            "reused": 0,
            "no_detections": 0,
        }
        return jsonify({
            "success": True,
            "summary": "No agent memory available yet.",
            "memory": {
                "events": [],
                "counts": empty_counts,
                "last_event": None,
            },
        })

    snapshot = agent.get_memory_snapshot(limit=limit)
    summary = agent.summarise_recent_events(limit=limit)
    return jsonify({"success": True, "summary": summary, "memory": snapshot})
