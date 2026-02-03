"""Monitoring control API endpoints."""

from flask import jsonify, request

from . import api_bp
from services.monitoring_service import get_monitoring_service
from services.camera_service import check_camera_availability
from services.inference_service import check_inference_backend_availability
from agents.vlm.task_types import TaskType
from core.config import get_config
from core.logging import get_logger

logger = get_logger(__name__)


@api_bp.route("/start_monitoring", methods=["POST"])
def start_monitoring():
    """Start camera monitoring."""
    service = get_monitoring_service()
    
    if not service.initialize():
        error_message = service.last_error or "Failed to initialize camera"
        return jsonify({"success": False, "error": error_message})

    if service.start_monitoring():
        return jsonify({"success": True, "message": "Monitoring started"})
    else:
        error_message = service.last_error or "Monitoring already active"
        return jsonify({"success": False, "error": error_message})


@api_bp.route("/stop_monitoring", methods=["POST"])
def stop_monitoring():
    """Stop camera monitoring."""
    service = get_monitoring_service()
    
    if service:
        service.stop_monitoring()
        return jsonify({"success": True, "message": "Monitoring stopped"})
    else:
        return jsonify({"success": False, "error": "No active monitoring"})


@api_bp.route("/status")
def get_status():
    """Get comprehensive system status including circuit breaker state."""
    config = get_config()
    service = get_monitoring_service()
    
    # Get agentic mode from prompt config
    prompt_config = service.get_active_prompt_config() if service else {}
    
    status = {
        "monitoring_active": service.is_monitoring if service else False,
        "camera_available": check_camera_availability(),
        "inference_backend": config.inference.backend,
        "inference_available": check_inference_backend_availability(),
        "agentic_mode": prompt_config.get("agentic_mode", False),
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


@api_bp.route("/agent/prompt", methods=["GET", "POST"])
def agent_prompt():
    """Get or set the active monitoring prompt configuration."""
    service = get_monitoring_service()
    
    if not service:
        return jsonify({"success": False, "error": "Camera agent not available"}), 500
    
    if request.method == "GET":
        config = service.get_active_prompt_config()
        return jsonify({"success": True, **config})
    
    # POST method - set active prompt
    data = request.get_json() or {}
    task_type_str = data.get("task_type", "package_detection")
    custom_prompt = data.get("custom_prompt", "")
    alerts_enabled = data.get("alerts_enabled", False)
    agentic_mode = data.get("agentic_mode", False)
    
    # Map string to TaskType enum
    task_type_map = {
        "package_detection": TaskType.PACKAGE_DETECTION,
        "ppe_detection": TaskType.PPE_DETECTION,
        "person_counting": TaskType.PERSON_COUNTING,
        "scene_description": TaskType.SCENE_DESCRIPTION,
        "custom": TaskType.CUSTOM,
    }
    
    task_type = task_type_map.get(task_type_str)
    if task_type is None:
        return jsonify({
            "success": False,
            "error": f"Invalid task_type: {task_type_str}"
        }), 400
    
    # Validate custom prompt for custom task
    if task_type == TaskType.CUSTOM and not custom_prompt:
        return jsonify({
            "success": False,
            "error": "Custom task requires a custom_prompt"
        }), 400
    
    # Get current config before update
    old_config = service.get_active_prompt_config()
    old_task_type = old_config.get("task_type")
    old_prompt = old_config.get("custom_prompt")
    
    prompt_changing = (old_task_type != task_type_str or old_prompt != custom_prompt)
    
    # Set the active prompt
    service.set_active_prompt(
        task_type=task_type,
        custom_prompt=custom_prompt,
        alerts_enabled=alerts_enabled,
        agentic_mode=agentic_mode,
    )
    
    new_config = service.get_active_prompt_config()
    
    return jsonify({
        "success": True,
        "message": f"Active prompt set to {task_type_str}" + (" (agentic mode)" if agentic_mode else ""),
        "task_type": task_type_str,
        "custom_prompt": custom_prompt,
        "alerts_enabled": alerts_enabled,
        "agentic_mode": agentic_mode,
        "prompt_version": new_config.get("prompt_version", 0),
        "queue_cleared": prompt_changing,
    })
