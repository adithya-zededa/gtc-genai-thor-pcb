"""Health check and readiness probe endpoints."""

from datetime import datetime
from flask import jsonify

from . import api_bp
from app.database import get_db_connection
from services.core.camera import check_camera_availability
from services.core.inference import check_inference_backend_availability
from core.config import get_config


@api_bp.route("/health")
def health_check():
    """Health check endpoint for container orchestration.
    
    Returns:
        JSON with health status and component availability.
    """
    config = get_config()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "inference_backend": "vllm",
        "components": {
            "database": False,
            "camera": False,
            "inference": False,
        }
    }
    
    # Check database
    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1")
        health_status["components"]["database"] = True
    except Exception:
        pass
    
    # Check camera
    health_status["components"]["camera"] = check_camera_availability()
    
    # Check inference backend
    health_status["components"]["inference"] = check_inference_backend_availability()
    
    # Determine overall status
    if not health_status["components"]["database"]:
        health_status["status"] = "unhealthy"
    elif not all(health_status["components"].values()):
        health_status["status"] = "degraded"
    
    status_code = 200 if health_status["status"] != "unhealthy" else 503
    return jsonify(health_status), status_code


@api_bp.route("/ready")
def readiness_check():
    """Readiness probe for Kubernetes.
    
    Returns:
        JSON indicating if the service is ready to accept traffic.
    """
    is_ready = True
    details = {}
    
    # Check if database is accessible
    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1")
        details["database"] = "ok"
    except Exception as e:
        is_ready = False
        details["database"] = str(e)
    
    if is_ready:
        return jsonify({"ready": True, "details": details})
    else:
        return jsonify({"ready": False, "details": details}), 503
