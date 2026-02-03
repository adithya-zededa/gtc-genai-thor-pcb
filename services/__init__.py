"""Service layer modules for business logic."""

from .camera_service import (
    CameraFrame,
    CameraFeedPublisher,
    get_camera_publisher,
    check_camera_availability,
)
from .inference_service import (
    check_inference_backend_availability,
    check_vllm_availability,
    check_ollama_availability,
)
from .config_service import (
    load_camera_config,
    save_camera_config,
    reset_camera_config,
    extract_recipients,
    apply_recipients,
    sanitize_config_payload,
    update_email_recipients,
)
from .monitoring_service import (
    get_monitoring_service,
    StreamlinedMonitoringService,
)
from .vlm_service import create_vlm_client_from_config

__all__ = [
    # Camera service
    "CameraFrame",
    "CameraFeedPublisher",
    "get_camera_publisher",
    "check_camera_availability",
    # Inference service
    "check_inference_backend_availability",
    "check_vllm_availability",
    "check_ollama_availability",
    # Config service
    "load_camera_config",
    "save_camera_config",
    "reset_camera_config",
    "extract_recipients",
    "apply_recipients",
    "sanitize_config_payload",
    "update_email_recipients",
    # Monitoring service
    "get_monitoring_service",
    "StreamlinedMonitoringService",
    # VLM service
    "create_vlm_client_from_config",
]
