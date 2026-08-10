"""Service layer modules for business logic.

Restructured layout:
    services/
    ├── core/           - Camera, monitoring, inference
    ├── infrastructure/ - Configuration, VLM client factory
    └── domains/        - PCB domain services

All symbols are re-exported here for backward compatibility.
"""

# Core services
from .core.camera import (
    CameraFrame,
    CameraFeedPublisher,
    get_camera_publisher,
    check_camera_availability,
)
from .core.inference import (
    check_agent_llm_availability,
    check_inference_backend_availability,
    check_inference_roles,
    check_vllm_availability,
)
from .core.monitoring import (
    get_monitoring_service,
    StreamlinedMonitoringService,
)

# Infrastructure services
from .infrastructure.config import (
    load_camera_config,
    save_camera_config,
    reset_camera_config,
    extract_recipients,
    apply_recipients,
    sanitize_config_payload,
    update_email_recipients,
)
from .infrastructure.vlm import create_vlm_client_from_config

# Domain services
from .domains.pcb import (
    record_defect as pcb_record_defect,
    should_alert as pcb_should_alert,
    generate_defect_report as pcb_generate_defect_report,
    classify_board_from_analysis as pcb_classify_board,
    extract_defects_from_analysis as pcb_extract_defects,
)

__all__ = [
    # Camera service
    "CameraFrame",
    "CameraFeedPublisher",
    "get_camera_publisher",
    "check_camera_availability",
    # Inference service
    "check_agent_llm_availability",
    "check_inference_backend_availability",
    "check_inference_roles",
    "check_vllm_availability",
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
    # PCB service
    "pcb_record_defect",
    "pcb_should_alert",
    "pcb_generate_defect_report",
    "pcb_classify_board",
    "pcb_extract_defects",
]
