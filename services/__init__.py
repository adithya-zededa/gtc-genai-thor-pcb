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
from .retail_service import (
    lookup_items as retail_lookup_items,
    lookup_item_by_sku as retail_lookup_item_by_sku,
    calculate_bill as retail_calculate_bill,
    generate_invoice_html as retail_generate_invoice_html,
    save_invoice as retail_save_invoice,
    save_and_send_invoice as retail_save_and_send_invoice,
)
from .pcb_service import (
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
    # Retail service
    "retail_lookup_items",
    "retail_lookup_item_by_sku",
    "retail_calculate_bill",
    "retail_generate_invoice_html",
    "retail_save_invoice",
    "retail_save_and_send_invoice",
    # PCB service
    "pcb_record_defect",
    "pcb_should_alert",
    "pcb_generate_defect_report",
    "pcb_classify_board",
    "pcb_extract_defects",
]
