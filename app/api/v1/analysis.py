"""Analysis API endpoints for VLM inference."""

import base64
import time
from flask import jsonify, request

import cv2
import numpy as np

from . import api_bp
from services.core.monitoring import get_monitoring_service
from services.infrastructure.config import load_camera_config
from services.infrastructure.vlm import create_vlm_client_from_config
from services.core.camera import get_camera_publisher
from agents.vlm.task_types import TaskType
from agents.tools.base import ToolExecutor
from app.database import DetectionLogRepository
from core.logging import get_logger

logger = get_logger(__name__)

# Canonical mapping from string to TaskType — used by multiple endpoints.
TASK_TYPE_MAP = {
    "package_detection": TaskType.PACKAGE_DETECTION,
    "ppe_detection": TaskType.PPE_DETECTION,
    "person_counting": TaskType.PERSON_COUNTING,
    "scene_description": TaskType.SCENE_DESCRIPTION,
    "custom": TaskType.CUSTOM,
}


def _parse_task_type(task_type_str: str) -> TaskType | None:
    """Resolve a task-type string to the enum, or *None* if invalid."""
    return TASK_TYPE_MAP.get(task_type_str)


def _capture_current_frame():
    """Capture and decode the latest camera frame.

    Returns:
        Tuple of ``(frame, error_response)``.  On success *error_response*
        is ``None``; on failure *frame* is ``None`` and *error_response*
        is a ready-to-return ``(jsonify(...), status_code)`` tuple.
    """
    try:
        publisher = get_camera_publisher()
        latest_frame = publisher.get_latest_frame()

        if not latest_frame or not latest_frame.image_b64:
            return None, (jsonify({"success": False, "error": "No frames available from camera"}), 500)

        frame_bytes = base64.b64decode(latest_frame.image_b64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is None:
            return None, (jsonify({"success": False, "error": "Failed to decode camera frame"}), 500)

        return frame, None
    except Exception as err:
        logger.error("Camera access failed: %s", err)
        return None, (jsonify({"success": False, "error": f"Camera access failed: {err}"}), 500)


def _log_detection_to_db(event, frame_metadata):
    """Log a detection event to the database."""
    try:
        DetectionLogRepository.create(
            timestamp=event.timestamp,
            confidence=event.confidence,
            response=event.full_response,
            image_path=event.image_path or "",
            frame_number=frame_metadata.get("frame_number"),
            reason=frame_metadata.get("reason", "Uploaded image analysis"),
            vision_description=getattr(event, "vision_description", ""),
            decision_details=event.decision_trace,
            tool_trace=getattr(event, "tool_trace", []),
        )
    except Exception as exc:
        logger.error("Failed to log detection to database: %s", exc)


@api_bp.route("/analyze_prompt", methods=["POST"])
def analyze_prompt():
    """Analyze the current camera frame with a dynamic prompt."""
    try:
        data = request.get_json() or {}
        task_type_str = data.get("task_type", "package_detection")
        custom_prompt = data.get("custom_prompt", "")
        
        task_type = _parse_task_type(task_type_str)
        if task_type is None:
            return jsonify({
                "success": False,
                "error": f"Invalid task_type: {task_type_str}"
            }), 400
        
        if task_type == TaskType.CUSTOM and not custom_prompt:
            return jsonify({
                "success": False,
                "error": "Custom task requires a custom_prompt"
            }), 400
        
        # Capture current frame
        frame, err_response = _capture_current_frame()
        if err_response is not None:
            return err_response
        
        # Initialize VLM client
        try:
            config = load_camera_config()
            vlm_client = create_vlm_client_from_config(config)
        except Exception as e:
            logger.error("Failed to initialize VLM client: %s", e)
            return jsonify({
                "success": False,
                "error": f"Failed to initialize VLM client: {str(e)}"
            }), 500
        
        # Run analysis
        result = vlm_client.analyze(
            frame=frame,
            task_type=task_type,
            user_query=custom_prompt if task_type == TaskType.CUSTOM else None,
        )
        
        if result is None:
            return jsonify({
                "success": False,
                "error": "Analysis failed - no result returned"
            }), 500
        
        return jsonify({
            "success": True,
            "result": {
                "task_type": task_type_str,
                "detected": result.detected,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "details": result.details,
            }
        })
        
    except Exception as e:
        logger.exception("Error in analyze_prompt API")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@api_bp.route("/analyze_single", methods=["POST"])
def analyze_single():
    """Run single-frame inference on the latest available frame."""
    service = get_monitoring_service()
    
    if not service:
        return jsonify({
            "success": False,
            "error": "Camera agent not initialized. Start monitoring first."
        }), 400
    
    data = request.get_json() or {}
    
    task_type = None
    task_type_str = data.get("task_type")
    if task_type_str:
        task_type = _parse_task_type(task_type_str)
        if task_type is None:
            return jsonify({
                "success": False,
                "error": f"Invalid task_type: {task_type_str}"
            }), 400
    
    custom_prompt = data.get("custom_prompt")
    
    if task_type == TaskType.CUSTOM and not custom_prompt:
        return jsonify({
            "success": False,
            "error": "Custom task requires a custom_prompt"
        }), 400
    
    try:
        event = service.analyze_single_frame(
            task_type=task_type,
            custom_prompt=custom_prompt,
        )
        
        if event is None:
            error_msg = service.last_error or "Inference failed - no result returned"
            return jsonify({
                "success": False,
                "error": error_msg
            }), 500
        
        prompt_config = service.get_active_prompt_config()
        effective_task_type = task_type_str or prompt_config.get("task_type", "unknown")
        
        return jsonify({
            "success": True,
            "single_frame": True,
            "result": {
                "task_type": effective_task_type,
                "detected": event.detected,
                "confidence": event.confidence,
                "reasoning": event.vision_description,
                "primary_label": event.primary_label,
                "should_alert": event.should_alert,
                "timestamp": event.timestamp,
                "decision_trace": event.decision_trace,
            }
        })
        
    except Exception as e:
        logger.exception("Error in analyze_single API")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@api_bp.route("/analyze_agentic", methods=["POST"])
def analyze_agentic():
    """Analyze the current camera frame with agentic tool calling."""
    try:
        data = request.get_json() or {}
        task_type_str = data.get("task_type", "package_detection")
        custom_prompt = data.get("prompt") or data.get("custom_prompt") or ""
        recipients = data.get("recipients", [])
        
        task_type = _parse_task_type(task_type_str)
        if task_type is None:
            return jsonify({
                "success": False,
                "error": f"Invalid task_type: {task_type_str}"
            }), 400
        
        # Capture current frame
        frame, err_response = _capture_current_frame()
        if err_response is not None:
            return err_response
        
        # Initialize VLM client
        try:
            config = load_camera_config()
            vlm_client = create_vlm_client_from_config(config)
            
            if not recipients:
                email_cfg = config.get("notifications", {}).get("email", {})
                recipients = email_cfg.get("recipients", [])
            
        except Exception as e:
            logger.error("Failed to initialize VLM client: %s", e)
            return jsonify({
                "success": False,
                "error": f"Failed to initialize VLM client: {str(e)}"
            }), 500
        
        tool_executor = ToolExecutor()
        
        result = vlm_client.analyze_with_tools(
            frame=frame,
            tool_executor=tool_executor,
            task_type=task_type,
            user_query=custom_prompt if custom_prompt else None,
            recipients=recipients,
        )
        
        if result is None:
            return jsonify({
                "success": False,
                "error": "Agentic analysis failed - no result returned"
            }), 500
        
        return jsonify({
            "success": True,
            "result": {
                "task_type": task_type_str,
                "analysis": result.analysis.to_dict() if result.analysis else None,
                "tools_called": result.tool_calls,
                "tool_results": result.tool_results,
                "tools_used": result.tools_used,
                "any_tools_called": result.any_tools_called,
                "all_tools_succeeded": result.all_tools_succeeded,
            }
        })
        
    except Exception as e:
        logger.exception("Error in analyze_agentic API")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@api_bp.route("/analyze_uploaded_image", methods=["POST"])
def analyze_uploaded_image():
    """Analyze an uploaded image using the agent."""
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided. Use 'image' field in multipart/form-data."
        }), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({
            "success": False,
            "error": "No image file selected"
        }), 400
    
    try:
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                "success": False,
                "error": "Failed to decode image."
            }), 400
    except Exception as e:
        logger.error("Failed to process uploaded image: %s", e)
        return jsonify({
            "success": False,
            "error": f"Failed to process image: {str(e)}"
        }), 400
    
    task_type_str = request.form.get("task_type")
    custom_prompt = request.form.get("custom_prompt") or request.form.get("prompt")
    agentic_mode_str = request.form.get("agentic_mode")
    
    task_type = None
    if task_type_str:
        task_type = _parse_task_type(task_type_str)
        if task_type is None:
            return jsonify({
                "success": False,
                "error": f"Invalid task_type: {task_type_str}"
            }), 400
    
    use_agentic = None
    if agentic_mode_str is not None:
        use_agentic = agentic_mode_str.lower() in ('true', '1', 'yes', 'on')
    
    service = get_monitoring_service()
    was_monitoring = service.is_monitoring if service else False
    
    try:
        config = load_camera_config()
        
        if service:
            if task_type is None:
                active_config = service.get_active_prompt_config()
                task_type_str = active_config.get("task_type", "package_detection")
                task_type = _parse_task_type(task_type_str) or TaskType.PACKAGE_DETECTION
                if custom_prompt is None:
                    custom_prompt = active_config.get("custom_prompt", "")
                if use_agentic is None:
                    use_agentic = active_config.get("agentic_mode", False)
            
            agent = service.agent
            if agent is None:
                if not service.initialize():
                    return jsonify({
                        "success": False,
                        "error": f"Failed to initialize agent: {service.last_error}"
                    }), 500
                agent = service.agent
        else:
            vlm_client = create_vlm_client_from_config(config)
            
            if task_type is None:
                task_type = TaskType.PACKAGE_DETECTION
                task_type_str = "package_detection"
            if use_agentic is None:
                use_agentic = False
            
            from agents.core.camera_agent import StreamlinedAgent, CircuitBreaker
            circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=120.0)
            agent = StreamlinedAgent(
                config=config,
                vlm_client=vlm_client,
                circuit_breaker=circuit_breaker,
            )
        
        metadata = {
            'frame_number': 0,
            'timestamp': time.time(),
            'uploaded_image': True,
            'filename': image_file.filename,
            'image_data': image_bytes,
        }
        
        start_time = time.time()
        
        if use_agentic:
            email_cfg = config.get("notifications", {}).get("email", {})
            recipients = email_cfg.get("recipients", [])
            
            event = agent.analyze_agentic(
                frame=frame,
                task_type=task_type,
                custom_prompt=custom_prompt,
                recipients=recipients,
            )
        else:
            if custom_prompt and task_type != TaskType.CUSTOM:
                task_type = TaskType.CUSTOM
                task_type_str = "custom"
            
            event = agent.analyze_with_prompt(
                frame=frame,
                task_type=task_type,
                custom_prompt=custom_prompt or "",
                metadata=metadata,
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        if event is None:
            return jsonify({
                "success": False,
                "error": "Analysis returned no result."
            }), 500
        
        _log_detection_to_db(event, metadata)
        
        return jsonify({
            "success": True,
            "uploaded_image": True,
            "filename": image_file.filename,
            "latency_ms": round(latency_ms, 1),
            "monitoring_resumed": was_monitoring,
            "result": {
                "task_type": task_type_str or (task_type.value if task_type else "unknown"),
                "agentic_mode": use_agentic,
                "detected": event.detected,
                "confidence": event.confidence,
                "reasoning": event.vision_description,
                "primary_label": event.primary_label,
                "should_alert": event.should_alert,
                "timestamp": event.timestamp,
                "decision_trace": event.decision_trace,
                "tools_used": event.tools_used,
                "tool_trace": event.tool_trace,
            }
        })
        
    except Exception as e:
        logger.exception("Error analyzing uploaded image")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# NOTE: Tools API endpoints are in mcp.py
