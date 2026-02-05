"""Core routes for the web application - main endpoints."""

import logging
import os
import time
import traceback
from typing import Any, Dict

import requests
from flask import Blueprint, jsonify, render_template, request
from werkzeug.utils import secure_filename

from processing import (
    detect_model_type,
    process_image_classification,
    process_keypoint_detection,
    process_object_detection,
    process_ocr,
    process_panoptic_segmentation,
    process_pose_estimation,
    process_segmentation,
)
from utils.files import allowed_file
from utils.logging import (
    clear_all_logs,
    endpoint_logs,
    endpoint_logs_lock,
    log_endpoint_call,
    log_processing_step,
    processing_logs,
    processing_logs_lock,
)
from utils.tensor import format_tensor_shape
from utils.errors import (
    BadRequestError,
    ServiceUnavailableError,
    create_success_response,
    handle_exceptions,
)

logger = logging.getLogger(__name__)

# Create Blueprint
core_bp = Blueprint('core', __name__)

# These will be set by the main app when registering the blueprint
_app_config: Dict[str, Any] = {}
_client = None


def init_core_routes(app_config: Dict[str, Any], client) -> None:
    """Initialize core routes with app configuration and client."""
    global _app_config, _client
    _app_config = app_config
    _client = client


def execute_prediction(filepath: str, file_bytes: bytes, model_name: str, task_type: str = 'auto') -> dict:
    """
    Execute prediction on an image file - reusable core logic.
    Returns complete results including full tensor information.
    
    Args:
        filepath: Path to image file (for visualization functions)
        file_bytes: Image file bytes (for preprocessing)
        model_name: Name of the model to use
        task_type: Task type ('auto', 'detection', 'classification', etc.)
        
    Returns:
        Dict with complete results including tensor_info, model_spec, and task-specific results
    """
    start_request_time = time.time()
    filename = os.path.basename(filepath)
    
    # Check if model is ready
    model_check_start = time.time()
    model_ready = _client.check_model_ready(model_name)
    model_check_time = time.time() - model_check_start
    
    if not model_ready:
        return {
            'success': False,
            'error': f'Model {model_name} is not ready'
        }
    
    # Get model metadata to check for DETR or other multi-input models
    metadata = _client.get_model_metadata(model_name)
    
    # Check if this is a DETR model (requires special handling)
    from processing.detr import is_detr_model, run_detr_inference
    if is_detr_model(model_name, metadata):
        logger.info(f"Detected DETR model: {model_name}, using specialized inference")
        
        # Get server URL from client
        server_url = _client.server_url
        
        # Run DETR inference with specialized processing
        result = run_detr_inference(
            server_url=server_url,
            model_name=model_name,
            image_bytes=file_bytes,
            threshold=0.5,  # Lower threshold to get more detections
        )
        
        if not result.get("success"):
            return result
        
        # Format DETR results to match expected output format
        detections = result.get("detections", [])
        timing = result.get("timing", {})
        
        return {
            'success': True,
            'model_name': model_name,
            'model_type': 'detection',
            'detected_type': 'detection',
            'auto_detected': True,
            'inference_time': timing.get('total_ms', 0) / 1000.0,
            'detections': [
                {
                    'class': det['label'],
                    'class_id': det.get('label_id', 0),
                    'confidence': det['score'],
                    'bbox': [
                        det['box']['xmin'],
                        det['box']['ymin'],
                        det['box']['xmax'],
                        det['box']['ymax']
                    ]
                }
                for det in detections
            ],
            'total_time': time.time() - start_request_time,
            'timing': timing,
            'original_size': result.get('original_size'),
        }
    
    # For other multi-input models that aren't DETR, return error
    if metadata:
        inputs = metadata.get('inputs', [])
        if len(inputs) > 1:
            input_names = [inp.get('name', 'unknown') for inp in inputs]
            return {
                'success': False,
                'error': f"Model '{model_name}' requires {len(inputs)} inputs ({', '.join(input_names)}). This multi-input model architecture is not yet supported."
            }
    
    # Get model input/output specs (auto-detected)
    input_spec = _client.get_model_input_spec(model_name)
    output_spec = _client.get_model_output_spec(model_name)
    
    # Preprocess image from bytes
    preprocess_start = time.time()
    image_array = _client.preprocess_image_bytes(file_bytes, model_name=model_name)
    preprocess_time = time.time() - preprocess_start
    
    if image_array is None:
        return {
            'success': False,
            'error': 'Failed to preprocess image'
        }
    
    # Send inference request
    inference_start = time.time()
    try:
        response = _client.send_inference_request(image_array, model_name, measure_latency=True)
    except Exception as e:
        return {
            'success': False,
            'error': f'Inference request failed: {str(e)}'
        }
    inference_time = time.time() - inference_start
    
    if response is None:
        return {
            'success': False,
            'error': 'Inference request failed - no response from server'
        }
    
    # Process prediction
    prediction_start = time.time()
    prediction = _client.process_prediction(response, model_name)
    prediction_time = time.time() - prediction_start
    
    if prediction is None:
        return {
            'success': False,
            'error': 'Failed to process prediction'
        }
    
    # Auto-detect model type if set to 'auto', otherwise use user selection
    actual_task_type = task_type
    if task_type == 'auto':
        # Get all output specs for better detection
        all_output_specs = None
        if response and 'outputs' in response:
            all_output_specs = [{'name': o.get('name', ''), 'shape': o.get('shape', [])} 
                               for o in response['outputs']]
        num_outputs = len(response.get('outputs', [])) if response else 1
        
        actual_task_type = detect_model_type(model_name, output_spec, num_outputs, all_output_specs)
    
    # Process based on task type
    if actual_task_type == 'detection':
        result = process_object_detection(
            prediction, response, filepath, filename, 
            model_name, inference_time, start_request_time,
            input_spec, output_spec, image_array
        )
    elif actual_task_type == 'pose':
        result = process_pose_estimation(
            prediction, response, filepath, filename,
            model_name, inference_time, start_request_time,
            input_spec, output_spec, image_array
        )
    elif actual_task_type == 'keypoint':
        result = process_keypoint_detection(
            prediction, response, filepath, filename,
            model_name, inference_time, start_request_time,
            input_spec, output_spec, image_array
        )
    elif actual_task_type == 'segmentation':
        result = process_segmentation(
            prediction, response, filepath, filename,
            model_name, inference_time, start_request_time,
            input_spec, output_spec, image_array
        )
    elif actual_task_type == 'panoptic':
        result = process_panoptic_segmentation(
            prediction, response, filepath, filename,
            model_name, inference_time, start_request_time,
            input_spec, output_spec, image_array
        )
    elif actual_task_type == 'ocr':
        result = process_ocr(
            prediction, response, filepath, filename,
            model_name, inference_time, start_request_time,
            input_spec, output_spec, image_array
        )
    else:
        # Image Classification processing (default)
        result = process_image_classification(
            prediction, response, filename, model_name, 
            inference_time, start_request_time,
            input_spec, output_spec, image_array,
            model_check_time, preprocess_time, prediction_time
        )
    
    # Add detected type info
    if task_type == 'auto':
        result['auto_detected'] = True
        result['detected_type'] = actual_task_type
    
    return result


@core_bp.route('/')
def index():
    """Main page - Chat UI"""
    log_processing_step("Page Load", "Loading chat interface", "info")
    
    return render_template('chat.html', config=_app_config)


@core_bp.route('/legacy')
def legacy_index():
    """Legacy page with upload form (for backward compatibility)"""
    log_processing_step("Page Load", "Loading legacy interface", "info")
    
    models_start = time.time()
    available_models = _client.get_available_models()
    models_time = time.time() - models_start
    
    if not available_models:
        available_models = []
        log_processing_step("Model Discovery", "No models available", "warning")
    else:
        log_endpoint_call("/v1/config", "GET", 200, models_time)
        log_processing_step("Model Discovery", f"Discovered {len(available_models)} models", "success")
    
    return render_template('index.html', 
                          models=available_models,
                          config=_app_config)


@core_bp.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with proper resource management."""
    start_request_time = time.time()
    filepath = None  # Track filepath for cleanup in finally block
    
    try:
        log_processing_step("Request Received", "Starting prediction request", "info")
        
        if 'image' not in request.files:
            log_processing_step("Validation Failed", "No image file provided", "error")
            return jsonify({
                'success': False,
                'error': 'No image file provided',
                'error_code': 'MISSING_IMAGE'
            }), 400
        
        file = request.files['image']
        model_name = request.form.get('model')
        task_type = request.form.get('task_type', 'classification')
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'error_code': 'EMPTY_FILENAME'
            }), 400
        
        if not model_name:
            return jsonify({
                'success': False,
                'error': 'No model selected',
                'error_code': 'MISSING_MODEL'
            }), 400
        
        log_processing_step("File Validation", f"Validating: {file.filename} (Task: {task_type})", "info")
        
        if not (file and allowed_file(file.filename, _app_config.get('allowed_extensions'))):
            return jsonify({
                'success': False,
                'error': 'Invalid file format',
                'error_code': 'INVALID_FILE_FORMAT'
            }), 400
        
        filename = secure_filename(file.filename)
        # Handle case where secure_filename returns empty string
        if not filename:
            filename = f"upload_{int(time.time())}.jpg"
        
        # Read image bytes directly from request
        file_bytes = file.read()
        
        # Save to disk for visualization functions that need a file path
        upload_folder = _app_config.get('upload_folder', '/tmp/uploads/')
        filepath = os.path.join(upload_folder, filename)
        with open(filepath, 'wb') as f:
            f.write(file_bytes)
        
        log_processing_step("File Upload", "File saved", "success")
        
        # Execute prediction using reusable function
        result = execute_prediction(filepath, file_bytes, model_name, task_type)
        
        # Check if prediction was successful
        if not result.get('success'):
            return jsonify(result), 500
        
        total_time = time.time() - start_request_time
        log_processing_step("Completion", f"Completed in {total_time:.3f}s", "success")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500
    
    finally:
        # Guaranteed cleanup of temporary file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.debug(f"Cleaned up temporary file: {filepath}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file {filepath}: {cleanup_error}")


@core_bp.route('/health')
@handle_exceptions("Health check failed")
def health():
    """Health check endpoint - works with both Triton and OpenVINO servers"""
    server_healthy, health_message = _client.check_server_health()
    server_type = _client.detect_server_type()
    server_info = _client.get_server_info()
    
    available_models = _client.get_available_models()
    
    if available_models:
        return jsonify(create_success_response({
            'status': 'healthy',
            'server_type': server_type,
            'server_info': server_info,
            'server_healthy': server_healthy,
            'available_models': available_models
        }))
    elif server_healthy:
        return jsonify(create_success_response({
            'status': 'degraded',
            'server_type': server_type,
            'server_info': server_info,
            'server_healthy': True,
            'available_models': [],
            'message': 'Server is ready but no models are available'
        })), 200
    else:
        raise ServiceUnavailableError(
            "No models available",
            details={
                'server_type': server_type,
                'server_healthy': server_healthy,
                'health_message': health_message
            }
        )


@core_bp.route('/server-info')
@handle_exceptions("Failed to get server info")
def get_server_info():
    """Get inference server information"""
    server_type = _client.detect_server_type()
    server_info = _client.get_server_info()
    server_healthy, health_message = _client.check_server_health()
    
    return jsonify(create_success_response({
        'server_type': server_type,
        'server_info': server_info,
        'server_healthy': server_healthy,
        'health_message': health_message
    }))


@core_bp.route('/models')
@handle_exceptions("Failed to get available models")
def get_models():
    """Get available models"""
    models = _client.get_available_models()
    server_type = _client.detect_server_type()
    return jsonify(create_success_response({
        'models': models,
        'server_type': server_type
    }))


@core_bp.route('/debug/config')
@handle_exceptions("Failed to get debug config")
def debug_config():
    """Debug endpoint to check raw v1/config response"""
    server_url = _client.server_url
    server_type = _client.detect_server_type()
    server_info = _client.get_server_info()
    known_models = _client._known_models
    
    # Try v1/config
    v1_config = None
    v1_config_error = None
    try:
        response = requests.get(f"{server_url}/v1/config", timeout=10)
        v1_config = {
            'status_code': response.status_code,
            'data': response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        v1_config_error = str(e)
    
    # Try v2/repository/index
    v2_index = None
    v2_index_error = None
    try:
        response = requests.post(f"{server_url}/v2/repository/index", timeout=10)
        v2_index = {
            'status_code': response.status_code,
            'data': response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        v2_index_error = str(e)
    
    return jsonify(create_success_response({
        'server_url': server_url,
        'server_type': server_type,
        'server_info': server_info,
        'known_models_from_env': known_models,
        'v1_config': v1_config,
        'v1_config_error': v1_config_error,
        'v2_repository_index': v2_index,
        'v2_repository_index_error': v2_index_error
    }))


@core_bp.route('/models/<model_name>/metadata')
@handle_exceptions("Failed to get model metadata")
def get_model_metadata(model_name):
    """Get detailed metadata for a specific model"""
    metadata = _client.get_model_metadata(model_name)
    input_spec = _client.get_model_input_spec(model_name)
    output_spec = _client.get_model_output_spec(model_name)
    
    detected_type = detect_model_type(model_name, output_spec)
    
    return jsonify(create_success_response({
        'model_name': model_name,
        'detected_type': detected_type,
        'metadata': metadata,
        'input_spec': input_spec,
        'output_spec': output_spec
    }))


@core_bp.route('/models/<model_name>/info')
@handle_exceptions("Failed to get model info")
def get_model_info(model_name):
    """Get comprehensive model information for display"""
    metadata = _client.get_model_metadata(model_name)
    input_spec = _client.get_model_input_spec(model_name)
    output_spec = _client.get_model_output_spec(model_name)
    all_output_specs = _client.get_all_output_specs(model_name)
    server_type = _client.detect_server_type()
    
    detected_type = detect_model_type(
        model_name, 
        output_spec, 
        num_outputs=len(all_output_specs),
        all_output_specs=all_output_specs
    )
    
    return jsonify(create_success_response({
        'model_name': model_name,
        'server_type': server_type,
        'detected_type': detected_type,
        'ready': _client.check_model_ready(model_name),
        'input': {
            'name': input_spec.get('name', 'input'),
            'shape': input_spec.get('shape', []),
            'shape_formatted': format_tensor_shape(input_spec.get('shape', [])),
            'datatype': input_spec.get('datatype', 'unknown'),
            'format': input_spec.get('format', 'unknown'),
            'width': input_spec.get('width'),
            'height': input_spec.get('height'),
            'channels': input_spec.get('channels', 3)
        },
        'output': {
            'name': output_spec.get('name', 'output'),
            'shape': output_spec.get('shape', []),
            'shape_formatted': format_tensor_shape(output_spec.get('shape', [])),
            'datatype': output_spec.get('datatype', 'unknown'),
            'num_classes': output_spec.get('num_classes')
        },
        'outputs': [
            {
                'name': spec.get('name', f'output_{i}'),
                'shape': spec.get('shape', []),
                'shape_formatted': format_tensor_shape(spec.get('shape', [])),
                'datatype': spec.get('datatype', 'unknown'),
                'num_classes': spec.get('num_classes')
            }
            for i, spec in enumerate(all_output_specs)
        ],
        'num_outputs': len(all_output_specs),
        'detection_disclaimer': 'Model type detection is based on heuristics and may be incorrect.'
    }))


@core_bp.route('/models/<model_name>/spec')
@handle_exceptions("Failed to get model spec")
def get_model_spec(model_name):
    """Get auto-detected input/output specifications for a model"""
    info = _client.get_full_model_info(model_name)
    
    return jsonify(create_success_response({
        'model_name': model_name,
        'ready': info['ready'],
        'input_spec': info['input_spec'],
        'output_spec': info['output_spec']
    }))


@core_bp.route('/models/<model_name>/endpoints')
@handle_exceptions("Failed to get model endpoints")
def get_model_endpoints(model_name):
    """Get API endpoint information for developers"""
    endpoints_info = _client.get_api_endpoints_info(model_name)
    
    return jsonify(create_success_response({
        'model_name': model_name,
        'endpoints': endpoints_info
    }))


@core_bp.route('/logs/endpoints')
def get_endpoint_logs():
    """Get recent endpoint call logs (thread-safe)"""
    with endpoint_logs_lock:
        return jsonify(create_success_response({'logs': list(endpoint_logs)}))


@core_bp.route('/logs/processing')
def get_processing_logs():
    """Get recent processing step logs (thread-safe)"""
    with processing_logs_lock:
        return jsonify(create_success_response({'logs': list(processing_logs)}))


@core_bp.route('/class_names', methods=['GET'])
def get_all_class_names():
    """Deprecated: Class names are now managed client-side."""
    return jsonify(create_success_response({
        'class_names': {},
        'message': 'Class names are now managed via client-side JSON upload.'
    }))


@core_bp.route('/class_names/<model_name>', methods=['GET'])
def get_model_class_names(model_name):
    """Deprecated: Class names are now managed client-side."""
    return jsonify(create_success_response({
        'model_name': model_name,
        'class_names': [],
        'message': 'Class names are now managed via client-side JSON upload.'
    }))


@core_bp.route('/class_names/<model_name>', methods=['POST'])
def update_model_class_names(model_name):
    """Deprecated: Class names are now managed client-side."""
    return jsonify({
        'success': False,
        'error': 'This endpoint is deprecated.',
        'error_code': 'ENDPOINT_DEPRECATED',
        'model_name': model_name
    }), 410


@core_bp.route('/logs/clear', methods=['POST'])
def clear_logs():
    """Clear all logs (thread-safe)"""
    clear_all_logs()
    return jsonify(create_success_response({'cleared': True}))


@core_bp.route('/config')
def get_config():
    """Get application configuration"""
    return jsonify(create_success_response(_app_config))
