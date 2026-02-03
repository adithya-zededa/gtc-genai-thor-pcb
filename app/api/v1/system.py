"""System status and environment API endpoints."""

import os
import time
import subprocess
from datetime import datetime
from pathlib import Path
from flask import jsonify, request

from . import api_bp
from services.camera_service import check_camera_availability
from services.inference_service import (
    check_inference_backend_availability,
    check_vllm_availability,
    check_ollama_availability,
)
from app.database import LogSettingsRepository
from core.config import get_config
from core.logging import get_logger, apply_log_preferences

logger = get_logger(__name__)

# Track application start time
APP_START_TIME = time.time()


def _get_jetson_gpu_stats():
    """Get GPU stats for NVIDIA Jetson platforms using sysfs."""
    try:
        gpu_devfreq_path = Path("/sys/class/devfreq/17000000.gpu")
        gpu_thermal_path = None
        
        thermal_base = Path("/sys/class/thermal")
        if thermal_base.exists():
            for zone in thermal_base.iterdir():
                if not zone.name.startswith("thermal_zone"):
                    continue
                type_path = zone / "type"
                if type_path.exists():
                    with open(type_path, 'r') as f:
                        if f.read().strip() == "gpu-thermal":
                            gpu_thermal_path = zone / "temp"
                            break
        
        if not gpu_devfreq_path.exists():
            return None
        
        cur_freq_path = gpu_devfreq_path / "cur_freq"
        max_freq_path = gpu_devfreq_path / "max_freq"
        
        if not cur_freq_path.exists() or not max_freq_path.exists():
            return None
        
        with open(cur_freq_path, 'r') as f:
            cur_freq = int(f.read().strip())
        with open(max_freq_path, 'r') as f:
            max_freq = int(f.read().strip())
        
        gpu_util = round((cur_freq / max_freq) * 100, 1) if max_freq > 0 else 0
        
        gpu_temp = None
        if gpu_thermal_path and gpu_thermal_path.exists():
            try:
                with open(gpu_thermal_path, 'r') as f:
                    gpu_temp = round(int(f.read().strip()) / 1000, 1)
            except (IOError, ValueError):
                pass
        
        import psutil
        mem = psutil.virtual_memory()
        mem_used_mb = round((mem.total - mem.available) / (1024 * 1024), 0)
        mem_total_mb = round(mem.total / (1024 * 1024), 0)
        
        return {
            "name": "NVIDIA Jetson GPU",
            "utilization": gpu_util,
            "memory_used_mb": mem_used_mb,
            "memory_total_mb": mem_total_mb,
            "memory_percent": round(mem.percent, 1),
            "temperature": gpu_temp,
            "power_watts": None,
            "freq_mhz": round(cur_freq / 1_000_000, 0),
            "max_freq_mhz": round(max_freq / 1_000_000, 0),
            "platform": "jetson",
        }
    except Exception as e:
        logger.debug(f"Failed to get Jetson GPU stats: {e}")
        return None


@api_bp.route("/system/status")
def system_status():
    """Get system status information including GPU metrics."""
    try:
        import psutil

        config = get_config()
        uptime_seconds = int(time.time() - APP_START_TIME)

        status = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "camera_available": check_camera_availability(),
            "inference_available": check_inference_backend_availability(),
            "inference_backend": config.inference.backend,
            "uptime_seconds": uptime_seconds,
        }
        
        try:
            gpu_info = _get_jetson_gpu_stats()
            if gpu_info:
                status["gpu"] = gpu_info
            else:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,name",
                        "--format=csv,noheader,nounits"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 6:
                        status["gpu"] = {
                            "name": parts[5],
                            "utilization": float(parts[0]),
                            "memory_used_mb": float(parts[1]),
                            "memory_total_mb": float(parts[2]),
                            "memory_percent": round((float(parts[1]) / float(parts[2])) * 100, 1) if float(parts[2]) > 0 else 0,
                            "temperature": float(parts[3]),
                            "power_watts": float(parts[4]) if parts[4] != "[N/A]" else None,
                        }
                else:
                    status["gpu"] = None
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            status["gpu"] = None
            logger.debug(f"GPU stats unavailable: {e}")
        
        return jsonify({"success": True, "status": status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/system/environment", methods=["GET", "POST"])
def system_environment():
    """Get or update system environment variables."""
    config = get_config()
    
    if request.method == "POST":
        try:
            data = request.get_json()
            if not isinstance(data, dict):
                return jsonify({"success": False, "error": "Invalid payload"}), 400
            for key, value in data.items():
                os.environ[key] = str(value)
            return jsonify({
                "success": True,
                "message": "Environment variables updated (in-memory only)",
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    try:
        env_vars = {
            "CAMERA_INDEX": os.getenv("CAMERA_INDEX", "0"),
            "INFERENCE_BACKEND": config.inference.backend,
            "VLLM_URL": config.inference.vllm_url if config.inference.backend.lower() == "vllm" else "",
            "OLLAMA_URL": config.inference.ollama_url if config.inference.backend.lower() != "vllm" else "",
            "VISION_MODEL": config.inference.model,
            "DIFF_THRESHOLD": os.getenv("DIFF_THRESHOLD", "0.80"),
            "CAPTURE_INTERVAL": os.getenv("CAPTURE_INTERVAL", "5"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        }
        return jsonify({"success": True, "environment": env_vars})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/system/logging", methods=["GET", "POST"])
def system_logging():
    """Retrieve or update logging preferences."""
    if request.method == "GET":
        try:
            settings = LogSettingsRepository.get()
            return jsonify({"success": True, "settings": settings.to_dict()})
        except Exception as exc:
            logger.error("Failed to fetch log settings: %s", exc, exc_info=True)
            return jsonify({
                "success": False,
                "error": "Unable to load log settings"
            }), 500

    # POST method
    data = request.get_json(silent=True) or {}
    try:
        from app.database import LogSettings
        
        current = LogSettingsRepository.get()
        
        # Update with provided values
        level_candidate = str(data.get("log_level", current.log_level)).upper()
        if level_candidate in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            current.log_level = level_candidate
        
        if "log_retention" in data:
            try:
                retention = int(data["log_retention"])
                if retention > 0:
                    current.log_retention = retention
            except (ValueError, TypeError):
                pass
        
        if "max_log_size" in data:
            try:
                max_size = int(data["max_log_size"])
                if max_size > 0:
                    current.max_log_size = max_size
            except (ValueError, TypeError):
                pass
        
        if "log_to_file" in data:
            current.log_to_file = bool(data["log_to_file"])
        if "log_to_console" in data:
            current.log_to_console = bool(data["log_to_console"])
        if "log_database" in data:
            current.log_database = bool(data["log_database"])
        
        updated = LogSettingsRepository.update(current)
        
        # Apply preferences at runtime
        apply_log_preferences(
            updated.log_level,
            updated.log_to_console,
            updated.log_to_file,
        )
        
        return jsonify({"success": True, "settings": updated.to_dict()})
    except Exception as exc:
        logger.error("Failed to update log settings: %s", exc, exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to save log settings"
        }), 500


@api_bp.route("/test_camera")
def test_camera():
    """Test camera functionality."""
    import cv2
    
    try:
        config = get_config()
        cap = cv2.VideoCapture(config.camera.index)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return jsonify({
                    "success": True,
                    "message": "Camera test successful"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to capture frame"
                })
        else:
            return jsonify({"success": False, "error": "Cannot open camera"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/test_inference")
def test_inference():
    """Test the configured inference backend."""
    config = get_config()
    backend = config.inference.backend.lower()
    
    if backend == "vllm":
        available = check_vllm_availability()
        return jsonify({
            "success": available,
            "backend": "vllm",
            "message": "vLLM connection successful" if available else "vLLM not available",
        })
    else:
        available = check_ollama_availability()
        return jsonify({
            "success": available,
            "backend": "ollama",
            "message": "Ollama connection successful" if available else "Ollama not available",
        })


@api_bp.route("/test_vllm")
def test_vllm():
    """Test vLLM connection."""
    import requests
    
    config = get_config()
    try:
        response = requests.get(
            f"{config.inference.vllm_url}/v1/models",
            timeout=config.http_timeout
        )
        response.raise_for_status()
        data = response.json()
        models = [m.get("id", "") for m in data.get("data", [])]
        return jsonify({
            "success": True,
            "message": "vLLM connection successful",
            "models": models,
            "url": config.inference.vllm_url
        })
    except requests.RequestException as exc:
        return jsonify({
            "success": False,
            "error": f"vLLM connectivity failed: {exc}"
        })


@api_bp.route("/test_ollama")
def test_ollama():
    """Test Ollama connection."""
    import requests
    
    config = get_config()
    try:
        response = requests.get(
            f"{config.inference.ollama_url}/api/version",
            timeout=config.http_timeout
        )
        response.raise_for_status()
        return jsonify({
            "success": True,
            "message": "Ollama connection successful"
        })
    except requests.RequestException as exc:
        return jsonify({
            "success": False,
            "error": f"Ollama connectivity failed: {exc}"
        })


@api_bp.route("/ollama_models")
def ollama_models():
    """List available Ollama models (vision-capable only)."""
    import requests
    
    config = get_config()
    try:
        response = requests.get(
            f"{config.inference.ollama_url}/api/tags",
            timeout=config.http_timeout
        )
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        vision_keywords = ["llava", "gemma3", "qwen", "minicpm", "llama3.2-vision", "moondream"]
        vision_models = [
            m.get("name", "") for m in models
            if any(kw in m.get("name", "").lower() for kw in vision_keywords)
        ]
        return jsonify({"success": True, "models": vision_models})
    except requests.RequestException as exc:
        return jsonify({"success": False, "error": str(exc), "models": []})
