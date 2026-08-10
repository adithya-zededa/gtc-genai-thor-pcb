"""System status and environment API endpoints."""

# pylint: disable=broad-exception-caught,import-outside-toplevel,logging-fstring-interpolation,unspecified-encoding,subprocess-run-check,unused-variable,too-many-locals,too-many-branches,no-else-return,unused-import,line-too-long

import os
import subprocess
import time
from pathlib import Path

from flask import jsonify, request

from app.database import LogSettingsRepository
from core.config import get_config
from core.logging import apply_log_preferences, get_logger
from services.core.camera import check_camera_availability, reset_publisher
from services.core.inference import (
    check_inference_roles,
    check_vllm_availability,
)

from . import api_bp

logger = get_logger(__name__)

# Track application start time
APP_START_TIME = time.time()

# Keys the /system/environment POST endpoint is allowed to write. This
# endpoint has no auth gate, so writes must be restricted to the specific
# runtime knobs the UI exposes rather than arbitrary process environment
# variables.
ALLOWED_ENV_VARS = frozenset({
    "CAMERA_INDEX",
    "DIFF_THRESHOLD",
    "CAPTURE_INTERVAL",
    "LOG_LEVEL",
})


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
                    with open(type_path, "r") as f:
                        if f.read().strip() == "gpu-thermal":
                            gpu_thermal_path = zone / "temp"
                            break

        if not gpu_devfreq_path.exists():
            return None

        cur_freq_path = gpu_devfreq_path / "cur_freq"
        max_freq_path = gpu_devfreq_path / "max_freq"

        if not cur_freq_path.exists() or not max_freq_path.exists():
            return None

        with open(cur_freq_path, "r") as f:
            cur_freq = int(f.read().strip())
        with open(max_freq_path, "r") as f:
            max_freq = int(f.read().strip())

        gpu_util = round((cur_freq / max_freq) * 100, 1) if max_freq > 0 else 0

        gpu_temp = None
        if gpu_thermal_path and gpu_thermal_path.exists():
            try:
                with open(gpu_thermal_path, "r") as f:
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

        inference_roles = check_inference_roles()
        status = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "camera_available": check_camera_availability(),
            "inference_available": all(inference_roles.values()),
            "inference_roles": inference_roles,
            "inference_backend": "vllm",
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
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if not result.returncode and result.stdout.strip():
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 6:
                        status["gpu"] = {
                            "name": parts[5],
                            "utilization": float(parts[0]),
                            "memory_used_mb": float(parts[1]),
                            "memory_total_mb": float(parts[2]),
                            "memory_percent": (
                                round((float(parts[1]) / float(parts[2])) * 100, 1)
                                if float(parts[2]) > 0
                                else 0
                            ),
                            "temperature": float(parts[3]),
                            "power_watts": (
                                float(parts[4]) if parts[4] != "[N/A]" else None
                            ),
                        }
                else:
                    status["gpu"] = None
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            status["gpu"] = None
            logger.debug(f"GPU stats unavailable: {e}")

        return jsonify({"success": True, "status": status})
    except Exception as e:
        logger.exception("Failed to get system status: %s", e)
        return jsonify({"success": False, "error": "Failed to get system status"}), 500


@api_bp.route("/system/environment", methods=["GET", "POST"])
def system_environment():
    """Get or update system environment variables."""
    config = get_config()

    if request.method == "POST":
        try:
            data = request.get_json()
            if not isinstance(data, dict):
                return jsonify({"success": False, "error": "Invalid payload"}), 400
            unknown = [key for key in data if key not in ALLOWED_ENV_VARS]
            if unknown:
                return jsonify({
                    "success": False,
                    "error": f"Unsupported environment variable(s): {', '.join(sorted(unknown))}",
                }), 400
            for key, value in data.items():
                os.environ[key] = str(value)
            return jsonify(
                {
                    "success": True,
                    "message": "Environment variables updated (in-memory only)",
                }
            )
        except Exception as e:
            logger.exception("Failed to update environment variables: %s", e)
            return jsonify({"success": False, "error": "Failed to update environment variables"}), 500

    try:
        env_vars = {
            "CAMERA_INDEX": os.getenv("CAMERA_INDEX", "0"),
            "INFERENCE_BACKEND": "vllm",
            "VLLM_URL": config.inference.vllm_url,
            "VISION_MODEL": config.inference.model,
            "DIFF_THRESHOLD": os.getenv("DIFF_THRESHOLD", "0.80"),
            "CAPTURE_INTERVAL": os.getenv("CAPTURE_INTERVAL", "5"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        }
        return jsonify({"success": True, "environment": env_vars})
    except Exception as e:
        logger.exception("Failed to get system environment: %s", e)
        return jsonify({"success": False, "error": "Failed to get environment variables"}), 500


@api_bp.route("/system/logging", methods=["GET", "POST"])
def system_logging():
    """Retrieve or update logging preferences."""
    if request.method == "GET":
        try:
            settings = LogSettingsRepository.get()
            return jsonify({"success": True, "settings": settings.to_dict()})
        except Exception as exc:
            logger.error("Failed to fetch log settings: %s", exc, exc_info=True)
            return (
                jsonify({"success": False, "error": "Unable to load log settings"}),
                500,
            )

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
        return jsonify({"success": False, "error": "Failed to save log settings"}), 500


@api_bp.route("/system/video_source", methods=["GET", "POST"])
def video_source():
    """Query or switch between live camera and simulated video feed."""
    config = get_config()

    if request.method == "GET":
        return jsonify({
            "success": True,
            "simulated": bool(config.camera.video_source),
            "video_source": config.camera.video_source or "",
        })

    # POST — toggle
    data = request.get_json(silent=True) or {}
    simulated = data.get("simulated", False)

    if simulated:
        source = data.get("path") or os.getenv("CAMERA_VIDEO_SOURCE", "")
        if not source:
            return jsonify({"success": False, "error": "No video file configured (set CAMERA_VIDEO_SOURCE)"}), 400

        # This endpoint has no auth gate, so the requested path must be
        # confined to the video directory the deployment ships/mounts —
        # otherwise a caller could point video capture at an arbitrary
        # file on disk.
        video_dir = config.video_dir.resolve()
        try:
            resolved_source = Path(source).resolve()
        except OSError:
            return jsonify({"success": False, "error": f"Invalid video path: {source}"}), 400

        if not resolved_source.is_relative_to(video_dir):
            return jsonify({
                "success": False,
                "error": f"Video path must be inside {video_dir}",
            }), 400
        if not resolved_source.is_file():
            return jsonify({"success": False, "error": f"Video file not found: {source}"}), 400
        config.camera.video_source = str(resolved_source)
    else:
        config.camera.video_source = None

    # Restart publisher so it picks up the new source
    reset_publisher()

    mode = "simulated video" if simulated else "live camera"
    logger.info("Video source switched to %s", mode)
    return jsonify({
        "success": True,
        "simulated": bool(config.camera.video_source),
        "video_source": config.camera.video_source or "",
        "message": f"Switched to {mode}",
    })


@api_bp.route("/test_camera")
def test_camera():
    """Test camera functionality."""
    import cv2

    try:
        config = get_config()
        cap = cv2.VideoCapture(config.camera.index)  # pylint: disable=no-member
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return jsonify({"success": True, "message": "Camera test successful"})
            else:
                return jsonify({"success": False, "error": "Failed to capture frame"})
        else:
            return jsonify({"success": False, "error": "Cannot open camera"})
    except Exception as e:
        logger.exception("Camera test failed: %s", e)
        return jsonify({"success": False, "error": "Camera test failed"}), 500


@api_bp.route("/test_inference")
def test_inference():
    """Test the configured inference backend."""
    available = check_vllm_availability()
    return jsonify(
        {
            "success": available,
            "backend": "vllm",
            "message": (
                "vLLM connection successful" if available else "vLLM not available"
            ),
        }
    )


@api_bp.route("/test_vllm")
def test_vllm():
    """Test vLLM connection."""
    import requests

    config = get_config()
    try:
        response = requests.get(
            f"{config.inference.vllm_url}/v1/models", timeout=config.http_timeout
        )
        response.raise_for_status()
        data = response.json()
        models = [m.get("id", "") for m in data.get("data", [])]
        return jsonify(
            {
                "success": True,
                "message": "vLLM connection successful",
                "models": models,
                "url": config.inference.vllm_url,
            }
        )
    except requests.RequestException as exc:
        return jsonify({"success": False, "error": f"vLLM connectivity failed: {exc}"})
