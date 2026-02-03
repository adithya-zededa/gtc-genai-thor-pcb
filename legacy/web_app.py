#!/usr/bin/env python3
"""
ZEDEDA Camera Monitoring Agent - Web Frontend
Modern Flask web interface for camera monitoring and configuration
"""

import os
import json
import yaml
import base64
import sqlite3
import logging
import copy
import csv
import io
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    Response,
    send_file,
    abort,
)
from flask_socketio import SocketIO, emit
import threading
import time
import requests
import cv2
import numpy as np

from camera_agent import StreamlinedAgent, DEFAULT_CONFIG_PATH
from agent_runtime.utils import (
    coerce_bool,
    dedupe_strings,
    ensure_directory,
)
from agent_runtime.publisher import get_camera_publisher
from agent_runtime.monitoring import StreamlinedMonitoringService
from agent_runtime.unified_vlm import UnifiedVLMClient, TaskType, AnalysisResult, VLMBackend
from agent_runtime.email_tools import send_email

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ENV_DATA_DIR = "CAMERA_AGENT_DATA_DIR"
ENV_SECRET_KEY = "FLASK_SECRET_KEY"
ENV_SOCKETIO_CORS = "SOCKETIO_CORS"
ENV_DB_PATH = "CAMERA_AGENT_DB"
ENV_DETECTED_DIR = "DETECTED_IMAGES_DIR"
ENV_PROCESSED_DIR = "PROCESSED_FRAMES_DIR"
ENV_OLLAMA_URL = "OLLAMA_URL"
ENV_VLLM_URL = "VLLM_URL"
ENV_INFERENCE_BACKEND = "INFERENCE_BACKEND"
ENV_HTTP_TIMEOUT = "HTTP_REQUEST_TIMEOUT"

DEFAULT_SECRET_KEY_BYTES = 24
DEFAULT_SOCKETIO_CORS = "*"
DEFAULT_HTTP_TIMEOUT = 5.0
IMAGE_GLOB_PATTERN = "*.jpg"


def _safe_int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _resolve_secret_key():
    secret = os.getenv(ENV_SECRET_KEY)
    return secret if secret else os.urandom(DEFAULT_SECRET_KEY_BYTES)


def _resolve_socketio_cors():
    raw = os.getenv(ENV_SOCKETIO_CORS)
    if not raw:
        return DEFAULT_SOCKETIO_CORS
    parsed = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return parsed or DEFAULT_SOCKETIO_CORS


DATA_DIR = Path(os.getenv(ENV_DATA_DIR, ".")).expanduser()
DATABASE_PATH = Path(
    os.getenv(ENV_DB_PATH, str(DATA_DIR / "camera_agent.db"))
).expanduser()
DETECTED_IMAGES_DIR = Path(
    os.getenv(ENV_DETECTED_DIR, str(DATA_DIR / "detected_images"))
).expanduser()
PROCESSED_FRAMES_DIR = Path(
    os.getenv(ENV_PROCESSED_DIR, str(DATA_DIR / "processed_frames"))
).expanduser()
OLLAMA_BASE_URL = os.getenv(ENV_OLLAMA_URL, "http://localhost:11434")
VLLM_BASE_URL = os.getenv(ENV_VLLM_URL, "http://localhost:8000")
INFERENCE_BACKEND = os.getenv(ENV_INFERENCE_BACKEND, "vllm")  # Default to vLLM


def _resolve_request_timeout() -> float:
    try:
        timeout_value = float(
            os.getenv(ENV_HTTP_TIMEOUT, DEFAULT_HTTP_TIMEOUT)
        )
        return timeout_value if timeout_value > 0 else DEFAULT_HTTP_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_HTTP_TIMEOUT


REQUEST_TIMEOUT = _resolve_request_timeout()


def _get_jetson_gpu_stats() -> Optional[Dict]:
    """Get GPU stats for NVIDIA Jetson platforms using sysfs.
    
    On Jetson, GPU and CPU share unified memory, so we report system memory usage.
    GPU utilization is approximated from frequency scaling (cur_freq vs max_freq).
    
    Returns:
        Dict with GPU stats if on Jetson, None otherwise
    """
    try:
        gpu_devfreq_path = Path("/sys/class/devfreq/17000000.gpu")
        gpu_thermal_path = None
        
        # Find GPU thermal zone
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
        
        # Check if this is a Jetson platform
        if not gpu_devfreq_path.exists():
            return None
        
        # Get GPU frequency info
        cur_freq_path = gpu_devfreq_path / "cur_freq"
        max_freq_path = gpu_devfreq_path / "max_freq"
        
        if not cur_freq_path.exists() or not max_freq_path.exists():
            return None
        
        with open(cur_freq_path, 'r') as f:
            cur_freq = int(f.read().strip())
        with open(max_freq_path, 'r') as f:
            max_freq = int(f.read().strip())
        
        # Approximate GPU utilization from frequency scaling
        # Higher frequency typically means higher GPU load
        gpu_util = round((cur_freq / max_freq) * 100, 1) if max_freq > 0 else 0
        
        # Get GPU temperature
        gpu_temp = None
        if gpu_thermal_path and gpu_thermal_path.exists():
            try:
                with open(gpu_thermal_path, 'r') as f:
                    # Temperature is in millidegrees Celsius
                    gpu_temp = round(int(f.read().strip()) / 1000, 1)
            except (IOError, ValueError):
                pass
        
        # Get system memory (shared with GPU on Jetson)
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
            "power_watts": None,  # Power info typically requires tegrastats
            "freq_mhz": round(cur_freq / 1_000_000, 0),
            "max_freq_mhz": round(max_freq / 1_000_000, 0),
            "platform": "jetson",
        }
    except Exception as e:
        logger.debug(f"Failed to get Jetson GPU stats: {e}")
        return None


# Rate limiting for expensive operations
_last_camera_check = 0.0
_last_camera_status = False
_camera_check_interval = 2.0  # seconds
_camera_check_lock = threading.Lock()


def _ollama_endpoint(path: str) -> str:
    base = OLLAMA_BASE_URL.rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def _create_vlm_client_from_config(config: Dict) -> UnifiedVLMClient:
    """Create a VLM client from configuration.
    
    Supports both Ollama and vLLM backends based on config and environment variables.
    Environment variables take precedence over config file settings.
    """
    # Check environment variable for backend selection
    env_backend = os.getenv(ENV_INFERENCE_BACKEND, "").lower()
    
    # Determine backend: env var > config > default (vllm)
    if env_backend == "vllm" or (not env_backend and config.get("vllm")):
        # Use vLLM backend
        vllm_cfg = config.get("vllm", {})
        vllm_url = os.getenv(ENV_VLLM_URL) or str(vllm_cfg.get("url", "http://localhost:8000")).rstrip("/")
        default_model = os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        vision_model = str(vllm_cfg.get("model", default_model))
        timeout = int(os.getenv("VLLM_TIMEOUT", vllm_cfg.get("timeout", 300)))
        temperature = float(os.getenv("VLLM_TEMPERATURE", vllm_cfg.get("temperature", 0.1)))
        
        logger.info("Creating vLLM client: url=%s, model=%s", vllm_url, vision_model)
        return UnifiedVLMClient(
            base_url=vllm_url,
            model=vision_model,
            timeout=timeout,
            backend=VLMBackend.VLLM,
            temperature=temperature,
        )
    
    # Fall back to Ollama config (legacy)
    ollama_cfg = config.get("ollama", {})
    ollama_url = os.getenv(ENV_OLLAMA_URL) or str(ollama_cfg.get("url", "http://localhost:11434")).rstrip("/")
    default_model = os.getenv("VISION_MODEL", "qwen3-vl:8b")
    vision_model = str(ollama_cfg.get("vision_model", default_model))
    timeout = int(ollama_cfg.get("timeout", 300))
    temperature = float(ollama_cfg.get("temperature", 0.1))
    
    logger.info("Creating Ollama client (legacy fallback): url=%s, model=%s", ollama_url, vision_model)
    return UnifiedVLMClient(
        base_url=ollama_url,
        model=vision_model,
        timeout=timeout,
        backend=VLMBackend.OLLAMA,
        temperature=temperature,
    )


app = Flask(__name__)
app.secret_key = _resolve_secret_key()
socketio = SocketIO(app, cors_allowed_origins=_resolve_socketio_cors())

# Global variables
camera_agent = None
camera_publisher = None  # Global publisher instance for camera viewing
APP_START_TIME = time.time()  # Track application start time
CONFIG_PATH = Path(
    os.getenv("CAMERA_AGENT_CONFIG", DEFAULT_CONFIG_PATH)
).expanduser()
CONFIG_LOCK = threading.RLock()

DEFAULT_LOG_SETTINGS = {
    "log_level": "INFO",
    "log_retention": 30,
    "max_log_size": 100,
    "log_to_file": True,
    "log_to_console": True,
    "log_database": False,
}
HANDLER_DISABLE_LEVEL = logging.CRITICAL + 10


def _coerce_bool_config(value, default: bool = False) -> bool:
    """Convert string/number representations to boolean values."""
    coerced = coerce_bool(value, default)
    return default if coerced is None else coerced


def _extract_email_intent(prompt: str) -> Optional[List[str]]:
    """Extract email addresses from user prompt if email action is requested.
    
    Looks for patterns like:
    - "send an email to user@example.com"
    - "email to: user@example.com"
    - "notify user@example.com"
    - "send an email to user1@example.com and to user2@example.com"
    
    Returns:
        List of email addresses if found, None otherwise
    """
    import re
    
    prompt_lower = prompt.lower()
    # Check for email action keywords
    email_keywords = ["send email", "send an email", "email to", "notify", "alert"]
    has_email_intent = any(kw in prompt_lower for kw in email_keywords)
    
    if not has_email_intent:
        return None
    
    # Extract all email addresses using regex
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    matches = re.findall(email_pattern, prompt)
    
    return matches if matches else None


def _execute_email_action(
    email_to: List[str],
    detection_result: "AnalysisResult",
    user_prompt: str,
    image_data: Optional[bytes] = None,
) -> str:
    """Execute email action based on detection result.
    
    Args:
        email_to: List of recipient email addresses
        detection_result: The VLM analysis result
        user_prompt: Original user prompt
        image_data: Optional JPEG image bytes to attach
        
    Returns:
        Status message from email sending
    """
    subject = f"Camera Agent Alert: {detection_result.task_type}"
    body = f"""Camera Agent Detection Alert

Detection Task: {detection_result.task_type}
User Query: {user_prompt}

Detection Results:
- Detected: {detection_result.detected}
- Confidence: {detection_result.confidence:.2%}
- Reasoning: {detection_result.reasoning}

Details:
{json.dumps(detection_result.details, indent=2)}

{"An image of the detection is attached." if image_data else ""}
---
This is an automated message from Camera Agent.
"""
    
    email_payload: Dict[str, Any] = {
        "to": email_to,
        "subject": subject,
        "body": body,
    }
    
    # Attach image if provided
    if image_data and isinstance(image_data, bytes):
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        email_payload["image_data"] = image_data
        email_payload["image_filename"] = f"detection_{timestamp_str}.jpg"
        logger.debug("Attaching image to email (%d bytes)", len(image_data))
    
    result = send_email(email_payload)
    
    logger.info("Email sent to %s: %s", email_to, result)
    return result


def ensure_config_directory() -> None:
    """Ensure the configuration directory exists."""
    try:
        ensure_directory(CONFIG_PATH.parent)
    except Exception as exc:
        logger.error(
            "Failed to create config directory %s: %s", CONFIG_PATH.parent, exc
        )


def ensure_database_directory() -> None:
    """Ensure the database directory exists before connecting."""
    try:
        ensure_directory(DATABASE_PATH.parent)
    except Exception as exc:
        logger.error(
            "Failed to prepare database directory %s: %s",
            DATABASE_PATH.parent,
            exc,
        )


def _sanitize_config_payload(config: Dict) -> Dict:
    """Return a sanitized copy of the configuration for persistence."""
    config_copy = copy.deepcopy(config) if isinstance(config, dict) else {}
    notifications = (
        config_copy.setdefault("notifications", {})
        if isinstance(config_copy, dict)
        else {}
    )
    if isinstance(notifications, dict):
        email_cfg = notifications.get("email") or {}
        if isinstance(email_cfg, dict):
            email_cfg.pop("sender_password", None)
    return config_copy


def load_camera_config() -> Dict:
    """Load the camera configuration from disk."""
    with CONFIG_LOCK:
        return StreamlinedAgent.load_config_from_path(CONFIG_PATH)


def save_camera_config(config: Dict) -> Dict:
    """Persist configuration to disk and return the sanitized payload."""
    canonical = copy.deepcopy(config) if isinstance(config, dict) else {}
    recipients = extract_recipients(canonical)
    apply_recipients(canonical, recipients)
    sanitized = _sanitize_config_payload(canonical)
    with CONFIG_LOCK:
        ensure_config_directory()
        with CONFIG_PATH.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(
                sanitized,
                config_file,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
    return sanitized


def reset_camera_config() -> Dict:
    """Reset the configuration to defaults and persist the change."""
    defaults = StreamlinedAgent.default_config()
    save_camera_config(defaults)
    return defaults


def extract_recipients(config: Dict) -> List[str]:
    """Collect the distinct email recipients from configuration payload."""
    recipients: List[str] = []
    notifications = (config or {}).get("notifications", {})
    email_cfg = (
        notifications.get("email", {})
        if isinstance(notifications, dict)
        else {}
    )
    email_recipients = (
        email_cfg.get("recipients") if isinstance(email_cfg, dict) else None
    )

    if isinstance(email_recipients, list):
        recipients.extend(
            [addr for addr in email_recipients if isinstance(addr, str)]
        )

    rules = config.get("rules") if isinstance(config, dict) else None
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            actions_email = (
                ((rule.get("actions") or {}).get("email"))
                if isinstance(rule.get("actions"), dict)
                else None
            )
            legacy_email = (
                rule.get("email")
                if isinstance(rule.get("email"), dict)
                else None
            )
            for container in (actions_email, legacy_email):
                if isinstance(container, dict):
                    recipients.extend(
                        [
                            addr
                            for addr in container.get("to", [])
                            if isinstance(addr, str)
                        ]
                    )

    return dedupe_strings(recipients)


def apply_recipients(config: Dict, recipients: List[str]) -> Dict:
    """Apply recipient list to notification config and rules."""
    sanitized_recipients = [
        email.strip()
        for email in recipients
        if isinstance(email, str) and email.strip()
    ]
    notifications = config.setdefault("notifications", {})
    email_cfg = notifications.setdefault("email", {})
    email_cfg["recipients"] = sanitized_recipients

    rules = config.get("rules")
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if "actions" in rule and isinstance(rule["actions"], dict):
                email_action = rule["actions"].setdefault("email", {})
                if isinstance(email_action, dict):
                    email_action["to"] = sanitized_recipients
            if "email" in rule and isinstance(rule["email"], dict):
                rule["email"]["to"] = sanitized_recipients

    return config


def _collect_images_from_directory(
    directory: Path, limit: int
) -> List[Dict[str, str]]:
    """Return encoded image metadata from a directory without duplicating loop logic."""
    if not directory.exists():
        return []

    images: List[Dict[str, str]] = []
    try:
        files = sorted(
            directory.glob(IMAGE_GLOB_PATTERN),
            key=lambda file_path: file_path.stat().st_mtime,
            reverse=True,
        )[: max(limit, 0)]
        for img_file in files:
            stat_info = img_file.stat()
            with img_file.open("rb") as img_stream:
                encoded = base64.b64encode(img_stream.read()).decode("utf-8")
            images.append(
                {
                    "filename": img_file.name,
                    "timestamp": datetime.fromtimestamp(
                        stat_info.st_mtime
                    ).isoformat(),
                    "image_b64": encoded,
                }
            )
    except Exception as exc:
        logger.error("Failed to collect images from %s: %s", directory, exc)
    return images


class WebCameraAgent(StreamlinedMonitoringService):
    """Flask-integrated camera monitoring agent with Socket.IO updates."""

    def __init__(self):
        super().__init__(
            subscriber_id="monitoring_agent", auto_start_publisher=False
        )

    def emit_event(self, event_name: str, payload):
        try:
            socketio.emit(event_name, payload)
        except Exception as exc:
            logger.error("Failed to emit %s event: %s", event_name, exc)

    def _record_detection(self, event, frame_metadata):
        """Persist detection event to the database for UI visibility."""
        conn = None
        new_log_id = None
        try:
            conn = get_db_connection()
            decision_details_json = json.dumps(event.decision_trace or {})
            tool_trace_json = json.dumps(getattr(event, "tool_trace", []) or [])
            cursor = conn.execute(
                """
                INSERT INTO detection_logs (
                    timestamp, confidence, response, image_path, frame_number,
                    reason, vision_description, decision_details, tool_trace
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.timestamp,
                    event.confidence,
                    event.full_response,
                    event.image_path or "",
                    frame_metadata.get("frame_number"),
                    frame_metadata.get("reason"),
                    getattr(event, "vision_description", ""),
                    decision_details_json,
                    tool_trace_json,
                ),
            )
            conn.commit()
            new_log_id = cursor.lastrowid
            logger.debug(
                "Recorded detection: frame=%s, label=%s, confidence=%.2f",
                frame_metadata.get("frame_number"),
                getattr(event, "primary_label", "unknown"),
                event.confidence if event.confidence else 0.0,
            )
            
            # Emit real-time event for live log updates
            if new_log_id:
                log_entry = {
                    "id": new_log_id,
                    "timestamp": event.timestamp,
                    "confidence": event.confidence,
                    "response": event.full_response,
                    "image_path": event.image_path or "",
                    "frame_number": frame_metadata.get("frame_number"),
                    "reason": frame_metadata.get("reason"),
                    "vision_description": getattr(event, "vision_description", ""),
                    "decision_details": event.decision_trace or {},
                    "tool_trace": getattr(event, "tool_trace", []) or [],
                    "detected": getattr(event, "detected", False),
                    "agentic_mode": (event.decision_trace or {}).get("classification") == "AGENTIC_ANALYSIS",
                }
                self.emit_event("new_log", log_entry)
                
        except Exception as exc:
            logger.error("Failed to record detection: %s", exc)
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass


def _log_detection_to_db(event, frame_metadata):
    """Log a detection event to the database (standalone helper).
    
    This is used for uploaded image analysis and other scenarios where
    the WebMonitoringService isn't the one running the analysis.
    
    Args:
        event: DetectionEvent object with analysis results
        frame_metadata: Dict with frame_number, reason, etc.
    """
    conn = None
    try:
        conn = get_db_connection()
        decision_details_json = json.dumps(event.decision_trace or {})
        tool_trace_json = json.dumps(getattr(event, "tool_trace", []) or [])
        cursor = conn.execute(
            """
            INSERT INTO detection_logs (
                timestamp, confidence, response, image_path, frame_number,
                reason, vision_description, decision_details, tool_trace
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.timestamp,
                event.confidence,
                event.full_response,
                event.image_path or "",
                frame_metadata.get("frame_number"),
                frame_metadata.get("reason", "Uploaded image analysis"),
                getattr(event, "vision_description", ""),
                decision_details_json,
                tool_trace_json,
            ),
        )
        conn.commit()
        logger.debug(
            "Logged uploaded image analysis: detected=%s, confidence=%.2f",
            getattr(event, "detected", False),
            event.confidence if event.confidence else 0.0,
        )
    except Exception as exc:
        logger.error("Failed to log detection to database: %s", exc)
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# Initialize database
def init_db():
    """Initialize SQLite database for user management and logs."""
    ensure_database_directory()
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Enable WAL mode for better concurrent access
    cursor.execute("PRAGMA journal_mode=WAL")

    # Users table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Detection logs table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            response TEXT,
            image_path TEXT,
            frame_number INTEGER,
            reason TEXT,
            vision_description TEXT,
            decision_details TEXT
        )
    """
    )
    
    # Create indexes for frequently queried columns
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_detection_logs_timestamp 
        ON detection_logs(timestamp DESC)
    """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_detection_logs_confidence 
        ON detection_logs(confidence)
    """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_users_active 
        ON users(active)
    """
    )

    # Ensure new columns exist when upgrading from previous schema
    try:
        cursor.execute(
            "ALTER TABLE detection_logs ADD COLUMN vision_description TEXT"
        )
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute(
            "ALTER TABLE detection_logs ADD COLUMN decision_details TEXT"
        )
    except sqlite3.OperationalError:
        pass
    
    # Add tool_trace column for agentic mode
    try:
        cursor.execute(
            "ALTER TABLE detection_logs ADD COLUMN tool_trace TEXT"
        )
    except sqlite3.OperationalError:
        pass

    # Configuration history table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS config_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            config_type TEXT,
            changes TEXT,
            user_email TEXT
        )
    """
    )

    ensure_log_settings_seed(conn)

    conn.commit()
    conn.close()


def get_db_connection() -> sqlite3.Connection:
    """Get database connection with row factory.
    
    Returns:
        sqlite3.Connection: A new database connection.
        
    Note:
        Caller is responsible for closing the connection.
        Prefer using 'with get_db_connection() as conn:' pattern.
    """
    ensure_database_directory()
    conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    conn.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
    return conn


def ensure_log_settings_seed(conn: sqlite3.Connection) -> None:
    """Guarantee the log_settings table exists with a canonical row."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS log_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            log_level TEXT NOT NULL DEFAULT 'INFO',
            log_retention INTEGER NOT NULL DEFAULT 30,
            max_log_size INTEGER NOT NULL DEFAULT 100,
            log_to_file INTEGER NOT NULL DEFAULT 1,
            log_to_console INTEGER NOT NULL DEFAULT 1,
            log_database INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
        INSERT INTO log_settings (id, log_level, log_retention, max_log_size, log_to_file, log_to_console, log_database)
        SELECT 1, ?, ?, ?, ?, ?, ?
        WHERE NOT EXISTS (SELECT 1 FROM log_settings WHERE id = 1)
    """,
        (
            DEFAULT_LOG_SETTINGS["log_level"],
            DEFAULT_LOG_SETTINGS["log_retention"],
            DEFAULT_LOG_SETTINGS["max_log_size"],
            int(DEFAULT_LOG_SETTINGS["log_to_file"]),
            int(DEFAULT_LOG_SETTINGS["log_to_console"]),
            int(DEFAULT_LOG_SETTINGS["log_database"]),
        ),
    )


def get_log_settings() -> Dict[str, object]:
    """Fetch persisted logging preferences or fall back to defaults."""
    conn = get_db_connection()
    ensure_log_settings_seed(conn)
    row = conn.execute(
        """
        SELECT log_level, log_retention, max_log_size, log_to_file, log_to_console, log_database
        FROM log_settings
        WHERE id = 1
    """
    ).fetchone()
    conn.commit()
    conn.close()

    if not row:
        return DEFAULT_LOG_SETTINGS.copy()

    return {
        "log_level": (
            row["log_level"] or DEFAULT_LOG_SETTINGS["log_level"]
        ).upper(),
        "log_retention": int(
            row["log_retention"]
            if row["log_retention"] is not None
            else DEFAULT_LOG_SETTINGS["log_retention"]
        ),
        "max_log_size": int(
            row["max_log_size"]
            if row["max_log_size"] is not None
            else DEFAULT_LOG_SETTINGS["max_log_size"]
        ),
        "log_to_file": bool(row["log_to_file"]),
        "log_to_console": bool(row["log_to_console"]),
        "log_database": bool(row["log_database"]),
    }


def _apply_log_preferences(settings: Dict[str, object]) -> None:
    """Apply log level and handler preferences at runtime."""
    level_name = str(
        settings.get("log_level", DEFAULT_LOG_SETTINGS["log_level"])
    ).upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    logger.setLevel(level)
    logging.getLogger("camera_agent").setLevel(level)
    logging.getLogger("email_agent").setLevel(level)

    log_to_console = bool(settings.get("log_to_console", True))
    log_to_file = bool(settings.get("log_to_file", True))

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(level if log_to_file else HANDLER_DISABLE_LEVEL)
        elif isinstance(handler, logging.StreamHandler):
            handler.setLevel(
                level if log_to_console else HANDLER_DISABLE_LEVEL
            )


def persist_log_settings(payload: Dict[str, object]) -> Dict[str, object]:
    """Validate and persist logging preferences."""
    sanitized = DEFAULT_LOG_SETTINGS.copy()

    level_candidate = str(
        payload.get("log_level", sanitized["log_level"])
    ).upper()
    if level_candidate in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        sanitized["log_level"] = level_candidate

    try:
        retention_value = payload.get("log_retention", sanitized["log_retention"])
        if retention_value is not None:
            retention = int(str(retention_value))
            if retention > 0:
                sanitized["log_retention"] = retention
    except (ValueError, TypeError):
        pass

    try:
        max_size_value = payload.get("max_log_size", sanitized["max_log_size"])
        if max_size_value is not None:
            max_size = int(str(max_size_value))
            if max_size > 0:
                sanitized["max_log_size"] = max_size
    except (ValueError, TypeError):
        pass
    sanitized["log_to_file"] = bool(
        payload.get("log_to_file", sanitized["log_to_file"])
    )
    sanitized["log_to_console"] = bool(
        payload.get("log_to_console", sanitized["log_to_console"])
    )
    sanitized["log_database"] = bool(
        payload.get("log_database", sanitized["log_database"])
    )

    conn = get_db_connection()
    ensure_log_settings_seed(conn)
    conn.execute(
        """
        UPDATE log_settings
        SET log_level = ?,
            log_retention = ?,
            max_log_size = ?,
            log_to_file = ?,
            log_to_console = ?,
            log_database = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = 1
    """,
        (
            sanitized["log_level"],
            sanitized["log_retention"],
            sanitized["max_log_size"],
            int(sanitized["log_to_file"]),
            int(sanitized["log_to_console"]),
            int(sanitized["log_database"]),
        ),
    )
    conn.commit()
    conn.close()

    _apply_log_preferences(sanitized)
    return sanitized


try:
    _apply_log_preferences(get_log_settings())
except Exception as exc:  # pragma: no cover - defensive path
    logger.warning("Unable to apply persisted logging preferences: %s", exc)


# Routes
@app.route("/")
def dashboard():
    """Main dashboard"""
    return render_template("dashboard.html")


@app.route("/health")
def health_check():
    """Health check endpoint for container orchestration.
    
    Returns:
        JSON with health status and component availability.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "inference_backend": INFERENCE_BACKEND,
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
        health_status["status"] = "degraded"
    
    # Check camera
    health_status["components"]["camera"] = check_camera_availability()
    
    # Check inference backend (vLLM or Ollama)
    health_status["components"]["inference"] = check_inference_backend_availability()
    
    # Determine overall status
    if not health_status["components"]["database"]:
        health_status["status"] = "unhealthy"
    elif not all(health_status["components"].values()):
        health_status["status"] = "degraded"
    
    status_code = 200 if health_status["status"] != "unhealthy" else 503
    return jsonify(health_status), status_code


@app.route("/ready")
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


@app.route("/monitoring")
def monitoring():
    """Live monitoring page"""
    return render_template("monitoring.html")


@app.route("/configuration")
def configuration():
    """Legacy configuration route kept for compatibility."""
    return redirect(url_for("settings"))


@app.route("/users")
def users():
    """User management"""
    with get_db_connection() as conn:
        users_list = conn.execute(
            "SELECT * FROM users ORDER BY created_at DESC"
        ).fetchall()
    return render_template("users.html", users=users_list)


@app.route("/logs")
def logs():
    """Detection logs and history"""
    with get_db_connection() as conn:
        detections = conn.execute(
            """
            SELECT * FROM detection_logs
            ORDER BY timestamp DESC
            LIMIT 100
        """
        ).fetchall()
    return render_template("logs.html", detections=detections)


@app.route("/logs/<int:log_id>")
def log_detail(log_id):
    """Single detection event detail view"""
    with get_db_connection() as conn:
        log_row = conn.execute(
            "SELECT * FROM detection_logs WHERE id = ?", (log_id,)
        ).fetchone()

    if not log_row:
        abort(404)

    log_data = dict(log_row)

    decision_details = {}
    if log_data.get("decision_details"):
        try:
            decision_details = json.loads(log_data["decision_details"])
        except json.JSONDecodeError:
            decision_details = {"raw": log_data["decision_details"]}

    image_url = None
    if log_data.get("image_path"):
        image_url = url_for(
            "api_serve_image", image_path=log_data["image_path"]
        )

    confidence_percent = None
    try:
        if log_data.get("confidence") is not None:
            confidence_percent = float(log_data["confidence"]) * 100
    except (TypeError, ValueError):
        confidence_percent = None

    log_data["decision_details"] = decision_details

    return render_template(
        "log_detail.html",
        log=log_data,
        image_url=image_url,
        confidence_percent=confidence_percent,
    )


@app.route("/settings")
def settings():
    """System settings"""
    return render_template("settings.html")


# API Routes
@app.route("/api/start_monitoring", methods=["POST"])
def api_start_monitoring():
    """Start monitoring via API"""
    global camera_agent
    if not camera_agent:
        camera_agent = WebCameraAgent()
        if not camera_agent.initialize():
            error_message = (
                camera_agent.last_error or "Failed to initialize camera"
            )
            camera_agent = None
            return jsonify({"success": False, "error": error_message})

    if camera_agent.start_monitoring():
        return jsonify({"success": True, "message": "Monitoring started"})
    else:
        error_message = camera_agent.last_error or "Monitoring already active"
        return jsonify({"success": False, "error": error_message})


@app.route("/api/stop_monitoring", methods=["POST"])
def api_stop_monitoring():
    """Stop monitoring via API"""
    if camera_agent:
        camera_agent.stop_monitoring()
        return jsonify({"success": True, "message": "Monitoring stopped"})
    else:
        return jsonify({"success": False, "error": "No active monitoring"})


@app.route("/api/status")
def api_status():
    """Get comprehensive system status including circuit breaker state."""
    # Get agentic mode from prompt config
    prompt_config = camera_agent.get_active_prompt_config() if camera_agent else {}
    
    status = {
        "monitoring_active": (
            camera_agent.is_monitoring if camera_agent else False
        ),
        "camera_available": check_camera_availability(),
        "inference_backend": INFERENCE_BACKEND,
        "inference_available": check_inference_backend_availability(),
        "agentic_mode": prompt_config.get("agentic_mode", False),
        "stats": camera_agent._serialize_stats() if camera_agent else {},
    }
    
    # Add circuit breaker status if agent is running
    if camera_agent and camera_agent.agent:
        agent = camera_agent.agent
        if hasattr(agent, 'circuit_breaker'):
            status["circuit_breaker"] = agent.circuit_breaker.get_stats()
        
        # Add memory summary
        if hasattr(agent, '_agent_memory'):
            memory_snapshot = agent.get_memory_snapshot(limit=5)
            status["recent_events_count"] = memory_snapshot.get("counts", {}).get("total", 0)

    return jsonify(status)


@app.route("/api/circuit_breaker/reset", methods=["POST"])
def api_reset_circuit_breaker():
    """Reset the circuit breaker to closed state.
    
    Use this when the VLM service has recovered and you want to
    immediately resume analysis without waiting for the recovery timeout.
    """
    if not camera_agent or not camera_agent.agent:
        return jsonify({
            "success": False,
            "error": "No active monitoring agent"
        }), 400
    
    agent = camera_agent.agent
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


@app.route("/api/agent/memory", methods=["GET", "DELETE"])
def api_agent_memory():
    """Expose recent agent memory and summarised activity."""
    agent = getattr(camera_agent, "agent", None) if camera_agent else None
    
    if request.method == "DELETE":
        # Clear agent memory
        if agent and hasattr(agent, 'clear_memory'):
            agent.clear_memory()
            return jsonify({"success": True, "message": "Agent memory cleared"})
        return jsonify({"success": True, "message": "No memory to clear"})
    
    # GET method
    limit = request.args.get("limit", type=int)
    agent = getattr(camera_agent, "agent", None) if camera_agent else None

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
        return jsonify(
            {
                "success": True,
                "summary": "No agent memory available yet.",
                "memory": {
                    "events": [],
                    "counts": empty_counts,
                    "last_event": None,
                },
            }
        )

    snapshot = agent.get_memory_snapshot(limit=limit)
    summary = agent.summarise_recent_events(limit=limit)
    return jsonify({"success": True, "summary": summary, "memory": snapshot})


@app.route("/api/agent/prompt", methods=["GET", "POST"])
def api_agent_prompt():
    """Get or set the active monitoring prompt configuration.
    
    GET: Returns current task_type, custom_prompt, and alerts_enabled
    POST: Sets the active prompt for monitoring
        - task_type: One of package_detection, ppe_detection, person_counting, scene_description, custom
        - custom_prompt: Required for custom task type
        - alerts_enabled: Whether to show visual alerts and send notifications
    """
    if not camera_agent:
        return jsonify({"success": False, "error": "Camera agent not available"}), 500
    
    if request.method == "GET":
        config = camera_agent.get_active_prompt_config()
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
    
    # Get current config before update for comparison
    old_config = camera_agent.get_active_prompt_config()
    old_task_type = old_config.get("task_type")
    old_prompt = old_config.get("custom_prompt")
    
    # Check if prompt is actually changing (will clear queue)
    prompt_changing = (old_task_type != task_type_str or old_prompt != custom_prompt)
    
    # Set the active prompt (this will clear queue if prompt changes)
    camera_agent.set_active_prompt(
        task_type=task_type,
        custom_prompt=custom_prompt,
        alerts_enabled=alerts_enabled,
        agentic_mode=agentic_mode,
    )
    
    # Get updated config with new version
    new_config = camera_agent.get_active_prompt_config()
    
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


@app.route("/api/users", methods=["GET", "POST"])
def api_users():
    """User management API"""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        required_fields = [
            field for field in ("email", "name") if not data.get(field)
        ]
        if required_fields:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Missing required fields: {', '.join(required_fields)}",
                    }),
                400,
            )

        try:
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO users (email, name, role)
                    VALUES (?, ?, ?)
                    """,
                    (data["email"], data["name"], data.get("role", "user")),
                )
        except sqlite3.IntegrityError as exc:
            return (
                jsonify(
                    {"success": False, "error": f"Unable to add user: {exc}"}
                ),
                400,
            )
        except sqlite3.Error as exc:
            return (
                jsonify({"success": False, "error": f"Database error: {exc}"}),
                500,
            )

        update_email_recipients()
        return jsonify({"success": True, "message": "User added successfully"})

    with get_db_connection() as conn:
        users = conn.execute("SELECT * FROM users WHERE active = 1").fetchall()
    return jsonify([dict(user) for user in users])


@app.route("/api/users/<int:user_id>", methods=["DELETE", "PUT"])
def api_user_modify(user_id: int):
    """Delete or update user.
    
    Args:
        user_id: The ID of the user to modify.
        
    Returns:
        JSON response indicating success or failure.
    """
    if request.method == "DELETE":
        try:
            with get_db_connection() as conn:
                conn.execute(
                    "UPDATE users SET active = 0 WHERE id = ?", (user_id,)
                )
        except sqlite3.Error as exc:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Failed to deactivate user: {exc}",
                    }
                ),
                500,
            )

        update_email_recipients()
        return jsonify({"success": True, "message": "User deactivated"})

    # PUT method
    data = request.get_json(silent=True) or {}
    required_fields = [
        field for field in ("email", "name") if not data.get(field)
    ]
    if required_fields:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Missing required fields: {', '.join(required_fields)}",
                }),
            400,
        )

    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE users
                SET email = ?, name = ?, role = ?
                WHERE id = ?
                """,
                (
                    data["email"],
                    data["name"],
                    data.get("role", "user"),
                    user_id,
                ),
            )
    except sqlite3.IntegrityError as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Unable to update user: {exc}",
                }
            ),
            400,
        )
    except sqlite3.Error as exc:
        return (
            jsonify({"success": False, "error": f"Database error: {exc}"}),
            500,
        )

    update_email_recipients()
    return jsonify(
        {"success": True, "message": "User updated successfully"}
    )


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    """Configuration management API"""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid configuration payload",
                    }
                ),
                400,
            )
        try:
            persisted = save_camera_config(data)

            # Log configuration change
            try:
                with get_db_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO config_history (config_type, changes)
                        VALUES (?, ?)
                        """,
                        ("camera_config", json.dumps(persisted)),
                    )
            except sqlite3.Error as db_exc:
                logger.error(
                    "Failed to record configuration history: %s", db_exc
                )

            # Refresh running agent configuration if active
            if camera_agent and camera_agent.agent:
                try:
                    camera_agent.agent.apply_config(persisted)
                    camera_agent.refresh_configuration(persisted)
                except Exception as exc:
                    logger.error(
                        f"Failed to apply updated configuration to agent: {exc}"
                    )

            notifications_cfg = (persisted or {}).get("notifications", {})
            email_cfg = (
                notifications_cfg.get("email", {})
                if isinstance(notifications_cfg, dict)
                else {}
            )
            if _coerce_bool_config(
                (email_cfg or {}).get("auto_sync_users"), True
            ):
                update_email_recipients()

            return jsonify(
                {
                    "success": True,
                    "message": "Configuration updated",
                    "config": persisted,
                }
            )
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    else:
        try:
            config = load_camera_config()
        except (
            Exception
        ) as exc:  # pragma: no cover - surface config load issues via API
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Failed to load configuration: {exc}",
                    }
                ),
                500,
            )

        sanitized = _sanitize_config_payload(config)
        return jsonify(sanitized)


@app.route("/api/config/defaults", methods=["GET"])
def api_config_defaults():
    """Return the default configuration without persisting it."""
    defaults = StreamlinedAgent.default_config()
    sanitized = _sanitize_config_payload(defaults)
    return jsonify({"success": True, "config": sanitized})


@app.route("/api/config/reset", methods=["POST"])
def api_config_reset():
    """Reset the configuration to defaults and persist the change."""
    defaults = reset_camera_config()

    if camera_agent and camera_agent.agent:
        try:
            camera_agent.agent.apply_config(defaults)
            camera_agent.refresh_configuration(defaults)
        except Exception as exc:
            logger.error(
                f"Failed to apply default configuration to agent: {exc}"
            )

    return jsonify(
        {
            "success": True,
            "message": "Configuration reset to defaults",
            "config": defaults,
        }
    )


@app.route("/api/notifications/recipients", methods=["GET", "PUT"])
def api_notification_recipients():
    """Manage notification email recipients."""
    if request.method == "GET":
        try:
            config = load_camera_config()
        except (
            Exception
        ) as exc:  # pragma: no cover - propagate load errors to client
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Failed to load configuration: {exc}",
                    }
                ),
                500,
            )

        recipients = extract_recipients(config)
        return jsonify({"success": True, "recipients": recipients})

    payload = request.get_json(silent=True) or {}
    recipients = payload.get("recipients", [])
    if not isinstance(recipients, list):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Recipients must be provided as a list",
                }
            ),
            400,
        )

    sanitized = []
    for email in recipients:
        if not isinstance(email, str):
            continue
        candidate = email.strip()
        if not candidate or "@" not in candidate:
            continue
        sanitized.append(candidate)

    try:
        config = load_camera_config()
    except (
        Exception
    ) as exc:  # pragma: no cover - propagate load errors to client
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Failed to load configuration: {exc}",
                }
            ),
            500,
        )
    apply_recipients(config, sanitized)
    persisted = save_camera_config(config)

    if camera_agent and camera_agent.agent:
        try:
            camera_agent.agent.apply_config(persisted)
            camera_agent.refresh_configuration(persisted)
        except Exception as exc:
            logger.error(f"Failed to apply recipient update to agent: {exc}")

    return jsonify(
        {"success": True, "recipients": extract_recipients(persisted)}
    )


@app.route("/api/test_camera")
def api_test_camera():
    """Test camera functionality"""
    try:
        # Try to open camera temporarily for testing
        cap = cv2.VideoCapture(_safe_int_env("CAMERA_INDEX", 0))
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return jsonify(
                    {"success": True, "message": "Camera test successful"}
                )
            else:
                return jsonify(
                    {"success": False, "error": "Failed to capture frame"}
                )
        else:
            return jsonify({"success": False, "error": "Cannot open camera"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/test_inference")
def api_test_inference():
    """Test the configured inference backend (vLLM or Ollama)"""
    backend = INFERENCE_BACKEND.lower()
    try:
        if backend == "vllm":
            response = requests.get(
                f"{VLLM_BASE_URL}/v1/models", timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            return jsonify({
                "success": True,
                "backend": "vllm",
                "message": "vLLM connection successful",
                "models": models
            })
        else:
            response = requests.get(
                _ollama_endpoint("api/version"), timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return jsonify({
                "success": True,
                "backend": "ollama",
                "message": "Ollama connection successful"
            })
    except requests.RequestException as exc:
        return jsonify({
            "success": False,
            "backend": backend,
            "error": f"{backend} connectivity failed: {exc}"
        })


@app.route("/api/test_vllm")
def api_test_vllm():
    """Test vLLM connection"""
    try:
        response = requests.get(
            f"{VLLM_BASE_URL}/v1/models", timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        models = [m.get("id", "") for m in data.get("data", [])]
        return jsonify({
            "success": True,
            "message": "vLLM connection successful",
            "models": models,
            "url": VLLM_BASE_URL
        })
    except requests.RequestException as exc:
        return jsonify(
            {"success": False, "error": f"vLLM connectivity failed: {exc}"}
        )


@app.route("/api/test_ollama")
def api_test_ollama():
    """Test Ollama connection"""
    try:
        response = requests.get(
            _ollama_endpoint("api/version"), timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return jsonify(
            {"success": True, "message": "Ollama connection successful"}
        )
    except requests.RequestException as exc:
        return jsonify(
            {"success": False, "error": f"Ollama connectivity failed: {exc}"}
        )


@app.route("/api/ollama_models")
def api_ollama_models():
    """List available Ollama models (vision-capable only)"""
    try:
        response = requests.get(
            _ollama_endpoint("api/tags"), timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        # Filter for vision-capable models and extract names
        vision_keywords = ["llava", "gemma3", "qwen", "minicpm", "llama3.2-vision", "moondream"]
        vision_models = [
            m.get("name", "") for m in models
            if any(kw in m.get("name", "").lower() for kw in vision_keywords)
        ]
        return jsonify({"success": True, "models": vision_models})
    except requests.RequestException as exc:
        return jsonify({"success": False, "error": str(exc), "models": []})


@app.route("/api/analyze_prompt", methods=["POST"])
def api_analyze_prompt():
    """Analyze the current camera frame with a dynamic prompt.
    
    Accepts:
        task_type: One of 'package_detection', 'ppe_detection', 'person_counting', 
                   'scene_description', 'custom'
        custom_prompt: Optional custom prompt for analysis (required for 'custom' task)
    
    Returns:
        JSON with analysis result including detected, confidence, reasoning, details
    """
    try:
        data = request.get_json() or {}
        task_type_str = data.get("task_type", "package_detection")
        custom_prompt = data.get("custom_prompt", "")
        
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
        
        # Capture current frame from camera publisher (not direct camera access)
        try:
            publisher = get_camera_publisher()
            latest_frame = publisher.get_latest_frame()
            
            if not latest_frame or not latest_frame.image_b64:
                return jsonify({
                    "success": False,
                    "error": "No frames available from camera"
                }), 500
            
            # Decode base64 frame to numpy array
            import base64
            frame_bytes = base64.b64decode(latest_frame.image_b64)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({
                    "success": False,
                    "error": "Failed to decode camera frame"
                }), 500
        except Exception as camera_err:
            logger.error("Camera access failed: %s", camera_err)
            return jsonify({
                "success": False,
                "error": f"Camera access failed: {str(camera_err)}"
            }), 500
        
        # Initialize VLM client with configuration
        try:
            config = load_camera_config()
            vlm_client = _create_vlm_client_from_config(config)
        except Exception as e:
            logger.error("Failed to initialize VLM client: %s", e)
            return jsonify({
                "success": False,
                "error": f"Failed to initialize VLM client: {str(e)}"
            }), 500
        
        # Run analysis with the specified task type
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
        
        # Parse the custom prompt for tool/action intents
        actions_executed = []
        if task_type == TaskType.CUSTOM and custom_prompt:
            # Check for email action intent
            email_match = _extract_email_intent(custom_prompt)
            if email_match and result.detected:
                try:
                    email_result = _execute_email_action(
                        email_to=email_match,
                        detection_result=result,
                        user_prompt=custom_prompt,
                        image_data=frame_bytes,  # Attach the detection image
                    )
                    actions_executed.append({
                        "action": "send_email",
                        "status": "success",
                        "details": email_result,
                    })
                except Exception as email_err:
                    logger.error("Email action failed: %s", email_err)
                    actions_executed.append({
                        "action": "send_email",
                        "status": "error",
                        "error": str(email_err),
                    })
        
        # Return the analysis result
        return jsonify({
            "success": True,
            "result": {
                "task_type": task_type_str,
                "detected": result.detected,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "details": result.details,
                "actions_executed": actions_executed,
            }
        })
        
    except Exception as e:
        logger.exception("Error in analyze_prompt API")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/analyze_single", methods=["POST"])
def api_analyze_single():
    """Run single-frame inference on the latest available frame.
    
    This endpoint provides on-demand inference that:
    1. Bypasses any batched/queued frames from continuous monitoring
    2. Runs immediately on the latest frame from the camera
    3. Does not block or delay scheduled inference
    4. Uses the active prompt configuration unless overridden
    
    Accepts (all optional):
        task_type: Override task type. Uses active config if not provided.
        custom_prompt: Override custom prompt. Uses active config if not provided.
    
    Returns:
        JSON with inference result including detected, confidence, reasoning, details.
        Returns error if no camera_agent is active or inference fails.
    """
    global camera_agent
    
    if not camera_agent:
        return jsonify({
            "success": False,
            "error": "Camera agent not initialized. Start monitoring first."
        }), 400
    
    data = request.get_json() or {}
    
    # Parse optional overrides
    task_type = None
    task_type_str = data.get("task_type")
    if task_type_str:
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
    
    custom_prompt = data.get("custom_prompt")
    
    # Validate custom prompt for custom task
    if task_type == TaskType.CUSTOM and not custom_prompt:
        return jsonify({
            "success": False,
            "error": "Custom task requires a custom_prompt"
        }), 400
    
    try:
        # Run single-frame inference (bypasses queue, runs immediately)
        event = camera_agent.analyze_single_frame(
            task_type=task_type,
            custom_prompt=custom_prompt,
        )
        
        if event is None:
            error_msg = camera_agent.last_error or "Inference failed - no result returned"
            return jsonify({
                "success": False,
                "error": error_msg
            }), 500
        
        # Get the active prompt config for response context
        prompt_config = camera_agent.get_active_prompt_config()
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


@app.route("/api/analyze_agentic", methods=["POST"])
def api_analyze_agentic():
    """Analyze the current camera frame with agentic tool calling.
    
    This endpoint enables the LLM to autonomously decide which tools to call
    based on its analysis of the image. The LLM can send emails, save evidence,
    log events, and more without explicit instructions.
    
    Accepts:
        task_type: One of 'package_detection', 'ppe_detection', 'person_counting', 
                   'scene_description', 'custom'
        custom_prompt: Optional custom prompt for analysis
        recipients: Optional list of email recipients (uses config default if not provided)
    
    Returns:
        JSON with analysis result and list of tools that were called
    """
    from agent_runtime.tools import ToolExecutor
    
    try:
        data = request.get_json() or {}
        task_type_str = data.get("task_type", "package_detection")
        # Accept both "prompt" and "custom_prompt" for convenience
        custom_prompt = data.get("prompt") or data.get("custom_prompt") or ""
        recipients = data.get("recipients", [])
        
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
        
        # Capture current frame from camera publisher
        try:
            publisher = get_camera_publisher()
            latest_frame = publisher.get_latest_frame()
            
            if not latest_frame or not latest_frame.image_b64:
                return jsonify({
                    "success": False,
                    "error": "No frames available from camera"
                }), 500
            
            # Decode base64 frame to numpy array
            import base64
            frame_bytes = base64.b64decode(latest_frame.image_b64)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({
                    "success": False,
                    "error": "Failed to decode camera frame"
                }), 500
        except Exception as camera_err:
            logger.error("Camera access failed: %s", camera_err)
            return jsonify({
                "success": False,
                "error": f"Camera access failed: {str(camera_err)}"
            }), 500
        
        # Initialize VLM client with configuration
        try:
            config = load_camera_config()
            vlm_client = _create_vlm_client_from_config(config)
            
            # Get recipients from config if not provided
            if not recipients:
                email_cfg = config.get("notifications", {}).get("email", {})
                recipients = email_cfg.get("recipients", [])
            
        except Exception as e:
            logger.error("Failed to initialize VLM client: %s", e)
            return jsonify({
                "success": False,
                "error": f"Failed to initialize VLM client: {str(e)}"
            }), 500
        
        # Create tool executor
        tool_executor = ToolExecutor()
        
        # Run agentic analysis with tool calling
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
        
        # Return the analysis result with tool information
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


@app.route("/api/analyze_uploaded_image", methods=["POST"])
def api_analyze_uploaded_image():
    """Analyze an uploaded image using the agent.
    
    This endpoint allows testing the agent with a user-uploaded image instead of
    the camera feed. The uploaded image takes priority over scheduled monitoring.
    After analysis completes, monitoring resumes automatically if it was active.
    
    Accepts (multipart/form-data):
        image: The image file to analyze (required)
        task_type: Task type for analysis (optional, defaults to active config)
        custom_prompt: Custom prompt for analysis (optional)
        agentic_mode: Whether to use agentic mode (optional, defaults to active config)
    
    Returns:
        JSON with analysis result including detected, confidence, reasoning, and tool calls.
    """
    global camera_agent
    from agent_runtime.tools import ToolExecutor
    
    # Check if image file is provided
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
    
    # Read and decode the image
    try:
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                "success": False,
                "error": "Failed to decode image. Please upload a valid image file (JPEG, PNG, etc.)"
            }), 400
    except Exception as e:
        logger.error("Failed to process uploaded image: %s", e)
        return jsonify({
            "success": False,
            "error": f"Failed to process image: {str(e)}"
        }), 400
    
    # Get form parameters
    task_type_str = request.form.get("task_type")
    custom_prompt = request.form.get("custom_prompt") or request.form.get("prompt")
    agentic_mode_str = request.form.get("agentic_mode")
    
    # Map task type string to enum
    task_type = None
    if task_type_str:
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
                "error": f"Invalid task_type: {task_type_str}. Valid options: {list(task_type_map.keys())}"
            }), 400
    
    # Determine if agentic mode should be used
    use_agentic = None
    if agentic_mode_str is not None:
        use_agentic = agentic_mode_str.lower() in ('true', '1', 'yes', 'on')
    
    # Track if monitoring was active (to resume after analysis)
    was_monitoring = False
    if camera_agent and camera_agent.is_monitoring:
        was_monitoring = True
        logger.info("📤 Uploaded image analysis requested - monitoring will continue after")
    
    try:
        # Try to use the existing camera_agent if available
        if camera_agent:
            # Get active config if task_type not specified
            if task_type is None:
                active_config = camera_agent.get_active_prompt_config()
                task_type_str = active_config.get("task_type", "package_detection")
                task_type_map = {
                    "package_detection": TaskType.PACKAGE_DETECTION,
                    "ppe_detection": TaskType.PPE_DETECTION,
                    "person_counting": TaskType.PERSON_COUNTING,
                    "scene_description": TaskType.SCENE_DESCRIPTION,
                    "custom": TaskType.CUSTOM,
                }
                task_type = task_type_map.get(task_type_str, TaskType.PACKAGE_DETECTION)
                if custom_prompt is None:
                    custom_prompt = active_config.get("custom_prompt", "")
                if use_agentic is None:
                    use_agentic = active_config.get("agentic_mode", False)
            
            agent = camera_agent.agent
            if agent is None:
                # Initialize if needed
                if not camera_agent.initialize():
                    return jsonify({
                        "success": False,
                        "error": f"Failed to initialize agent: {camera_agent.last_error}"
                    }), 500
                agent = camera_agent.agent
        else:
            # No camera_agent - create a temporary one
            config = load_camera_config()
            vlm_client = _create_vlm_client_from_config(config)
            
            # Default to package detection if not specified
            if task_type is None:
                task_type = TaskType.PACKAGE_DETECTION
                task_type_str = "package_detection"
            if use_agentic is None:
                use_agentic = False
            
            # Create a minimal agent for analysis
            from camera_agent import StreamlinedAgent, CircuitBreaker
            circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=120.0)
            agent = StreamlinedAgent(
                config=config,
                vlm_client=vlm_client,
                circuit_breaker=circuit_breaker,
            )
        
        # Create metadata for the analysis
        metadata = {
            'frame_number': 0,
            'timestamp': time.time(),
            'uploaded_image': True,
            'filename': image_file.filename,
            'image_data': image_bytes,
        }
        
        start_time = time.time()
        
        # Run the analysis
        if use_agentic:
            # Get recipients from config
            config = load_camera_config()
            email_cfg = config.get("notifications", {}).get("email", {})
            recipients = email_cfg.get("recipients", [])
            
            # Use agentic mode with tool calling
            logger.info("Running agentic analysis on uploaded image (task_type=%s)", task_type)
            event = agent.analyze_agentic(
                frame=frame,
                task_type=task_type,
                custom_prompt=custom_prompt,
                recipients=recipients,
            )
        else:
            # Use standard analysis
            # If custom_prompt is provided, ensure task_type is CUSTOM to use it
            if custom_prompt and task_type != TaskType.CUSTOM:
                logger.info("Custom prompt provided - switching to CUSTOM task type")
                task_type = TaskType.CUSTOM
                task_type_str = "custom"
            
            logger.info("Running standard analysis on uploaded image (task_type=%s, has_prompt=%s)", 
                       task_type, bool(custom_prompt))
            event = agent.analyze_with_prompt(
                frame=frame,
                task_type=task_type,
                custom_prompt=custom_prompt or "",
                metadata=metadata,
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        if event is None:
            logger.warning("Analysis returned None for uploaded image")
            return jsonify({
                "success": False,
                "error": "Analysis returned no result. Check if the AI server is running and accessible."
            }), 500
        
        # Log the result to database if we have a camera_agent
        if camera_agent:
            try:
                _log_detection_to_db(event, metadata)
                # Emit real-time update
                socketio.emit('detection', {
                    'event': event.to_dict(),
                    'stats': camera_agent._serialize_stats() if hasattr(camera_agent, '_serialize_stats') else {},
                    'source': 'uploaded_image',
                })
            except Exception as log_err:
                logger.warning("Failed to log uploaded image analysis: %s", log_err)
        
        logger.info(
            "📤 Uploaded image analysis completed in %.1fms (agentic=%s, detected=%s)",
            latency_ms,
            use_agentic,
            event.detected,
        )
        
        # Build response
        response = {
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
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.exception("Error analyzing uploaded image")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/tools", methods=["GET"])
def api_list_tools():
    """List all available tools for agentic analysis.
    
    Returns:
        JSON with list of tool definitions including name, description, and parameters
    """
    from agent_runtime.tools import TOOL_REGISTRY
    
    tools = []
    for tool in TOOL_REGISTRY.values():
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "required_params": tool.required_params,
        })
    
    return jsonify({
        "success": True,
        "tools": tools,
        "count": len(tools),
    })


@app.route("/api/recent_images")
def api_recent_images():
    """Get recent detection images"""
    try:
        detected_images = _collect_images_from_directory(
            DETECTED_IMAGES_DIR, 10
        )
        processed_images = _collect_images_from_directory(
            PROCESSED_FRAMES_DIR, 5
        )

        return jsonify(
            {"detected": detected_images, "processed": processed_images}
        )
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/logs", methods=["GET", "DELETE"])
def api_logs():
    """Get or clear detection logs.
    
    GET: Retrieve detection logs with pagination support.
        Query params:
            - page: Page number (default: 1)
            - per_page: Items per page (default: 50, max: 200)
            - detected_only: If 'true', only show detected events
    DELETE: Clear all detection logs from the database.
    
    Returns:
        JSON response with logs list and pagination info (GET) or success message (DELETE).
    """
    if request.method == "DELETE":
        try:
            with get_db_connection() as conn:
                conn.execute("DELETE FROM detection_logs")
            return jsonify({"success": True, "message": "All logs cleared"})
        except sqlite3.Error as exc:
            return (
                jsonify(
                    {"success": False, "error": f"Failed to clear logs: {exc}"}
                ),
                500,
            )

    # GET method with pagination
    try:
        page = max(1, request.args.get("page", 1, type=int))
        per_page = min(200, max(1, request.args.get("per_page", 50, type=int)))
        detected_only = request.args.get("detected_only", "").lower() == "true"
        offset = (page - 1) * per_page
        
        with get_db_connection() as conn:
            # Get total count for pagination
            count_query = "SELECT COUNT(*) FROM detection_logs"
            if detected_only:
                count_query += " WHERE confidence > 0"
            total_count = conn.execute(count_query).fetchone()[0]
            
            # Build query with filters
            query = """
                SELECT id, timestamp, confidence, response, image_path,
                       frame_number, reason, vision_description, decision_details, tool_trace
                FROM detection_logs
            """
            if detected_only:
                query += " WHERE confidence > 0"
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            
            logs = conn.execute(query, (per_page, offset)).fetchall()

        logs_list = []
        for log in logs:
            decision_details = {}
            if log["decision_details"]:
                try:
                    decision_details = json.loads(log["decision_details"])
                except json.JSONDecodeError:
                    decision_details = {"raw": log["decision_details"]}
            
            tool_trace = []
            try:
                tool_trace_raw = log["tool_trace"] if "tool_trace" in log.keys() else None
                if tool_trace_raw:
                    tool_trace = json.loads(tool_trace_raw)
            except (json.JSONDecodeError, KeyError):
                tool_trace = []

            is_agentic = decision_details.get("classification") == "AGENTIC_ANALYSIS"
            
            # Use the detected value from decision_details if available
            # Fall back to confidence-based inference only for old records without detected field
            if "detected" in decision_details:
                detected = decision_details.get("detected", False)
            else:
                # Legacy fallback for old records without detected field
                detected = log["confidence"] is not None and log["confidence"] > 0
            
            logs_list.append(
                {
                    "id": log["id"],
                    "timestamp": log["timestamp"],
                    "confidence": log["confidence"],
                    "response": log["response"],
                    "image_path": log["image_path"],
                    "frame_number": log["frame_number"],
                    "reason": log["reason"],
                    "vision_description": log["vision_description"],
                    "decision_details": decision_details,
                    "tool_trace": tool_trace,
                    "agentic_mode": is_agentic,
                    "detected": detected,
                }
            )

        total_pages = (total_count + per_page - 1) // per_page if per_page > 0 else 1
        
        return jsonify({
            "success": True,
            "logs": logs_list,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            }
        })
    except sqlite3.Error as exc:
        return (
            jsonify({"success": False, "error": f"Database error: {exc}"}),
            500,
        )


@app.route("/api/logs/<int:log_id>", methods=["DELETE"])
def api_delete_log(log_id):
    """Delete a specific log entry"""
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM detection_logs WHERE id = ?", (log_id,))
        return jsonify({"success": True, "message": "Log deleted"})
    except sqlite3.Error as exc:
        return (
            jsonify(
                {"success": False, "error": f"Failed to delete log: {exc}"}
            ),
            500,
        )


@app.route("/api/logs/export", methods=["POST"])
def api_export_logs():
    """Export logs as CSV"""
    try:
        data = request.get_json(silent=True) or {}
        format_type = data.get("format", "csv")

        with get_db_connection() as conn:
            logs = conn.execute(
                """
                SELECT timestamp, confidence, response, image_path, frame_number,
                       reason, vision_description, decision_details
                FROM detection_logs
                ORDER BY timestamp DESC
            """
            ).fetchall()

        if format_type == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(
                [
                    "Timestamp",
                    "Confidence",
                    "Response",
                    "Image Path",
                    "Frame Number",
                    "Reason",
                    "Vision Description",
                    "Decision Details",
                ]
            )

            for log in logs:
                writer.writerow(
                    [
                        log["timestamp"],
                        log["confidence"],
                        log["response"],
                        log["image_path"],
                        log["frame_number"],
                        log["reason"],
                        log["vision_description"],
                        log["decision_details"],
                    ]
                )

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename=detection_logs_{timestamp_str}.csv'
                    )
                },
            )
        else:
            logs_list = [dict(log) for log in logs]
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            return Response(
                json.dumps(logs_list, indent=2),
                mimetype="application/json",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename=detection_logs_{timestamp_str}.json'
                    )
                },
            )
    except sqlite3.Error as exc:
        return (
            jsonify(
                {"success": False, "error": f"Failed to export logs: {exc}"}
            ),
            500,
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/image/<path:image_path>")
def api_serve_image(image_path):
    """Serve detection images.
    
    Handles various path formats:
    - Absolute paths: /app/data/detected_images/image.jpg
    - Relative paths: detected_images/image.jpg
    - Filename only: image.jpg
    """
    try:
        # Sanitize and resolve the image path
        candidate_path = Path(image_path)
        
        # Check if path is within allowed directories
        detected_dir = DETECTED_IMAGES_DIR.resolve()
        processed_dir = PROCESSED_FRAMES_DIR.resolve()
        
        # Try multiple resolution strategies
        paths_to_try = []
        
        if candidate_path.is_absolute():
            # Absolute path - use as-is
            paths_to_try.append(candidate_path.resolve())
        else:
            # Relative path - try multiple base directories
            # 1. Try relative to DATA_DIR first
            paths_to_try.append((DATA_DIR / candidate_path).resolve())
            
            # 2. Try relative to current working directory
            paths_to_try.append(Path.cwd() / candidate_path)
            
            # 3. Try the filename directly in detected_images
            paths_to_try.append(detected_dir / candidate_path.name)
            
            # 4. Try the filename in processed_frames
            paths_to_try.append(processed_dir / candidate_path.name)
            
            # 5. If path starts with 'detected_images/' strip it and try in DETECTED_IMAGES_DIR
            path_str = str(candidate_path)
            if path_str.startswith("detected_images/"):
                stripped = path_str[len("detected_images/"):]
                paths_to_try.append(detected_dir / stripped)
            if path_str.startswith("processed_frames/"):
                stripped = path_str[len("processed_frames/"):]
                paths_to_try.append(processed_dir / stripped)

        # Find the first valid path
        resolved_path = None
        for try_path in paths_to_try:
            try:
                resolved = try_path.resolve()
                if resolved.exists() and resolved.is_file():
                    # Security check - ensure path is within allowed directories
                    if (
                        resolved.is_relative_to(detected_dir)
                        or resolved.is_relative_to(processed_dir)
                        or resolved.is_relative_to(DATA_DIR.resolve())
                    ):
                        resolved_path = resolved
                        break
            except Exception:
                continue

        if resolved_path:
            return send_file(resolved_path, mimetype="image/jpeg")
        else:
            logger.warning(f"Image not found: {image_path}. Tried paths: {[str(p) for p in paths_to_try[:3]]}")
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        logger.error(f"Error serving image {image_path}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/system/status")
def api_system_status():
    """Get system status information including GPU metrics"""
    try:
        import psutil
        import subprocess

        # Calculate uptime
        uptime_seconds = int(time.time() - APP_START_TIME)

        status = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "camera_available": check_camera_availability(),
            "inference_available": check_inference_backend_availability(),
            "inference_backend": INFERENCE_BACKEND,
            "uptime_seconds": uptime_seconds,
        }
        
        # Get GPU stats for Jetson (using sysfs) or desktop (nvidia-smi)
        try:
            gpu_info = _get_jetson_gpu_stats()
            if gpu_info:
                status["gpu"] = gpu_info
            else:
                # Fall back to nvidia-smi for non-Jetson systems
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
                        gpu_util = float(parts[0])
                        mem_used = float(parts[1])
                        mem_total = float(parts[2])
                        temp = float(parts[3])
                        power = float(parts[4]) if parts[4] != "[N/A]" else None
                        gpu_name = parts[5]
                        
                        status["gpu"] = {
                            "name": gpu_name,
                            "utilization": gpu_util,
                            "memory_used_mb": mem_used,
                            "memory_total_mb": mem_total,
                            "memory_percent": round((mem_used / mem_total) * 100, 1) if mem_total > 0 else 0,
                            "temperature": temp,
                            "power_watts": power,
                        }
                else:
                    status["gpu"] = None
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            # GPU stats unavailable
            status["gpu"] = None
            logger.debug(f"GPU stats unavailable: {e}")
        
        return jsonify({"success": True, "status": status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/system/environment", methods=["GET", "POST"])
def api_system_environment():
    """Get or update system environment variables.
    
    GET: Retrieve current environment variable values.
    POST: Update environment variables (in-memory only, not persisted).
    
    Returns:
        JSON response with environment variables or success message.
    """
    if request.method == "POST":
        try:
            data = request.get_json()
            if not isinstance(data, dict):
                return jsonify({"success": False, "error": "Invalid payload"}), 400
            # Note: This only updates in-memory; for persistence, update .env file
            for key, value in data.items():
                os.environ[key] = str(value)
            return jsonify(
                {
                    "success": True,
                    "message": "Environment variables updated (in-memory only)",
                }
            )
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    # GET method (default)
    try:
        env_vars = {
            "CAMERA_INDEX": os.getenv("CAMERA_INDEX", "0"),
            "INFERENCE_BACKEND": INFERENCE_BACKEND,
            "VLLM_URL": VLLM_BASE_URL if INFERENCE_BACKEND.lower() == "vllm" else "",
            "OLLAMA_URL": OLLAMA_BASE_URL if INFERENCE_BACKEND.lower() != "vllm" else "",
            "VISION_MODEL": os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
            "DIFF_THRESHOLD": os.getenv("DIFF_THRESHOLD", "0.80"),
            "CAPTURE_INTERVAL": os.getenv("CAPTURE_INTERVAL", "5"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        }
        return jsonify({"success": True, "environment": env_vars})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/system/logging", methods=["GET", "POST"])
def api_system_logging():
    """Retrieve or update logging preferences."""
    if request.method == "GET":
        try:
            settings = get_log_settings()
            return jsonify({"success": True, "settings": settings})
        except Exception as exc:
            logger.error(
                "Failed to fetch log settings: %s", exc, exc_info=True
            )
            return (
                jsonify(
                    {"success": False, "error": "Unable to load log settings"}
                ),
                500,
            )

    data = request.get_json(silent=True) or {}
    try:
        settings = persist_log_settings(data)
        return jsonify({"success": True, "settings": settings})
    except Exception as exc:
        logger.error("Failed to update log settings: %s", exc, exc_info=True)
        return (
            jsonify(
                {"success": False, "error": "Failed to save log settings"}
            ),
            500,
        )


@app.route("/api/video_feed")
def video_feed():
    """Stream live video frames from camera publisher.
    
    Returns a multipart JPEG stream suitable for <img> tags.
    Automatically cleans up subscription on client disconnect.
    """

    def generate():
        # Use thread ID for unique subscriber identification
        subscriber_id = f"video_feed_{threading.get_ident()}_{time.time_ns()}"
        subscribed = False
        consecutive_failures = 0
        max_failures = 10

        try:
            publisher = get_camera_publisher()
            if not publisher.subscribe(subscriber_id):
                logger.warning(f"Failed to subscribe video feed: {subscriber_id}")
                return
            subscribed = True

            while consecutive_failures < max_failures:
                try:
                    frame_obj = publisher.get_frame(subscriber_id, timeout=1.0)
                    if not frame_obj:
                        consecutive_failures += 1
                        continue
                    
                    consecutive_failures = 0  # Reset on success

                    # Yield frame in multipart format
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + frame_obj.image_data
                        + b"\r\n"
                    )
                except Exception as frame_err:
                    logger.debug(f"Frame error in video feed: {frame_err}")
                    consecutive_failures += 1
                    time.sleep(0.1)
                    
        except GeneratorExit:
            # Client disconnected - normal cleanup
            pass
        except Exception as e:
            logger.error(f"Video feed error: {e}")
        finally:
            # Cleanup: unsubscribe when done
            if subscribed:
                try:
                    publisher = get_camera_publisher()
                    publisher.unsubscribe(subscriber_id)
                except Exception as cleanup_err:
                    logger.debug(f"Cleanup error: {cleanup_err}")

    return Response(
        generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/capture_frame")
def capture_frame():
    """Get the latest captured frame"""
    try:
        publisher = get_camera_publisher()
        latest_frame = publisher.get_latest_frame()

        if latest_frame:
            return jsonify(
                {
                    "success": True,
                    "image_b64": latest_frame.image_b64,
                    "timestamp": latest_frame.timestamp,
                    "metadata": {
                        "frame_number": latest_frame.frame_number,
                        "source": "camera_publisher",
                    },
                }
            )

        return jsonify({"success": False, "error": "No frames available yet"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# Helper functions
def check_camera_availability() -> bool:
    """Check if camera is available with caching to reduce overhead.
    
    Returns:
        bool: True if camera is available, False otherwise.
    """
    global _last_camera_check, _last_camera_status
    
    current_time = time.time()
    with _camera_check_lock:
        if current_time - _last_camera_check < _camera_check_interval:
            return _last_camera_status
        
        try:
            publisher = get_camera_publisher()
            _last_camera_status = publisher.camera.isOpened() if publisher.camera else False
        except Exception:
            _last_camera_status = False
        
        _last_camera_check = current_time
        return _last_camera_status


def check_ollama_availability():
    """Check if Ollama is available"""
    try:
        response = requests.get(
            _ollama_endpoint("api/version"), timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def check_vllm_availability():
    """Check if vLLM server is available"""
    try:
        response = requests.get(
            f"{VLLM_BASE_URL}/v1/models", timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def check_inference_backend_availability():
    """Check if the configured inference backend is available"""
    backend = INFERENCE_BACKEND.lower()
    if backend == "vllm":
        return check_vllm_availability()
    return check_ollama_availability()


def update_email_recipients():
    """Update email recipients in configuration based on active users"""
    try:
        config = load_camera_config() or {}
        notifications = (
            config.get("notifications", {}) if isinstance(config, dict) else {}
        )
        email_cfg = (
            notifications.get("email", {})
            if isinstance(notifications, dict)
            else {}
        )
        auto_sync_enabled = _coerce_bool_config(
            (email_cfg or {}).get("auto_sync_users"), True
        )
        if not auto_sync_enabled:
            logger.debug(
                "Email auto-sync disabled in configuration; skipping recipient sync"
            )
            return

        # Get active users with valid emails
        with get_db_connection() as conn:
            users = conn.execute(
                'SELECT email FROM users WHERE active = 1 AND email IS NOT NULL AND email != ""'
            ).fetchall()

        seen = set()
        emails: List[str] = []
        for row in users:
            candidate = (row["email"] or "").strip()
            lower = candidate.lower()
            if not candidate or lower in seen:
                continue
            seen.add(lower)
            emails.append(candidate)

        emails.sort(key=lambda value: value.lower())

        current_recipients = extract_recipients(config)
        if emails == current_recipients:
            logger.debug("Email recipients already up to date; no sync needed")
            return

        apply_recipients(config, emails)
        persisted = save_camera_config(config)

        if camera_agent and camera_agent.agent:
            try:
                camera_agent.agent.apply_config(persisted)
                camera_agent.refresh_configuration(persisted)
            except Exception as exc:
                logger.error(
                    f"Failed to apply synced recipients to agent: {exc}"
                )

        logger.info(
            "Email recipients updated successfully via user sync. Recipients: %s",
            ", ".join(emails) if emails else "none",
        )

    except Exception as e:
        logger.error(f"Failed to update email recipients: {e}", exc_info=True)


# Socket.IO events
@socketio.on("connect")
def handle_connect():
    """Handle client connection"""
    logger.debug("Socket client connected")
    # Send current status
    status = {
        "message": "Connected to ZEDEDA Camera Agent",
        "camera_available": check_camera_availability(),
        "monitoring_active": (
            camera_agent.is_monitoring if camera_agent else False
        ),
    }
    emit("connected", status)


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""
    logger.debug("Socket client disconnected")


def broadcast_camera_status():
    """Background thread to broadcast camera availability status"""
    while True:
        try:
            time.sleep(2)  # Update every 2 seconds
            status = {
                "camera_available": check_camera_availability(),
                "monitoring_active": (
                    camera_agent.is_monitoring if camera_agent else False
                ),
                "timestamp": datetime.now().isoformat(),
            }

            # Only send if there are connected clients
            if camera_publisher:
                stats = camera_publisher.get_stats()
                status["publisher_stats"] = {
                    "frames_captured": stats.get("frames_captured", 0),
                    "subscribers": stats.get("subscribers_count", 0),
                }

            socketio.emit("camera_status", status)
        except Exception as e:
            logger.error(f"Error broadcasting camera status: {e}")
            time.sleep(5)


def initialize_camera_publisher():
    """Initialize camera publisher on app startup for live feed viewing"""
    global camera_publisher
    try:
        if camera_publisher and getattr(camera_publisher, "is_running", False):
            logger.info(
                "Camera publisher already running; skipping initialization"
            )
            return True
        camera_index = _safe_int_env("CAMERA_INDEX", 0)
        camera_publisher = get_camera_publisher(
            camera_index=camera_index,
            width=_safe_int_env("CAMERA_WIDTH", 640),
            height=_safe_int_env("CAMERA_HEIGHT", 480),
            fps=_safe_int_env("CAMERA_FPS", 30),
        )

        # Start the publisher so camera feed is always available
        if camera_publisher.start():
            logger.info("✅ Camera publisher started - live feed available")
            return True
        else:
            logger.error("❌ Failed to start camera publisher")
            return False
    except Exception as e:
        logger.error(f"❌ Camera publisher initialization failed: {e}")
        return False


def graceful_shutdown():
    """Perform graceful shutdown of all services."""
    global camera_agent, camera_publisher
    
    logger.info("🛑 Initiating graceful shutdown...")
    
    # Stop monitoring agent
    if camera_agent:
        try:
            camera_agent.stop_monitoring()
            logger.info("✅ Monitoring agent stopped")
        except Exception as e:
            logger.error(f"Error stopping monitoring agent: {e}")
    
    # Stop camera publisher
    if camera_publisher:
        try:
            camera_publisher.stop()
            logger.info("✅ Camera publisher stopped")
        except Exception as e:
            logger.error(f"Error stopping camera publisher: {e}")
    
    logger.info("👋 Shutdown complete")


if __name__ == "__main__":
    import signal
    import atexit
    
    debug_env = os.getenv("FLASK_DEBUG")
    debug_mode = (
        True
        if debug_env is None
        else debug_env.strip().lower() in {"1", "true", "yes", "on"}
    )

    # Register shutdown handlers
    atexit.register(graceful_shutdown)
    signal.signal(signal.SIGTERM, lambda sig, frame: graceful_shutdown())
    signal.signal(signal.SIGINT, lambda sig, frame: graceful_shutdown())

    # Initialize database schema before serving requests
    init_db()
    update_email_recipients()

    run_main_flag = os.getenv("WERKZEUG_RUN_MAIN")
    should_start_services = (run_main_flag == "true") or not debug_mode

    if should_start_services:
        if initialize_camera_publisher():
            status_thread = threading.Thread(
                target=broadcast_camera_status, daemon=True
            )
            status_thread.start()
        else:
            logger.warning(
                "⚠️  Camera publisher failed to start - live feed may not be available"
            )
    else:
        logger.info(
            "Skipping camera publisher startup in reloader bootstrap phase"
        )

    # Run the app
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = _safe_int_env("FLASK_RUN_PORT", 8080)
    app_url = os.getenv("FLASK_APP_URL")
    if not app_url:
        display_host = "localhost" if host in {"0.0.0.0", "::"} else host
        app_url = f"http://{display_host}:{port}"

    print("🌐 Starting ZEDEDA Camera Agent Web Interface...")
    print(f"📱 Access the interface at: {app_url}")
    print("📹 Live camera feed available (monitoring off by default)")
    print("🔍 Click 'Start Monitoring' to enable AI analysis")

    socketio.run(app, host=host, port=port, debug=debug_mode, allow_unsafe_werkzeug=True)
