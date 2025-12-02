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
from typing import List, Dict
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

from camera_agent import StreamlinedAgent, DEFAULT_CONFIG_PATH
from agent_runtime.utils import (
    coerce_bool,
    dedupe_strings,
    ensure_directory,
)
from agent_runtime.publisher import get_camera_publisher
from agent_runtime.monitoring import StreamlinedMonitoringService

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


def _resolve_request_timeout() -> float:
    try:
        timeout_value = float(
            os.getenv(ENV_HTTP_TIMEOUT, DEFAULT_HTTP_TIMEOUT)
        )
        return timeout_value if timeout_value > 0 else DEFAULT_HTTP_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_HTTP_TIMEOUT


REQUEST_TIMEOUT = _resolve_request_timeout()


def _ollama_endpoint(path: str) -> str:
    base = OLLAMA_BASE_URL.rstrip("/")
    return f"{base}/{path.lstrip('/')}"


app = Flask(__name__)
app.secret_key = _resolve_secret_key()
socketio = SocketIO(app, cors_allowed_origins=_resolve_socketio_cors())

# Global variables
camera_agent = None
camera_publisher = None  # Global publisher instance for camera viewing
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
            cursor = conn.execute(
                """
                INSERT INTO detection_logs (
                    timestamp, confidence, response, image_path, frame_number,
                    reason, vision_description, decision_details
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                    "detected": event.confidence is not None and event.confidence > 0,
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


# Initialize database
def init_db():
    """Initialize SQLite database for user management and logs"""
    ensure_database_directory()
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

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


def get_db_connection():
    """Get database connection"""
    ensure_database_directory()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
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
    """Get system status"""
    status = {
        "monitoring_active": (
            camera_agent.is_monitoring if camera_agent else False
        ),
        "camera_available": check_camera_availability(),
        "ollama_available": check_ollama_availability(),
        "stats": camera_agent._serialize_stats() if camera_agent else {},
    }

    return jsonify(status)


@app.route("/api/agent/memory")
def api_agent_memory():
    """Expose recent agent memory and summarised activity."""
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
    
    GET: Retrieve all detection logs with parsed decision details.
    DELETE: Clear all detection logs from the database.
    
    Returns:
        JSON response with logs list (GET) or success message (DELETE).
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

    # GET method (default)
    try:
        with get_db_connection() as conn:
            logs = conn.execute(
                """
                SELECT id, timestamp, confidence, response, image_path,
                       frame_number, reason, vision_description, decision_details
                FROM detection_logs
                ORDER BY timestamp DESC
            """
            ).fetchall()

        logs_list = []
        for log in logs:
            decision_details = {}
            if log["decision_details"]:
                try:
                    decision_details = json.loads(log["decision_details"])
                except json.JSONDecodeError:
                    decision_details = {"raw": log["decision_details"]}

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
                    "detected": log["confidence"] is not None
                    and log["confidence"] > 0,
                }
            )

        return jsonify({"success": True, "logs": logs_list})
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
    """Serve detection images"""
    try:
        # Sanitize and resolve the image path
        candidate_path = Path(image_path)
        if not candidate_path.is_absolute():
            candidate_path = (DATA_DIR / candidate_path).resolve()
        else:
            candidate_path = candidate_path.resolve()

        # Check if path is within allowed directories
        detected_dir = DETECTED_IMAGES_DIR.resolve()
        processed_dir = PROCESSED_FRAMES_DIR.resolve()

        if not (
            candidate_path.is_relative_to(detected_dir)
            or candidate_path.is_relative_to(processed_dir)
        ):
            return jsonify({"error": "Access denied"}), 403

        if candidate_path.exists() and candidate_path.is_file():
            return send_file(candidate_path, mimetype="image/jpeg")
        else:
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/system/status")
def api_system_status():
    """Get system status information"""
    try:
        import psutil

        status = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "camera_available": check_camera_availability(),
            "ollama_available": check_ollama_availability(),
        }
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
            "OLLAMA_URL": OLLAMA_BASE_URL,
            "VISION_MODEL": os.getenv("VISION_MODEL", "gemma3:4b"),
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
    """Stream live video frames from camera publisher"""

    def generate():
        # Subscribe to camera feed for streaming
        subscriber_id = f"video_feed_{id(generate)}"

        try:
            publisher = get_camera_publisher()
            if not publisher.subscribe(subscriber_id):
                return

            while True:
                frame_obj = publisher.get_frame(subscriber_id, timeout=1.0)
                if not frame_obj:
                    continue

                # Yield frame in multipart format
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_obj.image_data
                    + b"\r\n"
                )
        except GeneratorExit:
            # Client disconnected
            pass
        except Exception as e:
            print(f"Video feed error: {e}")
        finally:
            # Cleanup: unsubscribe when done
            try:
                publisher = get_camera_publisher()
                publisher.unsubscribe(subscriber_id)
            except Exception:
                pass

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
def check_camera_availability():
    """Check if camera is available"""
    try:
        publisher = get_camera_publisher()
        return publisher.camera.isOpened() if publisher.camera else False
    except Exception:
        return False


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


if __name__ == "__main__":
    debug_env = os.getenv("FLASK_DEBUG")
    debug_mode = (
        True
        if debug_env is None
        else debug_env.strip().lower() in {"1", "true", "yes", "on"}
    )

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
