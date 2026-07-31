"""Application configuration management.

Centralized configuration loaded from environment variables and YAML files.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Environment variable names
ENV_DATA_DIR = "CAMERA_AGENT_DATA_DIR"
ENV_SECRET_KEY = "FLASK_SECRET_KEY"
ENV_SOCKETIO_CORS = "SOCKETIO_CORS"
ENV_DB_PATH = "CAMERA_AGENT_DB"
ENV_DETECTED_DIR = "DETECTED_IMAGES_DIR"
ENV_PROCESSED_DIR = "PROCESSED_FRAMES_DIR"
ENV_VLLM_URL = "VLLM_URL"
ENV_HTTP_TIMEOUT = "HTTP_REQUEST_TIMEOUT"
ENV_CONFIG_PATH = "CAMERA_AGENT_CONFIG"

# Defaults
DEFAULT_SECRET_KEY_BYTES = 24
DEFAULT_SOCKETIO_CORS = "*"
DEFAULT_HTTP_TIMEOUT = 5.0
DEFAULT_CONFIG_PATH = "config.yaml"


def _safe_int_env(key: str, default: int) -> int:
    """Safely parse an integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _safe_float_env(key: str, default: float) -> float:
    """Safely parse a float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    path: Path = field(default_factory=lambda: Path("camera_agent.db"))


@dataclass
class CameraConfig:
    """Camera capture configuration."""
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    save_detection_images: bool = False
    detection_image_dir: Path = field(default_factory=lambda: Path("detected_images"))


@dataclass
class InferenceConfig:
    """vLLM inference configuration."""
    backend: str = "vllm"
    vllm_url: str = "http://localhost:8000"
    model: str = ""  # auto-detected from running server; set VISION_MODEL to override
    timeout: int = 300
    temperature: float = 0.1


@dataclass
class RouterConfig:
    """Agent LLM Router configuration.

    This is the text-only reasoning model (intent classification, tool
    selection, chat replies) — distinct from ``InferenceConfig``, which is
    the vision model used for frame analysis. Defaults to the same vLLM
    deployment as the vision model when AGENT_LLM_URL isn't set, so a
    single-model deployment keeps working unchanged.
    """
    enabled: bool = True  # Always enabled — vLLM is the sole provider
    use_for_classification: bool = True  # Use router for intent classification
    use_for_chat: bool = True  # Use router for conversational responses
    url: str = "http://localhost:8000"
    model: str = ""  # auto-detected from running server; set AGENT_MODEL to override
    timeout: int = 300
    temperature: float = 0.1


@dataclass
class FlaskConfig:
    """Flask web server configuration."""
    secret_key: str = ""
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    socketio_cors: str = "*"


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes
    """Main application configuration container."""

    data_dir: Path = field(default_factory=lambda: Path("."))
    config_path: Path = field(default_factory=lambda: Path(DEFAULT_CONFIG_PATH))
    http_timeout: float = DEFAULT_HTTP_TIMEOUT

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    flask: FlaskConfig = field(default_factory=FlaskConfig)
    router: RouterConfig = field(default_factory=RouterConfig)

    # Raw YAML config for backward compatibility
    _yaml_config: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @classmethod
    def from_environment(cls) -> "Config":
        """Create configuration from environment variables."""
        data_dir = Path(os.getenv(ENV_DATA_DIR, ".")).expanduser()

        config = cls(
            data_dir=data_dir,
            config_path=Path(os.getenv(ENV_CONFIG_PATH, DEFAULT_CONFIG_PATH)).expanduser(),
            http_timeout=_safe_float_env(ENV_HTTP_TIMEOUT, DEFAULT_HTTP_TIMEOUT),
            database=DatabaseConfig(
                path=Path(os.getenv(ENV_DB_PATH, str(data_dir / "camera_agent.db"))).expanduser(),
            ),
            camera=CameraConfig(
                index=_safe_int_env("CAMERA_INDEX", 0),
                width=_safe_int_env("CAMERA_WIDTH", 640),
                height=_safe_int_env("CAMERA_HEIGHT", 480),
                fps=_safe_int_env("CAMERA_FPS", 30),
                detection_image_dir=Path(
                    os.getenv(ENV_DETECTED_DIR, str(data_dir / "detected_images"))
                ).expanduser(),
            ),
            inference=InferenceConfig(
                backend="vllm",
                vllm_url=os.getenv(ENV_VLLM_URL, "http://localhost:8000"),
                model=os.getenv("VISION_MODEL", ""),  # resolved lazily via detect_model()
                timeout=_safe_int_env("VLLM_TIMEOUT", 300),
                temperature=_safe_float_env("VLLM_TEMPERATURE", 0.1),
            ),
            flask=FlaskConfig(
                secret_key=os.getenv(ENV_SECRET_KEY) or "",
                debug=os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"},
                host=os.getenv("FLASK_RUN_HOST", "0.0.0.0"),
                port=_safe_int_env("FLASK_RUN_PORT", 8080),
                socketio_cors=os.getenv(ENV_SOCKETIO_CORS, DEFAULT_SOCKETIO_CORS),
            ),
            router=RouterConfig(
                enabled=True,
                use_for_classification=(
                    os.getenv("LLM_ROUTER_FOR_CLASSIFICATION", "true").lower()
                    in {"1", "true", "yes"}
                ),
                use_for_chat=(
                    os.getenv("LLM_ROUTER_FOR_CHAT", "true").lower()
                    in {"1", "true", "yes"}
                ),
                url=os.getenv("AGENT_LLM_URL", os.getenv(ENV_VLLM_URL, "http://localhost:8000")),
                model=os.getenv("AGENT_MODEL", ""),
                timeout=_safe_int_env("AGENT_LLM_TIMEOUT", _safe_int_env("VLLM_TIMEOUT", 300)),
                temperature=_safe_float_env(
                    "AGENT_LLM_TEMPERATURE", _safe_float_env("VLLM_TEMPERATURE", 0.1)
                ),
            ),
        )

        # Generate secret key if not provided
        if not config.flask.secret_key:
            config.flask.secret_key = os.urandom(DEFAULT_SECRET_KEY_BYTES).hex()

        return config

    def load_yaml_config(self) -> Dict[str, Any]:
        """Load and cache the YAML configuration file."""
        with self._lock:
            if not self._yaml_config and self.config_path.exists():
                with self.config_path.open("r", encoding="utf-8") as f:
                    self._yaml_config = yaml.safe_load(f) or {}
            return self._yaml_config.copy()

    def save_yaml_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to YAML file."""
        with self._lock:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(
                    config,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            self._yaml_config = config.copy()

    def reload_yaml_config(self) -> Dict[str, Any]:
        """Force reload of YAML configuration."""
        with self._lock:
            self._yaml_config = {}
            return self.load_yaml_config()

    @property
    def detected_images_dir(self) -> Path:
        """Get the detected images directory path."""
        return self.camera.detection_image_dir

    @property
    def processed_frames_dir(self) -> Path:
        """Get the processed frames directory path."""
        return Path(
            os.getenv(ENV_PROCESSED_DIR, str(self.data_dir / "processed_frames"))
        ).expanduser()


# Global singleton instance
_config_state: dict[str, Optional[Config]] = {"config": None}
_config_lock = threading.Lock()


def get_config() -> Config:
    """Get the global configuration singleton."""
    if _config_state["config"] is None:
        with _config_lock:
            if _config_state["config"] is None:
                _config_state["config"] = Config.from_environment()
    config = _config_state["config"]
    if config is None:
        config = Config.from_environment()
        _config_state["config"] = config
    return config


def reset_config() -> None:
    """Reset the configuration singleton (for testing)."""
    with _config_lock:
        _config_state["config"] = None
