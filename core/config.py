"""Application configuration management.

Centralized configuration loaded from environment variables and YAML files.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Plain stdlib logger: core.logging configures handlers using values that come
# from here, so this module must not depend on it.
logger = logging.getLogger(__name__)


# Environment variable names
ENV_DATA_DIR = "CAMERA_AGENT_DATA_DIR"
ENV_SECRET_KEY = "FLASK_SECRET_KEY"
ENV_SOCKETIO_CORS = "SOCKETIO_CORS"
ENV_DB_PATH = "CAMERA_AGENT_DB"
ENV_DETECTED_DIR = "DETECTED_IMAGES_DIR"
ENV_PROCESSED_DIR = "PROCESSED_FRAMES_DIR"
ENV_VIDEO_DIR = "CAMERA_VIDEO_DIR"
ENV_VLLM_URL = "VLLM_URL"
ENV_AGENT_LLM_URL = "AGENT_LLM_URL"
ENV_AGENT_MODEL = "AGENT_MODEL"
ENV_HTTP_TIMEOUT = "HTTP_REQUEST_TIMEOUT"
ENV_CONFIG_PATH = "CAMERA_AGENT_CONFIG"

# Defaults
DEFAULT_SECRET_KEY_BYTES = 24
DEFAULT_SOCKETIO_CORS = None  # None => flask-socketio's same-origin-only default
DEFAULT_HTTP_TIMEOUT = 5.0
DEFAULT_CONFIG_PATH = "config.yaml"


def _get_or_create_secret_key(data_dir: Path) -> str:
    """Load a persisted Flask secret key, generating one on first run.

    Regenerating the key on every process start (the old behavior) invalidates
    signed tokens (e.g. chat-session tokens) across restarts, so the key is
    persisted to ``<data_dir>/.secret_key`` with owner-only permissions.
    """
    key_path = data_dir / ".secret_key"
    try:
        existing = key_path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass

    key = os.urandom(DEFAULT_SECRET_KEY_BYTES).hex()
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        key_path.write_text(key, encoding="utf-8")
        os.chmod(key_path, 0o600)
    except OSError:
        pass  # fall back to an in-memory-only key for this process
    return key


def _resolve_video_source() -> Optional[str]:
    """Resolve ``CAMERA_VIDEO_SOURCE``, ignoring a path that isn't there.

    The Docker image sets this to a baked-in simulator clip that is a local-only
    asset and may be absent from the build context. ``cv2.VideoCapture`` on a
    missing file just fails to open, leaving no feed at all, so an unusable
    path is dropped here and the live camera index is used instead. Non-path
    sources (e.g. an RTSP/HTTP URL) are passed through untouched.
    """
    source = os.getenv("CAMERA_VIDEO_SOURCE") or None
    if not source:
        return None
    if "://" in source:
        return source
    if Path(source).expanduser().is_file():
        return source
    logger.warning(
        "CAMERA_VIDEO_SOURCE=%s does not exist — falling back to the live camera",
        source,
    )
    return None


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
    video_source: Optional[str] = None  # Path/URL to video file; overrides camera index


@dataclass
class InferenceConfig:
    """Vision-model inference configuration (the VLM that sees frames)."""
    backend: str = "vllm"
    vllm_url: str = "http://localhost:8000"
    model: str = ""  # auto-detected from running server; set VISION_MODEL to override
    timeout: int = 300
    temperature: float = 0.1


@dataclass
class AgentInferenceConfig:
    """Text/agent-model inference configuration.

    The agent model is served by its own vLLM pod and handles intent
    classification, chat, and tool calling — everything that reasons over the
    vision model's output rather than over pixels.

    ``url``/``model`` empty means "no separate agent server": callers fall
    back to the vision endpoint, so a single-pod deployment keeps working.
    """
    url: str = ""
    model: str = ""
    timeout: int = 300
    temperature: float = 0.2

    @property
    def enabled(self) -> bool:
        """Whether a distinct agent endpoint is configured."""
        return bool(self.url)


@dataclass
class RouterConfig:
    """LLM Router configuration (vLLM-only)."""
    enabled: bool = True  # Always enabled — vLLM is the sole provider
    use_for_classification: bool = True  # Use router for intent classification
    use_for_chat: bool = True  # Use router for conversational responses


@dataclass
class FlaskConfig:
    """Flask web server configuration."""
    secret_key: str = ""
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    socketio_cors: Optional[str] = None


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes
    """Main application configuration container."""

    data_dir: Path = field(default_factory=lambda: Path("."))
    config_path: Path = field(default_factory=lambda: Path(DEFAULT_CONFIG_PATH))
    http_timeout: float = DEFAULT_HTTP_TIMEOUT

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    agent_inference: AgentInferenceConfig = field(default_factory=AgentInferenceConfig)
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
                video_source=_resolve_video_source(),
            ),
            inference=InferenceConfig(
                backend="vllm",
                vllm_url=os.getenv(ENV_VLLM_URL, "http://localhost:8000"),
                model=os.getenv("VISION_MODEL", ""),  # resolved lazily via detect_model()
                timeout=_safe_int_env("VLLM_TIMEOUT", 300),
                temperature=_safe_float_env("VLLM_TEMPERATURE", 0.1),
            ),
            agent_inference=AgentInferenceConfig(
                url=os.getenv(ENV_AGENT_LLM_URL, ""),
                model=os.getenv(ENV_AGENT_MODEL, ""),
                timeout=_safe_int_env("AGENT_LLM_TIMEOUT", _safe_int_env("VLLM_TIMEOUT", 300)),
                temperature=_safe_float_env("AGENT_LLM_TEMPERATURE", 0.2),
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
            ),
        )

        # Generate (or load a previously-persisted) secret key if not provided
        if not config.flask.secret_key:
            config.flask.secret_key = _get_or_create_secret_key(data_dir)

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

    @property
    def video_dir(self) -> Path:
        """Get the directory simulated-video-source files must live under.

        Matches the Dockerfile's ``/app/video`` (relative ``video`` when the
        app's cwd is ``/app``), independent of ``data_dir`` since the video
        simulator asset ships baked into the image, not on the data volume.
        """
        return Path(os.getenv(ENV_VIDEO_DIR, "video")).expanduser()


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
