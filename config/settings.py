"""Pydantic-based settings models for type-safe configuration.

Supports:
- Environment variable overrides (via env_prefix or explicit env names)
- Nested config with ``__`` delimiter (e.g. ``VLLM__URL``)
- .env file loading
- Validation built-in
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class DatabaseSettings(BaseModel):
    """Database configuration."""
    path: Path = Field(
        default=Path("camera_agent.db"),
        description="Path to SQLite database file",
    )


class CameraSettings(BaseModel):
    """Camera capture configuration."""
    device_index: int = Field(default=0, description="Camera device index")
    width: int = Field(default=640, ge=160, le=3840, description="Capture width")
    height: int = Field(default=480, ge=120, le=2160, description="Capture height")
    fps: int = Field(default=30, ge=1, le=120, description="Capture FPS")
    capture_interval: int = Field(default=30, ge=1, description="Seconds between captures")
    save_detection_images: bool = Field(default=True, description="Save images on detection")
    detection_image_dir: Path = Field(
        default=Path("detected_images"),
        description="Directory for saved detection images",
    )


class VLLMSettings(BaseModel):
    """vLLM inference backend configuration."""
    url: str = Field(
        default="http://localhost:8000",
        description="vLLM server URL",
    )
    model: str = Field(
        default="",
        description="Vision model name (auto-detected from server when empty)",
    )
    timeout: int = Field(default=300, ge=1, description="Request timeout in seconds")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")


class InferenceSettings(BaseModel):
    """vLLM inference backend configuration."""
    vllm: VLLMSettings = Field(default_factory=VLLMSettings)


class FlaskSettings(BaseModel):
    """Flask web server configuration."""
    secret_key: str = Field(default="", description="Flask secret key")
    debug: bool = Field(default=False, description="Enable debug mode")
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8080, ge=1, le=65535, description="Bind port")
    socketio_cors: str = Field(default="*", description="SocketIO CORS origins")


class RouterSettings(BaseModel):
    """LLM Router configuration (vLLM-only)."""
    enabled: bool = Field(default=True, description="Router enabled (always True)")
    use_for_classification: bool = Field(default=True, description="Use for intent classification")
    use_for_chat: bool = Field(default=True, description="Use for chat responses")


class MemorySettings(BaseModel):
    """Agent memory configuration."""
    max_events: int = Field(default=50, ge=1, description="Max events in memory ring buffer")
    summary_window: int = Field(default=10, ge=1, description="Events for summary generation")


class NotificationSettings(BaseModel):
    """Notification configuration."""
    email_enabled: bool = Field(default=True, description="Enable email alerts")
    desktop_enabled: bool = Field(default=True, description="Enable desktop alerts")


class TTSSettings(BaseModel):
    """Text-to-Speech configuration for invoice audio summaries."""
    enabled: bool = Field(default=True, description="Enable TTS for invoice generation")
    language: str = Field(default="en", description="TTS language code")
    slow: bool = Field(default=False, description="Use slower speech rate")


class Settings(BaseModel):
    """Root settings container.

    Usage::

        settings = get_settings()
        print(settings.camera.device_index)
        print(settings.inference.vllm.url)
    """
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    camera: CameraSettings = Field(default_factory=CameraSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    flask: FlaskSettings = Field(default_factory=FlaskSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    http_timeout: float = Field(default=5.0, ge=0.1, description="HTTP request timeout")

    @classmethod
    def from_environment(cls) -> "Settings":
        """Build settings from environment variables with sensible defaults."""
        return cls(
            database=DatabaseSettings(
                path=Path(os.getenv("CAMERA_AGENT_DB", "camera_agent.db")),
            ),
            camera=CameraSettings(
                device_index=int(os.getenv("CAMERA_INDEX", "0")),
                width=int(os.getenv("CAMERA_WIDTH", "640")),
                height=int(os.getenv("CAMERA_HEIGHT", "480")),
                capture_interval=int(os.getenv("CAPTURE_INTERVAL", "30")),
                save_detection_images=os.getenv("SAVE_DETECTION_IMAGES", "true").lower()
                in {"1", "true", "yes"},
                detection_image_dir=Path(
                    os.getenv("DETECTED_IMAGES_DIR", "detected_images")
                ),
            ),
            inference=InferenceSettings(
                vllm=VLLMSettings(
                    url=os.getenv("VLLM_URL", "http://localhost:8000"),
                    model=os.getenv("VISION_MODEL", ""),  # auto-detected at runtime
                    timeout=int(os.getenv("VLLM_TIMEOUT", "300")),
                    temperature=float(os.getenv("VLLM_TEMPERATURE", "0.1")),
                ),
            ),
            flask=FlaskSettings(
                secret_key=os.getenv("FLASK_SECRET_KEY", ""),
                debug=os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"},
                host=os.getenv("FLASK_RUN_HOST", "0.0.0.0"),
                port=int(os.getenv("FLASK_RUN_PORT", "8080")),
                socketio_cors=os.getenv("SOCKETIO_CORS", "*"),
            ),
            router=RouterSettings(
                enabled=True,
            ),
            http_timeout=float(os.getenv("HTTP_REQUEST_TIMEOUT", "5.0")),
        )


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_settings: Optional[Settings] = None
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    """Return the global settings singleton, creating from env if needed."""
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = Settings.from_environment()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (primarily for testing)."""
    global _settings
    with _settings_lock:
        _settings = None
