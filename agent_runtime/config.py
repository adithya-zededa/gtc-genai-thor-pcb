"""Typed configuration helpers for the camera monitoring agent."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(os.getenv("CAMERA_AGENT_CONFIG", "config.yaml")).expanduser()


@dataclass(slots=True)
class OllamaModels:
    """Ollama model configuration."""

    url: str
    vision_model: str
    decision_model: str
    timeout: int = 60
    decision_timeout: int = 30
    temperature: float = 0.1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OllamaModels":
        url = str(payload.get("url", "")).rstrip("/")
        vision_model = str(payload.get("vision_model", ""))
        decision_model = str(payload.get("decision_model", ""))
        timeout = int(payload.get("timeout", 60) or 60)
        decision_timeout = int(payload.get("decision_timeout", payload.get("decision_llm_timeout", 30) or 30))
        temperature = float(payload.get("temperature", 0.1) if payload.get("temperature") is not None else 0.1)

        if not url:
            raise ValueError("Ollama configuration requires a non-empty 'url'.")
        if not vision_model:
            raise ValueError("Ollama configuration requires a non-empty 'vision_model'.")
        if not decision_model:
            raise ValueError("Ollama configuration requires a non-empty 'decision_model'.")

        return cls(
            url=url,
            vision_model=vision_model,
            decision_model=decision_model,
            timeout=timeout,
            decision_timeout=decision_timeout,
            temperature=temperature,
        )


@dataclass(slots=True)
class CameraPreprocessing:
    diff_threshold: float = 0.8

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "CameraPreprocessing":
        if not isinstance(payload, Mapping):
            return cls()
        diff_threshold = float(payload.get("diff_threshold", 0.8) or 0.8)
        return cls(diff_threshold=diff_threshold)


@dataclass(slots=True)
class CameraSettings:
    device_index: int = 0
    capture_interval: float = 5.0
    save_detection_images: bool = True
    save_processed_frames: bool = True
    preprocessing: CameraPreprocessing = field(default_factory=CameraPreprocessing)

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "CameraSettings":
        if not isinstance(payload, Mapping):
            return cls()
        device_index = int(payload.get("device_index", 0) or 0)
        capture_interval = float(payload.get("capture_interval", 5.0) or 5.0)
        save_detection_images = bool(payload.get("save_detection_images", True))
        save_processed_frames = bool(payload.get("save_processed_frames", True))
        preprocessing = CameraPreprocessing.from_mapping(payload.get("preprocessing"))
        return cls(
            device_index=device_index,
            capture_interval=capture_interval,
            save_detection_images=save_detection_images,
            save_processed_frames=save_processed_frames,
            preprocessing=preprocessing,
        )


@dataclass(slots=True)
class AdvancedSettings:
    agent_ssim_skip_threshold: float = 0.95
    agent_ssim_recheck_seconds: float = 30.0
    agent_ssim_cache_ttl_seconds: float = 30.0
    max_concurrent_analyses: int = 1
    max_pending_analyses: int = 12
    motion_burst_interval: float = 0.2
    motion_burst_window: float = 12.0
    motion_burst_ssim: float = 0.75
    pending_dedupe_ssim: float = 0.92

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "AdvancedSettings":
        if not isinstance(payload, Mapping):
            return cls()
        default_instance = cls()
        data = dataclasses.asdict(default_instance)
        for key in data.keys():
            if key in payload and payload[key] is not None:
                candidate = payload[key]
                if key.startswith("agent_ssim") or key.endswith("interval") or key.endswith("window"):
                    data[key] = float(candidate)
                else:
                    data[key] = float(candidate) if isinstance(data[key], float) else int(candidate)
        return cls(**data)


@dataclass(slots=True)
class EmailNotifications:
    recipients: List[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "EmailNotifications":
        if not isinstance(payload, Mapping):
            return cls()
        recipients_raw = payload.get("recipients", [])
        recipients = [str(addr).strip() for addr in recipients_raw if isinstance(addr, str) and addr.strip()]
        return cls(recipients=recipients)


@dataclass(slots=True)
class Notifications:
    email: EmailNotifications = field(default_factory=EmailNotifications)

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "Notifications":
        if not isinstance(payload, Mapping):
            return cls()
        email_cfg = EmailNotifications.from_mapping(payload.get("email"))
        return cls(email=email_cfg)


@dataclass(slots=True)
class MemorySettings:
    max_events: int = 50
    summary_window: int = 10

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "MemorySettings":
        if not isinstance(payload, Mapping):
            return cls()
        max_events = int(payload.get("max_events", 50) or 50)
        summary_window = int(payload.get("summary_window", 10) or 10)
        return cls(max_events=max_events, summary_window=summary_window)


@dataclass(slots=True)
class AgentSettings:
    ollama: OllamaModels
    camera: CameraSettings = field(default_factory=CameraSettings)
    advanced: AdvancedSettings = field(default_factory=AdvancedSettings)
    notifications: Notifications = field(default_factory=Notifications)
    memory: MemorySettings = field(default_factory=MemorySettings)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AgentSettings":
        ollama = OllamaModels.from_mapping(payload.get("ollama", {}))
        camera = CameraSettings.from_mapping(payload.get("camera"))
        advanced = AdvancedSettings.from_mapping(payload.get("advanced"))
        notifications = Notifications.from_mapping(payload.get("notifications"))
        memory = MemorySettings.from_mapping(payload.get("memory"))
        raw_copy = dict(payload)
        return cls(
            ollama=ollama,
            camera=camera,
            advanced=advanced,
            notifications=notifications,
            memory=memory,
            raw=raw_copy,
        )

    def merge_updates(self, updates: Mapping[str, Any]) -> "AgentSettings":
        merged = dict(self.raw)
        for key, value in updates.items():
            if isinstance(value, MutableMapping) and isinstance(merged.get(key), MutableMapping):
                merged[key] = _deep_merge_dict(merged.get(key, {}), value)
            else:
                merged[key] = value
        return AgentSettings.from_mapping(merged)


def _deep_merge_dict(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(path: Optional[Path | str] = None) -> AgentSettings:
    """Load agent settings from YAML, returning a typed configuration object."""
    target_path = Path(path or DEFAULT_CONFIG_PATH).expanduser()
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {target_path}")

    with target_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Configuration root must be a mapping")

    return AgentSettings.from_mapping(payload)
