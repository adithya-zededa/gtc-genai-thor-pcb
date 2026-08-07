"""Validation schemas for configuration payloads.

Provides JSON Schema-compatible validation for API-submitted configuration
updates and startup validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CameraConfigPayload(BaseModel):
    """Schema for camera configuration updates via API."""
    device_index: Optional[int] = Field(None, ge=0, le=10)
    capture_interval: Optional[int] = Field(None, ge=1, le=3600)
    save_detection_images: Optional[bool] = None
    detection_image_dir: Optional[str] = None


class VLLMConfigPayload(BaseModel):
    """Schema for vLLM configuration updates via API."""
    url: Optional[str] = None
    model: Optional[str] = None
    timeout: Optional[int] = Field(None, ge=1, le=600)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class NotificationEmailPayload(BaseModel):
    """Schema for email notification configuration."""
    enabled: Optional[bool] = None
    recipients: Optional[List[str]] = None
    sender_email: Optional[str] = None
    smtp_server: Optional[str] = None
    smtp_port: Optional[int] = Field(None, ge=1, le=65535)
    use_tls: Optional[bool] = None
    auto_sync_users: Optional[bool] = None


class MemoryConfigPayload(BaseModel):
    """Schema for agent memory configuration."""
    max_events: Optional[int] = Field(None, ge=1, le=10000)
    summary_window: Optional[int] = Field(None, ge=1, le=1000)


class ConfigUpdatePayload(BaseModel):
    """Top-level schema for configuration updates."""
    camera: Optional[CameraConfigPayload] = None
    vllm: Optional[VLLMConfigPayload] = None
    memory: Optional[MemoryConfigPayload] = None
    notifications: Optional[Dict[str, Any]] = None
    detection: Optional[Dict[str, Any]] = None
    rules: Optional[List[Dict[str, Any]]] = None


def validate_config_update(payload: Dict[str, Any]) -> ConfigUpdatePayload:
    """Validate a configuration update payload.
    
    Args:
        payload: Raw configuration update dictionary.
        
    Returns:
        Validated ConfigUpdatePayload model.
        
    Raises:
        ValueError: If validation fails.
    """
    return ConfigUpdatePayload.model_validate(payload)
