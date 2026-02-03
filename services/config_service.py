"""Configuration management service."""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any, Dict, List

import yaml

from core.config import get_config
from core.utils import coerce_bool, dedupe_strings, ensure_directory
from core.logging import get_logger

logger = get_logger(__name__)

# Thread lock for config operations
CONFIG_LOCK = threading.RLock()


def load_camera_config() -> Dict[str, Any]:
    """Load the camera configuration from disk."""
    config = get_config()
    
    with CONFIG_LOCK:
        if not config.config_path.exists():
            raise FileNotFoundError(f"Config not found: {config.config_path}")

        with config.config_path.open('r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError(f"Config must be a mapping: {config.config_path}")

        return cfg


def save_camera_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Persist configuration to disk and return the sanitized payload."""
    config = get_config()
    canonical = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
    recipients = extract_recipients(canonical)
    apply_recipients(canonical, recipients)
    sanitized = sanitize_config_payload(canonical)
    
    with CONFIG_LOCK:
        ensure_directory(config.config_path.parent)
        with config.config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                sanitized,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
    return sanitized


def reset_camera_config() -> Dict[str, Any]:
    """Reset the configuration to defaults and persist the change."""
    from agents.camera_agent import StreamlinedAgent
    
    defaults = StreamlinedAgent.default_config()
    save_camera_config(defaults)
    return defaults


def sanitize_config_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a sanitized copy of the configuration for persistence."""
    config_copy = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
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


def extract_recipients(cfg: Dict[str, Any]) -> List[str]:
    """Collect the distinct email recipients from configuration payload."""
    recipients: List[str] = []
    notifications = (cfg or {}).get("notifications", {})
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

    rules = cfg.get("rules") if isinstance(cfg, dict) else None
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


def apply_recipients(cfg: Dict[str, Any], recipients: List[str]) -> Dict[str, Any]:
    """Apply recipient list to notification config and rules."""
    sanitized_recipients = [
        email.strip()
        for email in recipients
        if isinstance(email, str) and email.strip()
    ]
    notifications = cfg.setdefault("notifications", {})
    email_cfg = notifications.setdefault("email", {})
    email_cfg["recipients"] = sanitized_recipients

    rules = cfg.get("rules")
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

    return cfg


def update_email_recipients() -> None:
    """Update email recipients in configuration based on active users."""
    from app.database import UserRepository
    from services.monitoring_service import get_monitoring_service
    
    try:
        cfg = load_camera_config() or {}
        notifications = cfg.get("notifications", {}) if isinstance(cfg, dict) else {}
        email_cfg = notifications.get("email", {}) if isinstance(notifications, dict) else {}
        auto_sync_enabled = coerce_bool((email_cfg or {}).get("auto_sync_users"), True)
        
        if not auto_sync_enabled:
            logger.debug("Email auto-sync disabled in configuration")
            return

        # Get active users with valid emails
        emails = UserRepository.get_active_emails()
        emails.sort(key=lambda value: value.lower())

        current_recipients = extract_recipients(cfg)
        if emails == current_recipients:
            logger.debug("Email recipients already up to date")
            return

        apply_recipients(cfg, emails)
        persisted = save_camera_config(cfg)

        service = get_monitoring_service()
        if service and service.agent:
            try:
                service.agent.apply_config(persisted)
                service.refresh_configuration(persisted)
            except Exception as exc:
                logger.error(f"Failed to apply synced recipients to agent: {exc}")

        logger.info(
            "Email recipients updated via user sync. Recipients: %s",
            ", ".join(emails) if emails else "none",
        )

    except Exception as e:
        logger.error(f"Failed to update email recipients: {e}", exc_info=True)
