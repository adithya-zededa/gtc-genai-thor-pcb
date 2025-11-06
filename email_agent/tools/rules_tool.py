"""Business rules evaluation helper for the Email Agent."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "/app/config.yaml"))


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        logger.warning("Configuration file %s not found; using empty rules list.", CONFIG_PATH)
        return {"rules": []}

    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {"rules": []}


def _render_template(template: str, payload: Dict[str, Any]) -> str:
    rendered = template
    for key, value in payload.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def _compare(value: Any, expected: Any, operator: str) -> bool:
    try:
        if operator in {"lt", "less_than"}:
            return float(value) < float(expected)
        if operator in {"lte", "less_than_or_equal"}:
            return float(value) <= float(expected)
        if operator in {"gt", "greater_than"}:
            return float(value) > float(expected)
        if operator in {"gte", "greater_than_or_equal"}:
            return float(value) >= float(expected)
        if operator in {"ne", "not_equal"}:
            return value != expected
        # default equality comparison
        return value == expected
    except (TypeError, ValueError):
        return False


def _build_email_payload(rule: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    email_cfg = rule.get("email", {})
    body_template = email_cfg.get("body", "")
    subject_template = email_cfg.get("subject", "Notification from Email Agent")

    return {
        "to": email_cfg.get("to", []),
        "cc": email_cfg.get("cc", []),
        "bcc": email_cfg.get("bcc", []),
        "subject": _render_template(subject_template, event),
        "body": _render_template(body_template, event),
    }


def evaluate_rules(event: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate configured rules against the incoming event payload."""

    config = _load_config()
    triggered: List[Dict[str, Any]] = []

    for rule in config.get("rules", []):
        condition = rule.get("condition", {})
        field = condition.get("field")
        operator = condition.get("operator", "eq")
        expected = condition.get("value")

        if not field:
            logger.debug("Skipping rule %s - no field defined", rule.get("id"))
            continue

        value = event.get(field)
        if value is None:
            continue

        if _compare(value, expected, operator):
            triggered.append(
                {
                    "rule_id": rule.get("id"),
                    "description": rule.get("description"),
                    "email_payload": _build_email_payload(rule, event),
                }
            )

    logger.info("evaluate_rules triggered %d rule(s)", len(triggered))
    return {"triggered": triggered}
