"""Email automation orchestration built on the Ollama LLaVA 1.5 model."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from agent_runtime.email_tools import send_email, evaluate_rules

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Wrapper for automation outcomes."""

    raw: Dict[str, Any]

    @property
    def output_text(self) -> Optional[str]:
        return self.raw.get("output_text")

    def to_dict(self) -> Dict[str, Any]:
        return self.raw


class EmailAutomationAgent:
    """Coordinates rule evaluation, email delivery, and LLM summarisation."""

    def __init__(self, model: str, base_url: str) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    def run(
        self, payload: Dict[str, Any], context: Optional[str] = None
    ) -> AgentResponse:
        """Execute the automation workflow for the provided payload."""

        rules_result = evaluate_rules(payload)
        triggered = rules_result.get("triggered", [])
        email_results: List[Dict[str, Any]] = []

        for rule in triggered:
            email_payload = rule.get("email_payload") or {}
            if not email_payload.get("to"):
                logger.warning(
                    "Rule %s triggered but no recipients defined.",
                    rule.get("rule_id"),
                )
                continue

            try:
                result = send_email(email_payload)
                email_results.append(
                    {"rule_id": rule.get("rule_id"), "status": result}
                )
            except Exception as exc:
                logger.error("Failed to send email for rule %s: %s", rule.get("rule_id"), exc)
                email_results.append(
                    {"rule_id": rule.get("rule_id"), "error": str(exc)}
                )

        # If context is provided, we might want to use the LLM to summarize or generate content
        # For now, we just return the rules execution result
        
        response_data = {
            "rules_triggered": len(triggered),
            "email_results": email_results,
            "output_text": f"Processed {len(triggered)} rules.",
        }

        return AgentResponse(raw=response_data)
