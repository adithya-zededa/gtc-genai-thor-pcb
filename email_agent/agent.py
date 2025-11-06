"""Email automation orchestration built on the Ollama LLaVA 1.5 model."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from email_agent.tools.email_tool import send_email
from email_agent.tools.rules_tool import evaluate_rules

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

    def run(self, payload: Dict[str, Any], context: Optional[str] = None) -> AgentResponse:
        """Execute the automation workflow for the provided payload."""

        rules_result = evaluate_rules(payload)
        triggered = rules_result.get("triggered", [])
        email_results: List[Dict[str, Any]] = []

        for rule in triggered:
            email_payload = rule.get("email_payload") or {}
            if not email_payload.get("to"):
                email_results.append(
                    {
                        "rule_id": rule.get("rule_id"),
                        "result": "Skipped email (no recipients).",
                    }
                )
                continue

            try:
                result = send_email(email_payload)
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.exception("Email delivery failed for rule %s", rule.get("rule_id"))
                result = f"Email send failed: {exc}"

            email_results.append(
                {
                    "rule_id": rule.get("rule_id"),
                    "result": result,
                }
            )

        summary_text = self._summarise(payload, triggered, email_results, context)

        raw_response = {
            "output_text": summary_text,
            "triggered_rules": triggered,
            "email_results": email_results,
        }

        return AgentResponse(raw=raw_response)

    def _summarise(
        self,
        event: Dict[str, Any],
        triggered: List[Dict[str, Any]],
        email_results: List[Dict[str, Any]],
        context: Optional[str],
    ) -> str:
        summary_payload = {
            "event": event,
            "triggered_rules": triggered,
            "email_results": email_results,
            "context": context,
        }

        try:
            text = self._generate_summary(summary_payload)
            return text.strip() if text else self._fallback_summary(triggered, email_results)
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning("Ollama summary generation failed: %s", exc, exc_info=True)
            return self._fallback_summary(triggered, email_results)

    def _generate_summary(self, payload: Dict[str, Any]) -> str:
        prompt = (
            "You are an operations analyst creating concise status updates for automation events. "
            "Summarize the triggered rules, highlight any skipped notifications, and suggest logical next steps.\n\n"
            f"Event details JSON: {json.dumps(payload, ensure_ascii=False)}"
        )

        response = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    @staticmethod
    def _fallback_summary(
        triggered: List[Dict[str, Any]],
        email_results: List[Dict[str, Any]],
    ) -> str:
        if not triggered:
            return "No rules were triggered for the submitted event."

        triggered_ids = ", ".join(filter(None, [rule.get("rule_id") for rule in triggered])) or "unknown"
        delivery_notes = "; ".join(
            f"{result.get('rule_id')}: {result.get('result')}" for result in email_results
        )
        return f"Triggered rule(s): {triggered_ids}. Email delivery summary: {delivery_notes}."


def build_agent(model: str, base_url: str) -> EmailAutomationAgent:
    """Factory helper used by the application."""
    return EmailAutomationAgent(model=model, base_url=base_url)
