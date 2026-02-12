"""LLM-based intent classifier for MCP domain routing and tool selection.

Uses a single LLM inference call that returns structured JSON to classify
    user messages into domains and tools.  The vLLM backend is the sole
    inference provider — there are no keyword or regex fallbacks.
Output schema::

    {
        "domain": "pcb" | "general",
        "tool":   "<tool_name or null>",
        "confidence": 0.0 – 1.0,
        "params": { ... extracted parameters ... }
    }
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

from core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Structured output of the LLM intent classifier."""

    domain: str                          # pcb | general
    tool: Optional[str] = None           # e.g. "scan_tray_items"
    confidence: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    source: str = "llm"                  # "llm" or "keyword_fallback"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "tool": self.tool,
            "confidence": self.confidence,
            "params": self.params,
            "rationale": self.rationale,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

_CLASSIFICATION_PROMPT = """\
You are an intent-classification assistant for a PCB conveyor inspection system.

Given the user message below, decide:
1. Which **domain** the request belongs to.
2. Which **tool** (if any) should be called.
3. Extract any **parameters** the user provided (emails, board types, severities, etc.).

## Domains

| domain   | description |
|----------|-------------|
| pcb      | PCB inspection on a conveyor, board presence, motion/stopped state, board defects, board-type identification, defect logging/reporting. |
| general  | Session control, greetings, help, status, and other non-inspection camera control tasks. |

Scope constraint:
- The system is PCB-only.
- Any user request about non-PCB domains must be routed to `general` with `tool=null`.

## Tools

### PCB domain tools
| tool                   | when to pick |
|------------------------|--------------|
| inspect_pcb            | User wants to visually inspect / check / scan a PCB for defects. |
| classify_board         | User wants to identify what type of board is visible. |
| send_defect_alert      | User wants to email/alert someone about a defect. |
| log_defect             | User wants to record / save / log a defect. |
| generate_defect_report | User wants a report or summary of past defects. |

### General domain tools
| tool                      | when to pick |
|---------------------------|--------------|
| start_monitoring_session  | User wants to start / begin / activate camera monitoring. |
| end_session               | User wants to stop / end / deactivate monitoring or end the session. |
| go_idle                   | User wants to pause monitoring or put the agent in idle / take a break. |
| get_agent_status          | User asks about the agent's status, whether it's running, current state. |
| analyze_current_frame     | User wants to analyze / look at / describe / examine what's in the camera view. |
| query_history             | User asks about recent detections, past events, history. |
| get_session_summary       | User wants a summary / recap of the monitoring session. |
| set_detection_task        | User wants to set PCB inspection task behavior (pcb_inspection or custom). |
| send_alert_email          | User wants to send an alert email about a detection. |
| shutdown_agent            | User wants to completely shut down the agent. |
| acknowledge_error         | User wants to acknowledge / dismiss an error state. |

Pick `null` for the tool when the message is a greeting, help request, thanks, or casual conversation that doesn't need any tool.

## Parameter extraction rules
- **emails**: extract all email addresses from the text.
- **board_type**: if a specific board is mentioned (arduino, raspberry pi, esp32, stm32, jetson, etc.), extract it.
- **severity**: if a severity level (low/medium/high) is mentioned, extract it.
- **recipient_email**: the primary email to send something to.
- **recipients**: list of emails for alerts.
- **query**: the original user message (useful as context for scan/inspect tools).

## Output format
Respond ONLY with a JSON object — no explanation, no markdown fences:

{"domain":"<pcb|general>","tool":"<tool_name or null>","confidence":<0.0-1.0>,"params":{...},"rationale":"<one sentence>"}

## User message
"""


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------

class LLMIntentClassifier:
    """Classifies user messages via a single LLM inference call.

    Falls back to a low-confidence general result when the vLLM backend
    is unreachable or returns an unparsable response.

    Classification requests are sent through the LLM router, which wraps
    the vLLM adapter with resilience (retry, circuit-breaking).
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 15,
    ):
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "camera-agent-classifier/1.0"
        self._lock = threading.Lock()

        # Simple circuit breaker: skip LLM after N consecutive failures
        self._consecutive_failures = 0
        self._max_failures = 3
        self._circuit_open_until: float = 0.0  # timestamp
        self._backoff_seconds = 30.0

        # Router integration
        self._router = None
        self._router_checked = False

    # ------------------------------------------------------------------
    # Lazy resolution of URL / model from env
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        if self._base_url:
            return self._base_url
        from core.config import get_config
        cfg = get_config()
        return os.getenv("VLLM_URL", cfg.inference.vllm_url).rstrip("/")

    @property
    def model(self) -> str:
        if self._model:
            return self._model
        # Prefer a smaller / faster model for classification if set
        classifier_model = os.getenv("CLASSIFIER_MODEL")
        if classifier_model:
            return classifier_model
        from core.model_detect import detect_model
        return detect_model(backend="vllm", base_url=self._base_url, wait=False)

    # ------------------------------------------------------------------
    # Router integration
    # ------------------------------------------------------------------

    def _get_router(self):
        """Lazily initialise and return the LLM router, or None."""
        if self._router_checked:
            return self._router
        self._router_checked = True
        try:
            from router import get_router
            self._router = get_router()
            logger.info("LLM classifier using vLLM router")
        except Exception as exc:
            logger.warning("Failed to initialise LLM router for classifier: %s", exc)
        return self._router

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, message: str) -> ClassificationResult:
        """Classify a user message, returning domain + tool + params.

        Uses the LLM exclusively.  Returns a low-confidence general
        result if the LLM is unreachable.
        """
        if not message or not message.strip():
            return ClassificationResult(
                domain="general", tool=None, confidence=1.0,
                rationale="Empty message",
            )

        # Circuit breaker check
        if self._is_circuit_open():
            logger.debug("LLM classifier circuit open — returning general")
            return ClassificationResult(
                domain="general", tool=None, confidence=0.2,
                rationale="LLM circuit breaker open",
                source="circuit_breaker",
            )

        try:
            result = self._call_llm(message)
            self._record_success()
            return result
        except Exception as exc:
            self._record_failure()
            logger.warning(
                "LLM classification failed (%s), returning general domain",
                exc,
            )
            return ClassificationResult(
                domain="general", tool=None, confidence=0.1,
                rationale=f"LLM unavailable: {exc}",
                source="llm_error",
            )

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, message: str) -> ClassificationResult:
        prompt = _CLASSIFICATION_PROMPT + message

        # Try the router first if available
        router = self._get_router()
        if router:
            try:
                raw = self._router_completion(router, prompt)
            except Exception as exc:
                logger.warning("Router classification failed (%s), falling back to direct backend", exc)
                raw = self._direct_completion(prompt)
        else:
            raw = self._direct_completion(prompt)

        parsed = self._parse_response(raw)
        if parsed is None:
            raise ValueError(f"Unparsable LLM response: {raw[:200]}")

        # Validate domain
        domain = parsed.get("domain", "general")
        if domain not in ("pcb", "general"):
            domain = "general"

        llm_params = parsed.get("params") or {}

        return ClassificationResult(
            domain=domain,
            tool=parsed.get("tool"),
            confidence=min(float(parsed.get("confidence", 0.8)), 1.0),
            params=llm_params,
            rationale=parsed.get("rationale", ""),
            source="llm",
        )

    def _router_completion(self, router, prompt: str) -> str:
        """Send classification request through the LLM router."""
        from router.config import ChatResponse
        response: ChatResponse = router.chat(
            messages=[{"role": "user", "content": prompt}],
            # Disable Qwen3 thinking mode for classification
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        logger.debug(
            "Router classification via %s/%s",
            response.provider, response.model,
        )
        return response.content

    def _direct_completion(self, prompt: str) -> str:
        """Send classification request directly to vLLM."""
        return self._vllm_completion(prompt)

    def _vllm_completion(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            # Disable Qwen3 thinking mode for structured classification output
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = self._session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            # Prefer content; fall back to reasoning_content for Qwen3 thinking models
            content = msg.get("content", "")
            if not content and msg.get("reasoning_content"):
                content = msg["reasoning_content"]
            return content or ""
        return ""

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(text: str) -> Optional[Dict[str, Any]]:
        """Extract the first JSON object from the LLM output."""
        if not text:
            return None
        text = text.strip()
        # Strip markdown fences if present
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        # Strip thinking tokens
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text, flags=re.DOTALL)
        text = text.strip()

        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _is_circuit_open(self) -> bool:
        if self._consecutive_failures < self._max_failures:
            return False
        if time.time() >= self._circuit_open_until:
            # Half-open: allow one attempt
            return False
        return True

    def _record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def _record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_failures:
                self._circuit_open_until = time.time() + self._backoff_seconds
                logger.warning(
                    "LLM classifier circuit OPEN — will use keyword fallback for %.0fs",
                    self._backoff_seconds,
                )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_classifier: Optional[LLMIntentClassifier] = None
_clf_lock = threading.Lock()


def get_classifier() -> LLMIntentClassifier:
    """Return the global ``LLMIntentClassifier`` singleton."""
    global _classifier
    if _classifier is None:
        with _clf_lock:
            if _classifier is None:
                _classifier = LLMIntentClassifier()
                logger.info("LLMIntentClassifier singleton created (vLLM)")
    return _classifier
