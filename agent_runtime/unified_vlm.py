"""Unified Vision Language Model client for streamlined detection pipeline.

This module provides a single VLM client that handles both image analysis
and decision-making in one inference call, eliminating the need for a
separate decision LLM stage.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result from the unified VLM detection pipeline."""
    
    detected: bool
    confidence: float
    reasoning: str
    box_count: int
    shipping_label_present: Optional[bool]
    should_alert: bool
    raw_response: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "box_count": self.box_count,
            "shipping_label_present": self.shipping_label_present,
            "should_alert": self.should_alert,
            "raw_response": self.raw_response,
        }


# Streamlined prompt that asks for JSON directly from the VLM
DEFAULT_DETECTION_PROMPT = """Analyze this image for shipping/packaging boxes.

TASK: Look for brown cardboard shipping boxes and check if they have shipping labels.

A SHIPPING LABEL is: White/light paper sticker with printed address, barcode, or tracking info.
NOT a shipping label: Product logos, handwritten text, tape, or markings printed on cardboard.

Respond with ONLY valid JSON (no other text):
{
  "detected": boolean (true or false),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Brief description of what you see",
  "box_count": integer,
  "shipping_label_present": boolean or null
}

RULES:
- detected=true if ANY cardboard shipping box is visible
- shipping_label_present=false if ANY box lacks a proper shipping label
- shipping_label_present=true only if ALL boxes have shipping labels
- If no boxes, set detected=false, box_count=0, shipping_label_present=null
"""


class UnifiedVLMClient:
    """Unified Vision Language Model client for detection and decision-making.
    
    This client combines image analysis and decision logic into a single
    VLM inference call, reducing latency and complexity.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 300,
        prompt: Optional[str] = None,
    ):
        if not model:
            raise ValueError("Vision model name must be provided")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.prompt = prompt or DEFAULT_DETECTION_PROMPT
        self.session = requests.Session()
        user_agent = os.getenv("CAMERA_AGENT_USER_AGENT", "camera-agent/1.0")
        self.session.headers.update({"User-Agent": user_agent})
        
        logger.info(
            "Initialized Unified VLM client for model '%s' at %s (timeout=%ds)",
            model,
            self.base_url,
            timeout,
        )
        self._ensure_model_available()

    def _check_model_exists(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            for model_info in models:
                model_name = model_info.get("name", "")
                if model_name == self.model or model_name.startswith(f"{self.model}:"):
                    return True
            return False
        except Exception as exc:
            logger.warning("Failed to check if model exists: %s", exc)
            return False

    def _pull_model(self) -> bool:
        try:
            logger.info("🔄 Pulling model '%s'...", self.model)
            payload = {"name": self.model, "stream": False}
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=600,
            )
            response.raise_for_status()
            logger.info("✅ Successfully pulled model '%s'", self.model)
            return True
        except Exception as exc:
            logger.error("❌ Failed to pull model '%s': %s", self.model, exc)
            return False

    def _ensure_model_available(self) -> None:
        if not self._check_model_exists():
            logger.warning("Model '%s' not found. Attempting to pull...", self.model)
            if not self._pull_model():
                logger.error(
                    "Could not pull model '%s'. Run 'ollama pull %s' manually.",
                    self.model,
                    self.model,
                )
        else:
            logger.info("Model '%s' is available", self.model)

    def test_connection(self) -> bool:
        """Test connectivity to Ollama."""
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=10)
            response.raise_for_status()
            logger.info("Ollama connection test successful")
            return True
        except Exception as exc:
            logger.error("Ollama connection test failed: %s", exc)
            return False

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode numpy frame to base64 JPEG."""
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode frame to JPEG")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from model response, handling various formats."""
        if not response_text:
            return None
        
        # Clean up common VLM artifacts
        text = response_text.strip()
        text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        
        # Find JSON object
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            logger.warning("No JSON object found in response: %s", text[:200])
            return None
        
        json_str = text[start_idx:end_idx + 1]
        
        # Try parsing
        for attempt_text in [json_str, json_str.replace("'", '"')]:
            try:
                return json.loads(attempt_text)
            except json.JSONDecodeError:
                continue
        
        logger.warning("Failed to parse JSON from response: %s", json_str[:200])
        return None

    def _coerce_bool(self, value: Any) -> Optional[bool]:
        """Convert various values to boolean."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower().strip()
            if lowered in ("true", "yes", "1"):
                return True
            if lowered in ("false", "no", "0"):
                return False
            if lowered in ("null", "none", "unknown"):
                return None
        return None

    def analyze_frame(
        self,
        frame: np.ndarray,
        cv_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[DetectionResult]:
        """Analyze a frame and return detection result.
        
        Args:
            frame: OpenCV/numpy array (BGR format)
            cv_context: Optional context from computer vision preprocessing
            
        Returns:
            DetectionResult or None if analysis failed
        """
        try:
            base64_image = self._encode_frame(frame)
            
            # Build prompt with optional CV context
            prompt = self.prompt
            if cv_context:
                box_count = cv_context.get("packaging_box_count", 0)
                rfdet_hint = cv_context.get("rfdet_hint", "")
                if box_count > 0 or rfdet_hint:
                    context_line = f"\n\nNote: Object detector found approximately {box_count} potential box(es)."
                    if rfdet_hint:
                        context_line += f" {rfdet_hint}"
                    prompt = prompt + context_line
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
            }
            
            logger.debug("Sending request to VLM...")
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            
            if response.status_code == 404:
                logger.warning("Model not found, attempting to pull...")
                if self._pull_model():
                    response = self.session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self.timeout,
                    )
                else:
                    raise RuntimeError(f"Model '{self.model}' not available")
            
            response.raise_for_status()
            result = response.json()
            raw_response = result.get("response", "")
            
            # Parse JSON from response
            parsed = self._parse_json_response(raw_response)
            if not parsed:
                logger.warning("Could not parse VLM response as JSON")
                # Create a fallback result based on raw text analysis
                return self._create_fallback_result(raw_response)
            
            # Extract fields with defaults
            detected = bool(parsed.get("detected", False))
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", raw_response[:200]))
            box_count = int(parsed.get("box_count", 0) or 0)
            shipping_label_present = self._coerce_bool(parsed.get("shipping_label_present"))
            
            # Determine if alert should be triggered
            # Alert when: boxes detected AND no shipping labels on at least one
            should_alert = detected and (shipping_label_present is False)
            
            return DetectionResult(
                detected=detected,
                confidence=confidence,
                reasoning=reasoning,
                box_count=box_count,
                shipping_label_present=shipping_label_present,
                should_alert=should_alert,
                raw_response=raw_response,
            )
            
        except requests.exceptions.Timeout:
            logger.error("VLM request timed out after %ds", self.timeout)
            raise
        except requests.exceptions.RequestException as exc:
            logger.error("VLM request failed: %s", exc)
            raise
        except Exception as exc:
            logger.error("VLM analysis failed: %s", exc)
            raise

    def _create_fallback_result(self, raw_response: str) -> DetectionResult:
        """Create a fallback result when JSON parsing fails."""
        raw_lower = raw_response.lower()
        
        # Try to infer from text
        has_box = any(word in raw_lower for word in ["box", "package", "cardboard", "shipping"])
        has_label = any(word in raw_lower for word in ["label", "sticker", "barcode", "address"])
        no_box = any(phrase in raw_lower for phrase in ["no box", "no package", "empty", "no cardboard"])
        
        if no_box or not has_box:
            return DetectionResult(
                detected=False,
                confidence=0.3,
                reasoning=f"Fallback: {raw_response[:150]}",
                box_count=0,
                shipping_label_present=None,
                should_alert=False,
                raw_response=raw_response,
            )
        
        return DetectionResult(
            detected=True,
            confidence=0.5,
            reasoning=f"Fallback: {raw_response[:150]}",
            box_count=1,
            shipping_label_present=has_label,
            should_alert=not has_label,
            raw_response=raw_response,
        )
