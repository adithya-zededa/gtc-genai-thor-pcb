"""Unified Vision Language Model client for streamlined detection pipeline.

This module provides a single VLM client that handles both image analysis
and decision-making in one inference call, eliminating the need for a
separate decision LLM stage.

Supported backends:
- Ollama: Local LLM server with /api/generate endpoint
- vLLM: High-performance inference with OpenAI-compatible /v1/chat/completions endpoint
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np
import requests

from core.logging import get_logger
from .task_types import TaskType
from .prompts import TASK_PROMPTS, DEFAULT_DETECTION_PROMPT, CUSTOM_QUERY_TEMPLATE

if TYPE_CHECKING:
    from agents.tools.base import ToolExecutor, ToolCall

logger = get_logger(__name__)


class VLMBackend(Enum):
    """Supported VLM backend types."""
    OLLAMA = "ollama"
    VLLM = "vllm"


@dataclass
class AnalysisResult:
    """Generic result from any VLM analysis task."""
    
    task_type: str
    detected: bool
    confidence: float
    reasoning: str
    should_alert: bool
    raw_response: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "detected": self.detected,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "should_alert": self.should_alert,
            "raw_response": self.raw_response,
            "details": self.details,
        }
    
    @property
    def box_count(self) -> int:
        return self.details.get("box_count", 0)
    
    @property
    def shipping_label_present(self) -> Optional[bool]:
        return self.details.get("shipping_label_present")
    
    @property
    def person_count(self) -> int:
        return self.details.get("person_count", 0)
    
    @property
    def helmet_count(self) -> int:
        return self.details.get("helmet_count", 0)
    
    @property
    def no_helmet_count(self) -> int:
        return self.details.get("no_helmet_count", 0)


@dataclass
class AgenticResult:
    """Result from an agentic analysis with tool calling."""
    
    analysis: Optional[AnalysisResult]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "raw_response": self.raw_response,
        }
    
    @property
    def tools_used(self) -> List[str]:
        return [tc.get("tool", "") for tc in self.tool_calls]
    
    @property
    def any_tools_called(self) -> bool:
        return len(self.tool_calls) > 0
    
    @property
    def all_tools_succeeded(self) -> bool:
        return all(r.get("success", False) for r in self.tool_results)


@dataclass
class DetectionResult:
    """Result from the unified VLM detection pipeline (legacy)."""
    
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
    
    def to_analysis_result(self) -> AnalysisResult:
        return AnalysisResult(
            task_type=TaskType.PACKAGE_DETECTION.value,
            detected=self.detected,
            confidence=self.confidence,
            reasoning=self.reasoning,
            should_alert=self.should_alert,
            raw_response=self.raw_response,
            details={
                "box_count": self.box_count,
                "shipping_label_present": self.shipping_label_present,
            },
        )


# =============================================================================
# ALERT CONDITION FUNCTIONS
# =============================================================================

def _package_alert_condition(parsed: Dict[str, Any]) -> bool:
    """Determine if alert should trigger for package detection."""
    detected = bool(parsed.get("detected", False))
    shipping_label_present = parsed.get("shipping_label_present")
    return detected and (shipping_label_present is False)


def _ppe_alert_condition(parsed: Dict[str, Any]) -> bool:
    """Determine if alert should trigger for PPE detection."""
    compliance_status = parsed.get("compliance_status", "").lower()
    no_helmet = int(parsed.get("no_helmet_count", 0) or 0)
    no_vest = int(parsed.get("no_reflective_vest_count", 0) or 0)
    return compliance_status == "non_compliant" or no_helmet > 0 or no_vest > 0


def _person_counting_alert_condition(parsed: Dict[str, Any]) -> bool:
    """Determine if alert should trigger for person counting (never by default)."""
    return False


def _scene_description_alert_condition(parsed: Dict[str, Any]) -> bool:
    """Determine if alert should trigger for scene description (never by default)."""
    return False


def _default_alert_condition(parsed: Dict[str, Any]) -> bool:
    """Default alert condition - trigger if explicitly marked."""
    return bool(parsed.get("should_alert", False))


ALERT_CONDITIONS: Dict[TaskType, Callable[[Dict[str, Any]], bool]] = {
    TaskType.PACKAGE_DETECTION: _package_alert_condition,
    TaskType.PPE_DETECTION: _ppe_alert_condition,
    TaskType.PERSON_COUNTING: _person_counting_alert_condition,
    TaskType.SCENE_DESCRIPTION: _scene_description_alert_condition,
    TaskType.CUSTOM: _default_alert_condition,
}


class UnifiedVLMClient:
    """Unified Vision Language Model client for detection and decision-making."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 300,
        prompt: Optional[str] = None,
        default_task_type: TaskType = TaskType.PACKAGE_DETECTION,
        backend: VLMBackend = VLMBackend.VLLM,
        temperature: float = 0.1,
    ):
        if not model:
            raise ValueError("Vision model name must be provided")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.default_task_type = default_task_type
        self.backend = backend
        self.temperature = temperature
        self.prompt = prompt or TASK_PROMPTS.get(default_task_type, DEFAULT_DETECTION_PROMPT)
        self.session = requests.Session()
        user_agent = os.getenv("CAMERA_AGENT_USER_AGENT", "camera-agent/1.0")
        self.session.headers.update({"User-Agent": user_agent})
        
        backend_name = backend.value if isinstance(backend, VLMBackend) else backend
        logger.info(
            "Initialized Unified VLM client for model '%s' at %s (backend=%s, timeout=%ds)",
            model,
            self.base_url,
            backend_name,
            timeout,
        )
        self._ensure_model_available()

    def _check_model_exists(self) -> bool:
        """Check if model exists on the backend server."""
        if self.backend == VLMBackend.VLLM:
            return self._check_model_exists_vllm()
        return self._check_model_exists_ollama()

    def _check_model_exists_ollama(self) -> bool:
        """Check if model exists on Ollama."""
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

    def _check_model_exists_vllm(self) -> bool:
        """Check if model exists on vLLM server."""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            for model_info in models:
                model_id = model_info.get("id", "")
                if model_id == self.model or self.model in model_id:
                    return True
            return len(models) > 0
        except Exception as exc:
            logger.warning("Failed to check if model exists on vLLM: %s", exc)
            return False

    def _pull_model(self) -> bool:
        """Pull model - only applicable for Ollama backend."""
        if self.backend == VLMBackend.VLLM:
            logger.info("vLLM does not support model pulling - model must be pre-loaded")
            return False
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
        """Ensure the model is available on the backend."""
        if not self._check_model_exists():
            if self.backend == VLMBackend.VLLM:
                logger.warning(
                    "Model '%s' not found on vLLM server. Ensure vLLM is running with: vllm serve '%s'",
                    self.model,
                    self.model,
                )
            else:
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
        """Test connectivity to the VLM backend."""
        if self.backend == VLMBackend.VLLM:
            return self._test_connection_vllm()
        return self._test_connection_ollama()

    def _test_connection_ollama(self) -> bool:
        """Test connectivity to Ollama."""
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=10)
            response.raise_for_status()
            logger.info("Ollama connection test successful")
            return True
        except Exception as exc:
            logger.error("Ollama connection test failed: %s", exc)
            return False

    def _test_connection_vllm(self) -> bool:
        """Test connectivity to vLLM."""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            logger.info("vLLM connection test successful")
            return True
        except Exception as exc:
            logger.error("vLLM connection test failed: %s", exc)
            return False

    def _resize_for_inference(self, frame: np.ndarray, max_dimension: int = 1024) -> np.ndarray:
        """Resize image for efficient inference while maintaining aspect ratio."""
        height, width = frame.shape[:2]
        
        if max(height, width) <= max_dimension:
            return frame
        
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.debug("Resized image from %dx%d to %dx%d for inference", width, height, new_width, new_height)
        return resized

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode numpy frame to base64 JPEG with optional resizing."""
        resized_frame = self._resize_for_inference(frame)
        success, buffer = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode frame to JPEG")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from model response, handling various formats."""
        if not response_text:
            return None
        
        text = response_text.strip()
        text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.strip()
        
        if text.startswith("```json"):
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        elif text.startswith("```") and not text.startswith("```tool_call"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        
        tool_call_start = text.find("```tool_call")
        if tool_call_start > 0:
            text = text[:tool_call_start].strip()
        
        start_idx = text.find("{")
        if start_idx == -1:
            logger.warning("No JSON object found in response: %s", text[:200])
            return None
        
        depth = 0
        end_idx = -1
        for i, char in enumerate(text[start_idx:], start=start_idx):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            logger.warning("No matching closing brace in response: %s", text[:200])
            return None
        
        json_str = text[start_idx:end_idx + 1]
        
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

    def _build_prompt(
        self,
        task_type: TaskType,
        cv_context: Optional[Dict[str, Any]] = None,
        user_query: Optional[str] = None,
    ) -> str:
        """Build the prompt for a given task type."""
        if task_type == TaskType.CUSTOM:
            if not user_query:
                raise ValueError("user_query is required for CUSTOM task type")
            prompt = CUSTOM_QUERY_TEMPLATE.format(user_query=user_query)
        else:
            prompt = TASK_PROMPTS.get(task_type, DEFAULT_DETECTION_PROMPT)

        if cv_context:
            context_parts = []
            
            box_count = cv_context.get("packaging_box_count", 0)
            rfdet_hint = cv_context.get("rfdet_hint", "")
            if box_count > 0:
                context_parts.append(f"Object detector found approximately {box_count} potential box(es).")
            if rfdet_hint:
                context_parts.append(rfdet_hint)
            
            person_count = cv_context.get("person_count", 0)
            if person_count > 0:
                context_parts.append(f"Person detector found {person_count} person(s).")
            
            helmet_count = cv_context.get("helmet_count", 0)
            no_helmet_count = cv_context.get("no_helmet_count", 0)
            if helmet_count > 0 or no_helmet_count > 0:
                context_parts.append(
                    f"PPE detector found {helmet_count} with helmet, {no_helmet_count} without."
                )
            
            ml_confidence = cv_context.get("ml_confidence")
            if ml_confidence is not None:
                context_parts.append(f"ML confidence: {ml_confidence:.2f}")
            
            if context_parts:
                prompt = prompt + "\n\nNote: " + " ".join(context_parts)
        
        return prompt

    def _send_vlm_request(self, prompt: str, base64_image: str) -> str:
        """Send request to VLM and return raw response."""
        if self.backend == VLMBackend.VLLM:
            return self._send_vllm_request(prompt, base64_image)
        return self._send_ollama_request(prompt, base64_image)

    def _send_vllm_request(self, prompt: str, base64_image: str) -> str:
        """Send request to vLLM using OpenAI-compatible chat completions API."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1024,
        }
        
        logger.debug("Sending request to vLLM (task prompt length: %d)", len(prompt))
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        
        if not response.ok:
            try:
                error_body = response.json()
                logger.error("vLLM request failed with status %d: %s", response.status_code, error_body)
            except Exception:
                logger.error("vLLM request failed with status %d: %s", response.status_code, response.text[:500])
        
        response.raise_for_status()
        result = response.json()
        
        choices = result.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")
        return ""

    def _send_ollama_request(self, prompt: str, base64_image: str) -> str:
        """Send request to Ollama using the generate API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
        }
        
        logger.debug("Sending request to Ollama (task prompt length: %d)", len(prompt))
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
        return result.get("response", "")

    def analyze(
        self,
        frame: np.ndarray,
        task_type: Optional[TaskType] = None,
        cv_context: Optional[Dict[str, Any]] = None,
        user_query: Optional[str] = None,
        custom_alert_condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Optional[AnalysisResult]:
        """Analyze a frame with dynamic task type selection."""
        effective_task_type = task_type or self.default_task_type

        try:
            base64_image = self._encode_frame(frame)
            prompt = self._build_prompt(effective_task_type, cv_context, user_query)
            raw_response = self._send_vlm_request(prompt, base64_image)
            
            parsed = self._parse_json_response(raw_response)
            if not parsed:
                logger.warning("Could not parse VLM response as JSON")
                return self._create_fallback_analysis_result(effective_task_type, raw_response)
            
            detected = bool(parsed.get("detected", False))
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", raw_response[:200]))
            
            if custom_alert_condition:
                should_alert = custom_alert_condition(parsed)
            else:
                alert_fn = ALERT_CONDITIONS.get(effective_task_type, _default_alert_condition)
                should_alert = alert_fn(parsed)
            
            details = {k: v for k, v in parsed.items() 
                      if k not in ("detected", "confidence", "reasoning")}
            
            for bool_field in ["shipping_label_present"]:
                if bool_field in details:
                    details[bool_field] = self._coerce_bool(details[bool_field])
            
            return AnalysisResult(
                task_type=effective_task_type.value,
                detected=detected,
                confidence=confidence,
                reasoning=reasoning,
                should_alert=should_alert,
                raw_response=raw_response,
                details=details,
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

    def _create_fallback_analysis_result(
        self,
        task_type: TaskType,
        raw_response: str,
    ) -> AnalysisResult:
        """Create a fallback AnalysisResult when JSON parsing fails."""
        raw_lower = raw_response.lower()
        
        if task_type == TaskType.PACKAGE_DETECTION:
            has_box = any(word in raw_lower for word in ["box", "package", "cardboard", "shipping"])
            has_label = any(word in raw_lower for word in ["label", "sticker", "barcode", "address"])
            no_box = any(phrase in raw_lower for phrase in ["no box", "no package", "empty"])
            
            if no_box or not has_box:
                return AnalysisResult(
                    task_type=task_type.value,
                    detected=False,
                    confidence=0.3,
                    reasoning=f"Fallback: {raw_response[:150]}",
                    should_alert=False,
                    raw_response=raw_response,
                    details={"box_count": 0, "shipping_label_present": None},
                )
            return AnalysisResult(
                task_type=task_type.value,
                detected=True,
                confidence=0.5,
                reasoning=f"Fallback: {raw_response[:150]}",
                should_alert=not has_label,
                raw_response=raw_response,
                details={"box_count": 1, "shipping_label_present": has_label},
            )
        
        elif task_type == TaskType.PPE_DETECTION:
            has_person = any(word in raw_lower for word in ["person", "people", "worker", "man", "woman"])
            has_helmet = any(word in raw_lower for word in ["helmet", "hard hat", "hardhat"])
            no_helmet = "no helmet" in raw_lower or "without helmet" in raw_lower
            
            return AnalysisResult(
                task_type=task_type.value,
                detected=has_person,
                confidence=0.3,
                reasoning=f"Fallback: {raw_response[:150]}",
                should_alert=no_helmet,
                raw_response=raw_response,
                details={
                    "person_count": 1 if has_person else 0,
                    "helmet_count": 1 if has_helmet and not no_helmet else 0,
                    "no_helmet_count": 1 if no_helmet else 0,
                },
            )
        
        elif task_type == TaskType.PERSON_COUNTING:
            numbers = re.findall(r'\b(\d+)\s*(?:person|people|individual)', raw_lower)
            count = int(numbers[0]) if numbers else (1 if "person" in raw_lower else 0)
            
            return AnalysisResult(
                task_type=task_type.value,
                detected=count > 0,
                confidence=0.3,
                reasoning=f"Fallback: {raw_response[:150]}",
                should_alert=False,
                raw_response=raw_response,
                details={"person_count": count},
            )
        
        return AnalysisResult(
            task_type=task_type.value,
            detected=True,
            confidence=0.3,
            reasoning=f"Fallback: {raw_response[:150]}",
            should_alert=False,
            raw_response=raw_response,
            details={},
        )

    def analyze_frame(
        self,
        frame: np.ndarray,
        cv_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[DetectionResult]:
        """Analyze a frame and return detection result (legacy)."""
        result = self.analyze(frame, task_type=TaskType.PACKAGE_DETECTION, cv_context=cv_context)
        
        if result is None:
            return None
        
        return DetectionResult(
            detected=result.detected,
            confidence=result.confidence,
            reasoning=result.reasoning,
            box_count=result.details.get("box_count", 0),
            shipping_label_present=result.details.get("shipping_label_present"),
            should_alert=result.should_alert,
            raw_response=result.raw_response,
        )

    def analyze_with_tools(
        self,
        frame: np.ndarray,
        tool_executor: "ToolExecutor",
        task_type: Optional[TaskType] = None,
        cv_context: Optional[Dict[str, Any]] = None,
        user_query: Optional[str] = None,
        max_tool_rounds: int = 3,
        recipients: Optional[List[str]] = None,
    ) -> AgenticResult:
        """Analyze a frame with tool-calling capability."""
        from agents.tools.base import ToolCall
        
        if user_query:
            effective_task_type = TaskType.CUSTOM
        else:
            effective_task_type = task_type or self.default_task_type
        
        base64_image = self._encode_frame(frame)
        image_bytes = self._frame_to_bytes(frame)
        
        tool_executor.context["image_data"] = image_bytes
        tool_executor.context["recipients"] = recipients or []
        
        if effective_task_type == TaskType.CUSTOM:
            agentic_base = f"""Analyze this image based on the following instructions:

{user_query}

First, provide your analysis as JSON:
{{
  "detected": boolean (true if the condition in the instructions is met),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Your analysis and findings",
  "should_alert": boolean (true if action should be taken),
  "details": {{any additional structured data}}
}}

Then, if the user's instructions require an action (like sending an email), you MUST use the tools below to complete that action."""
        else:
            agentic_base = self._build_prompt(effective_task_type, cv_context, user_query)
        
        tools_prompt = self._get_tools_prompt(tool_executor)
        
        agentic_prompt = f"""{agentic_base}

---

{tools_prompt}

IMPORTANT: Follow the user's instructions exactly. If they ask you to send an email to a specific address, use THAT EXACT address in the tool call.

After your JSON analysis, if the condition in the user's instructions is met, call the appropriate tools to complete the action.
"""
        
        all_tool_calls: List[ToolCall] = []
        all_tool_results: List[Dict[str, Any]] = []
        analysis_result: Optional[AnalysisResult] = None
        raw_response = ""
        
        for round_num in range(max_tool_rounds):
            raw_response = self._send_vlm_request(agentic_prompt, base64_image)
            
            if round_num == 0:
                parsed = self._parse_json_response(raw_response)
                if parsed:
                    detected = bool(parsed.get("detected", False))
                    confidence = float(parsed.get("confidence", 0.5))
                    reasoning = str(parsed.get("reasoning", raw_response[:200]))
                    
                    if effective_task_type == TaskType.CUSTOM:
                        should_alert = bool(parsed.get("should_alert", detected))
                    else:
                        alert_fn = ALERT_CONDITIONS.get(effective_task_type, _default_alert_condition)
                        should_alert = alert_fn(parsed)
                    
                    details = {k: v for k, v in parsed.items() 
                              if k not in ("detected", "confidence", "reasoning")}
                    
                    analysis_result = AnalysisResult(
                        task_type=effective_task_type.value,
                        detected=detected,
                        confidence=confidence,
                        reasoning=reasoning,
                        should_alert=should_alert,
                        raw_response=raw_response,
                        details=details,
                    )
                else:
                    analysis_result = self._create_fallback_analysis_result(
                        effective_task_type, raw_response
                    )
            
            tool_calls = self._parse_tool_calls(raw_response)
            
            if not tool_calls:
                break
            
            results = tool_executor.execute_many(tool_calls)
            all_tool_calls.extend(tool_calls)
            all_tool_results.extend([r.to_dict() for r in results])
            
            results_text = "\n".join([
                f"Tool '{r.tool_name}' result: {json.dumps(r.result)}"
                for r in results
            ])
            
            agentic_prompt = f"""Previous tool calls completed:

{results_text}

Based on these results, do you need to take any additional actions?
If yes, make more tool calls. If no, summarize what was done.
"""
        
        return AgenticResult(
            analysis=analysis_result,
            tool_calls=[{"tool": tc.tool_name, "arguments": tc.arguments} for tc in all_tool_calls],
            tool_results=all_tool_results,
            raw_response=raw_response,
        )

    def _get_tools_prompt(self, tool_executor: "ToolExecutor") -> str:
        """Get the tools prompt from the executor."""
        from agents.tools.base import TOOL_REGISTRY
        
        tools_desc = []
        for name, tool in TOOL_REGISTRY.items():
            params_desc = json.dumps(tool.parameters, indent=2)
            tools_desc.append(f"Tool: {name}\nDescription: {tool.description}\nParameters: {params_desc}")
        
        return """You have access to the following tools:

""" + "\n\n".join(tools_desc) + """

To call a tool, use this format:
```tool_call
{"tool": "tool_name", "arguments": {"param1": "value1"}}
```
"""

    def _parse_tool_calls(self, response: str) -> List["ToolCall"]:
        """Parse tool calls from LLM response."""
        from agents.tools.base import ToolCall
        
        calls = []
        pattern = r'```tool_call\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                calls.append(ToolCall(
                    tool_name=data.get("tool", ""),
                    arguments=data.get("arguments", {}),
                ))
            except json.JSONDecodeError:
                logger.warning("Failed to parse tool call: %s", match[:100])
        
        return calls

    def _frame_to_bytes(self, frame: np.ndarray) -> bytes:
        """Convert numpy frame to JPEG bytes."""
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buffer.tobytes()
