"""Unified Vision Language Model client for streamlined detection pipeline.

This module provides a single VLM client that handles both image analysis
and decision-making in one inference call, eliminating the need for a
separate decision LLM stage.

The client supports dynamic prompting for multi-purpose analysis:
- Package/shipping box detection
- PPE (Personal Protective Equipment) detection
- General scene description
- Custom user-defined queries
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np
import requests

# Type checking import to avoid circular dependency
if TYPE_CHECKING:
    from agent_runtime.memory import ConversationalMemory

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Supported analysis task types."""
    PACKAGE_DETECTION = "package_detection"
    PPE_DETECTION = "ppe_detection"
    PERSON_COUNTING = "person_counting"
    SCENE_DESCRIPTION = "scene_description"
    CUSTOM = "custom"


@dataclass
class AnalysisResult:
    """Generic result from any VLM analysis task.
    
    This flexible dataclass supports multiple use cases by storing
    task-specific data in the `details` dictionary.
    """
    
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
    
    # Convenience accessors for common fields
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


# Legacy alias for backward compatibility
@dataclass
class DetectionResult:
    """Result from the unified VLM detection pipeline.
    
    DEPRECATED: Use AnalysisResult for new code.
    This class is maintained for backward compatibility.
    """
    
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
        """Convert to the new AnalysisResult format."""
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
# PROMPT TEMPLATES FOR DIFFERENT TASK TYPES
# =============================================================================

# Package/Shipping Box Detection
PACKAGE_DETECTION_PROMPT = """Analyze this image for shipping/packaging boxes.

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

# PPE (Personal Protective Equipment) Detection
PPE_DETECTION_PROMPT = """Analyze this image for Personal Protective Equipment (PPE) compliance.

TASK: Identify all people in the scene and check if they are wearing required safety equipment.

PPE items to detect:
- Hard hat / Safety helmet
- Reflective vest / High-visibility jacket
- Safety glasses / Goggles
- Gloves

Respond with ONLY valid JSON (no other text):
{
  "detected": boolean (true if any people visible),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Brief description of PPE compliance status",
  "person_count": integer,
  "helmet_count": integer (people wearing hard hats),
  "no_helmet_count": integer (people without hard hats),
  "reflective_vest_count": integer,
  "no_reflective_vest_count": integer,
  "compliance_status": "compliant" or "non_compliant" or "no_people"
}

RULES:
- detected=true if ANY person is visible
- compliance_status="non_compliant" if ANY person lacks required PPE
- compliance_status="compliant" only if ALL people have required PPE
- If no people, set detected=false, all counts=0, compliance_status="no_people"
"""

# Person Counting
PERSON_COUNTING_PROMPT = """Count the number of people visible in this image.

TASK: Accurately count all visible people, including partially visible individuals.

Respond with ONLY valid JSON (no other text):
{
  "detected": boolean (true if any people visible),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Brief description of what you see",
  "person_count": integer,
  "fully_visible": integer (people fully in frame),
  "partially_visible": integer (people partially in frame)
}

RULES:
- Count each person only once
- Include people in the background if clearly identifiable as human
- detected=true if person_count > 0
"""

# General Scene Description
SCENE_DESCRIPTION_PROMPT = """Describe what you see in this image.

TASK: Provide a detailed but concise description of the scene.

Respond with ONLY valid JSON (no other text):
{
  "detected": true,
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Detailed description of the scene",
  "scene_type": "indoor" or "outdoor" or "unknown",
  "objects": ["list", "of", "main", "objects"],
  "activity": "Description of any activity or action happening"
}
"""

# Custom query template (user provides the task)
CUSTOM_QUERY_TEMPLATE = """Analyze this image based on the following instructions:

{user_query}

Respond with ONLY valid JSON (no other text):
{{
  "detected": boolean (true if the condition in the instructions is met),
  "confidence": number between 0.0 and 1.0,
  "reasoning": "Your analysis and findings",
  "should_alert": boolean (true if user should be alerted based on the instructions),
  "details": {{any additional structured data}}
}}
"""

# Registry mapping task types to their prompts
TASK_PROMPTS: Dict[TaskType, str] = {
    TaskType.PACKAGE_DETECTION: PACKAGE_DETECTION_PROMPT,
    TaskType.PPE_DETECTION: PPE_DETECTION_PROMPT,
    TaskType.PERSON_COUNTING: PERSON_COUNTING_PROMPT,
    TaskType.SCENE_DESCRIPTION: SCENE_DESCRIPTION_PROMPT,
}

# Legacy alias
DEFAULT_DETECTION_PROMPT = PACKAGE_DETECTION_PROMPT


# =============================================================================
# ALERT CONDITION FUNCTIONS
# =============================================================================

def _package_alert_condition(parsed: Dict[str, Any]) -> bool:
    """Determine if alert should trigger for package detection."""
    detected = bool(parsed.get("detected", False))
    shipping_label_present = parsed.get("shipping_label_present")
    # Alert when: boxes detected AND no shipping labels on at least one
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


# Registry mapping task types to their alert conditions
ALERT_CONDITIONS: Dict[TaskType, Callable[[Dict[str, Any]], bool]] = {
    TaskType.PACKAGE_DETECTION: _package_alert_condition,
    TaskType.PPE_DETECTION: _ppe_alert_condition,
    TaskType.PERSON_COUNTING: _person_counting_alert_condition,
    TaskType.SCENE_DESCRIPTION: _scene_description_alert_condition,
    TaskType.CUSTOM: _default_alert_condition,
}


class UnifiedVLMClient:
    """Unified Vision Language Model client for detection and decision-making.
    
    This client combines image analysis and decision logic into a single
    VLM inference call, reducing latency and complexity.
    
    Supports multiple task types:
    - PACKAGE_DETECTION: Detect shipping boxes and labels
    - PPE_DETECTION: Detect personal protective equipment compliance
    - PERSON_COUNTING: Count people in the scene
    - SCENE_DESCRIPTION: General scene understanding
    - CUSTOM: User-defined queries with custom prompts
    
    Example usage:
        client = UnifiedVLMClient(base_url, model)
        
        # Package detection (default)
        result = client.analyze_frame(frame)
        
        # PPE detection
        result = client.analyze(frame, task_type=TaskType.PPE_DETECTION)
        
        # Custom query
        result = client.analyze(
            frame,
            task_type=TaskType.CUSTOM,
            user_query="Count all red objects in the scene"
        )
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 300,
        prompt: Optional[str] = None,
        default_task_type: TaskType = TaskType.PACKAGE_DETECTION,
    ):
        if not model:
            raise ValueError("Vision model name must be provided")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.default_task_type = default_task_type
        # Use provided prompt or get from task registry
        self.prompt = prompt or TASK_PROMPTS.get(default_task_type, DEFAULT_DETECTION_PROMPT)
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

    def _build_prompt(
        self,
        task_type: TaskType,
        cv_context: Optional[Dict[str, Any]] = None,
        user_query: Optional[str] = None,
        memory_context: Optional[str] = None,
    ) -> str:
        """Build the prompt for a given task type.
        
        Args:
            task_type: The type of analysis task
            cv_context: Optional context from CV preprocessing
            user_query: Custom query for CUSTOM task type
            memory_context: Optional context from conversation memory
            
        Returns:
            The complete prompt string
        """
        if task_type == TaskType.CUSTOM:
            if not user_query:
                raise ValueError("user_query is required for CUSTOM task type")
            prompt = CUSTOM_QUERY_TEMPLATE.format(user_query=user_query)
        else:
            prompt = TASK_PROMPTS.get(task_type, DEFAULT_DETECTION_PROMPT)
        
        # Add memory context if provided (at the beginning for better coherence)
        if memory_context:
            prompt = f"Context from previous observations:\n{memory_context}\n\n---\n\n{prompt}"
        
        # Add CV context if provided
        if cv_context:
            context_parts = []
            
            # Package detection context
            box_count = cv_context.get("packaging_box_count", 0)
            rfdet_hint = cv_context.get("rfdet_hint", "")
            if box_count > 0:
                context_parts.append(f"Object detector found approximately {box_count} potential box(es).")
            if rfdet_hint:
                context_parts.append(rfdet_hint)
            
            # PPE detection context
            person_count = cv_context.get("person_count", 0)
            if person_count > 0:
                context_parts.append(f"Person detector found {person_count} person(s).")
            
            helmet_count = cv_context.get("helmet_count", 0)
            no_helmet_count = cv_context.get("no_helmet_count", 0)
            if helmet_count > 0 or no_helmet_count > 0:
                context_parts.append(
                    f"PPE detector found {helmet_count} with helmet, {no_helmet_count} without."
                )
            
            # Generic context
            ml_confidence = cv_context.get("ml_confidence")
            if ml_confidence is not None:
                context_parts.append(f"ML confidence: {ml_confidence:.2f}")
            
            if context_parts:
                prompt = prompt + "\n\nNote: " + " ".join(context_parts)
        
        return prompt

    def _send_vlm_request(self, prompt: str, base64_image: str) -> str:
        """Send request to VLM and return raw response.
        
        Args:
            prompt: The prompt to send
            base64_image: Base64 encoded image
            
        Returns:
            Raw response text from VLM
            
        Raises:
            RuntimeError: If model is not available
            requests.exceptions.RequestException: If request fails
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
        }
        
        logger.debug("Sending request to VLM (task prompt length: %d)", len(prompt))
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
        memory: Optional["ConversationalMemory"] = None,
        include_memory_context: bool = True,
    ) -> Optional[AnalysisResult]:
        """Analyze a frame with dynamic task type selection.
        
        This is the main entry point for multi-purpose analysis. It supports:
        - Predefined task types (PACKAGE_DETECTION, PPE_DETECTION, etc.)
        - Custom user queries with arbitrary prompts
        - Custom alert conditions
        - Memory-enhanced prompts with conversation context
        
        Args:
            frame: OpenCV/numpy array (BGR format)
            task_type: Type of analysis to perform (defaults to client's default)
            cv_context: Optional context from CV preprocessing
            user_query: Custom query string (required for CUSTOM task type)
            custom_alert_condition: Optional function to determine alerting
            memory: Optional ConversationalMemory for context
            include_memory_context: Whether to include memory context in prompt
            
        Returns:
            AnalysisResult or None if analysis failed
            
        Example:
            # PPE detection
            result = client.analyze(frame, task_type=TaskType.PPE_DETECTION)
            
            # Custom query with memory
            result = client.analyze(
                frame,
                task_type=TaskType.CUSTOM,
                user_query="Count all vehicles in the parking lot",
                memory=agent_memory
            )
        """
        effective_task_type = task_type or self.default_task_type
        
        # Build memory context if available
        memory_context = None
        if memory and include_memory_context:
            try:
                memory_context = memory.get_context_for_prompt()
            except Exception as e:
                logger.warning("Failed to get memory context: %s", e)
        
        try:
            base64_image = self._encode_frame(frame)
            prompt = self._build_prompt(
                effective_task_type, 
                cv_context, 
                user_query,
                memory_context=memory_context,
            )
            raw_response = self._send_vlm_request(prompt, base64_image)
            
            # Parse JSON from response
            parsed = self._parse_json_response(raw_response)
            if not parsed:
                logger.warning("Could not parse VLM response as JSON")
                result = self._create_fallback_analysis_result(
                    effective_task_type, raw_response
                )
                # Record in memory even for fallback
                if memory:
                    memory.record_analysis(
                        task_type=effective_task_type.value,
                        detected=result.detected,
                        confidence=result.confidence,
                        reasoning=result.reasoning,
                        should_alert=result.should_alert,
                        details=result.details,
                        user_prompt=user_query,
                    )
                return result
            
            # Extract common fields
            detected = bool(parsed.get("detected", False))
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", raw_response[:200]))
            
            # Determine alert condition
            if custom_alert_condition:
                should_alert = custom_alert_condition(parsed)
            else:
                alert_fn = ALERT_CONDITIONS.get(effective_task_type, _default_alert_condition)
                should_alert = alert_fn(parsed)
            
            # Build details dict with all parsed fields (excluding common ones)
            details = {k: v for k, v in parsed.items() 
                      if k not in ("detected", "confidence", "reasoning")}
            
            # Coerce boolean fields
            for bool_field in ["shipping_label_present"]:
                if bool_field in details:
                    details[bool_field] = self._coerce_bool(details[bool_field])
            
            result = AnalysisResult(
                task_type=effective_task_type.value,
                detected=detected,
                confidence=confidence,
                reasoning=reasoning,
                should_alert=should_alert,
                raw_response=raw_response,
                details=details,
            )
            
            # Record successful analysis in memory
            if memory:
                memory.record_analysis(
                    task_type=effective_task_type.value,
                    detected=detected,
                    confidence=confidence,
                    reasoning=reasoning,
                    should_alert=should_alert,
                    details=details,
                    user_prompt=user_query,
                )
            
            return result
            
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
        
        # Task-specific fallback logic
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
            # Try to extract numbers
            import re
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
        
        # Generic fallback
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
        """Analyze a frame and return detection result.
        
        DEPRECATED: Use analyze() for new code. This method is maintained
        for backward compatibility with the package detection pipeline.
        
        Args:
            frame: OpenCV/numpy array (BGR format)
            cv_context: Optional context from computer vision preprocessing
            
        Returns:
            DetectionResult or None if analysis failed
        """
        # Use the new analyze method with package detection
        result = self.analyze(
            frame,
            task_type=TaskType.PACKAGE_DETECTION,
            cv_context=cv_context,
        )
        
        if result is None:
            return None
        
        # Convert to legacy DetectionResult
        return DetectionResult(
            detected=result.detected,
            confidence=result.confidence,
            reasoning=result.reasoning,
            box_count=result.details.get("box_count", 0),
            shipping_label_present=result.details.get("shipping_label_present"),
            should_alert=result.should_alert,
            raw_response=result.raw_response,
        )

    def _create_fallback_result(self, raw_response: str) -> DetectionResult:
        """Create a fallback result when JSON parsing fails.
        
        DEPRECATED: Use _create_fallback_analysis_result() for new code.
        """
        result = self._create_fallback_analysis_result(
            TaskType.PACKAGE_DETECTION, raw_response
        )
        return DetectionResult(
            detected=result.detected,
            confidence=result.confidence,
            reasoning=result.reasoning,
            box_count=result.details.get("box_count", 0),
            shipping_label_present=result.details.get("shipping_label_present"),
            should_alert=result.should_alert,
            raw_response=result.raw_response,
        )
