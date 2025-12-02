"""Inference gateways for vision and decision LLM stages."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from pydantic import ValidationError

from .models import DetectionDecision

logger = logging.getLogger(__name__)

ollama: Any | None
try:  # pragma: no cover - optional dependency
    import ollama  # type: ignore
except ImportError:  # pragma: no cover - best effort fallback
    ollama = None  # type: ignore


class OllamaVisionClient:
    """Client for interacting with Ollama vision models."""

    def __init__(self, base_url: str, model: str, timeout: int = 60, temperature: float = 0.1) -> None:
        if not model:
            raise ValueError("Vision model name must be provided")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.session = requests.Session()
        user_agent = os.getenv("CAMERA_AGENT_USER_AGENT", "camera-agent/1.0")
        self.session.headers.update({"User-Agent": user_agent})
        logger.info(
            "Initialized Ollama client for vision model '%s' at %s (temperature=%.2f)",
            model,
            self.base_url,
            self.temperature,
        )

        self._ensure_model_available()

    def _check_model_exists(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            for model in models:
                model_name = model.get("name", "")
                if model_name == self.model or model_name.startswith(f"{self.model}:"):
                    return True
            return False
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("Failed to check if model exists: %s", exc)
            return False

    def _pull_model(self) -> bool:
        try:
            logger.info(
                "🔄 Pulling Ollama model '%s'... This may take a few minutes.",
                self.model,
            )
            payload: Dict[str, Any] = {"name": self.model, "stream": False}
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=600,
            )
            response.raise_for_status()
            logger.info("✅ Successfully pulled model '%s'", self.model)
            return True
        except Exception as exc:  # pragma: no cover - network path
            logger.error("❌ Failed to pull model '%s': %s", self.model, exc)
            return False

    def _ensure_model_available(self) -> None:
        if not self._check_model_exists():
            logger.warning(
                "Model '%s' not found locally. Attempting to pull...",
                self.model,
            )
            if not self._pull_model():
                logger.error(
                    "Could not pull model '%s'. Please run 'ollama pull %s' manually.",
                    self.model,
                    self.model,
                )
        else:
            logger.info("Model '%s' is already available", self.model)

    def analyze_image(self, image_data: bytes, prompt: str) -> Dict[str, Any]:
        try:
            base64_image = base64.b64encode(image_data).decode("utf-8")
            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                },
            }
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 404:
                logger.warning(
                    "Model '%s' not found. Attempting to pull it now...",
                    self.model,
                )
                if self._pull_model():
                    response = self.session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self.timeout,
                    )
                else:
                    raise requests.exceptions.HTTPError(
                        f"Model '{self.model}' not available and could not be pulled",
                        response=response,
                    )
            response.raise_for_status()
            result = response.json()
            logger.debug("Ollama response: %s", result)
            return result
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to analyze image with Ollama: %s", exc)
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Unexpected error during image analysis: %s", exc)
            raise

    def analyze_frame(self, frame_array: Any, prompt: str) -> Dict[str, Any]:
        """
        Optimized method that accepts numpy array directly.
        
        This eliminates redundant decode/encode cycles by accepting the frame
        as a numpy array and encoding to Base64 only once at the API boundary.
        
        Args:
            frame_array: OpenCV/numpy array (BGR format)
            prompt: Text prompt for the vision model
            
        Returns:
            Dict containing the model's response
        """
        try:
            import cv2
            
            # Encode numpy array directly to JPEG bytes (more efficient than PNG)
            success, buffer = cv2.imencode('.jpg', frame_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise ValueError("Failed to encode frame to JPEG")
            
            # Convert to base64 only once
            image_bytes = buffer.tobytes()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                },
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            
            if response.status_code == 404:
                logger.warning(
                    "Model '%s' not found. Attempting to pull it now...",
                    self.model,
                )
                if self._pull_model():
                    response = self.session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self.timeout,
                    )
                else:
                    raise requests.exceptions.HTTPError(
                        f"Model '{self.model}' not available and could not be pulled",
                        response=response,
                    )
            
            response.raise_for_status()
            result = response.json()
            logger.debug("Ollama frame analysis response received")
            return result
            
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to analyze frame with Ollama: %s", exc)
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Unexpected error during frame analysis: %s", exc)
            raise

    def test_connection(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=10)
            response.raise_for_status()
            logger.info("Ollama connection test successful")
            return True
        except Exception as exc:
            logger.error("Ollama connection test failed: %s", exc)
            return False


class DecisionLLM:
    """Second-stage LLM for packaging box decisions with tool calling capability."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        system_prompt: str,
        user_prompt_template: str,
        tools: List[Dict[str, Any]],
        timeout: int = 30,
        temperature: float = 0.1,
    ) -> None:
        if not model:
            raise ValueError("Decision model name must be provided")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.session = requests.Session()
        user_agent = os.getenv("CAMERA_AGENT_USER_AGENT", "camera-agent/1.0")
        self.session.headers.update({"User-Agent": user_agent})
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.tools = tools or []
        self._tool_mode_supported = True
        self._python_client_available = False
        self._ollama_python_client = None
        self._captured_tool_calls: List[Dict[str, Any]] = []
        self._tool_function_map: Dict[str, Callable[..., str]] = {}
        self._tool_function_list: List[Callable[..., str]] = []
        logger.info("Initialized Decision LLM model '%s' at %s (temperature=%.2f)", model, self.base_url, self.temperature)

        if ollama is not None:
            try:
                self._ollama_python_client = ollama.Client(host=self.base_url)
                self._python_client_available = True
            except Exception as exc:  # pragma: no cover - optional path
                logger.warning("Unable to initialize Ollama Python client: %s", exc)
        else:  # pragma: no cover - optional dependency
            logger.debug("Ollama Python package not available; HTTP fallback will be used")

        if self.tools:
            self._tool_function_map, self._tool_function_list = self._prepare_tool_functions(self.tools)

        self._ensure_model_available()

    def _prepare_tool_functions(
        self, tool_specs: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Callable[..., str]], List[Callable[..., str]]]:
        function_map: Dict[str, Callable[..., str]] = {}

        for spec in tool_specs:
            if not isinstance(spec, dict):
                continue
            function_spec = spec.get("function")
            if not isinstance(function_spec, dict):
                continue

            name = function_spec.get("name")
            if not name:
                continue

            description = str(function_spec.get("description", "")).strip()
            parameters = {}
            required: List[str] = []
            param_block = function_spec.get("parameters")
            if isinstance(param_block, dict):
                params = param_block.get("properties")
                if isinstance(params, dict):
                    parameters = params
                req = param_block.get("required")
                if isinstance(req, list):
                    required = [str(item) for item in req if isinstance(item, str)]

            docstring = self._tool_docstring(name, description, parameters, required)

            if name == "trigger_packaging_alert":
                function_map[name] = self._build_trigger_alert_tool(docstring)
            elif name == "record_no_detection":
                function_map[name] = self._build_record_no_detection_tool(docstring)

        function_list = list(function_map.values())
        return function_map, function_list

    @staticmethod
    def _tool_docstring(
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: List[str],
    ) -> str:
        lines: List[str] = []
        summary = description or f"Tool '{name}'"
        lines.append(summary)

        if parameters:
            lines.append("\nArgs:")
            for param_name, schema in parameters.items():
                schema_desc = ""
                schema_type = ""
                if isinstance(schema, dict):
                    schema_desc = str(schema.get("description", "")).strip()
                    schema_type = schema.get("type") if isinstance(schema.get("type"), str) else ""
                required_flag = " (required)" if param_name in required else ""
                type_fragment = f"{schema_type} " if schema_type else ""
                line = f"    {param_name}: {type_fragment}{required_flag}".rstrip()
                if schema_desc:
                    line += f" - {schema_desc}"
                lines.append(line)

        return "\n".join(lines)

    def _register_tool_call(self, name: str, arguments: Dict[str, Any]) -> str:
        captured = {key: arguments.get(key) for key in arguments}
        self._captured_tool_calls.append({"name": name, "arguments": captured})
        logger.debug("Captured tool call '%s' with args %s", name, captured)
        try:
            response_body = json.dumps({"action": name, "arguments": captured})
        except TypeError:
            safe_arguments = {key: str(value) for key, value in captured.items()}
            response_body = json.dumps({"action": name, "arguments": safe_arguments})
        return response_body

    def _build_trigger_alert_tool(self, docstring: str) -> Callable[..., str]:
        def trigger_packaging_alert(
            confidence: Optional[float] = None,
            reasoning: Optional[str] = None,
            box_count: Optional[int] = None,
            label_count: Optional[int] = None,
            shipping_label_present: Optional[bool] = None,
            labels_per_box: Optional[float] = None,
            box_description: Optional[str] = None,
            notes: Optional[str] = None,
            **extra: Any,
        ) -> str:
            payload: Dict[str, Any] = {
                "confidence": confidence,
                "reasoning": reasoning,
                "box_count": box_count,
                "label_count": label_count,
                "shipping_label_present": shipping_label_present,
                "labels_per_box": labels_per_box,
                "box_description": box_description,
                "notes": notes,
            }
            for key, value in extra.items():
                if key not in payload:
                    payload[key] = value
            return self._register_tool_call("trigger_packaging_alert", payload)

        trigger_packaging_alert.__name__ = "trigger_packaging_alert"
        trigger_packaging_alert.__doc__ = docstring
        return trigger_packaging_alert

    def _build_record_no_detection_tool(self, docstring: str) -> Callable[..., str]:
        def record_no_detection(
            reasoning: Optional[str] = None,
            confidence: Optional[float] = None,
            shipping_label_present: Optional[bool] = None,
            box_count: Optional[int] = None,
            label_count: Optional[int] = None,
            labels_per_box: Optional[float] = None,
            notes: Optional[str] = None,
            **extra: Any,
        ) -> str:
            payload: Dict[str, Any] = {
                "reasoning": reasoning,
                "confidence": confidence,
                "shipping_label_present": shipping_label_present,
                "box_count": box_count,
                "label_count": label_count,
                "labels_per_box": labels_per_box,
                "notes": notes,
            }
            for key, value in extra.items():
                if key not in payload:
                    payload[key] = value
            return self._register_tool_call("record_no_detection", payload)

        record_no_detection.__name__ = "record_no_detection"
        record_no_detection.__doc__ = docstring
        return record_no_detection

    def _check_model_exists(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            for model in models:
                model_name = model.get("name", "")
                if model_name == self.model or model_name.startswith(f"{self.model}:"):
                    return True
            return False
        except Exception as exc:
            logger.warning("Failed to check if model exists: %s", exc)
            return False

    def _pull_model(self) -> bool:
        try:
            logger.info(
                "🔄 Pulling Ollama model '%s'... This may take a few minutes.",
                self.model,
            )
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
            logger.warning(
                "Model '%s' not found locally. Attempting to pull...",
                self.model,
            )
            if not self._pull_model():
                logger.error(
                    "Could not pull model '%s'. Please run 'ollama pull %s' manually.",
                    self.model,
                    self.model,
                )
        else:
            logger.info("Model '%s' is already available", self.model)

    @staticmethod
    def _parse_json_content_block(content: str) -> Optional[Dict[str, Any]]:
        if not isinstance(content, str):
            return None
        stripped = content.strip()
        if not stripped:
            return None
        stripped = stripped.replace("<|im_start|>assistant\n", "")
        stripped = stripped.replace("<|im_end|>", "")
        stripped = stripped.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines:
                lines = lines[1:]
            for idx in range(len(lines) - 1, -1, -1):
                if lines[idx].strip().startswith("```"):
                    lines = lines[:idx]
                    break
            stripped = "\n".join(lines).strip()
        start_idx = stripped.find("{")
        end_idx = stripped.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            return None
        json_blob = stripped[start_idx : end_idx + 1]
        for attempt in [json_blob, json_blob.replace("\n", " "), json_blob.replace("'", '"')]:
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict) and any(
                    key in parsed for key in ["tool", "action", "decision", "function"]
                ):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _normalize_tool_name(action_value: Any) -> str:
        if not isinstance(action_value, str):
            return "record_no_detection"
        normalized = action_value.strip().lower()
        if not normalized:
            return "record_no_detection"
        if "trigger" in normalized and "alert" in normalized:
            return "trigger_packaging_alert"
        if normalized in {"alert", "trigger_packaging_alert", "trigger_alert", "raise_alert"}:
            return "trigger_packaging_alert"
        if "no" in normalized and "detection" in normalized:
            return "record_no_detection"
        if normalized in {"record_no_detection", "no_detection", "skip", "log_only"}:
            return "record_no_detection"
        return "record_no_detection"

    def make_detection_decision(
        self,
        vision_description: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        tool_trace_entries: List[Dict[str, Any]] = []
        context_lines: List[str] = []
        packaging_hint = None
        rfdet_hint = None

        if isinstance(extra_context, dict):
            packaging_hint = extra_context.get("packaging_hint")
            rfdet_hint = extra_context.get("rfdet_hint")

            if packaging_hint:
                context_lines.append(f"Classical packaging analysis: {packaging_hint}")
            if rfdet_hint:
                context_lines.append(f"RF-DETR analysis: {rfdet_hint}")

            box_count_ctx = extra_context.get("packaging_box_count")
            rfdet_box_ctx = extra_context.get("rfdet_box_count")
            rfdet_avg_conf = extra_context.get("rfdet_average_confidence")
            packaging_conf_ctx = extra_context.get("packaging_confidence")
            packaging_candidates_ctx = extra_context.get("packaging_candidate_count")

            metrics_parts: List[str] = []
            if box_count_ctx is not None:
                metrics_parts.append(f"boxes≈{box_count_ctx}")
            if rfdet_box_ctx is not None:
                metrics_parts.append(f"rfdet_boxes≈{rfdet_box_ctx}")
            if packaging_conf_ctx is not None:
                metrics_parts.append(f"packaging_conf≈{packaging_conf_ctx}")
            if packaging_candidates_ctx is not None:
                metrics_parts.append(f"candidates≈{packaging_candidates_ctx}")
            if rfdet_avg_conf is not None:
                metrics_parts.append(f"rfdet_conf≈{rfdet_avg_conf}")
            if metrics_parts:
                context_lines.append("Scene metrics: " + ", ".join(str(part) for part in metrics_parts))

        context_block = ""
        if context_lines:
            context_block = "\n\nAdditional context:\n" + "\n".join(context_lines)

        base_user_prompt = self.user_prompt_template.format(
            vision_description=vision_description,
            extra_context=context_block,
        )

        # json_instruction removed as we are using native Ollama tools

        def _coerce_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _coerce_int(value: Any) -> Optional[int]:
            try:
                converted = int(round(float(value)))
                return converted
            except (TypeError, ValueError):
                return None

        try:
            attempt_tool_mode = bool(self.tools) and self._tool_mode_supported
            assistant_message: Dict[str, Any] = {}
            response_payload: Optional[Dict[str, Any]] = None
            python_tool_mode_used = False
            self._captured_tool_calls = []

            while True:
                active_prompt = base_user_prompt
                use_python_tools = (
                    attempt_tool_mode
                    and self._python_client_available
                    and bool(self._tool_function_list)
                    and self._ollama_python_client is not None
                )

                if use_python_tools:
                    logger.info(
                        "🤖 Decision LLM requesting unified tool call via ollama.chat()"
                    )
                    try:
                        self._captured_tool_calls = []
                        if self._ollama_python_client is None:
                            raise RuntimeError("Ollama Python client is not initialized")
                        response_payload = self._ollama_python_client.chat(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": active_prompt},
                            ],
                            tools=self._tool_function_list,
                            stream=False,
                            options={"temperature": self.temperature},
                        )
                        if hasattr(response_payload, "message"):
                            assistant_message = getattr(response_payload, "message", {}) or {}
                        elif isinstance(response_payload, dict):
                            assistant_message = response_payload.get("message", {})
                        else:
                            assistant_message = {}
                        python_tool_mode_used = True
                    except Exception as exc:
                        logger.warning("Ollama Python client tool-call failed: %s", exc)
                        attempt_tool_mode = False
                        continue
                else:
                    logger.info("🤖 Decision LLM requesting JSON decision via HTTP generate")
                    self._captured_tool_calls = []
                    payload = {
                        "model": self.model,
                        "system": self.system_prompt,
                        "prompt": active_prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                        },
                    }
                    response = self.session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self.timeout,
                    )
                    if response.status_code == 404 and attempt_tool_mode:
                        logger.warning("Decision model '%s' not found; retrying without tool mode", self.model)
                        if self._pull_model():
                            response = self.session.post(
                                f"{self.base_url}/api/generate",
                                json=payload,
                                timeout=self.timeout,
                            )
                        else:
                            break
                    response.raise_for_status()
                    assistant_message = response.json()

                tool_payload: Optional[Dict[str, Any]] = None
                if python_tool_mode_used and self._captured_tool_calls:
                    tool_payload = self._captured_tool_calls[-1]
                else:
                    content = assistant_message.get("message", {}).get("content")
                    if not content and isinstance(assistant_message, dict):
                        content = assistant_message.get("response")
                    parsed = self._parse_json_content_block(content) if content else None
                    if parsed:
                        normalized_tool = self._normalize_tool_name(parsed.get("tool") or parsed.get("action"))
                        parsed["tool"] = normalized_tool
                        tool_payload = {"name": normalized_tool, "arguments": parsed}

                if tool_payload is None and attempt_tool_mode:
                    logger.warning("Decision model response missing tool call; falling back to JSON mode")
                    attempt_tool_mode = False
                    python_tool_mode_used = False
                    continue

                if tool_payload is None:
                    logger.warning("Decision model response missing structured payload")
                    return None

                tool_trace_entries.append(tool_payload)
                arguments = tool_payload.get("arguments", {}) if isinstance(tool_payload, dict) else {}
                tool_name = tool_payload.get("name") if isinstance(tool_payload, dict) else None
                normalized_tool = self._normalize_tool_name(tool_name)

                confidence = _coerce_float(arguments.get("confidence"), 0.0)
                box_count = _coerce_int(arguments.get("box_count")) or 0
                label_count = _coerce_int(arguments.get("label_count")) or 0
                labels_per_box = arguments.get("labels_per_box")
                if labels_per_box is not None:
                    labels_per_box = _coerce_float(labels_per_box, 0.0)
                shipping_label_present_raw = arguments.get("shipping_label_present")
                if isinstance(shipping_label_present_raw, bool):
                    shipping_label_present = shipping_label_present_raw
                elif isinstance(shipping_label_present_raw, str):
                    lowered = shipping_label_present_raw.lower().strip()
                    if lowered in {"true", "yes"}:
                        shipping_label_present = True
                    elif lowered in {"false", "no"}:
                        shipping_label_present = False
                    else:
                        shipping_label_present = None
                else:
                    shipping_label_present = None

                payload = {
                    "tool": normalized_tool,
                    "confidence": confidence,
                    "reasoning": arguments.get("reasoning"),
                    "box_count": box_count,
                    "label_count": label_count,
                    "shipping_label_present": shipping_label_present,
                    "labels_per_box": labels_per_box,
                    "notes": arguments.get("notes"),
                    "raw_response": assistant_message,
                    "tool_trace": tool_trace_entries,
                    "tool_payload": arguments,
                    "packaging_hint": packaging_hint,
                    "rfdet_hint": rfdet_hint,
                }
                return payload
        except requests.exceptions.RequestException as exc:
            logger.error("Decision LLM request failed: %s", exc)
            raise
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error("Decision LLM processing failed: %s", exc, exc_info=True)
            return None



    def wait_for_cooldown(self, elapsed_seconds: float, cooldown_seconds: float) -> None:
        remaining = max(0.0, cooldown_seconds - elapsed_seconds)
        if remaining > 0:
            time.sleep(remaining)
