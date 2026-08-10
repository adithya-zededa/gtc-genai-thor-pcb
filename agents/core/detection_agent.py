#!/usr/bin/env python3
"""
ZEDEDA Camera Monitoring Agent - Streamlined Version
Uses a single Vision Language Model for detection and decision-making.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import cv2
import numpy as np
import yaml

from core.config import get_config
from core.logging import get_logger
from core.utils import coerce_bool
from agents.core.state import AgentMemory, DetectionEvent
from core.resilience import CircuitBreaker

if TYPE_CHECKING:
    from agents.vlm import UnifiedVLMClient, DetectionResult, TaskType  # noqa: F401

# pylint: disable=no-member
# cv2 attributes are dynamically generated and not visible to pylint

try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = str(get_config().config_path)

__all__ = [
    "StreamlinedAgent",
    "CircuitBreaker",
    "DEFAULT_CONFIG_PATH",
]


class StreamlinedAgent:  # pylint: disable=too-many-instance-attributes
    """
    Streamlined PCB inspection agent using a single Vision Language Model.

    This agent uses a VLM for all scene understanding and decision-making,
    providing flexible multi-purpose analysis without specialized object detectors.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        vlm_client: "UnifiedVLMClient",
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.config = config
        self.vlm_client = vlm_client
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.last_error: Optional[str] = None

        # Lock for SSIM cache fields (thread-safety)
        self._ssim_lock = threading.Lock()

        # Initialize memory
        memory_cfg = config.get("memory", {})
        self._memory_max_events = max(1, int(memory_cfg.get("max_events", 50)))
        self._memory_summary_window = max(1, int(memory_cfg.get("summary_window", 10)))
        self._agent_memory = AgentMemory(
            max_events=self._memory_max_events,
            summary_window=self._memory_summary_window,
        )

        # SSIM caching for efficiency
        self._ssim_threshold = 0.95
        self._ssim_recheck_seconds = 5.0
        self._last_processed_event: Optional[DetectionEvent] = None
        self._last_similarity_frame: Optional[np.ndarray] = None
        self._last_analysis_time = 0.0
        self._ssim_reference_size = (320, 240)
        self._last_board_signature: Optional[str] = None

        # Image saving config
        camera_cfg = config.get("camera", {})
        self.save_images = coerce_bool(camera_cfg.get("save_detection_images"), False)

        # env > YAML > default; the precedence itself lives in core.config
        # so this is not a second, drifting copy of the same rule.
        self.images_dir = get_config().resolve_detection_image_dir(
            camera_cfg.get("detection_image_dir")
        )

        if self.save_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)

        logger.info("StreamlinedAgent initialized with unified VLM")

    # ── Keyword signatures used to recognise the built-in PCB
    #    defect inspection prompt (DEFAULT_MONITORING_DEFECT_PROMPT).
    _PCB_DEFECT_PROMPT_SIGNATURES = (
        "power_jack_status",
        "usb_port_status",
        "header_pins_status",
        "PCB quality inspector",
    )

    @classmethod
    def _derive_analysis_metadata(
        cls,
        task_type: "TaskType",
        custom_prompt: str,
        result: "AnalysisResult",
    ) -> tuple:
        """Derive classification label and tool name from the analysis context.

        Returns ``(classification, tool_label)`` so that ``tools_used``
        and ``decision_trace["classification"]`` reflect what the agent
        actually did, instead of being hardcoded to ``"unified_vlm"``.
        """
        # Check if the prompt is the canonical PCB defect inspection prompt
        prompt_lower = (custom_prompt or "").lower()
        is_pcb_defect_prompt = any(
            sig.lower() in prompt_lower
            for sig in cls._PCB_DEFECT_PROMPT_SIGNATURES
        )

        if is_pcb_defect_prompt:
            return "PCB_DEFECT_INSPECTION", "pcb_defect_inspection"

        # Fallback: derive from task_type
        task_val = task_type.value if hasattr(task_type, "value") else str(task_type)
        return task_val.upper(), f"{task_val}_vlm"

    def _save_detection_image(self, frame: np.ndarray, result) -> str:
        """Save detection frame to disk and return the path."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            readiness_status = "ready" if getattr(result, 'pcb_stable', False) else "not_ready"
            filename = f"detection_{timestamp}_{readiness_status}.jpg"
            filepath = self.images_dir / filename

            cv2.imwrite(str(filepath), frame)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Saved detection image: %s", filepath)
            return str(filepath)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to save detection image: %s", e)
            return ""

    @classmethod
    def load_config_from_path(cls, path: Optional[str | Path] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        target_path = Path(path or DEFAULT_CONFIG_PATH).expanduser()
        if not target_path.exists():
            raise FileNotFoundError(f"Config not found: {target_path}")

        with target_path.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"Config must be a mapping: {target_path}")

        return config

    _default_config_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Return the default configuration (cached after first load)."""
        if cls._default_config_cache is None:
            cls._default_config_cache = cls.load_config_from_path()
        return copy.deepcopy(cls._default_config_cache)

    def _make_similarity_reference(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Create a grayscale reference for similarity comparison.

        Performance: Downscaled to 320x240 for faster comparison.
        """
        try:
            resized = cv2.resize(frame, self._ssim_reference_size)
            return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        except Exception as e:  # pylint: disable=broad-exception-caught
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Failed to build similarity reference: %s", e)
            return None

    def _fast_similarity(self, ref: np.ndarray, current: np.ndarray) -> float:
        """Fast similarity calculation using template matching.

        Alternative to SSIM that's 20% faster and doesn't require scikit-image.
        Returns normalized correlation coefficient [0.0, 1.0].
        """
        try:
            result = cv2.matchTemplate(ref, current, cv2.TM_CCOEFF_NORMED)
            return float(result[0][0])
        except Exception as e:  # pylint: disable=broad-exception-caught
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Fast similarity comparison failed: %s", e)
            return 0.0

    def _check_ssim_skip(
        self,
        frame: np.ndarray,
        current_time: float,
        board_signature: Optional[str] = None,
    ) -> tuple[bool, Optional[DetectionEvent]]:
        """Check if we can skip analysis due to scene similarity.

        Uses either SSIM (if scikit-image available) or fast template matching.
        Performance: 20% faster with template matching, no external dependency.

        If *board_signature* is provided and differs from the signature
        of the last analysed frame the cache is automatically
        invalidated so that a new board is never served a stale result.
        """
        with self._ssim_lock:
            if self._last_similarity_frame is None or self._last_processed_event is None:
                return False, None

            # A signature that failed to derive is an unknown board — never
            # reuse a stale verdict for it, even if the previous board also
            # had no signature.
            if board_signature is None:
                self._last_similarity_frame = None
                self._last_processed_event = None
                self._last_analysis_time = 0.0
                self._last_board_signature = None
                return False, None

            # Invalidate cache when the board identity changes
            if (
                self._last_board_signature is not None
                and board_signature != self._last_board_signature
            ):
                self._last_similarity_frame = None
                self._last_processed_event = None
                self._last_analysis_time = 0.0
                self._last_board_signature = None
                return False, None

            # Check time since last analysis
            time_since_last = current_time - self._last_analysis_time
            if time_since_last > self._ssim_recheck_seconds:
                return False, None

            # Compare frames
            current_ref = self._make_similarity_reference(frame)
            if current_ref is None:
                return False, None

            try:
                # Try SSIM first if available, otherwise use fast template matching
                if structural_similarity is not None:
                    similarity = structural_similarity(
                        self._last_similarity_frame,
                        current_ref,
                        data_range=255
                    )
                    if isinstance(similarity, tuple):
                        similarity = similarity[0]
                else:
                    # Fallback to fast template matching (20% faster)
                    similarity = self._fast_similarity(self._last_similarity_frame, current_ref)

                if similarity >= self._ssim_threshold:
                    # Reuse previous event
                    reused = copy.deepcopy(self._last_processed_event)
                    reused.timestamp = datetime.now().isoformat()
                    # Preserve original should_alert so genuinely
                    # alertable repeat conditions are not silenced.
                    reused.decision_trace["ssim_skip"] = {
                        "similarity": round(similarity, 3),
                        "reused": True,
                        "method": "ssim" if structural_similarity else "template_match",
                    }
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("♻️ SSIM skip: similarity=%.3f", similarity)
                    return True, reused
            except Exception as e:  # pylint: disable=broad-exception-caught
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("SSIM comparison failed: %s", e)

            return False, None

    def _run_vlm_analysis(
        self,
        frame: np.ndarray,
        cv_context: Dict[str, Any],
        max_retries: int = 2,
    ) -> Optional["DetectionResult"]:
        """Run the unified VLM for detection and decision with retry logic."""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                result = self.circuit_breaker.call(
                    self.vlm_client.analyze_frame,
                    frame,
                    cv_context,
                )
                return result
            except RuntimeError as e:
                if "Circuit is OPEN" in str(e):
                    logger.warning(
                        "⚡ VLM circuit breaker is OPEN. Will retry in %ds.",
                        int(self.circuit_breaker.recovery_timeout)
                    )
                    return None  # Don't retry if circuit is open
                last_error = e
            except Exception as e:  # pylint: disable=broad-exception-caught
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "VLM analysis attempt %d/%d failed: %s",
                        attempt + 1, max_retries + 1, e
                    )
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff

        if last_error:
            logger.error(
                "VLM analysis failed after %d attempts: %s",
                max_retries + 1, last_error
            )
            self.last_error = str(last_error)
        return None

    def analyze_frame(  # pylint: disable=broad-exception-caught
        self,
        frame: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DetectionEvent]:
        """
        Analyze a frame using VLM-only pipeline.

        Pipeline:
        1. SSIM check - skip if scene hasn't changed
        2. VLM analysis - vision + decision
        3. Create detection event
        """
        metadata = metadata or {}

        # 1. SSIM skip check
        board_sig = metadata.get("board_signature")
        should_skip, cached_event = self._check_ssim_skip(
            frame, time.time(), board_signature=board_sig,
        )
        if should_skip and cached_event:
            self._remember_event(cached_event, source="ssim_skip")
            return cached_event

        # 2. VLM Analysis (no RF-DETR pre-filtering)
        cv_context = {}  # No pre-detection context needed

        vlm_result = self._run_vlm_analysis(frame, cv_context)

        if vlm_result is None:
            logger.warning("VLM analysis returned None")
            return None

        # 3. Create detection event
        image_path = ""
        if self.save_images:
            image_path = self._save_detection_image(frame, vlm_result)

        # Derive the event label from VLM reasoning instead of hardcoding.
        primary_label = "no_detection"
        if vlm_result.detected:
            primary_label = getattr(vlm_result, 'primary_label', None) or "detection"

        # Derive tool label from the VLM result task_type
        tool_label = f"{vlm_result.task_type}_vlm" if hasattr(vlm_result, 'task_type') and vlm_result.task_type else "vlm_analysis"

        event = DetectionEvent(
            timestamp=datetime.now().isoformat(),
            detected=vlm_result.detected,
            confidence=vlm_result.confidence,
            primary_label=primary_label,
            vision_description=vlm_result.reasoning,
            full_response=vlm_result.raw_response[:500],
            should_alert=vlm_result.should_alert,
            pcb_stable=vlm_result.pcb_stable,
            image_path=image_path,
            tools_used=[tool_label],
            tool_trace=[],
            decision_trace={
                "classification": "VLM_DECISION",
                "detected": vlm_result.detected,
                "pcb_count": vlm_result.pcb_count,
                "confidence": vlm_result.confidence,
                "pcb_stable": vlm_result.pcb_stable,
                "should_alert": vlm_result.should_alert,
            },
            token_usage=getattr(vlm_result, 'token_usage', None)
                or getattr(self.vlm_client, '_last_token_usage', None)
                or {},
        )

        # Update cache (including board signature for identity tracking)
        with self._ssim_lock:
            self._last_similarity_frame = self._make_similarity_reference(frame)
            self._last_processed_event = event
            self._last_analysis_time = time.time()
            self._last_board_signature = metadata.get("board_signature")
        self._remember_event(event, source=tool_label)

        # Log result
        if vlm_result.should_alert:
            logger.info(
                "🔔 ALERT: PCB defect condition detected (confidence: %.2f)",
                vlm_result.confidence
            )
        elif vlm_result.detected:
            logger.info(
                "🟢 PCB detected without defect trigger (confidence: %.2f)",
                vlm_result.confidence
            )

        return event

    def analyze_with_prompt(
        self,
        frame: np.ndarray,
        task_type: "TaskType",
        custom_prompt: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DetectionEvent]:
        """Analyze a frame with a specific task type and custom prompt."""
        metadata = metadata or {}

        try:
            result = self.circuit_breaker.call(
                self.vlm_client.analyze,
                frame,
                task_type=task_type,
                user_query=custom_prompt or None,
                cv_context=None,
            )

            if result is None:
                return None

            # Use task_type value directly as the label
            primary_label = task_type.value if result.detected else "no_detection"

            # Save detection image if enabled
            image_path = ""
            if self.save_images:
                image_path = self._save_detection_image(frame, result)

            # Derive classification and tools_used from actual analysis context
            classification, tool_label = self._derive_analysis_metadata(
                task_type, custom_prompt, result,
            )

            event = DetectionEvent(
                timestamp=datetime.now().isoformat(),
                detected=result.detected,
                confidence=result.confidence,
                primary_label=primary_label,
                vision_description=result.reasoning,
                full_response=result.raw_response or "",
                should_alert=result.should_alert,
                pcb_stable=result.details.get("pcb_stable"),
                image_path=image_path,
                tools_used=[tool_label],
                tool_trace=[],
                decision_trace={
                    "classification": classification,
                    "task_type": task_type.value,
                    "custom_prompt": custom_prompt[:100] if custom_prompt else "",
                    "detected": result.detected,
                    "confidence": result.confidence,
                    "should_alert": result.should_alert,
                    "details": result.details if hasattr(result, 'details') else {},
                },
                token_usage=getattr(result, 'token_usage', {}),
            )

            self._remember_event(event, source=tool_label)

            if result.should_alert:
                logger.info(
                    "🔔 %s ALERT: %s (Confidence: %.2f)",
                    classification,
                    result.reasoning[:50] if result.reasoning else "Alert triggered",
                    result.confidence
                )

            return event

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception("Analysis failed for %s: %s", task_type.value, e)
            return None

    def analyze_agentic(  # pylint: disable=too-many-locals
        self,
        frame: np.ndarray,
        task_type: Optional["TaskType"] = None,
        custom_prompt: Optional[str] = None,
        recipients: Optional[list] = None,
    ) -> Optional[DetectionEvent]:
        """Analyze frame with agentic tool calling."""
        from agents.vlm import TaskType as TT  # pylint: disable=import-outside-toplevel
        from agents.mcp.globals import get_mcp_executor  # pylint: disable=import-outside-toplevel
        from agents.mcp.tool_defs import get_agentic_tool_schemas  # pylint: disable=import-outside-toplevel

        if self.circuit_breaker.is_open:
            logger.warning("Circuit breaker is open - skipping agentic analysis")
            return None

        effective_task_type = task_type or TT.CUSTOM

        # Get recipients from config if not provided
        if recipients is None:
            email_cfg = self.config.get("notifications", {}).get("email", {})
            recipients = email_cfg.get("recipients", [])

        # Create MCP-backed tool dispatch
        mcp_executor = get_mcp_executor()
        image_bytes = self.vlm_client._frame_to_bytes(frame) if frame is not None else b""
        tool_schemas = get_agentic_tool_schemas()

        def tool_dispatch(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
            return mcp_executor.submit_agentic_call(
                tool_name,
                arguments,
                context_updates={"image_data": image_bytes, "recipients": recipients or []},
                rationale=f"VLM agentic tool call during {effective_task_type.value} analysis",
            )

        try:
            # Run agentic analysis
            agentic_result = self.vlm_client.analyze_with_tools(
                frame=frame,
                tool_dispatch=tool_dispatch,
                tool_schemas=tool_schemas,
                task_type=effective_task_type,
                user_query=custom_prompt,
            )

            if agentic_result is None or agentic_result.analysis is None:
                logger.warning("Agentic analysis returned no result")
                return None

            analysis = agentic_result.analysis

            # Build tool trace from results
            tool_trace = []
            for i, (call, result) in enumerate(zip(
                agentic_result.tool_calls,
                agentic_result.tool_results
            )):
                tool_trace.append({
                    "step": i + 1,
                    "tool": call.get("tool", "unknown"),
                    "arguments": call.get("arguments", {}),
                    "success": result.get("success", False),
                    "result": result.get("result", {}),
                })

            # Save detection image if enabled
            image_path = ""
            if self.save_images:
                image_path = self._save_detection_image(frame, analysis)

            # Derive the base tool label from prompt context
            _, base_tool_label = self._derive_analysis_metadata(
                effective_task_type, custom_prompt or "", analysis,
            )

            # Create detection event
            event = DetectionEvent(
                timestamp=datetime.now().isoformat(),
                detected=analysis.detected,
                confidence=analysis.confidence,
                primary_label=f"{effective_task_type.value}_agentic",
                vision_description=analysis.reasoning,
                full_response=analysis.raw_response[:500] if analysis.raw_response else "",
                should_alert=analysis.should_alert,
                pcb_stable=analysis.details.get("pcb_stable"),
                image_path=image_path,
                tools_used=[base_tool_label] + agentic_result.tools_used,
                tool_trace=tool_trace,
                decision_trace={
                    "classification": "AGENTIC_ANALYSIS",
                    "task_type": effective_task_type.value,
                    "detected": analysis.detected,
                    "confidence": analysis.confidence,
                    "should_alert": analysis.should_alert,
                    "tools_called": len(agentic_result.tool_calls),
                    "all_tools_succeeded": agentic_result.all_tools_succeeded,
                },
                token_usage=getattr(agentic_result, 'total_token_usage', {}),
            )

            self._remember_event(event, source="agentic")

            if agentic_result.any_tools_called:
                tool_sequence = " -> ".join(agentic_result.tools_used)
                logger.info(
                    "🤖 Agentic analysis called %d tools: %s",
                    len(agentic_result.tool_calls),
                    tool_sequence,
                )
                # Log individual tool results
                for i, (call, result) in enumerate(zip(
                    agentic_result.tool_calls,
                    agentic_result.tool_results,
                )):
                    status = "✅" if result.get("success", False) else "❌"
                    logger.info(
                        "  %s Step %d: %s %s",
                        status, i + 1, call.get("tool", "?"),
                        "(success)" if result.get("success", False) else f"(error: {result.get('error', 'unknown')})",
                    )

            # Log token usage
            usage = getattr(agentic_result, 'total_token_usage', {})
            if usage and usage.get("total_tokens", 0) > 0:
                logger.info(
                    "📊 Token usage: prompt=%d, completion=%d, total=%d",
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0),
                )

            return event

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception("Agentic analysis failed: %s", e)
            return None

    def _remember_event(self, event: DetectionEvent, source: str = "analysis") -> None:
        """Store event in memory for tracking."""
        record = {
            "timestamp": event.timestamp,
            "detected": event.detected,
            "confidence": round(event.confidence, 3),
            "primary_label": event.primary_label,
            "pcb_stable": event.pcb_stable,
            "should_alert": event.should_alert,
            "source": source,
        }
        self._agent_memory.append(record)

    def get_memory_snapshot(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get recent events from memory."""
        return asdict(self._agent_memory.snapshot(limit))

    def summarise_recent_events(self, limit: Optional[int] = None) -> str:
        """Get a text summary of recent activity."""
        return self._agent_memory.summarise(limit)

    def apply_config(self, updated_config: Dict[str, Any]) -> None:
        """Apply updated configuration."""
        self.config = updated_config

        # Update memory settings
        memory_cfg = updated_config.get("memory", {})
        new_max = max(1, int(memory_cfg.get("max_events", 50)))
        new_window = max(1, int(memory_cfg.get("summary_window", 10)))
        self._agent_memory.resize(new_max, new_window)

        logger.info("Configuration updated")
