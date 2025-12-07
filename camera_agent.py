#!/usr/bin/env python3
"""
ZEDEDA Camera Monitoring Agent - Streamlined Version
Uses a single Vision Language Model for detection and decision-making.
"""

import copy
import logging
import os
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum, auto
from abc import ABC, abstractmethod

import numpy as np
import cv2
import yaml
from dotenv import load_dotenv

# Agent runtime imports
from agent_runtime.state import AgentMemory, DetectionEvent
from agent_runtime.utils import coerce_bool as _coerce_bool
from agent_runtime.alerting import AlertManager
from agent_runtime.unified_vlm import UnifiedVLMClient, DetectionResult, TaskType

try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None

# Public exports
__all__ = [
    'StreamlinedAgent',
    'CircuitBreaker',
    'DEFAULT_CONFIG_PATH',
]

# Load environment variables
load_dotenv()

# Configure logging
DEFAULT_LOG_FILE = os.getenv("CAMERA_AGENT_LOG_FILE", "camera_agent.log")
log_handlers = [logging.StreamHandler(sys.stdout)]

if DEFAULT_LOG_FILE:
    log_path = Path(DEFAULT_LOG_FILE).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(log_path, mode='a'))
    except OSError as exc:
        sys.stderr.write(f"Warning: unable to use log file {log_path}: {exc}\n")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.getenv("CAMERA_AGENT_CONFIG", "config.yaml")


class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """Circuit breaker pattern for resilience.
    
    Protects against cascading failures by temporarily blocking calls
    to a failing service.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 120.0):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0.0
        self.total_calls = 0
        self.successful_calls = 0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        """Check if circuit is currently open."""
        with self._lock:
            return self.state == CircuitState.OPEN

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "state": self.state.name,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "success_rate": (
                    round(self.successful_calls / self.total_calls * 100, 2)
                    if self.total_calls > 0 else 0.0
                ),
            }

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            logger.info("Circuit breaker manually reset to CLOSED")

    def call(self, func, *args, **kwargs):
        with self._lock:
            self.total_calls += 1
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF-OPEN state")
                else:
                    raise RuntimeError("Circuit is OPEN")

        try:
            result = func(*args, **kwargs)
            with self._lock:
                self.successful_calls += 1
                if self.state != CircuitState.CLOSED:
                    logger.info("Circuit breaker recovering to CLOSED state")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker tripped to OPEN (failures: {self.failure_count}). Error: {e}"
                    )
            raise


class StreamlinedAgent:
    """
    Streamlined detection agent using a single Vision Language Model.
    
    This agent uses a VLM for all scene understanding and decision-making,
    providing flexible multi-purpose analysis without specialized object detectors.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        vlm_client: UnifiedVLMClient,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.config = config
        self.vlm_client = vlm_client
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.last_error: Optional[str] = None
        
        # Initialize memory
        memory_cfg = config.get("memory", {})
        self._memory_max_events = max(1, int(memory_cfg.get("max_events", 50)))
        self._memory_summary_window = max(1, int(memory_cfg.get("summary_window", 10)))
        self._agent_memory = AgentMemory(
            max_events=self._memory_max_events,
            summary_window=self._memory_summary_window,
        )
        
        # Initialize alert manager
        self.alert_manager = AlertManager(config)
        
        # SSIM caching for efficiency
        self._ssim_threshold = 0.95
        self._ssim_recheck_seconds = 30.0
        self._last_processed_event: Optional[DetectionEvent] = None
        self._last_similarity_frame: Optional[np.ndarray] = None
        self._last_analysis_time = 0.0
        self._ssim_reference_size = (320, 240)
        
        # Image saving config
        camera_cfg = config.get("camera", {})
        self.save_images = _coerce_bool(camera_cfg.get("save_detection_images"), False)
        
        # Resolve images directory: prefer DETECTED_IMAGES_DIR env var, fall back to config
        env_images_dir = os.getenv("DETECTED_IMAGES_DIR")
        if env_images_dir:
            self.images_dir = Path(env_images_dir)
        else:
            config_images_dir = camera_cfg.get("detection_image_dir", "detected_images")
            # If relative, make it relative to DATA_DIR
            data_dir = Path(os.getenv("CAMERA_AGENT_DATA_DIR", "."))
            images_path = Path(config_images_dir)
            if not images_path.is_absolute():
                self.images_dir = data_dir / images_path
            else:
                self.images_dir = images_path
        
        if self.save_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("StreamlinedAgent initialized with unified VLM")

    def _save_detection_image(self, frame: np.ndarray, result) -> str:
        """Save detection frame to disk and return the path."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            label_status = "labeled" if result.shipping_label_present else "unlabeled"
            filename = f"detection_{timestamp}_{label_status}.jpg"
            filepath = self.images_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            logger.debug(f"Saved detection image: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save detection image: {e}")
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

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Return the default configuration."""
        return copy.deepcopy(cls.load_config_from_path())

    def _make_similarity_reference(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Create a grayscale reference for SSIM comparison."""
        try:
            resized = cv2.resize(frame, self._ssim_reference_size)
            return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

    def _check_ssim_skip(
        self,
        frame: np.ndarray,
        current_time: float,
    ) -> tuple[bool, Optional[DetectionEvent]]:
        """Check if we can skip analysis due to scene similarity."""
        if structural_similarity is None:
            return False, None
        
        if self._last_similarity_frame is None or self._last_processed_event is None:
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
            similarity = structural_similarity(
                self._last_similarity_frame,
                current_ref,
                data_range=255
            )
            if isinstance(similarity, tuple):
                similarity = similarity[0]
            
            if similarity >= self._ssim_threshold:
                # Reuse previous event
                reused = copy.deepcopy(self._last_processed_event)
                reused.timestamp = datetime.now().isoformat()
                reused.should_alert = False  # Don't re-alert
                reused.decision_trace["ssim_skip"] = {
                    "similarity": round(similarity, 3),
                    "reused": True,
                }
                logger.debug(f"♻️ SSIM skip: similarity={similarity:.3f}")
                return True, reused
        except Exception as e:
            logger.debug(f"SSIM comparison failed: {e}")
        
        return False, None

    def _run_vlm_analysis(
        self,
        frame: np.ndarray,
        cv_context: Dict[str, Any],
        max_retries: int = 2,
    ) -> Optional[DetectionResult]:
        """Run the unified VLM for detection and decision with retry logic.
        
        Args:
            frame: The image frame to analyze.
            cv_context: Optional context dictionary (unused in VLM-only mode).
            max_retries: Maximum number of retry attempts.
            
        Returns:
            DetectionResult if successful, None otherwise.
        """
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
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "VLM analysis attempt %d/%d failed: %s",
                        attempt + 1, max_retries + 1, e
                    )
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        if last_error:
            logger.error(f"VLM analysis failed after {max_retries + 1} attempts: {last_error}")
            self.last_error = str(last_error)
        return None

    def analyze_frame(
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
        analysis_start = time.time()
        
        # 1. SSIM skip check
        should_skip, cached_event = self._check_ssim_skip(frame, analysis_start)
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
        if self.save_images and vlm_result.detected:
            image_path = self._save_detection_image(frame, vlm_result)
        
        event = DetectionEvent(
            timestamp=datetime.now().isoformat(),
            detected=vlm_result.detected,
            confidence=vlm_result.confidence,
            primary_label="packaging_box" if vlm_result.detected else "no_detection",
            vision_description=vlm_result.reasoning,
            full_response=vlm_result.raw_response[:500],
            should_alert=vlm_result.should_alert,
            shipping_label_present=vlm_result.shipping_label_present,
            image_path=image_path,
            tools_used=["unified_vlm"],
            tool_trace=[],
            decision_trace={
                "classification": "VLM_DECISION",
                "box_count": vlm_result.box_count,
                "confidence": vlm_result.confidence,
                "shipping_label_present": vlm_result.shipping_label_present,
                "should_alert": vlm_result.should_alert,
            },
        )
        
        # Update cache
        self._last_similarity_frame = self._make_similarity_reference(frame)
        self._last_processed_event = event
        self._last_analysis_time = analysis_start
        self._remember_event(event, source="unified_vlm")
        
        # Log result
        if vlm_result.should_alert:
            logger.info(
                "🔔 ALERT: Unlabeled box detected! Confidence: %.2f",
                vlm_result.confidence
            )
        elif vlm_result.detected:
            logger.info(
                "📦 Box detected with label. Confidence: %.2f",
                vlm_result.confidence
            )
        
        return event

    def analyze_with_prompt(
        self,
        frame: np.ndarray,
        task_type: TaskType,
        custom_prompt: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DetectionEvent]:
        """Analyze a frame with a specific task type and custom prompt.
        
        This method supports dynamic prompt configuration for multi-purpose analysis.
        
        Args:
            frame: The image frame to analyze.
            task_type: The TaskType enum for the analysis.
            custom_prompt: Custom user query (required for CUSTOM task type).
            metadata: Optional frame metadata.
            
        Returns:
            DetectionEvent if analysis succeeds, None otherwise.
        """
        metadata = metadata or {}
        analysis_start = time.time()
        
        # All task types go through VLM for proper analysis and logging
        try:
            result = self.circuit_breaker.call(
                self.vlm_client.analyze,
                frame,
                task_type=task_type,
                user_query=custom_prompt if task_type == TaskType.CUSTOM else None,
                cv_context=None,
            )
            
            if result is None:
                return None
            
            # Map task type to label
            task_label_map = {
                TaskType.PACKAGE_DETECTION: "package_detection",
                TaskType.PPE_DETECTION: "ppe_detection",
                TaskType.PERSON_COUNTING: "person_count",
                TaskType.SCENE_DESCRIPTION: "scene_description",
                TaskType.CUSTOM: "custom_detection",
            }
            primary_label = task_label_map.get(task_type, "detection") if result.detected else "no_detection"
            
            event = DetectionEvent(
                timestamp=datetime.now().isoformat(),
                detected=result.detected,
                confidence=result.confidence,
                primary_label=primary_label,
                vision_description=result.reasoning,
                full_response=result.raw_response[:500] if result.raw_response else "",
                should_alert=result.should_alert,
                shipping_label_present=None,
                tools_used=["unified_vlm"],
                tool_trace=[],
                decision_trace={
                    "classification": task_type.value.upper(),
                    "task_type": task_type.value,
                    "custom_prompt": custom_prompt[:100] if custom_prompt else "",
                    "detected": result.detected,
                    "confidence": result.confidence,
                    "should_alert": result.should_alert,
                    "details": result.details if hasattr(result, 'details') else {},
                },
            )
            
            self._remember_event(event, source=f"{task_type.value}_vlm")
            
            if result.should_alert:
                logger.info(
                    "🔔 %s ALERT: %s (Confidence: %.2f)",
                    task_type.value.upper(),
                    result.reasoning[:50] if result.reasoning else "Alert triggered",
                    result.confidence
                )
            
            return event
            
        except Exception as e:
            logger.error(f"Analysis failed for {task_type.value}: {e}")
            return None

    def analyze_agentic(
        self,
        frame: np.ndarray,
        task_type: Optional[TaskType] = None,
        custom_prompt: Optional[str] = None,
        recipients: Optional[list] = None,
    ) -> Optional[DetectionEvent]:
        """Analyze frame with agentic tool calling.
        
        This method allows the LLM to autonomously decide which tools to call
        based on its analysis. For example, it may decide to:
        - Send an alert email if something concerning is detected
        - Save evidence images for later review
        - Log events for auditing
        - Query history for context
        
        Args:
            frame: OpenCV frame to analyze
            task_type: Type of analysis (defaults to PACKAGE_DETECTION)
            custom_prompt: Optional custom analysis prompt
            recipients: Email recipients for alerts (uses config if not provided)
            
        Returns:
            DetectionEvent with tool call information in tool_trace
        """
        from agent_runtime.tools import ToolExecutor
        
        if self.circuit_breaker.is_open:
            logger.warning("Circuit breaker is open - skipping agentic analysis")
            return None
        
        effective_task_type = task_type or TaskType.PACKAGE_DETECTION
        
        # Get recipients from config if not provided
        if recipients is None:
            email_cfg = self.config.get("notifications", {}).get("email", {})
            recipients = email_cfg.get("recipients", [])
        
        # Create tool executor with current context
        tool_executor = ToolExecutor()
        
        try:
            # Encode frame for tool context
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_bytes = buffer.tobytes()
            
            # Run agentic analysis
            agentic_result = self.vlm_client.analyze_with_tools(
                frame=frame,
                tool_executor=tool_executor,
                task_type=effective_task_type,
                user_query=custom_prompt,
                recipients=recipients,
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
            if self.save_images and analysis.detected:
                image_path = self._save_detection_image(frame, analysis)
            
            # Create detection event
            event = DetectionEvent(
                timestamp=datetime.now().isoformat(),
                detected=analysis.detected,
                confidence=analysis.confidence,
                primary_label=f"{effective_task_type.value}_agentic",
                vision_description=analysis.reasoning,
                full_response=analysis.raw_response[:500] if analysis.raw_response else "",
                should_alert=analysis.should_alert,
                shipping_label_present=analysis.details.get("shipping_label_present"),
                image_path=image_path,
                tools_used=["unified_vlm"] + agentic_result.tools_used,
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
            )
            
            self._remember_event(event, source="agentic")
            
            if agentic_result.any_tools_called:
                logger.info(
                    "🤖 Agentic analysis called %d tools: %s",
                    len(agentic_result.tool_calls),
                    ", ".join(agentic_result.tools_used)
                )
            
            return event
            
        except Exception as e:
            logger.error(f"Agentic analysis failed: {e}")
            return None

    def process_detection(self, event: DetectionEvent) -> bool:
        """Process detection event and send alerts if needed."""
        if not event.should_alert:
            return False
        
        logger.info(f"Processing alert: {event.primary_label}")
        alerts_sent = 0
        
        # Process rules
        for rule in self.config.get("rules", []):
            if not rule.get("enabled", True):
                continue
            if self.alert_manager.send_email_alert(event, rule):
                alerts_sent += 1
        
        # Desktop notification
        if self.config.get("notifications", {}).get("desktop", {}).get("enabled"):
            if self.alert_manager.send_desktop_notification(event):
                alerts_sent += 1
        
        return alerts_sent > 0

    def _remember_event(self, event: DetectionEvent, source: str = "analysis") -> None:
        """Store event in memory for tracking."""
        record = {
            "timestamp": event.timestamp,
            "detected": event.detected,
            "confidence": round(event.confidence, 3),
            "primary_label": event.primary_label,
            "shipping_label_present": event.shipping_label_present,
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
        self.alert_manager.refresh_config(updated_config)
        
        # Update memory settings
        memory_cfg = updated_config.get("memory", {})
        new_max = max(1, int(memory_cfg.get("max_events", 50)))
        new_window = max(1, int(memory_cfg.get("summary_window", 10)))
        self._agent_memory.resize(new_max, new_window)
        
        logger.info("Configuration updated")
