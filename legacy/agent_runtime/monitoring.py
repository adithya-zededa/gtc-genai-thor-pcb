#!/usr/bin/env python3
"""
Streamlined camera monitoring service.

This module provides the StreamlinedMonitoringService that subscribes to the
camera feed and runs the unified VLM-based detection pipeline.
"""

import logging
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Dict, Callable, Any, List
from uuid import uuid4

import cv2
import numpy as np

from camera_agent import (
    StreamlinedAgent,
    CircuitBreaker,
    DEFAULT_CONFIG_PATH,
)
from agent_runtime.unified_vlm import UnifiedVLMClient, TaskType, VLMBackend
from agent_runtime.publisher import get_camera_publisher
from agent_runtime.state import DetectionEvent
from agent_runtime.email_tools import send_email

logger = logging.getLogger(__name__)

# Environment variable names
ENV_INFERENCE_BACKEND = "INFERENCE_BACKEND"
ENV_VLLM_URL = "VLLM_URL"
ENV_OLLAMA_URL = "OLLAMA_URL"

# Export for easy importing
__all__ = ['StreamlinedMonitoringService']


class StreamlinedMonitoringService:
    """
    Camera monitoring controller using the streamlined VLM pipeline.
    
    This service:
    1. Subscribes to the camera feed publisher
    2. Runs detection analysis on captured frames
    3. Records events and sends alerts
    4. Supports dynamic prompt/task type changes during monitoring
    """

    def __init__(
        self,
        subscriber_id: Optional[str] = None,
        publisher_getter: Callable = get_camera_publisher,
        auto_start_publisher: bool = True,
    ) -> None:
        self.agent: Optional[StreamlinedAgent] = None
        self.subscriber_id = subscriber_id or f"monitor_{uuid4().hex[:8]}"
        self.is_monitoring = False
        self.last_frame: Optional[Dict[str, object]] = None
        self.last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._publisher_getter = publisher_getter
        self.publisher = None
        self.auto_start_publisher = auto_start_publisher
        
        # Dynamic prompt configuration with versioning
        self._current_task_type: TaskType = TaskType.PACKAGE_DETECTION
        self._custom_prompt: str = ""
        self._alerts_enabled: bool = False  # Only show alerts when user explicitly requests
        self._agentic_mode: bool = False  # Use tool-calling LLM agent
        self._prompt_version: int = 0  # Incremented on prompt change to discard stale frames
        self._prompt_lock = threading.Lock()
        
        # Single-frame inference lock (prevents concurrent on-demand + scheduled inference)
        self._inference_lock = threading.Lock()
        self._single_frame_in_progress = False
        
        # Timing configuration
        self.last_processed_time = 0.0
        self.capture_interval = 30.0  # Process every 30 seconds by default (scheduled inference)
        self.analysis_workers = 1
        self.max_pending_analyses = 3
        
        # Analysis executor
        self._analysis_executor: Optional[ThreadPoolExecutor] = None
        self._analysis_futures = deque()
        self._futures_lock = threading.Lock()  # Thread-safe access to futures deque
        
        # Motion detection
        self.ssim_threshold = 0.80
        self.motion_burst_interval = 0.2
        self._last_backpressure_log = 0.0
        
        # Stats tracking (thread-safe with lock)
        self._stats_lock = threading.Lock()
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections': 0,
            'alerts_sent': 0,
            'dropped_frames': 0,
            'single_frame_requests': 0,
            'analysis_avg_ms': 0.0,
            'analysis_samples': 0,
            'uptime_start': datetime.now().isoformat()
        }

    # Dynamic prompt management
    def set_active_prompt(
        self,
        task_type: TaskType,
        custom_prompt: str = "",
        alerts_enabled: bool = False,
        agentic_mode: bool = False,
    ) -> None:
        """Set the active task type and prompt for monitoring.
        
        When the prompt changes, this method:
        1. Increments the prompt version to invalidate stale frames
        2. Clears any pending inference queue
        3. Updates the active prompt configuration
        
        Args:
            task_type: The TaskType enum value for the analysis task.
            custom_prompt: Custom prompt for CUSTOM task type.
            alerts_enabled: Whether to show visual alerts on detection.
            agentic_mode: Whether to use tool-calling LLM agent mode.
        """
        with self._prompt_lock:
            # Check if prompt actually changed
            prompt_changed = (
                self._current_task_type != task_type or
                self._custom_prompt != custom_prompt
            )
            
            if prompt_changed:
                # Increment version to discard stale frames
                self._prompt_version += 1
                logger.info(
                    "Prompt changed - incrementing version to %d, clearing queue",
                    self._prompt_version
                )
                # Clear pending inference queue (stale frames from old prompt)
                self._clear_pending_queue()
            
            self._current_task_type = task_type
            self._custom_prompt = custom_prompt
            self._alerts_enabled = alerts_enabled
            self._agentic_mode = agentic_mode
        
        logger.info(
            "Active prompt updated: task=%s, alerts=%s, agentic=%s, custom=%s",
            task_type.value,
            alerts_enabled,
            agentic_mode,
            custom_prompt[:50] if custom_prompt else "None"
        )

    def _clear_pending_queue(self) -> None:
        """Clear all pending analysis futures (called when prompt changes).
        
        This discards frames queued under the old prompt to ensure
        inference results always correspond to the active prompt.
        """
        cancelled_count = 0
        with self._futures_lock:
            while self._analysis_futures:
                future = self._analysis_futures.popleft()
                if not future.done():
                    future.cancel()
                    cancelled_count += 1
        
        if cancelled_count > 0:
            logger.info("Cleared %d pending analyses from queue", cancelled_count)

    def get_active_prompt_config(self) -> Dict[str, Any]:
        """Get the current prompt configuration.
        
        Returns:
            Dictionary with task_type, custom_prompt, alerts_enabled, agentic_mode, and prompt_version.
        """
        with self._prompt_lock:
            return {
                "task_type": self._current_task_type.value,
                "custom_prompt": self._custom_prompt,
                "alerts_enabled": self._alerts_enabled,
                "agentic_mode": self._agentic_mode,
                "prompt_version": self._prompt_version,
            }

    def analyze_single_frame(
        self,
        task_type: Optional[TaskType] = None,
        custom_prompt: Optional[str] = None,
    ) -> Optional[DetectionEvent]:
        """Run on-demand single-frame inference on the latest available frame.
        
        This method:
        1. Bypasses any batched/queued frames
        2. Runs immediately on the latest frame from the camera
        3. Does not block scheduled inference (uses separate lock)
        4. Always uses the active prompt unless overridden
        
        Args:
            task_type: Optional override for the task type. Uses active config if None.
            custom_prompt: Optional override for custom prompt. Uses active config if None.
            
        Returns:
            DetectionEvent with the inference result, or None if inference fails.
        """
        # Acquire inference lock to prevent race conditions with scheduled inference
        if not self._inference_lock.acquire(timeout=5.0):
            logger.warning("Single-frame inference timed out waiting for lock")
            self.last_error = "Inference lock timeout - another inference in progress"
            return None
        
        try:
            self._single_frame_in_progress = True
            self._increment_stat('single_frame_requests')
            
            # Get the latest frame directly from publisher (bypass queue)
            publisher = self.publisher or self._publisher_getter()
            if not publisher:
                self.last_error = "Camera publisher not available"
                return None
            
            latest_frame = publisher.get_latest_frame()
            if not latest_frame or not latest_frame.image_data:
                self.last_error = "No frame available from camera"
                return None
            
            # Decode image
            try:
                nparr = np.frombuffer(latest_frame.image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    self.last_error = "Failed to decode camera frame"
                    return None
            except Exception as e:
                self.last_error = f"Image decode error: {e}"
                return None
            
            # Get prompt config - use overrides if provided, else active config
            with self._prompt_lock:
                effective_task_type = task_type if task_type is not None else self._current_task_type
                effective_prompt = custom_prompt if custom_prompt is not None else self._custom_prompt
                agentic_mode = self._agentic_mode
                prompt_version = self._prompt_version
            
            agent = self.agent
            if agent is None:
                self.last_error = "Agent not initialized"
                return None
            
            # Build metadata
            metadata = {
                'frame_number': latest_frame.frame_number,
                'timestamp': time.time(),
                'single_frame': True,
                'prompt_version': prompt_version,
                'image_data': latest_frame.image_data,
            }
            
            start_time = time.time()
            
            # Run inference (synchronously, bypass queue)
            if agentic_mode:
                event = self._run_agentic_analysis(
                    agent, frame, effective_task_type, effective_prompt, metadata
                )
            else:
                event = agent.analyze_with_prompt(
                    frame, effective_task_type, effective_prompt or "", metadata
                )
            
            # Update latency stats
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency_stats(latency_ms)
            
            if event:
                logger.info(
                    "🎯 Single-frame inference completed in %.1fms (frame %d)",
                    latency_ms,
                    latest_frame.frame_number
                )
                # Record the detection for logging
                self._record_detection(event, metadata)
            else:
                logger.warning("Single-frame inference returned no result")
            
            return event
            
        except Exception as e:
            logger.error("Single-frame inference failed: %s", e)
            self.last_error = str(e)
            return None
        finally:
            self._single_frame_in_progress = False
            self._inference_lock.release()

    # Hooks for subclasses
    def emit_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Emit realtime events (override in subclasses).
        
        Args:
            event_name: Name of the event to emit.
            payload: Event data to send.
        """
        pass

    def _record_detection(self, event: DetectionEvent, frame_metadata: Dict[str, Any]) -> None:
        """Record detection to storage (override in subclasses).
        
        Args:
            event: The detection event to record.
            frame_metadata: Metadata about the frame that was analyzed.
        """
        pass

    # Configuration
    def _apply_configuration_settings(self, config: Optional[Dict]) -> None:
        """Apply configuration to runtime parameters."""
        if not isinstance(config, dict):
            return

        camera_cfg = config.get("camera", {})
        if isinstance(camera_cfg, dict):
            interval = camera_cfg.get("capture_interval")
            if interval:
                try:
                    self.capture_interval = max(1.0, float(interval))
                except (TypeError, ValueError):
                    pass

        advanced_cfg = config.get("advanced", {})
        if isinstance(advanced_cfg, dict):
            workers = advanced_cfg.get("max_concurrent_analyses")
            if workers:
                try:
                    self.analysis_workers = max(1, int(workers))
                except (TypeError, ValueError):
                    pass
            
            pending = advanced_cfg.get("max_pending_analyses")
            if pending:
                try:
                    self.max_pending_analyses = max(1, int(pending))
                except (TypeError, ValueError):
                    pass

    def refresh_configuration(self, config: Optional[Dict] = None) -> None:
        """Refresh settings when configuration changes."""
        if config is None and self.agent:
            config = self.agent.config
        self._apply_configuration_settings(config)

    # Lifecycle
    def initialize(self) -> bool:
        """Initialize the monitoring agent."""
        try:
            config = StreamlinedAgent.load_config_from_path(DEFAULT_CONFIG_PATH)
            
            # Determine backend from environment or config
            env_backend = os.getenv(ENV_INFERENCE_BACKEND, "").lower()
            
            # Get custom prompt if configured
            detection_cfg = config.get("detection", {})
            prompt = detection_cfg.get("prompt") if isinstance(detection_cfg, dict) else None
            
            # Initialize VLM client based on backend
            if env_backend == "vllm" or (not env_backend and config.get("vllm")):
                # Use vLLM backend
                vllm_cfg = config.get("vllm", {})
                vllm_url = os.getenv(ENV_VLLM_URL) or str(vllm_cfg.get("url", "http://localhost:8000")).rstrip("/")
                default_model = os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
                vision_model = str(vllm_cfg.get("model", default_model))
                timeout = int(os.getenv("VLLM_TIMEOUT", vllm_cfg.get("timeout", 300)))
                temperature = float(os.getenv("VLLM_TEMPERATURE", vllm_cfg.get("temperature", 0.1)))
                
                logger.info("Initializing vLLM client: url=%s, model=%s", vllm_url, vision_model)
                vlm_client = UnifiedVLMClient(
                    base_url=vllm_url,
                    model=vision_model,
                    timeout=timeout,
                    prompt=prompt,
                    backend=VLMBackend.VLLM,
                    temperature=temperature,
                )
                
                if not vlm_client.test_connection():
                    self.last_error = f"Cannot connect to vLLM at {vllm_url}"
                    return False
            else:
                # Use Ollama backend (legacy fallback)
                ollama_cfg = config.get("ollama", {})
                ollama_url = os.getenv(ENV_OLLAMA_URL) or str(ollama_cfg.get("url", "http://localhost:11434")).rstrip("/")
                default_model = os.getenv("VISION_MODEL", "qwen3-vl:8b")
                vision_model = str(ollama_cfg.get("vision_model", default_model))
                timeout = int(ollama_cfg.get("timeout", 300))
                
                logger.info("Initializing Ollama client (legacy): url=%s, model=%s", ollama_url, vision_model)
                vlm_client = UnifiedVLMClient(
                    base_url=ollama_url,
                    model=vision_model,
                    timeout=timeout,
                    prompt=prompt,
                    backend=VLMBackend.OLLAMA,
                )
                
                if not vlm_client.test_connection():
                    self.last_error = f"Cannot connect to Ollama at {ollama_url}"
                    return False
            
            # Initialize circuit breaker with higher tolerance
            circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=120.0,
            )
            
            # Create agent (VLM-only pipeline)
            self.agent = StreamlinedAgent(
                config=config,
                vlm_client=vlm_client,
                circuit_breaker=circuit_breaker,
            )
            
            self._apply_configuration_settings(config)
            
            # Reset executor
            if self._analysis_executor:
                self._analysis_executor.shutdown(wait=False, cancel_futures=True)
            self._analysis_executor = None
            with self._futures_lock:
                self._analysis_futures.clear()
            
            self.last_error = None
            logger.info("Camera monitoring service initialized")
            return True
            
        except Exception as exc:
            self.last_error = str(exc)
            logger.exception("Failed to initialize monitoring service")
            return False

    def start_monitoring(self) -> bool:
        """Start monitoring the camera feed."""
        if self.is_monitoring:
            return False
        
        if not self.agent:
            if not self.initialize():
                return False
        
        publisher = self._publisher_getter()
        if publisher is None:
            self.last_error = "Camera publisher not available"
            return False
        
        if not getattr(publisher, 'is_running', False):
            if not self.auto_start_publisher:
                self.last_error = "Camera publisher not running"
                return False
            if not publisher.start():
                self.last_error = "Failed to start camera publisher"
                return False
        
        if not publisher.subscribe(self.subscriber_id):
            self.last_error = "Failed to subscribe to camera feed"
            return False
        
        self.publisher = publisher
        with self._futures_lock:
            self._analysis_futures.clear()
        self._ensure_executor()
        self._last_backpressure_log = 0.0
        self.last_processed_time = time.time() - self.capture_interval
        
        self.is_monitoring = True
        thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        thread.start()
        self._thread = thread
        
        logger.info("Monitoring started")
        return True

    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.is_monitoring = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        
        if self.publisher:
            self.publisher.unsubscribe(self.subscriber_id)
        
        if self._analysis_executor:
            self._analysis_executor.shutdown(wait=False, cancel_futures=True)
            self._analysis_executor = None
        with self._futures_lock:
            self._analysis_futures.clear()
        
        logger.info("Monitoring stopped")

    # Executor management
    def _ensure_executor(self) -> None:
        if self._analysis_executor:
            return
        self._analysis_executor = ThreadPoolExecutor(
            max_workers=self.analysis_workers,
            thread_name_prefix="vlm-analysis",
        )

    def _count_pending(self) -> int:
        with self._futures_lock:
            return sum(1 for f in self._analysis_futures if not f.done())

    def _update_latency_stats(self, latency_ms: float) -> None:
        with self._stats_lock:
            samples = self.stats.get('analysis_samples', 0)
            avg = self.stats.get('analysis_avg_ms', 0.0)
            new_samples = samples + 1
            self.stats['analysis_avg_ms'] = ((avg * samples) + latency_ms) / new_samples
            self.stats['analysis_samples'] = new_samples

    def _increment_stat(self, key: str, amount: int = 1) -> None:
        """Thread-safe increment of a stats counter."""
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + amount

    # Main loop
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for scheduled inference (every 30s by default).
        
        This loop:
        1. Processes completed analyses
        2. Triggers scheduled inference at capture_interval (default 30s)
        3. Yields to single-frame requests (doesn't block on-demand inference)
        4. Tracks prompt version to discard stale frames
        """
        publisher = self.publisher
        if not publisher:
            self.last_error = "Publisher unavailable"
            return
        
        prev_frame = None
        
        while self.is_monitoring:
            try:
                # Process completed analyses
                self._drain_results()
                
                current_time = time.time()
                
                # Get frame from publisher
                frame_obj = publisher.get_frame(self.subscriber_id, timeout=0.5)
                if not frame_obj:
                    continue
                
                self._increment_stat('total_frames')
                frame_count = frame_obj.frame_number
                
                self.last_frame = {
                    'image_b64': frame_obj.image_b64,
                    'timestamp': frame_obj.timestamp,
                    'frame_number': frame_count,
                }
                
                self.emit_event('monitoring_update', {
                    'frame_number': frame_count,
                    'stats': self._serialize_stats(),
                })
                
                # Check if enough time has passed for scheduled inference
                time_since_last = current_time - self.last_processed_time
                if time_since_last < self.capture_interval:
                    continue
                
                # Skip if single-frame inference is in progress (don't block it)
                if self._single_frame_in_progress:
                    logger.debug("Skipping scheduled inference - single-frame in progress")
                    continue
                
                # Check backpressure
                pending = self._count_pending()
                if pending >= self.max_pending_analyses:
                    if current_time - self._last_backpressure_log >= 5.0:
                        logger.warning(
                            "⏳ Backpressure: %d pending analyses (max %d)",
                            pending, self.max_pending_analyses
                        )
                        self._last_backpressure_log = current_time
                    self._increment_stat('dropped_frames')
                    continue
                
                # Get current prompt version for staleness check
                with self._prompt_lock:
                    prompt_version = self._prompt_version
                
                # Submit for scheduled analysis
                frame_metadata = {
                    'frame_number': frame_count,
                    'timestamp': current_time,
                    'queued_at': current_time,
                    'image_data': frame_obj.image_data,  # Store for email attachments
                    'prompt_version': prompt_version,  # Track for staleness check
                    'scheduled': True,
                }
                
                # Ensure executor is available
                if self._analysis_executor is None:
                    self._ensure_executor()
                
                future = self._analysis_executor.submit(
                    self._analyze_frame_task,
                    frame_obj.image_data,
                    frame_metadata,
                )
                future.frame_metadata = frame_metadata
                with self._futures_lock:
                    self._analysis_futures.append(future)
                
                self.last_processed_time = current_time
                self._increment_stat('processed_frames')
                
                logger.debug(f"🔍 Submitted frame {frame_count} for analysis")
                
            except Exception as exc:
                logger.error(f"Monitoring loop error: {exc}")
                self.last_error = str(exc)
                time.sleep(2)
        
        # Flush remaining results
        self._drain_results(flush=True)

    def _analyze_frame_task(self, image_data: bytes, metadata: Dict) -> Optional[DetectionEvent]:
        """Analysis task that runs in thread pool for scheduled inference.
        
        This method checks prompt version to discard stale frames from
        previous prompt configurations.
        
        Args:
            image_data: JPEG encoded image bytes.
            metadata: Frame metadata including frame number, timestamp, and prompt_version.
            
        Returns:
            DetectionEvent if analysis succeeds, None otherwise.
            
        Raises:
            RuntimeError: If the agent is not initialized.
        """
        agent = self.agent
        if agent is None:
            raise RuntimeError("Agent not initialized")
        
        # Check for stale frame (prompt changed since frame was queued)
        frame_prompt_version = metadata.get('prompt_version')
        with self._prompt_lock:
            current_version = self._prompt_version
        
        if frame_prompt_version is not None and frame_prompt_version < current_version:
            frame_num = metadata.get('frame_number', '?')
            logger.info(
                "🗑️ Discarding stale frame %s (prompt version %d < current %d)",
                frame_num, frame_prompt_version, current_version
            )
            return None  # Discard - result would not match active prompt
        
        # Decode image
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Failed to decode image")
                return None
        except Exception as e:
            logger.warning(f"Image decode error: {e}")
            return None
        
        # Get current prompt configuration
        with self._prompt_lock:
            task_type = self._current_task_type
            custom_prompt = self._custom_prompt
            agentic_mode = self._agentic_mode
        
        # Use agentic mode if enabled (LLM decides which tools to call)
        if agentic_mode:
            return self._run_agentic_analysis(agent, frame, task_type, custom_prompt, metadata)
        
        # Use analyze_with_prompt for all task types
        # This properly handles PPE, Person Counting, Scene Description, and Custom
        return agent.analyze_with_prompt(frame, task_type, custom_prompt or "", metadata)

    def _run_agentic_analysis(
        self,
        agent: StreamlinedAgent,
        frame: np.ndarray,
        task_type: TaskType,
        custom_prompt: str,
        metadata: Dict,
    ) -> Optional[DetectionEvent]:
        """Run agentic analysis with tool calling.
        
        The LLM agent decides which tools to call based on its analysis.
        Tools include: send_alert_email, save_evidence, log_event, etc.
        
        Args:
            agent: The StreamlinedAgent instance.
            frame: The decoded image frame.
            task_type: The task type for analysis.
            custom_prompt: Custom prompt for the analysis.
            metadata: Frame metadata.
            
        Returns:
            DetectionEvent with tool trace information.
        """
        from datetime import datetime
        
        try:
            # Use the agentic analysis method
            event = agent.analyze_agentic(
                frame=frame,
                task_type=task_type,
                custom_prompt=custom_prompt or None,
            )
            
            if event:
                # Log tool usage
                tools_used = event.tools_used or []
                if tools_used:
                    logger.info(
                        "🤖 Agentic analysis completed. Tools called: %s",
                        ", ".join(tools_used)
                    )
                else:
                    logger.info("🤖 Agentic analysis completed. No tools called.")
            
            return event
            
        except Exception as e:
            logger.error("Agentic analysis failed: %s", e)
            # Return a failed event
            return DetectionEvent(
                timestamp=datetime.now().isoformat(),
                detected=False,
                confidence=0.0,
                primary_label="agentic_analysis_failed",
                vision_description=f"Agentic analysis failed: {e}",
                full_response="",
                decision_trace={"error": str(e), "agentic_mode": True},
            )

    def _drain_results(self, flush: bool = False) -> None:
        """Process completed analysis results."""
        # Thread-safe check for empty queue
        with self._futures_lock:
            if not self._analysis_futures:
                return
        
        still_pending = deque()
        completed_futures = []
        
        # Collect futures under lock, process outside lock to avoid blocking
        with self._futures_lock:
            while self._analysis_futures:
                future = self._analysis_futures.popleft()
                
                if not future.done():
                    if flush:
                        future.cancel()
                    else:
                        still_pending.append(future)
                    continue
                
                if not future.cancelled():
                    completed_futures.append(future)
            
            # Put pending futures back
            self._analysis_futures.extend(still_pending)
        
        # Process completed futures outside of lock
        for future in completed_futures:
            metadata = getattr(future, 'frame_metadata', {})
            
            try:
                event = future.result()
            except Exception as exc:
                frame_num = metadata.get('frame_number', '?')
                logger.warning(f"Analysis failed for frame {frame_num}: {exc}")
                # Record failure
                failed_event = DetectionEvent(
                    timestamp=datetime.now().isoformat(),
                    detected=False,
                    confidence=0.0,
                    primary_label="analysis_failed",
                    full_response=f"Analysis failed: {exc}",
                    vision_description="",
                    decision_trace={"error": str(exc)},
                )
                self._record_detection(failed_event, metadata)
                continue
            
            self._handle_result(event, metadata)

    def _handle_result(self, event: Optional[DetectionEvent], metadata: Dict) -> None:
        """Handle analysis result."""
        frame_num = metadata.get('frame_number', '?')
        
        # Calculate latency
        queued_at = metadata.get('queued_at')
        if queued_at:
            latency_ms = (time.time() - queued_at) * 1000
            self._update_latency_stats(latency_ms)
            metadata['analysis_latency_ms'] = latency_ms
        
        if not event:
            logger.warning(f"⚠️ Analysis returned None for frame {frame_num}")
            return
        
        # Record the detection
        self._record_detection(event, metadata)
        
        # Check if alerts are enabled by user and get custom prompt
        with self._prompt_lock:
            alerts_enabled = self._alerts_enabled
            custom_prompt = self._custom_prompt
            task_type = self._current_task_type
        
        # Update stats and send alerts
        if event.detected:
            self._increment_stat('detections')
            
            # Only send alerts and show visual indicators if alerts are enabled
            effective_should_alert = event.should_alert and alerts_enabled
            
            if effective_should_alert:
                # Get image_data from metadata for email attachments
                image_data = metadata.get('image_data') if metadata else None
                
                # Check for email in custom prompt and send if requested
                if task_type == TaskType.CUSTOM and custom_prompt:
                    # Pass metadata for image attachment
                    email_sent = self._send_custom_email_alert(custom_prompt, event, metadata)
                    if email_sent:
                        self._increment_stat('alerts_sent')
                        logger.info("📧 Custom email alert sent")
                elif self.agent and self.agent.process_detection(event, image_data=image_data):
                    self._increment_stat('alerts_sent')
                    logger.info("🔔 Alert sent for detection")
            
            # Emit event to connected clients
            # Override should_alert based on user's alerts_enabled preference
            self.emit_event('detection_event', {
                'event': {
                    'timestamp': event.timestamp,
                    'confidence': event.confidence,
                    'response': event.vision_description[:200] if event.vision_description else "",
                    'shipping_label_present': event.shipping_label_present,
                    'should_alert': effective_should_alert,  # Only true if alerts enabled
                    'show_overlay': alerts_enabled,  # Frontend uses this to show/hide overlay
                    'task_type': task_type.value,  # Include task type for dynamic UI
                    'primary_label': event.primary_label,  # Include label for display
                },
                'stats': self._serialize_stats(),
            })
        else:
            logger.debug(f"No detection in frame {frame_num}")

    def _extract_email_from_prompt(self, prompt: str) -> Optional[List[str]]:
        """Extract email addresses from a custom prompt.
        
        Looks for patterns like:
        - "send email to user@example.com"
        - "email user@example.com"
        - "notify user@example.com"
        - "send email to user1@example.com and to user2@example.com"
        
        Args:
            prompt: The custom prompt string.
            
        Returns:
            List of email addresses if found, None otherwise.
        """
        # Common email regex pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        matches = re.findall(email_pattern, prompt, re.IGNORECASE)
        if matches:
            return matches
        return None

    def _send_custom_email_alert(
        self, 
        custom_prompt: str, 
        event: DetectionEvent, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send email alert based on custom prompt with optional image attachment.
        
        Args:
            custom_prompt: The user's custom prompt containing email(s).
            event: The detection event that triggered the alert.
            metadata: Optional frame metadata containing image_data for attachment.
            
        Returns:
            True if email was sent successfully, False otherwise.
        """
        email_addresses = self._extract_email_from_prompt(custom_prompt)
        if not email_addresses:
            logger.warning("No email address found in custom prompt: %s", custom_prompt[:50])
            return False
        
        # Compose email
        subject = f"[Camera Agent Alert] Detection at {event.timestamp}"
        body = f"""
Camera Agent Custom Alert

Custom Query: {custom_prompt}

Detection Details:
- Timestamp: {event.timestamp}
- Confidence: {event.confidence:.2f}
- Description: {event.vision_description or 'N/A'}

This alert was triggered based on your custom monitoring query.

An image of the detection is attached below.
"""
        
        # Build email payload
        email_payload: Dict[str, Any] = {
            "to": email_addresses,
            "subject": subject,
            "body": body,
        }
        
        # Attach the image - try multiple sources
        image_attached = False
        
        # Source 1: image_data from metadata (preferred - raw bytes)
        if metadata and metadata.get('image_data'):
            image_data = metadata['image_data']
            if isinstance(image_data, bytes):
                timestamp_str = event.timestamp.replace(":", "-").replace(" ", "_")[:19]
                email_payload["image_data"] = image_data
                email_payload["image_filename"] = f"detection_{timestamp_str}.jpg"
                logger.info("Attaching image to email (%d bytes) from frame metadata", len(image_data))
                image_attached = True
        
        # Source 2: image_path from event (fallback - read from disk)
        if not image_attached and event.image_path:
            try:
                from pathlib import Path
                image_path = Path(event.image_path).expanduser()
                if image_path.exists():
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()
                    email_payload["image_data"] = image_bytes
                    email_payload["image_filename"] = image_path.name
                    logger.info("Attaching image to email (%d bytes) from file: %s", len(image_bytes), image_path)
                    image_attached = True
                else:
                    logger.warning("Image path %s does not exist", image_path)
            except Exception as e:
                logger.error("Failed to read image from path %s: %s", event.image_path, e)
        
        if not image_attached:
            logger.warning("No image available to attach to email alert")
        
        try:
            result = send_email(email_payload)
            if "sent" in result.lower() or "success" in result.lower():
                logger.info("📧 Email %s sent to %s for custom alert", 
                           "with image" if image_attached else "without image", email_addresses)
                return True
            else:
                logger.warning("Email result: %s", result)
                return "sent" in result.lower()
        except Exception as e:
            logger.error("Email sending failed: %s", e)
            return False

    def _serialize_stats(self) -> Dict[str, object]:
        """Serialize stats for API response."""
        with self._stats_lock:
            stats = self.stats.copy()
        if 'analysis_avg_ms' in stats:
            stats['analysis_avg_ms'] = round(stats['analysis_avg_ms'], 2)
        stats['last_update'] = datetime.now().isoformat()
        return stats
