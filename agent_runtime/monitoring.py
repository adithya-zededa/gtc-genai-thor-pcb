#!/usr/bin/env python3
"""
Streamlined camera monitoring service.

This module provides the StreamlinedMonitoringService that subscribes to the
camera feed and runs the unified VLM-based detection pipeline.
"""

import logging
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Dict, Callable, Any
from uuid import uuid4

import cv2
import numpy as np

from camera_agent import (
    StreamlinedAgent,
    RFDetrAdapter,
    CircuitBreaker,
    DEFAULT_CONFIG_PATH,
)
from agent_runtime.unified_vlm import UnifiedVLMClient, TaskType
from agent_runtime.publisher import get_camera_publisher
from agent_runtime.state import DetectionEvent
from agent_runtime.email_tools import send_email

logger = logging.getLogger(__name__)

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
        
        # Dynamic prompt configuration
        self._current_task_type: TaskType = TaskType.PACKAGE_DETECTION
        self._custom_prompt: str = ""
        self._alerts_enabled: bool = False  # Only show alerts when user explicitly requests
        self._prompt_lock = threading.Lock()
        
        # Timing configuration
        self.last_processed_time = 0.0
        self.capture_interval = 30.0  # Process every 30 seconds by default
        self.analysis_workers = 1
        self.max_pending_analyses = 3
        
        # Analysis executor
        self._analysis_executor: Optional[ThreadPoolExecutor] = None
        self._analysis_futures = deque()
        
        # Motion detection
        self.ssim_threshold = 0.80
        self.motion_burst_interval = 0.2
        self._last_backpressure_log = 0.0
        
        # Stats tracking
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections': 0,
            'alerts_sent': 0,
            'dropped_frames': 0,
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
    ) -> None:
        """Set the active task type and prompt for monitoring.
        
        Args:
            task_type: The TaskType enum value for the analysis task.
            custom_prompt: Custom prompt for CUSTOM task type.
            alerts_enabled: Whether to show visual alerts on detection.
        """
        with self._prompt_lock:
            self._current_task_type = task_type
            self._custom_prompt = custom_prompt
            self._alerts_enabled = alerts_enabled
        
        logger.info(
            "Active prompt updated: task=%s, alerts=%s, custom=%s",
            task_type.value,
            alerts_enabled,
            custom_prompt[:50] if custom_prompt else "None"
        )

    def get_active_prompt_config(self) -> Dict[str, Any]:
        """Get the current prompt configuration.
        
        Returns:
            Dictionary with task_type, custom_prompt, and alerts_enabled.
        """
        with self._prompt_lock:
            return {
                "task_type": self._current_task_type.value,
                "custom_prompt": self._custom_prompt,
                "alerts_enabled": self._alerts_enabled,
            }

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
            
            # Initialize unified VLM client
            ollama_cfg = config.get("ollama", {})
            ollama_url = str(ollama_cfg.get("url", "http://localhost:11434")).rstrip("/")
            vision_model = str(ollama_cfg.get("vision_model", "qwen3-vl:4b"))
            timeout = int(ollama_cfg.get("timeout", 300))
            
            # Get custom prompt if configured
            detection_cfg = config.get("detection", {})
            prompt = detection_cfg.get("prompt") if isinstance(detection_cfg, dict) else None
            
            vlm_client = UnifiedVLMClient(
                base_url=ollama_url,
                model=vision_model,
                timeout=timeout,
                prompt=prompt,
            )
            
            if not vlm_client.test_connection():
                self.last_error = f"Cannot connect to Ollama at {ollama_url}"
                return False
            
            # Initialize RF-DETR detector
            rfdet_cfg = config.get("analysis", {}).get("rf_detr", {})
            detector = RFDetrAdapter(rfdet_cfg)
            
            # Initialize circuit breaker with higher tolerance
            circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=120.0,
            )
            
            # Create agent
            self.agent = StreamlinedAgent(
                config=config,
                vlm_client=vlm_client,
                detector=detector,
                circuit_breaker=circuit_breaker,
            )
            
            self._apply_configuration_settings(config)
            
            # Reset executor
            if self._analysis_executor:
                self._analysis_executor.shutdown(wait=False, cancel_futures=True)
            self._analysis_executor = None
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
        return sum(1 for f in self._analysis_futures if not f.done())

    def _update_latency_stats(self, latency_ms: float) -> None:
        samples = self.stats.get('analysis_samples', 0)
        avg = self.stats.get('analysis_avg_ms', 0.0)
        new_samples = samples + 1
        self.stats['analysis_avg_ms'] = ((avg * samples) + latency_ms) / new_samples
        self.stats['analysis_samples'] = new_samples

    # Main loop
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
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
                
                self.stats['total_frames'] += 1
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
                
                # Check if enough time has passed
                time_since_last = current_time - self.last_processed_time
                if time_since_last < self.capture_interval:
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
                    self.stats['dropped_frames'] += 1
                    continue
                
                # Submit for analysis
                frame_metadata = {
                    'frame_number': frame_count,
                    'timestamp': current_time,
                    'queued_at': current_time,
                }
                
                future = self._analysis_executor.submit(
                    self._analyze_frame_task,
                    frame_obj.image_data,
                    frame_metadata,
                )
                future.frame_metadata = frame_metadata
                self._analysis_futures.append(future)
                
                self.last_processed_time = current_time
                self.stats['processed_frames'] += 1
                
                logger.debug(f"🔍 Submitted frame {frame_count} for analysis")
                
            except Exception as exc:
                logger.error(f"Monitoring loop error: {exc}")
                self.last_error = str(exc)
                time.sleep(2)
        
        # Flush remaining results
        self._drain_results(flush=True)

    def _analyze_frame_task(self, image_data: bytes, metadata: Dict) -> Optional[DetectionEvent]:
        """Analysis task that runs in thread pool.
        
        Args:
            image_data: JPEG encoded image bytes.
            metadata: Frame metadata including frame number and timestamp.
            
        Returns:
            DetectionEvent if analysis succeeds, None otherwise.
            
        Raises:
            RuntimeError: If the agent is not initialized.
        """
        agent = self.agent
        if agent is None:
            raise RuntimeError("Agent not initialized")
        
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
        
        # Use custom analysis if task type is custom with a prompt
        if task_type == TaskType.CUSTOM and custom_prompt:
            return agent.analyze_with_prompt(frame, task_type, custom_prompt, metadata)
        
        # Otherwise use standard analysis
        return agent.analyze_frame(frame, metadata)

    def _drain_results(self, flush: bool = False) -> None:
        """Process completed analysis results."""
        if not self._analysis_futures:
            return
        
        still_pending = deque()
        
        while self._analysis_futures:
            future = self._analysis_futures.popleft()
            
            if not future.done():
                if flush:
                    future.cancel()
                else:
                    still_pending.append(future)
                continue
            
            metadata = getattr(future, 'frame_metadata', {})
            
            if future.cancelled():
                continue
            
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
        
        self._analysis_futures.extend(still_pending)

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
            self.stats['detections'] += 1
            
            # Only send alerts and show visual indicators if alerts are enabled
            effective_should_alert = event.should_alert and alerts_enabled
            
            if effective_should_alert:
                # Check for email in custom prompt and send if requested
                if task_type == TaskType.CUSTOM and custom_prompt:
                    email_sent = self._send_custom_email_alert(custom_prompt, event)
                    if email_sent:
                        self.stats['alerts_sent'] += 1
                        logger.info("📧 Custom email alert sent")
                elif self.agent and self.agent.process_detection(event):
                    self.stats['alerts_sent'] += 1
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
                },
                'stats': self._serialize_stats(),
            })
        else:
            logger.debug(f"No detection in frame {frame_num}")

    def _extract_email_from_prompt(self, prompt: str) -> Optional[str]:
        """Extract email address from a custom prompt.
        
        Looks for patterns like:
        - "send email to user@example.com"
        - "email user@example.com"
        - "notify user@example.com"
        
        Args:
            prompt: The custom prompt string.
            
        Returns:
            Email address if found, None otherwise.
        """
        # Common email regex pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(email_pattern, prompt, re.IGNORECASE)
        if match:
            return match.group(0)
        return None

    def _send_custom_email_alert(self, custom_prompt: str, event: DetectionEvent) -> bool:
        """Send email alert based on custom prompt.
        
        Args:
            custom_prompt: The user's custom prompt containing email.
            event: The detection event that triggered the alert.
            
        Returns:
            True if email was sent successfully, False otherwise.
        """
        email_address = self._extract_email_from_prompt(custom_prompt)
        if not email_address:
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
"""
        
        try:
            # send_email takes a payload dict, not keyword arguments
            result = send_email({
                "to": email_address,
                "subject": subject,
                "body": body,
            })
            # send_email returns a string, not a dict
            if "sent" in result.lower() or "success" in result.lower():
                logger.info("📧 Email sent to %s for custom alert", email_address)
                return True
            else:
                logger.warning("Email result: %s", result)
                return "sent" in result.lower()
        except Exception as e:
            logger.error("Email sending failed: %s", e)
            return False

    def _serialize_stats(self) -> Dict[str, object]:
        """Serialize stats for API response."""
        stats = self.stats.copy()
        if 'analysis_avg_ms' in stats:
            stats['analysis_avg_ms'] = round(stats['analysis_avg_ms'], 2)
        stats['last_update'] = datetime.now().isoformat()
        return stats
