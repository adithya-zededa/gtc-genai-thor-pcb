#!/usr/bin/env python3
"""Shared camera monitoring logic for consuming camera feed frames and running analysis."""

import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from datetime import datetime
from typing import Optional, Dict, Callable
from uuid import uuid4

import cv2

try:
    from skimage.metrics import structural_similarity
except ImportError:  # pragma: no cover - optional dependency
    structural_similarity = None

from camera_agent import MonitorDetectionAgent
from camera_feed_publisher import get_camera_publisher

logger = logging.getLogger(__name__)


class CameraMonitoringService:
    """Camera monitoring controller that subscribes to a camera feed and runs AI analysis."""

    def __init__(
        self,
        subscriber_id: Optional[str] = None,
        publisher_getter: Callable = get_camera_publisher,
        auto_start_publisher: bool = True,
    ) -> None:
        self.agent: Optional[MonitorDetectionAgent] = None
        self.subscriber_id = subscriber_id or f"monitoring_agent_{uuid4().hex[:8]}"
        self.is_monitoring = False
        self.last_frame: Optional[Dict[str, object]] = None
        self.last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._publisher_getter = publisher_getter
        self.publisher = None
        self.auto_start_publisher = auto_start_publisher
        self.last_processed_time = 0.0
        self.capture_interval = 5.0
        self.analysis_workers = 1
        self.max_pending_analyses = 12
        self._analysis_executor: Optional[ThreadPoolExecutor] = None
        self._analysis_futures = deque()
        self.ssim_threshold = 0.80
        self.motion_burst_interval = 0.2
        self.motion_burst_window = 12.0
        self.motion_burst_ssim = 0.75
        self.pending_similarity_threshold = 0.92
        self._pending_reference_frames = deque()
        self._last_dedupe_log = 0.0
        self.last_motion_time = 0.0
        self._last_backpressure_log = 0.0
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections': 0,
            'alerts_sent': 0,
            'dropped_frames': 0,
            'deduped_frames': 0,
            'analysis_avg_ms': 0.0,
            'analysis_samples': 0,
            'uptime_start': datetime.now().isoformat()
        }

    # Hooks for subclasses / consumers -------------------------------------------------
    def emit_event(self, event_name: str, payload: Dict[str, object]) -> None:
        """Emit realtime events to observers (default is no-op)."""
        return

    def _record_detection(self, event, frame_metadata) -> None:  # pragma: no cover - default no-op
        """Persist detection events (override in subclasses)."""
        return

    # Configuration --------------------------------------------------------------------
    def _apply_configuration_settings(self, config: Optional[Dict]) -> None:
        """Apply configuration values to runtime parameters."""
        if not isinstance(config, dict):
            return

        camera_config = config.get("camera")
        if isinstance(camera_config, dict):
            interval = camera_config.get("capture_interval")
            self.capture_interval = self._safe_positive_float(interval, self.capture_interval)

            preprocessing_cfg = camera_config.get("preprocessing")
            threshold_cfg = preprocessing_cfg.get("diff_threshold") if isinstance(preprocessing_cfg, dict) else None
            env_threshold = os.getenv("SSIM_DIFF_THRESHOLD")
            threshold_source = env_threshold if env_threshold is not None else threshold_cfg
            if threshold_source is not None:
                self.ssim_threshold = self._clamp(threshold_source, 0.0, 1.0, self.ssim_threshold)

        advanced_cfg = config.get("advanced")
        if isinstance(advanced_cfg, dict):
            configured_workers = self._safe_positive_int(advanced_cfg.get("max_concurrent_analyses"), self.analysis_workers)
            env_workers = os.getenv("MAX_CONCURRENT_ANALYSES")
            self.analysis_workers = self._safe_positive_int(env_workers, configured_workers)

            default_pending = max(self.analysis_workers * 3, self.max_pending_analyses, 1)
            configured_pending = self._safe_positive_int(advanced_cfg.get("max_pending_analyses"), default_pending)
            env_pending = os.getenv("MAX_PENDING_ANALYSES")
            self.max_pending_analyses = self._safe_positive_int(env_pending, configured_pending)

            motion_interval_source = os.getenv("MOTION_BURST_INTERVAL")
            if motion_interval_source is None:
                motion_interval_source = advanced_cfg.get("motion_burst_interval")
            motion_interval = self._safe_float(motion_interval_source, self.motion_burst_interval)
            self.motion_burst_interval = motion_interval if motion_interval > 0 else 0.0

            motion_window_source = os.getenv("MOTION_BURST_WINDOW")
            if motion_window_source is None:
                motion_window_source = advanced_cfg.get("motion_burst_window")
            motion_window = self._safe_float(motion_window_source, self.motion_burst_window)
            self.motion_burst_window = motion_window if motion_window > 0 else 0.0

            burst_ssim_source = os.getenv("MOTION_BURST_SSIM")
            if burst_ssim_source is None:
                burst_ssim_source = advanced_cfg.get("motion_burst_ssim")
            if burst_ssim_source is not None:
                self.motion_burst_ssim = self._clamp(
                    burst_ssim_source,
                    0.0,
                    1.0,
                    self.motion_burst_ssim
                )

            dedupe_ssim_source = os.getenv("PENDING_DEDUPE_SSIM")
            if dedupe_ssim_source is None:
                dedupe_ssim_source = advanced_cfg.get("pending_dedupe_ssim")
            if dedupe_ssim_source is not None:
                self.pending_similarity_threshold = self._clamp(
                    dedupe_ssim_source,
                    0.0,
                    1.0,
                    self.pending_similarity_threshold
                )

        self.last_motion_time = 0.0

    def refresh_configuration(self, config: Optional[Dict] = None) -> None:
        """Refresh derived runtime settings when configuration changes."""
        if config is None and self.agent is not None:
            config = self.agent.config
        self._apply_configuration_settings(config)

    # Utility helpers ------------------------------------------------------------------
    @staticmethod
    def _safe_positive_int(value, default):
        try:
            val = int(value)
            return val if val > 0 else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_positive_float(value, default):
        try:
            val = float(value)
            return val if val > 0 else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value, minimum, maximum, default):
        try:
            val = float(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, val))

    @staticmethod
    def _make_reference_frame(frame):
        try:
            target_size = (320, 240)
            resized = cv2.resize(frame, target_size)
            return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

    # Lifecycle ------------------------------------------------------------------------
    def initialize(self) -> bool:
        """Initialize the monitoring agent components."""
        try:
            self.agent = MonitorDetectionAgent()
            if not self.agent.ollama_client.test_connection():
                self.last_error = (
                    f"Failed to connect to Ollama service at {self.agent.ollama_client.base_url}"
                )
                return False

            if not self.agent.ensure_rfdet_ready():
                self.last_error = "RF-DETR package detector failed to load"
                return False

            self._apply_configuration_settings(self.agent.config)

            if self._analysis_executor:
                self._analysis_executor.shutdown(wait=False, cancel_futures=True)
                self._analysis_executor = None
            self._analysis_futures.clear()

            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            logger.exception("Failed to initialize camera agent")
            return False

    def start_monitoring(self) -> bool:
        """Start monitoring by subscribing to camera feed for analysis."""
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
                self.last_error = "Camera publisher failed to start"
                return False

        if not publisher.subscribe(self.subscriber_id):
            self.last_error = "Failed to subscribe to camera feed"
            return False

        self.publisher = publisher
        self._analysis_futures.clear()
        self._ensure_analysis_executor()
        self._last_backpressure_log = 0.0
        now = time.time()
        self.last_processed_time = now - self.capture_interval
        self.last_motion_time = now - (self.motion_burst_interval if self.motion_burst_interval > 0 else 0.0)
        self._pending_reference_frames.clear()
        self._last_dedupe_log = 0.0

        self.is_monitoring = True
        thread = threading.Thread(target=self._monitoring_loop)
        thread.daemon = True
        thread.start()
        self._thread = thread
        return True

    def stop_monitoring(self) -> None:
        """Stop monitoring and release subscription."""
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
        self._pending_reference_frames.clear()

    # Analysis management --------------------------------------------------------------
    def _ensure_analysis_executor(self) -> None:
        if self._analysis_executor:
            return
        try:
            self._analysis_executor = ThreadPoolExecutor(
                max_workers=self.analysis_workers,
                thread_name_prefix="ollama-analysis"
            )
        except ValueError:
            self.analysis_workers = max(self.analysis_workers, 1)
            self._analysis_executor = ThreadPoolExecutor(
                max_workers=self.analysis_workers,
                thread_name_prefix="ollama-analysis"
            )

    def _count_pending_futures(self) -> int:
        return sum(1 for future in self._analysis_futures if not future.done())

    def _is_duplicate_pending(self, reference_frame):
        if reference_frame is None or not self._pending_reference_frames:
            return False, 0.0
        if structural_similarity is None:
            return False, 0.0

        max_similarity = 0.0
        for future, pending_frame in list(self._pending_reference_frames):
            if future.cancelled():
                continue
            if pending_frame is None:
                continue
            try:
                similarity = structural_similarity(pending_frame, reference_frame, data_range=255)
            except Exception:
                continue

            if similarity > max_similarity:
                max_similarity = similarity

            if similarity >= self.pending_similarity_threshold:
                return True, similarity

        return False, max_similarity

    def _prune_pending_reference(self, future) -> None:
        if not self._pending_reference_frames:
            return
        cleaned = deque()
        for existing_future, ref_frame in self._pending_reference_frames:
            if existing_future is future or existing_future.cancelled():
                continue
            cleaned.append((existing_future, ref_frame))
        self._pending_reference_frames = cleaned

    def _cleanup_pending_references(self) -> None:
        if not self._pending_reference_frames:
            return
        cleaned = deque()
        for future, ref_frame in self._pending_reference_frames:
            if future.cancelled() or future.done():
                continue
            cleaned.append((future, ref_frame))
        self._pending_reference_frames = cleaned

    def _record_backpressure(self, current_time, frame_number, pending) -> None:
        self.stats['dropped_frames'] += 1
        if current_time - self._last_backpressure_log >= 1.0:
            print(
                f"⏳ Skipping frame {frame_number}; {pending} tasks pending (limit {self.max_pending_analyses})"
            )
            self._last_backpressure_log = current_time

    def _update_latency_stats(self, latency_ms: float) -> None:
        samples = self.stats.get('analysis_samples', 0)
        average = self.stats.get('analysis_avg_ms', 0.0)
        new_samples = samples + 1
        self.stats['analysis_avg_ms'] = ((average * samples) + latency_ms) / new_samples
        self.stats['analysis_samples'] = new_samples

    # Monitoring loop ------------------------------------------------------------------
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that consumes frames from the publisher."""
        publisher = self.publisher
        if not publisher:
            self.last_error = "Camera publisher unavailable"
            return
        prev_frame_for_ssim = None

        while self.is_monitoring:
            try:
                self._drain_analysis_results()
                current_time = time.time()

                frame_obj = publisher.get_frame(self.subscriber_id, timeout=0.5)
                if not frame_obj:
                    continue

                frame_count = frame_obj.frame_number
                self.stats['total_frames'] += 1

                self.last_frame = {
                    'image_b64': frame_obj.image_b64,
                    'timestamp': frame_obj.timestamp,
                    'frame_number': frame_count
                }

                self.emit_event('monitoring_update', {
                    'frame_number': frame_count,
                    'stats': self._serialize_stats()
                })

                similarity = None
                motion_detected = True
                reason = "initial_frame"
                reference_frame = None

                if prev_frame_for_ssim is not None:
                    if structural_similarity is not None:
                        try:
                            target_size = (320, 240)
                            current_small = cv2.resize(frame_obj.raw_frame, target_size)
                            prev_small = cv2.resize(prev_frame_for_ssim, target_size)

                            current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
                            prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

                            similarity = structural_similarity(prev_gray, current_gray, data_range=255)
                            motion_detected = similarity < self.ssim_threshold
                            reference_frame = current_gray
                            if motion_detected:
                                reason = f"scene_change (SSIM={similarity:.3f})"
                            else:
                                reason = f"scene_similar (SSIM={similarity:.3f})"
                        except Exception as exc:
                            print(f"SSIM check failed: {exc}")
                            motion_detected = True
                            reason = "ssim_fallback"
                    else:
                        reason = "ssim_unavailable"

                if reference_frame is None:
                    reference_frame = self._make_reference_frame(frame_obj.raw_frame)

                min_interval = self.motion_burst_interval if motion_detected else self.capture_interval
                min_interval = max(min_interval, 0.0)
                last_time = self.last_motion_time if motion_detected else self.last_processed_time
                allow_wait = motion_detected

                ready_to_submit = (min_interval == 0.0) or ((current_time - last_time) >= min_interval)

                if ready_to_submit:
                    frame_metadata = {
                        'frame_number': frame_count,
                        'timestamp': current_time,
                        'reason': reason,
                        'motion': motion_detected
                    }
                    if similarity is not None:
                        frame_metadata['similarity_score'] = similarity

                    submitted, pending, deduped = self._submit_frame_for_analysis(
                        frame_obj.image_data,
                        frame_metadata,
                        current_time,
                        allow_wait=allow_wait,
                        reference_frame=reference_frame
                    )

                    if submitted:
                        if motion_detected:
                            self.last_motion_time = current_time
                        else:
                            self.last_processed_time = current_time

                        self.stats['processed_frames'] += 1
                        frame_metadata['processed_count'] = self.stats['processed_frames']
                        print(f"🔍 Sending frame {frame_count} to agent for analysis: {reason}")
                    elif not deduped:
                        self._record_backpressure(current_time, frame_count, pending)
                elif motion_detected and current_time - self._last_backpressure_log >= 1.0:
                    remaining = max(min_interval - (current_time - last_time), 0.0)
                    print(f"⏱ Waiting {remaining:.2f}s before next burst submission")
                    self._last_backpressure_log = current_time

                prev_frame_for_ssim = frame_obj.raw_frame.copy()

            except Exception as exc:
                print(f"Monitoring loop error: {exc}")
                self.last_error = str(exc)
                time.sleep(2)

        self._drain_analysis_results(flush=True)

    # Analysis pipeline ----------------------------------------------------------------
    def _submit_frame_for_analysis(self, image_data, frame_metadata, enqueue_time, allow_wait=False, reference_frame=None):
        if not self._analysis_executor:
            return False, 0, False

        is_duplicate = False
        duplicate_similarity = 0.0
        if reference_frame is not None:
            is_duplicate, duplicate_similarity = self._is_duplicate_pending(reference_frame)

        if is_duplicate:
            self.stats['deduped_frames'] += 1
            if enqueue_time - self._last_dedupe_log >= 1.0:
                print(f"🗑️ Dropping similar frame (SSIM={duplicate_similarity:.3f}) already pending")
                self._last_dedupe_log = enqueue_time
            return False, self._count_pending_futures(), True

        pending = self._count_pending_futures()
        if self.max_pending_analyses > 0:
            while pending >= self.max_pending_analyses:
                if not allow_wait:
                    return False, pending, False
                self._drain_analysis_results(wait=True)
                pending = self._count_pending_futures()

        frame_metadata['queued_at'] = enqueue_time
        frame_metadata['submitted_at'] = enqueue_time
        future = self._analysis_executor.submit(self._analyze_frame_task, image_data, frame_metadata)
        future.frame_metadata = frame_metadata  # type: ignore[attr-defined]
        if reference_frame is not None:
            future.reference_frame = reference_frame  # type: ignore[attr-defined]
            self._pending_reference_frames.append((future, reference_frame))
        else:
            self._pending_reference_frames.append((future, None))
        self._analysis_futures.append(future)
        return True, pending, False

    def _analyze_frame_task(self, image_data, frame_metadata):
        if not self.agent:
            raise RuntimeError("MonitorDetectionAgent not initialized")
        return self.agent.analyze_frame(image_data, frame_metadata)

    def _drain_analysis_results(self, flush=False, wait=False, wait_timeout=0.25):
        if not self._analysis_futures:
            if wait:
                time.sleep(wait_timeout)
            return

        processed_any = False
        still_pending = deque()

        while self._analysis_futures:
            future = self._analysis_futures.popleft()

            if not future.done():
                if flush:
                    future.cancel()
                else:
                    still_pending.append(future)
                continue

            processed_any = True
            frame_metadata = getattr(future, 'frame_metadata', {})

            if future.cancelled():
                self._prune_pending_reference(future)
                continue

            try:
                event = future.result()
            except Exception as exc:
                frame_number = frame_metadata.get('frame_number', 'unknown') if isinstance(frame_metadata, dict) else 'unknown'
                print(f"Analysis task failed for frame {frame_number}: {exc}")
                self._prune_pending_reference(future)
                continue

            self._prune_pending_reference(future)
            self._handle_analysis_result(event, frame_metadata if isinstance(frame_metadata, dict) else {})

        self._analysis_futures.extend(still_pending)
        self._cleanup_pending_references()

        if wait and not processed_any and still_pending:
            futures_list = list(still_pending)
            try:
                wait(futures_list, timeout=wait_timeout, return_when=FIRST_COMPLETED)
            except Exception:
                time.sleep(wait_timeout)

    # Post-processing ------------------------------------------------------------------
    def _handle_analysis_result(self, event, frame_metadata):
        frame_number = frame_metadata.get('frame_number', 'unknown')

        latency_ms = None
        queued_at = frame_metadata.get('queued_at')
        if isinstance(queued_at, (int, float)):
            latency_ms = (time.time() - queued_at) * 1000.0
        elif isinstance(queued_at, str):
            try:
                latency_ms = (time.time() - float(queued_at)) * 1000.0
            except ValueError:
                latency_ms = None

        if latency_ms is not None:
            frame_metadata['analysis_latency_ms'] = latency_ms
            self._update_latency_stats(latency_ms)

        if not event:
            print(f"⚠️  Analysis failed for frame {frame_number}")
            return

        if latency_ms is not None and isinstance(event.decision_trace, dict):
            event.decision_trace['analysis_latency_ms'] = latency_ms

        self._record_detection(event, frame_metadata)

        decision_trace = event.decision_trace if isinstance(event.decision_trace, dict) else {}
        skip_meta = decision_trace.get('agent_similarity_skip') if isinstance(decision_trace, dict) else None
        skip_counts = True
        if isinstance(skip_meta, dict):
            skip_counts = skip_meta.get('count_as_detection', True)

        if event.detected:
            if skip_counts:
                self.stats['detections'] += 1

            label_status = event.shipping_label_present
            if isinstance(skip_meta, dict):
                similarity_display = skip_meta.get('similarity')
                try:
                    similarity_text = f"{float(similarity_display):.3f}"
                except (TypeError, ValueError):
                    similarity_text = str(similarity_display) if similarity_display is not None else "unknown"
                prior_label = skip_meta.get('previous_primary_label', event.primary_label)
                print(
                    f"🟡 Similar frame reused previous decision ({prior_label}) - SSIM {similarity_text}; no duplicate alert"
                )
            else:
                if label_status is True:
                    print(f"ℹ️  Packaging box detected with shipping label (confidence {event.confidence:.2f})")
                elif label_status is False:
                    print(f"🔔 Unlabeled packaging box detected! Confidence: {event.confidence:.2f}")
                else:
                    print(f"🔔 Packaging box detected (label status unknown). Confidence: {event.confidence:.2f}")

            alert_sent = False
            if self.agent and event.should_alert and skip_counts:
                if self.agent.process_detection(event):
                    self.stats['alerts_sent'] += 1
                    alert_sent = True

            preview = event.full_response
            if isinstance(preview, str) and len(preview) > 200:
                preview = preview[:200] + '...'

            label_analysis_summary = None
            packaging_analysis_summary = None
            tools_used = list(event.tools_used) if isinstance(event.tools_used, list) else []
            tool_trace = event.tool_trace if isinstance(event.tool_trace, list) else []
            if isinstance(event.decision_trace, dict):
                label_analysis = event.decision_trace.get('shipping_label_analysis')
                if isinstance(label_analysis, dict):
                    label_analysis_summary = label_analysis.get('summary')
                packaging_analysis = event.decision_trace.get('packaging_analysis')
                if isinstance(packaging_analysis, dict):
                    packaging_analysis_summary = packaging_analysis.get('summary')
                if not tools_used:
                    trace_tools = event.decision_trace.get('tools_used')
                    if isinstance(trace_tools, list):
                        tools_used = [str(tool) for tool in trace_tools]
                    elif trace_tools:
                        tools_used = [str(trace_tools)]
                if not tool_trace:
                    trace_entries = event.decision_trace.get('tool_trace')
                    if isinstance(trace_entries, list):
                        tool_trace = trace_entries

            if label_analysis_summary:
                print(f"   Local label analysis: {label_analysis_summary}")
            if packaging_analysis_summary:
                print(f"   Packaging analysis: {packaging_analysis_summary}")
            if tools_used:
                print(f"   Tools used: {', '.join(tools_used)}")

            self.emit_event('detection_event', {
                'event': {
                    'timestamp': event.timestamp,
                    'confidence': event.confidence,
                    'response': preview,
                    'shipping_label_present': label_status,
                    'should_alert': event.should_alert,
                    'alert_sent': alert_sent,
                    'primary_label': event.primary_label,
                    'label_analysis_summary': label_analysis_summary,
                    'packaging_analysis_summary': packaging_analysis_summary,
                    'tools_used': tools_used,
                    'tool_trace': tool_trace,
                    'skip_reused': bool(skip_meta)
                },
                'stats': self._serialize_stats()
            })
        else:
            classification = event.decision_trace.get('classification') if event.decision_trace else None
            classification_display = classification or 'NO_DETECTION'
            print(f"   No detection in frame {frame_number} (Decision: {classification_display})")

    def _serialize_stats(self) -> Dict[str, object]:
        """Return stats dict with JSON-serializable values."""
        stats_copy = self.stats.copy()
        uptime_start = stats_copy.get('uptime_start')
        if isinstance(uptime_start, datetime):
            stats_copy['uptime_start'] = uptime_start.isoformat()
        avg_latency = stats_copy.get('analysis_avg_ms')
        if isinstance(avg_latency, (int, float)):
            stats_copy['analysis_avg_ms'] = round(avg_latency, 2)
        stats_copy['last_update'] = datetime.now().isoformat()
        return stats_copy
