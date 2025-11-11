#!/usr/bin/env python3
"""Run the camera monitoring agent without the web application."""

import logging
import signal
import sys
import time
from typing import Any, Dict

from camera_monitoring import CameraMonitoringService
from camera_feed_publisher import get_camera_publisher


class HeadlessCameraAgent(CameraMonitoringService):
    """Camera monitoring agent that logs events to the console."""

    def __init__(self) -> None:
        super().__init__(subscriber_id="monitoring_agent_cli")

    def emit_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if event_name == 'detection_event':
            event = payload.get('event', {}) if isinstance(payload, dict) else {}
            confidence = event.get('confidence')
            timestamp = event.get('timestamp')
            logging.info(
                "Detection event: confidence=%s timestamp=%s",
                f"{confidence:.2f}" if isinstance(confidence, (int, float)) else confidence,
                timestamp,
            )
        elif event_name == 'monitoring_update':
            stats = payload.get('stats') if isinstance(payload, dict) else None
            if isinstance(stats, dict):
                logging.debug(
                    "Monitoring update: frame=%s total=%s processed=%s",
                    payload.get('frame_number'),
                    stats.get('total_frames'),
                    stats.get('processed_frames'),
                )

    def _record_detection(self, event, frame_metadata) -> None:
        if not getattr(event, 'detected', False):
            return
        logging.info(
            "Detection recorded: label=%s confidence=%.2f frame=%s",
            getattr(event, 'primary_label', 'unknown'),
            getattr(event, 'confidence', 0.0),
            frame_metadata.get('frame_number'),
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info("Starting headless camera monitoring agent")

    publisher = get_camera_publisher()
    if not getattr(publisher, 'is_running', False):
        if not publisher.start():
            logging.error("Failed to start camera publisher")
            return 1
        logging.info("Camera publisher started")
    else:
        logging.info("Camera publisher already running")

    agent = HeadlessCameraAgent()
    if not agent.initialize():
        logging.error("Agent initialization failed: %s", agent.last_error)
        return 1

    if not agent.start_monitoring():
        logging.error("Failed to start monitoring: %s", agent.last_error)
        return 1

    logging.info("Monitoring active. Press Ctrl+C to stop.")

    stop_requested = False

    def _handle_signal(signum, frame):  # type: ignore[unused-argument]
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop_requested:
            time.sleep(1)
    finally:
        logging.info("Stopping monitoring")
        agent.stop_monitoring()

    logging.info("Headless camera monitoring agent stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
