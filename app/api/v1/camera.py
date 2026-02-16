"""Camera feed API endpoints."""

# pylint: disable=broad-exception-caught,logging-fstring-interpolation,cyclic-import

import threading
import time

from flask import Response, jsonify

from core.logging import get_logger
from services.core.camera import get_camera_publisher

from . import api_bp

logger = get_logger(__name__)


@api_bp.route("/video_feed")
def video_feed():
    """Stream live video frames from camera publisher."""

    def generate():
        subscriber_id = f"video_feed_{threading.get_ident()}_{time.time_ns()}"
        subscribed = False
        consecutive_failures = 0
        max_failures = 10

        try:
            publisher = get_camera_publisher()

            # Auto-start the publisher if not running
            if not publisher.is_running:
                logger.info("Auto-starting camera publisher for video_feed")
                if not publisher.start():
                    logger.error("Failed to start camera for video_feed")
                    return

            if not publisher.subscribe(subscriber_id):
                logger.warning(f"Failed to subscribe video feed: {subscriber_id}")
                return
            subscribed = True

            while consecutive_failures < max_failures:
                try:
                    frame_obj = publisher.get_frame(subscriber_id, timeout=1.0)
                    if not frame_obj:
                        consecutive_failures += 1
                        continue

                    consecutive_failures = 0

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + frame_obj.image_data
                        + b"\r\n"
                    )
                except Exception as frame_err:
                    logger.debug(f"Frame error in video feed: {frame_err}")
                    consecutive_failures += 1
                    time.sleep(0.1)

        except GeneratorExit:
            pass
        except Exception as e:
            logger.error(f"Video feed error: {e}")
        finally:
            if subscribed:
                try:
                    publisher = get_camera_publisher()
                    publisher.unsubscribe(subscriber_id)
                except Exception as cleanup_err:
                    logger.debug(f"Cleanup error: {cleanup_err}")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@api_bp.route("/capture_frame")
def capture_frame():
    """Get the latest captured frame."""
    try:
        publisher = get_camera_publisher()

        # Auto-start the publisher if not running
        if not publisher.is_running:
            logger.info("Auto-starting camera publisher for capture_frame")
            if not publisher.start():
                return jsonify({"success": False, "error": "Failed to start camera"})

        latest_frame = publisher.get_latest_frame()

        if latest_frame:
            return jsonify(
                {
                    "success": True,
                    "image_b64": latest_frame.image_b64,
                    "timestamp": latest_frame.timestamp,
                    "metadata": {
                        "frame_number": latest_frame.frame_number,
                        "source": "camera_publisher",
                    },
                }
            )

        return jsonify({"success": False, "error": "No frames available yet"})
    except Exception as e:
        logger.error(f"capture_frame error: {e}")
        return jsonify({"success": False, "error": str(e)})
