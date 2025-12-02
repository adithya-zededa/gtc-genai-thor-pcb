import base64
import time
from typing import List

import numpy as np
import pytest

import cv2

from agent_runtime.publisher import CameraFeedPublisher, CameraFrame


class FakeCapture:
    """Minimal cv2.VideoCapture replacement for deterministic testing."""

    def __init__(self, frame: np.ndarray) -> None:
        self._opened = True
        self._frame = frame

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        # Return a fresh copy so mutations in tested code do not bleed across
        # calls
        return True, self._frame.copy()

    def release(self) -> None:
        self._opened = False

    def set(
        self, *_args, **_kwargs
    ) -> bool:  # pragma: no cover - setter just needs to exist
        return True

    def get(self, prop_id):  # pragma: no cover - queried during initialization
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._frame.shape[1])
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._frame.shape[0])
        return 0.0


@pytest.fixture(autouse=True)
def patch_video_capture(monkeypatch):
    """Ensure camera interactions use the deterministic fake capture."""
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def factory(_index):
        return FakeCapture(test_frame)

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    return test_frame


@pytest.fixture
def publisher() -> CameraFeedPublisher:
    instance = CameraFeedPublisher(
        camera_index=0, width=640, height=480, fps=10
    )
    try:
        yield instance
    finally:
        instance.stop()


def test_start_stop_publisher_produces_frames(publisher):
    assert publisher.start() is True
    time.sleep(0.2)  # Allow capture loop to run at least once
    latest = publisher.get_latest_frame()
    assert latest is not None
    assert latest.width == 640
    assert latest.height == 480
    publisher.stop()
    assert publisher.is_running is False


def test_publish_frame_reaches_subscribers(publisher):
    frame_numbers: List[int] = []
    assert publisher.subscribe(
        "client",
        callback=lambda frame: frame_numbers.append(frame.frame_number),
    )

    raw_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    jpeg_bytes = cv2.imencode(".jpg", raw_frame)[1].tobytes()
    frame = CameraFrame(
        frame_number=1,
        timestamp="2025-01-01T00:00:00",
        image_data=jpeg_bytes,
        image_b64=base64.b64encode(jpeg_bytes).decode("utf-8"),
        raw_frame=raw_frame,
        width=10,
        height=10,
    )

    publisher._publish_frame(frame)  # pylint: disable=protected-access
    queue = publisher._subscribers[
        "client"
    ]  # pylint: disable=protected-access
    retrieved = queue.get(timeout=0.1)

    assert retrieved is frame
    assert frame_numbers == [1]


def test_stats_reflect_subscribers_and_frames(publisher):
    publisher.subscribe("client")
    publisher.start()
    time.sleep(0.2)
    stats = publisher.get_stats()

    assert stats["is_running"] is True
    assert stats["subscribers_count"] == 1
    assert stats["frames_captured"] >= 1
    assert isinstance(stats["camera_available"], bool)
    publisher.stop()
