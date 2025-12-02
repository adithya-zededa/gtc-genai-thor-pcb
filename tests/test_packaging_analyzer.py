import cv2
import numpy as np

from agent_runtime.detection import PackagingBoxAnalyzer


def test_packaging_analyzer_blank_frame_returns_low_confidence():
    analyzer = PackagingBoxAnalyzer()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blank, cv2.COLOR_BGR2HSV)

    result = analyzer.analyze(blank, gray, hsv)

    assert result is not None
    assert result.detected is False
    assert result.confidence <= 0.5
