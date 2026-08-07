"""Tests for the focus-based settle window in MonitoringLoop.

The settle window replaced a fixed 4s blocking wait. These tests pin the
behaviour that matters: it exits early once focus plateaus, it still honours
the upper bound when focus never settles, and it only ever nominates a frame
that actually shows a stopped board.
"""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from agents.core.monitoring_loop import MonitoringLoop, Observation


INSTRUCTION = "Monitor the conveyor for PCB defects"


@pytest.fixture(autouse=True)
def _offline_classifier(monkeypatch):
    """Constructing a MonitoringLoop scope-validates the instruction through
    the LLM classifier. These tests exercise CV code, so stub it out rather
    than waiting on a vLLM backend that isn't running."""
    stub = SimpleNamespace(
        classify=lambda text: SimpleNamespace(domain="pcb", tool=None)
    )
    monkeypatch.setattr(
        "agents.core.monitoring_loop.get_classifier", lambda: stub
    )


def _observation(*, in_zone=True, moving=False, sharpness=100.0):
    return Observation(
        frame_number=1,
        timestamp="2026-08-06T00:00:00",
        motion_score=0.1,
        motion_moving=moving,
        edge_density=20.0,
        board_in_zone=in_zone,
        board_signature="board-test",
        frame_quality={"sharpness": sharpness, "brightness": 120.0, "contrast": 40.0},
    )


def _sharp_frame(seed: int = 0) -> np.ndarray:
    """A frame with real high-frequency content (non-trivial focus score)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)


def _blurred(frame: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def _loop(**config):
    return MonitoringLoop(
        instruction=INSTRUCTION,
        publisher_getter=lambda: None,
        on_board_ready=lambda frame, ctx: None,
        config={"frame_interval_seconds": 0.0, **config},
    )


class _FrameFeeder:
    """Drives _settle_for_focus with a scripted sequence of frames."""

    def __init__(self, loop, frames, observations=None):
        self._frames = list(frames)
        self._observations = observations
        self._index = 0
        loop._acquire_frame = self._acquire  # noqa: SLF001
        loop._observe = self._observe  # noqa: SLF001
        loop._store_frame = lambda *a, **kw: None  # noqa: SLF001

    def _acquire(self):
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        return SimpleNamespace(raw_frame=frame, frame_number=self._index, timestamp="t")

    def _observe(self, frame_obj):
        obs = (
            self._observations[self._index]
            if self._observations
            else _observation()
        )
        self._index += 1
        return obs

    @property
    def consumed(self):
        return self._index


def test_focus_score_ranks_sharp_above_blurred():
    loop = _loop()
    sharp = _sharp_frame()
    assert loop._focus_score(sharp) > loop._focus_score(_blurred(sharp, 9))


def test_focus_score_handles_missing_frame():
    assert _loop()._focus_score(None) is None


def test_settle_exits_early_once_focus_plateaus():
    """A camera that is already focused should not burn the full window."""
    loop = _loop(
        settle_max_seconds=5.0,
        settle_min_seconds=0.0,
        settle_focus_plateau_frames=3,
    )
    base = _sharp_frame()
    # Same sharpness throughout => no improvement => plateau immediately.
    feeder = _FrameFeeder(loop, [base] * 50)

    result = loop._settle_for_focus()

    assert result["exit_reason"] == "focus_plateau"
    assert result["seconds"] < 5.0
    # First frame improves on the 0.0 baseline, then 3 plateau frames.
    assert feeder.consumed == 4
    assert result["frame"] is not None


def test_settle_keeps_the_sharpest_frame_not_the_last():
    loop = _loop(
        settle_max_seconds=5.0,
        settle_min_seconds=0.0,
        settle_focus_plateau_frames=2,
    )
    base = _sharp_frame()
    sharpest = base
    # Focus improves, peaks, then degrades — the peak must win.
    frames = [_blurred(base, 9), _blurred(base, 5), sharpest,
              _blurred(base, 5), _blurred(base, 9), _blurred(base, 9)]
    _FrameFeeder(loop, frames)

    result = loop._settle_for_focus()

    assert result["frame"] is sharpest
    assert result["focus_score"] == pytest.approx(loop._focus_score(sharpest), rel=1e-6)


def test_settle_respects_minimum_duration():
    """settle_min_seconds prevents exiting before auto-focus reacts."""
    loop = _loop(
        settle_max_seconds=5.0,
        settle_min_seconds=0.25,
        settle_focus_plateau_frames=1,
    )
    _FrameFeeder(loop, [_sharp_frame()] * 200)

    result = loop._settle_for_focus()

    assert result["seconds"] >= 0.25


def test_settle_honours_the_upper_bound_when_focus_never_plateaus():
    loop = _loop(
        settle_max_seconds=0.3,
        settle_min_seconds=0.0,
        settle_focus_plateau_frames=1000,
    )
    _FrameFeeder(loop, [_sharp_frame(i) for i in range(500)])

    result = loop._settle_for_focus()

    assert result["exit_reason"] == "deadline"
    assert result["seconds"] >= 0.3


def test_settle_ignores_frames_without_a_stopped_board():
    """A frame with no board (or a moving one) is never an inspection candidate."""
    loop = _loop(settle_max_seconds=0.3, settle_min_seconds=0.0)
    frames = [_sharp_frame(i) for i in range(20)]
    observations = [_observation(in_zone=False) for _ in frames]
    _FrameFeeder(loop, frames, observations)

    result = loop._settle_for_focus()

    assert result["frame"] is None
    assert result["focus_score"] == 0.0


def test_settle_aborts_when_the_loop_is_stopped():
    loop = _loop(settle_max_seconds=5.0)
    _FrameFeeder(loop, [_sharp_frame()] * 10)
    loop._stop_event.set()

    result = loop._settle_for_focus()

    assert result["exit_reason"] == "stopped"


def test_settle_telemetry_reaches_the_board_ready_context():
    captured = {}
    loop = MonitoringLoop(
        instruction=INSTRUCTION,
        publisher_getter=lambda: None,
        on_board_ready=lambda frame, ctx: captured.update(ctx),
        config={"frame_interval_seconds": 0.0},
    )

    loop._fire_board_ready(
        _sharp_frame(),
        _observation(),
        {"frame": _sharp_frame(), "seconds": 0.9, "exit_reason": "focus_plateau",
         "frames_observed": 9, "focus_score": 1234.5, "max_seconds": 4.0},
    )

    assert captured["settle"]["exit_reason"] == "focus_plateau"
    assert captured["settle"]["seconds"] == 0.9
    # The frame itself must not be serialised into the LLM/DB context.
    assert "frame" not in captured["settle"]
