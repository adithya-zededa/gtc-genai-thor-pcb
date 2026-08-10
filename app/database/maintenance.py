"""Periodic data retention.

The appliance is meant to run unattended for weeks on a fixed-size data
volume, but every write path here is append-only: inspections store a
frame plus its JPEG, detections store a log row plus its JPEG. Without a
sweep the PVC fills and the pod starts failing writes.

``PCBFrameStoreRepository.cleanup_old()`` and the ``log_retention``
setting both existed before this module; neither had a caller. This is
that caller — one daemon thread, started from the process entry point,
running a bounded amount of SQL on a slow interval.

Retention has two sources because the two datasets mean different things:

* **Frame store** — a working buffer the inspection pipeline consumes
  from. It is sized in *seconds and rows* (``RetentionConfig``), not days;
  frames older than the current board are of no interest.
* **Detection logs** — the operator-visible history behind ``/logs``. Its
  window is the ``log_retention`` value from the settings UI, in days, so
  changing it in the UI takes effect on the next sweep.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from core.config import get_config
from core.logging import get_logger

from .repositories import DetectionLogRepository, LogSettingsRepository, PCBFrameStoreRepository

logger = get_logger(__name__)


def run_retention_sweep() -> Dict[str, int]:
    """Run one retention pass. Returns per-dataset deletion counts.

    Each dataset is swept independently so one failure cannot stop the
    others — a corrupt image path in the frame store should not also
    strand the detection log at its high-water mark.
    """
    config = get_config()
    results: Dict[str, int] = {}

    try:
        results["frames"] = PCBFrameStoreRepository.cleanup_old(
            max_age_seconds=config.retention.frame_max_age_seconds,
            max_rows=config.retention.frame_max_rows,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Frame-store retention sweep failed: %s", exc)
        results["frames"] = 0

    try:
        retention_days = LogSettingsRepository.get().log_retention
        results["detection_logs"] = DetectionLogRepository.delete_older_than(
            retention_days
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Detection-log retention sweep failed: %s", exc)
        results["detection_logs"] = 0

    if any(results.values()):
        logger.info(
            "Retention sweep: removed %d frames, %d detection logs",
            results["frames"],
            results["detection_logs"],
        )
    return results


class RetentionWorker:
    """Daemon thread that calls :func:`run_retention_sweep` on an interval.

    Sleeps on an ``Event`` rather than ``time.sleep`` so shutdown is
    immediate instead of waiting out a full interval.
    """

    def __init__(self, interval_seconds: Optional[float] = None) -> None:
        self.interval_seconds = (
            interval_seconds
            if interval_seconds is not None
            else get_config().retention.interval_seconds
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="retention-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Retention worker started (every %.0fs)", self.interval_seconds
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def _loop(self) -> None:
        # Sweep once at startup: a crash-restart loop would otherwise never
        # reach the first interval, and that is exactly when the volume is
        # most likely to be full.
        while True:
            run_retention_sweep()
            if self._stop.wait(timeout=self.interval_seconds):
                break
        logger.info("Retention worker stopped")


_worker: Optional[RetentionWorker] = None
_worker_lock = threading.Lock()


def start_retention_worker() -> Optional[RetentionWorker]:
    """Start the process-wide retention worker, if enabled.

    Called from the process entry points (``run.py`` / ``wsgi.py``) rather
    than from ``create_app()``, so importing the app in a test does not
    spawn a background thread that mutates the database.
    """
    global _worker  # pylint: disable=global-statement

    if not get_config().retention.enabled:
        logger.info("Retention worker disabled by configuration")
        return None

    with _worker_lock:
        if _worker is None:
            _worker = RetentionWorker()
        _worker.start()
        return _worker


def stop_retention_worker() -> None:
    """Stop the process-wide retention worker (tests, graceful shutdown)."""
    global _worker  # pylint: disable=global-statement

    with _worker_lock:
        if _worker is not None:
            _worker.stop()
            _worker = None
