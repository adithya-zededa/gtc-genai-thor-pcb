"""Retention sweep: rows *and* their image files must actually go away.

The bug these cover is that ``cleanup_old`` and the ``log_retention``
setting both existed with no caller, so nothing bounded the data volume.
"""

from __future__ import annotations

import pytest

from core import config as core_config


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the config + connection pool at a throwaway database."""
    from app.database import connection as conn_mod

    monkeypatch.setenv("CAMERA_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CAMERA_AGENT_DB", str(tmp_path / "test.db"))
    core_config.reset_config()
    conn_mod._POOL_STATE["connection_pool"] = None  # pylint: disable=protected-access

    from app.database import init_db

    init_db()
    yield tmp_path

    pool = conn_mod._POOL_STATE["connection_pool"]  # pylint: disable=protected-access
    if pool is not None:
        pool.close_all()
    conn_mod._POOL_STATE["connection_pool"] = None  # pylint: disable=protected-access
    core_config.reset_config()


def _make_image(tmp_path, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(b"not-really-a-jpeg")
    return str(path)


def test_frame_cleanup_removes_rows_and_files(temp_db):
    from app.database import get_db_connection
    from app.database.repositories import PCBFrameStoreRepository

    old_image = _make_image(temp_db, "old.jpg")
    fresh_image = _make_image(temp_db, "fresh.jpg")

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO pcb_frame_store (timestamp, image_path, motion_score,
                                         board_signature, quality_score)
            VALUES (datetime('now', '-2 hours'), ?, 0.0, 'sig-old', 0.5)
            """,
            (old_image,),
        )
        conn.execute(
            """
            INSERT INTO pcb_frame_store (timestamp, image_path, motion_score,
                                         board_signature, quality_score)
            VALUES (datetime('now'), ?, 0.0, 'sig-new', 0.9)
            """,
            (fresh_image,),
        )
        conn.commit()

    removed = PCBFrameStoreRepository.cleanup_old(max_age_seconds=3600, max_rows=100)

    assert removed == 1
    import os

    assert not os.path.exists(old_image), "aged-out frame's file should be unlinked"
    assert os.path.exists(fresh_image), "in-window frame must be untouched"

    with get_db_connection() as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM pcb_frame_store").fetchone()[0]
    assert remaining == 1


def test_frame_cleanup_trims_to_max_rows(temp_db):
    from app.database import get_db_connection
    from app.database.repositories import PCBFrameStoreRepository

    with get_db_connection() as conn:
        for i in range(5):
            conn.execute(
                """
                INSERT INTO pcb_frame_store (timestamp, image_path, motion_score,
                                             board_signature, quality_score)
                VALUES (datetime('now', ? || ' seconds'), '', 0.0, ?, 0.5)
                """,
                (f"-{i}", f"sig-{i}"),
            )
        conn.commit()

    removed = PCBFrameStoreRepository.cleanup_old(max_age_seconds=86400, max_rows=2)

    assert removed == 3
    with get_db_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM pcb_frame_store").fetchone()[0] == 2


def test_detection_log_retention_deletes_rows_and_images(temp_db):
    from app.database import get_db_connection
    from app.database.repositories import DetectionLogRepository

    stale_image = _make_image(temp_db, "stale.jpg")
    recent_image = _make_image(temp_db, "recent.jpg")

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO detection_logs (timestamp, confidence, response, image_path)
            VALUES (datetime('now', '-45 days'), 0.9, 'old', ?)
            """,
            (stale_image,),
        )
        conn.execute(
            """
            INSERT INTO detection_logs (timestamp, confidence, response, image_path)
            VALUES (datetime('now', '-2 days'), 0.9, 'new', ?)
            """,
            (recent_image,),
        )
        conn.commit()

    removed = DetectionLogRepository.delete_older_than(30)

    assert removed == 1
    import os

    assert not os.path.exists(stale_image)
    assert os.path.exists(recent_image)


def test_detection_log_retention_of_zero_days_deletes_nothing(temp_db):
    """A misconfigured retention must skip the sweep, never wipe the table."""
    from app.database import get_db_connection
    from app.database.repositories import DetectionLogRepository

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO detection_logs (timestamp, confidence, response, image_path)
            VALUES (datetime('now', '-999 days'), 0.9, 'ancient', '')
            """
        )
        conn.commit()

    assert DetectionLogRepository.delete_older_than(0) == 0
    assert DetectionLogRepository.delete_older_than(-5) == 0

    with get_db_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM detection_logs").fetchone()[0] == 1


def test_sweep_reports_both_datasets(temp_db):
    from app.database.maintenance import run_retention_sweep

    results = run_retention_sweep()

    assert set(results) == {"frames", "detection_logs"}


def test_sweep_survives_a_failing_dataset(temp_db, monkeypatch):
    """One broken repository must not strand the other dataset."""
    from app.database import maintenance

    def _boom(**_kwargs):
        raise RuntimeError("frame store exploded")

    monkeypatch.setattr(
        maintenance.PCBFrameStoreRepository, "cleanup_old", staticmethod(_boom)
    )

    results = maintenance.run_retention_sweep()

    assert results["frames"] == 0
    assert results["detection_logs"] == 0  # ran, found nothing — did not raise


def test_worker_start_stop_is_idempotent(temp_db):
    from app.database.maintenance import RetentionWorker

    worker = RetentionWorker(interval_seconds=3600)
    worker.start()
    worker.start()  # second call is a no-op, not a second thread
    assert worker.is_running

    worker.stop()
    assert not worker.is_running
