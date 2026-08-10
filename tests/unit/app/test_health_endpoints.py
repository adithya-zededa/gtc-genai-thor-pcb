"""Probe endpoints must be able to fail.

The Helm chart previously pointed both probes at ``/api/status``, which
returns 200 unconditionally — readiness could never pull a broken pod out
of the Service. These pin the semantics the chart now depends on.
"""

from __future__ import annotations

import pytest

from core import config as core_config


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERA_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CAMERA_AGENT_DB", str(tmp_path / "test.db"))
    core_config.reset_config()

    from app.database import connection as conn_mod, init_db

    conn_mod._POOL_STATE["connection_pool"] = None  # pylint: disable=protected-access
    init_db()

    from app import create_app

    yield create_app().test_client()

    pool = conn_mod._POOL_STATE["connection_pool"]  # pylint: disable=protected-access
    if pool is not None:
        pool.close_all()
    conn_mod._POOL_STATE["connection_pool"] = None  # pylint: disable=protected-access
    core_config.reset_config()


def test_liveness_has_no_dependencies(client, monkeypatch):
    """A dead database and a dead vLLM must not restart a live process."""
    from app.api.v1 import health as health_mod

    def _explode(*_a, **_kw):
        raise RuntimeError("database is gone")

    monkeypatch.setattr(health_mod, "get_db_connection", _explode)
    monkeypatch.setattr(health_mod, "check_inference_roles", _explode)

    response = client.get("/api/health/live")

    assert response.status_code == 200
    assert response.get_json() == {"alive": True}


def test_readiness_is_200_when_the_database_works(client):
    response = client.get("/api/ready")

    assert response.status_code == 200
    assert response.get_json()["ready"] is True


def test_readiness_is_503_when_the_database_is_broken(client, monkeypatch):
    from app.api.v1 import health as health_mod

    def _explode(*_a, **_kw):
        raise RuntimeError("no such table")

    monkeypatch.setattr(health_mod, "get_db_connection", _explode)

    response = client.get("/api/ready")

    assert response.status_code == 503
    assert response.get_json()["ready"] is False


def test_readiness_ignores_inference(client, monkeypatch):
    """A model still loading must not deregister the pod from the Service."""
    from app.api.v1 import health as health_mod

    monkeypatch.setattr(
        health_mod, "check_inference_roles", lambda: {"vision": False, "agent": False}
    )

    assert client.get("/api/ready").status_code == 200


def test_health_reports_each_inference_role(client, monkeypatch):
    """A live vision pod must not mask a dead agent pod."""
    from app.api.v1 import health as health_mod

    monkeypatch.setattr(
        health_mod, "check_inference_roles", lambda: {"vision": True, "agent": False}
    )
    monkeypatch.setattr(health_mod, "check_camera_availability", lambda: True)

    payload = client.get("/api/health").get_json()

    assert payload["inference_roles"] == {"vision": True, "agent": False}
    assert payload["components"]["inference"] is False
    assert payload["status"] == "degraded"


def test_health_is_503_only_when_the_database_is_down(client, monkeypatch):
    from app.api.v1 import health as health_mod

    def _explode(*_a, **_kw):
        raise RuntimeError("database is gone")

    monkeypatch.setattr(health_mod, "get_db_connection", _explode)
    monkeypatch.setattr(
        health_mod, "check_inference_roles", lambda: {"vision": True, "agent": True}
    )

    response = client.get("/api/health")

    assert response.status_code == 503
    assert response.get_json()["status"] == "unhealthy"
