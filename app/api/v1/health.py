"""Health, liveness, and readiness endpoints.

Three endpoints with three different jobs — they are not
interchangeable, and the Helm chart wires each to the matching probe:

``/api/health/live``
    Liveness. Answers one question: is this process still able to serve a
    request? It touches nothing else, because a liveness probe that
    depends on the database or on a peer pod turns *their* outage into a
    restart loop here.
``/api/ready``
    Readiness. Local dependencies the app cannot serve without — today
    that is the SQLite database. Returns 503 so Kubernetes pulls the pod
    out of the Service rather than sending it traffic it cannot handle.
``/api/health``
    Diagnostics for humans and dashboards: every component, including
    per-role inference reachability. Not wired to a probe, so it is free
    to make outbound calls.
"""

# pylint: disable=broad-exception-caught

from datetime import datetime

from flask import jsonify

from app.database import get_db_connection
from services.core.camera import check_camera_availability
from services.core.inference import check_inference_roles

from . import api_bp


def _database_ok() -> tuple[bool, str]:
    """Return (ok, detail) for a trivial database round-trip."""
    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1")
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


@api_bp.route("/health/live")
def liveness_check():
    """Liveness probe — process is up and the WSGI stack is responding.

    Deliberately dependency-free. Anything added here becomes a reason
    for Kubernetes to kill a process that is working fine.
    """
    return jsonify({"alive": True}), 200


@api_bp.route("/ready")
def readiness_check():
    """Readiness probe — local dependencies are usable.

    Inference reachability is intentionally *not* part of this: the UI,
    the logs, and the history APIs all work while vLLM is still loading
    its weights, and gating the whole Service on a ten-minute model load
    would make every rollout an outage.
    """
    db_ok, db_detail = _database_ok()
    details = {"database": db_detail}

    if db_ok:
        return jsonify({"ready": True, "details": details})
    return jsonify({"ready": False, "details": details}), 503


@api_bp.route("/health")
def health_check():
    """Full component health, including per-role inference status."""
    db_ok, _ = _database_ok()
    inference_roles = check_inference_roles()

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "inference_backend": "vllm",
        "components": {
            "database": db_ok,
            "camera": check_camera_availability(),
            "inference": all(inference_roles.values()),
        },
        # Broken out so a green "inference" cannot hide a dead agent pod.
        "inference_roles": inference_roles,
    }

    if not db_ok:
        health_status["status"] = "unhealthy"
    elif not all(health_status["components"].values()):
        health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] != "unhealthy" else 503
    return jsonify(health_status), status_code
