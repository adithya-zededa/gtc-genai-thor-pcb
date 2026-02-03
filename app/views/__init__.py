"""View routes for rendering HTML templates.

These routes handle page rendering for the web interface.
"""

from flask import Blueprint, render_template, redirect, url_for, abort

from app.database import get_db_connection, DetectionLogRepository

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
def dashboard():
    """Main dashboard page."""
    return render_template("dashboard.html")


@views_bp.route("/monitoring")
def monitoring():
    """Live monitoring page."""
    return render_template("monitoring.html")


@views_bp.route("/configuration")
def configuration():
    """Legacy configuration route - redirects to settings."""
    return redirect(url_for("views.settings"))


@views_bp.route("/users")
def users():
    """User management page."""
    with get_db_connection() as conn:
        users_list = conn.execute(
            "SELECT * FROM users ORDER BY created_at DESC"
        ).fetchall()
    return render_template("users.html", users=users_list)


@views_bp.route("/logs")
def logs():
    """Detection logs page."""
    with get_db_connection() as conn:
        detections = conn.execute(
            """
            SELECT * FROM detection_logs
            ORDER BY timestamp DESC
            LIMIT 100
            """
        ).fetchall()
    return render_template("logs.html", detections=detections)


@views_bp.route("/logs/<int:log_id>")
def log_detail(log_id: int):
    """Single detection event detail view."""
    import json
    
    log = DetectionLogRepository.get_by_id(log_id)
    if not log:
        abort(404)
    
    log_data = log.to_dict()
    
    image_url = None
    if log.image_path:
        image_url = url_for("api_v1.serve_image", image_path=log.image_path)
    
    confidence_percent = None
    if log.confidence is not None:
        try:
            confidence_percent = float(log.confidence) * 100
        except (TypeError, ValueError):
            pass
    
    return render_template(
        "log_detail.html",
        log=log_data,
        image_url=image_url,
        confidence_percent=confidence_percent,
    )


@views_bp.route("/settings")
def settings():
    """System settings page."""
    return render_template("settings.html")
