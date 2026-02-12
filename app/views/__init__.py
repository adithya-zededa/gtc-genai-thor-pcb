"""View routes for rendering HTML templates.

These routes handle page rendering for the web interface.
"""

# pylint: disable=import-outside-toplevel

from flask import Blueprint, abort, redirect, render_template, url_for

from app.database import DetectionLogRepository, UserRepository

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
def dashboard():
    """Main dashboard page."""
    return render_template("dashboard.html")


@views_bp.route("/monitoring")
def monitoring():
    """Live monitoring page - redirects to chat interface."""
    return redirect(url_for("views.chat"))


@views_bp.route("/chat")
def chat():
    """Chat-based monitoring interface with MCP protocol."""
    return render_template("chat.html")


@views_bp.route("/chat/v2")
def chat_v2():
    """Chat-based monitoring interface - redirects to main chat."""
    return redirect(url_for("views.chat"))


@views_bp.route("/configuration")
def configuration():
    """Legacy configuration route - redirects to settings."""
    return redirect(url_for("views.settings"))


@views_bp.route("/users")
def users():
    """User management page."""
    users_list = UserRepository.get_all(active_only=False)
    return render_template("users.html", users=users_list)


@views_bp.route("/logs")
def logs():
    """Detection logs page."""
    detections, _total = DetectionLogRepository.get_paginated(page=1, per_page=100)
    return render_template("logs.html", detections=detections)


@views_bp.route("/logs/<int:log_id>")
def log_detail(log_id: int):
    """Single detection event detail view."""
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
    import os

    from core.config import get_config

    cfg = get_config()
    vllm_url = os.getenv("VLLM_URL", cfg.inference.vllm_url)
    return render_template("settings.html", vllm_url=vllm_url)
