"""Flask application factory and initialization.

This module provides the application factory pattern for creating
Flask app instances with proper configuration and extensions.
"""

# pylint: disable=import-outside-toplevel

from flask import Flask
from flask_socketio import SocketIO

from core.config import get_config

# Global SocketIO instance
socketio = SocketIO()


def create_app(config_override: dict = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_override: Optional configuration overrides for testing.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    # Load configuration
    config = get_config()
    app.secret_key = config.flask.secret_key
    app.config["DEBUG"] = config.flask.debug

    # Apply any overrides
    if config_override:
        app.config.update(config_override)

    # Initialize SocketIO
    cors_origins = config.flask.socketio_cors
    if cors_origins and "," in cors_origins:
        cors_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
    socketio.init_app(app, cors_allowed_origins=cors_origins or "*")

    # Register blueprints
    from app.api import register_api_versions
    from app.views import views_bp
    from app.websocket import register_handlers

    app.register_blueprint(views_bp)
    register_api_versions(app)

    # Register WebSocket handlers
    register_handlers(socketio)

    return app


# Re-export for convenience
__all__ = ["create_app", "socketio"]
