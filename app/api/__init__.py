"""API blueprint registration and versioning.

Provides a central function to register all API version blueprints
with the Flask application.  Adding a v2 in the future is as simple
as creating ``app/api/v2/`` and registering it here.
"""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

from flask import Flask


def register_api_versions(app: Flask) -> None:
    """Register all API version blueprints with the application.

    Args:
        app: The Flask application instance.

    Example — adding a v2::

        from app.api.v2 import api_v2_bp
        app.register_blueprint(api_v2_bp, url_prefix="/api/v2")
    """
    from app.api.v1 import (
        analysis,
        camera,
        config,
        defects,
        health,
        llm,
        logs,
        mcp,
        monitoring,
        system,
        users,
    )
    from app.api.v1 import api_bp as api_v1_bp

    _ = (
        analysis,
        camera,
        config,
        defects,
        health,
        llm,
        logs,
        mcp,
        monitoring,
        system,
        users,
    )

    # v1 keeps the legacy ``/api`` prefix for backward compatibility.
    app.register_blueprint(api_v1_bp, url_prefix="/api")

    # Future versions:
    # from app.api.v2 import api_bp as api_v2_bp
    # app.register_blueprint(api_v2_bp, url_prefix="/api/v2")
