"""API v1 Blueprint registration and initialization."""

from flask import Blueprint

api_bp = Blueprint("api_v1", __name__)

# Import routes to register them with the blueprint
from . import (
    analysis,
    camera,
    config,
    health,
    llm,
    logs,
    mcp,
    monitoring,
    retail,
    system,
    users,
)

__all__ = ["api_bp"]
