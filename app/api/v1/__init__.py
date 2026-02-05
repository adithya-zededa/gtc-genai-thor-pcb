"""API v1 Blueprint registration and initialization."""

from flask import Blueprint

api_bp = Blueprint("api_v1", __name__)

# Import routes to register them with the blueprint
from . import (
    health,
    monitoring,
    users,
    config,
    logs,
    analysis,
    system,
    camera,
    mcp,
    retail,
    llm,
)

__all__ = ["api_bp"]
