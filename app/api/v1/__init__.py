"""API v1 Blueprint registration and initialization."""

from flask import Blueprint

api_bp = Blueprint("api_v1", __name__)

__all__ = ["api_bp"]
