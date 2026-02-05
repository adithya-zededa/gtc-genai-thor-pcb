"""Route modules (Flask Blueprints) for the web application."""

from .core import core_bp
from .agent import agent_bp
from .llm import llm_bp

__all__ = [
    'core_bp',
    'agent_bp',
    'llm_bp',
]
