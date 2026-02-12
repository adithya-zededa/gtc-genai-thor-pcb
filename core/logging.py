"""Centralized logging configuration for the Camera Agent application."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Default settings
DEFAULT_LOG_FILE = os.getenv("CAMERA_AGENT_LOG_FILE", "camera_agent.log")
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Log level that effectively disables a handler
HANDLER_DISABLE_LEVEL = logging.CRITICAL + 10

_LOGGING_STATE: dict[str, bool] = {"initialized": False}


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    log_to_file: bool = True,
) -> logging.Logger:
    """Set up application-wide logging configuration.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file. None disables file logging.
        log_to_console: Whether to log to stdout.
        log_to_file: Whether to log to file.

    Returns:
        The root logger instance.
    """

    level = level or DEFAULT_LOG_LEVEL
    log_file = log_file if log_file is not None else DEFAULT_LOG_FILE

    handlers: list[logging.Handler] = []

    if log_to_console:
        handlers.append(logging.StreamHandler(sys.stdout))

    if log_to_file and log_file:
        log_path = Path(log_file).expanduser()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path, mode='a'))
        except OSError as exc:
            sys.stderr.write(f"Warning: unable to use log file {log_path}: {exc}\n")

    # Only configure if not already done or if explicitly called
    if not _LOGGING_STATE["initialized"] or handlers:
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format=DEFAULT_LOG_FORMAT,
            handlers=handlers or [logging.StreamHandler(sys.stdout)],
            force=True,  # Override any existing configuration
        )
        _LOGGING_STATE["initialized"] = True

    return logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    if not _LOGGING_STATE["initialized"]:
        setup_logging()
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """Update the log level for all application loggers.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    level_value = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level_value)

    # Update specific application loggers
    for logger_name in ["camera_agent", "email_agent", "app", "agents", "services"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level_value)


def apply_log_preferences(
    log_level: str,
    log_to_console: bool = True,
    log_to_file: bool = True,
) -> None:
    """Apply log level and handler preferences at runtime.

    Args:
        log_level: Log level string.
        log_to_console: Whether console logging is enabled.
        log_to_file: Whether file logging is enabled.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(level if log_to_file else HANDLER_DISABLE_LEVEL)
        elif isinstance(handler, logging.StreamHandler):
            handler.setLevel(level if log_to_console else HANDLER_DISABLE_LEVEL)
