"""Minimal logger shim.

The real application uses core.logging.get_logger, which configures file
handlers, log level from env vars, etc. None of that is relevant to
extracting tool schemas — this just returns a plain stdlib logger so
state_machine.py and registry.py don't need the rest of the app's
logging infrastructure.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
