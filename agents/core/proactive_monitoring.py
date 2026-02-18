"""Backward-compat re-export — canonical location is monitoring_loop.py.

.. deprecated::
    Import from ``agents.core.monitoring_loop`` instead.
"""

from agents.core.monitoring_loop import (  # noqa: F401
    MonitoringLoop as ProactiveMonitoringAgent,
    MonitoringLoop,
    Observation as ObservationResult,
    MonitoringContext,
)

__all__ = [
    "ProactiveMonitoringAgent",
    "MonitoringLoop",
    "ObservationResult",
    "MonitoringContext",
]
