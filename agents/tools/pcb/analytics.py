"""PCB defect analytics and query tools."""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.logging import get_logger
from agents.tools.validation import safe_error as _safe_error

logger = get_logger(__name__)


def tool_get_monitoring_status() -> Dict[str, Any]:
    """Get current and historical monitoring status including defect counts.

    Returns agent state, session info, total defects, and recent activity.
    Also includes trend direction and threshold status for quick overview.
    """
    logger.info("tool_get_monitoring_status invoked")

    from services.core.monitoring import get_monitoring_service
    from services.domains.pcb.defect_store import (
        count_defects,
        get_latest_defect,
        get_defect_trend,
        check_threshold_alerts,
    )

    service = get_monitoring_service()
    monitoring_active = False
    monitoring_mode = "idle"
    if service:
        monitoring_mode = service.get_active_monitoring_mode() if hasattr(service, "get_active_monitoring_mode") else "unknown"
        monitoring_active = monitoring_mode != "idle"

    total_defects = count_defects()
    defects_24h = count_defects(hours=24)
    defects_1h = count_defects(hours=1)
    latest = get_latest_defect()

    # Add trend and threshold status
    trend_info = None
    threshold_info = None
    try:
        trend = get_defect_trend(window_hours=168.0)
        trend_info = {
            "direction": trend.get("trend", "unknown"),
            "total_in_window": trend.get("total_defects", 0),
        }
    except Exception:
        pass

    try:
        threshold = check_threshold_alerts(window_hours=24.0)
        threshold_info = {
            "exceeded": threshold.get("threshold_exceeded", False),
            "alerts": [a["message"] for a in threshold.get("alerts", [])],
        }
    except Exception:
        pass

    logger.info(
        "Monitoring status: active=%s mode=%s total_defects=%d 24h=%d 1h=%d trend=%s",
        monitoring_active, monitoring_mode, total_defects, defects_24h, defects_1h,
        trend_info.get("direction") if trend_info else "N/A",
    )

    return {
        "success": True,
        "message": f"Monitoring is {'active' if monitoring_active else 'inactive'}. "
                   f"{defects_24h} defect(s) in the last 24h, {total_defects} total.",
        "data": {
            "monitoring_active": monitoring_active,
            "monitoring_mode": monitoring_mode,
            "total_defects": total_defects,
            "defects_last_24h": defects_24h,
            "defects_last_1h": defects_1h,
            "latest_defect": latest,
            "trend": trend_info,
            "threshold_status": threshold_info,
        },
    }


def tool_get_defect_summary(
    hours: Optional[float] = None,
    board_type: Optional[str] = None,
    severity: Optional[str] = None,
) -> Dict[str, Any]:
    """Get a summary of defects identified within a time range or for a board type.

    Parameters
    ----------
    hours : float, optional
        Time window in hours (e.g., 24 for last day, 168 for last week).
    board_type : str, optional
        Filter by specific board type.
    severity : str, optional
        Filter by severity level.
    """
    logger.info(
        "tool_get_defect_summary invoked (hours=%s, board_type=%s, severity=%s)",
        hours, board_type, severity,
    )

    from services.domains.pcb.defect_store import (
        get_defects_in_range,
        _time_window,
        count_defects,
        get_defect_trend,
    )

    start_time = None
    if hours is not None:
        if hours <= 0:
            return {"success": False, "message": "hours must be > 0"}
        start_time, _ = _time_window(hours)

    defects = get_defects_in_range(
        start_time=start_time,
        board_type=board_type,
        severity=severity,
    )

    # Match breakdown to the same filtered defect set used for totals
    type_counts: Dict[str, int] = {}
    for defect in defects:
        defect_type = defect.get("defect_type", "unknown")
        type_counts[defect_type] = type_counts.get(defect_type, 0) + 1

    total_for_pct = len(defects) or 1
    type_breakdown = [
        {
            "defect_type": defect_type,
            "count": count,
            "percentage": round((count / total_for_pct) * 100, 2),
        }
        for defect_type, count in sorted(type_counts.items(), key=lambda item: item[1], reverse=True)
    ]

    # Severity counts
    sev_counts: Dict[str, int] = {}
    for d in defects:
        s = d.get("severity", "unknown")
        sev_counts[s] = sev_counts.get(s, 0) + 1

    # Include trend data for context
    trend_info = None
    try:
        trend_data = get_defect_trend(window_hours=hours or 168.0)
        trend_info = {
            "direction": trend_data.get("trend", "unknown"),
            "first_half_rate": trend_data.get("first_half_rate", 0),
            "second_half_rate": trend_data.get("second_half_rate", 0),
        }
    except Exception:
        pass

    # Compare to previous period for delta
    delta_info = None
    if hours:
        try:
            prev_count = count_defects(hours=hours * 2) - len(defects)
            if prev_count > 0:
                change_pct = round(((len(defects) - prev_count) / prev_count) * 100, 1)
            else:
                change_pct = 100.0 if len(defects) > 0 else 0.0
            delta_info = {
                "current_period": len(defects),
                "previous_period": prev_count,
                "change_percent": change_pct,
            }
        except Exception:
            pass

    time_desc = f"last {hours}h" if hours else "all time"
    filters = []
    if board_type:
        filters.append(f"board={board_type}")
    if severity:
        filters.append(f"severity={severity}")
    filter_str = f" ({', '.join(filters)})" if filters else ""

    logger.info(
        "Defect summary: %d defect(s) for %s%s, severities=%s, types=%d",
        len(defects), time_desc, filter_str, sev_counts, len(type_counts),
    )

    return {
        "success": True,
        "message": f"{len(defects)} defect(s) found for {time_desc}{filter_str}.",
        "data": {
            "total": len(defects),
            "time_window": time_desc,
            "severity_breakdown": sev_counts,
            "defect_types": type_breakdown,
            "trend": trend_info,
            "period_comparison": delta_info,
            "recent_defects": [d for d in defects[:10]],
        },
    }


def tool_count_defective_pcbs(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Count how many defective PCBs were detected overall or within a time window.

    Parameters
    ----------
    hours : float, optional
        Time window in hours. Omit for all-time count.
    """
    logger.info("tool_count_defective_pcbs invoked (hours=%s)", hours)

    from services.domains.pcb.defect_store import count_defects

    total = count_defects(hours=hours)
    high = count_defects(hours=hours, severity="high")
    medium = count_defects(hours=hours, severity="medium")
    low = count_defects(hours=hours, severity="low")

    time_desc = f"last {hours}h" if hours else "all time"

    logger.info(
        "Defect count (%s): total=%d, high=%d, medium=%d, low=%d",
        time_desc, total, high, medium, low,
    )

    return {
        "success": True,
        "message": f"{total} defective PCB(s) detected ({time_desc}): "
                   f"{high} high, {medium} medium, {low} low severity.",
        "data": {
            "total": total,
            "high": high,
            "medium": medium,
            "low": low,
            "time_window": time_desc,
        },
    }


def tool_get_latest_defect() -> Dict[str, Any]:
    """Get the most recently identified defect with full details."""
    logger.info("tool_get_latest_defect invoked")

    from services.domains.pcb.defect_store import get_latest_defect

    defect = get_latest_defect()
    if not defect:
        logger.info("No defects recorded yet")
        return {
            "success": True,
            "message": "No defects have been recorded yet.",
            "data": None,
        }

    logger.info(
        "Latest defect: type=%s severity=%s board=%s confidence=%.2f timestamp=%s",
        defect.get("defect_type", "unknown"),
        defect.get("severity", "unknown"),
        defect.get("board_type", "unknown"),
        defect.get("confidence", 0.0),
        defect.get("timestamp", "unknown"),
    )

    return {
        "success": True,
        "message": f"Most recent defect: {defect.get('defect_type', 'unknown')} "
                   f"(severity: {defect.get('severity', 'unknown')}) at {defect.get('timestamp', 'unknown')}.",
        "data": defect,
    }


def tool_get_defect_type_breakdown(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Get what types of defects have been identified and how frequently each occurs.

    Parameters
    ----------
    hours : float, optional
        Time window in hours. Omit for all-time breakdown.
    """
    logger.info("tool_get_defect_type_breakdown invoked (hours=%s)", hours)

    from services.domains.pcb.defect_store import get_defect_type_breakdown

    breakdown = get_defect_type_breakdown(hours=hours)
    if not breakdown:
        logger.info("Defect type breakdown: no defects recorded")
        return {
            "success": True,
            "message": "No defects recorded yet.",
            "data": {"types": []},
        }

    time_desc = f"last {hours}h" if hours else "all time"
    summary_parts = [f"{t['defect_type']}: {t['count']} ({t['percentage']}%)" for t in breakdown[:5]]
    total_types = len(breakdown)

    logger.info(
        "Defect type breakdown (%s): %d type(s), top=%s",
        time_desc, total_types, summary_parts[:3],
    )

    return {
        "success": True,
        "message": f"Defect type breakdown ({time_desc}): {'; '.join(summary_parts)}.",
        "data": {"types": breakdown, "time_window": time_desc, "total_types": total_types},
    }


def tool_get_defect_trend(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Analyze whether defect rates are increasing, decreasing, or stable.

    Parameters
    ----------
    hours : float, optional
        Analysis window in hours (default: 168 = 1 week).
    """
    logger.info("tool_get_defect_trend invoked (hours=%s)", hours)

    from services.domains.pcb.defect_store import get_defect_trend

    window = hours or 168.0
    trend = get_defect_trend(window_hours=window)

    logger.info(
        "Defect trend (%sh window): direction=%s, total=%d, "
        "first_half_rate=%.2f, second_half_rate=%.2f, buckets=%d",
        window, trend["trend"], trend["total_defects"],
        trend["first_half_rate"], trend["second_half_rate"],
        len(trend.get("buckets", [])),
    )

    return {
        "success": True,
        "message": f"Defect trend over the last {window}h: {trend['trend'].upper()}. "
                   f"First half rate: {trend['first_half_rate']}/period, "
                   f"second half rate: {trend['second_half_rate']}/period. "
                   f"Total: {trend['total_defects']} defects.",
        "data": trend,
    }


def tool_get_most_severe_defect(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Get the most severe / highest-priority defect detected.

    Parameters
    ----------
    hours : float, optional
        Time window in hours. Omit for all-time.
    """
    logger.info("tool_get_most_severe_defect invoked (hours=%s)", hours)

    from services.domains.pcb.defect_store import get_most_severe_defect

    defect = get_most_severe_defect(hours=hours)
    if not defect:
        time_desc = f"last {hours}h" if hours else "all time"
        logger.info("No defects found (%s)", time_desc)
        return {
            "success": True,
            "message": f"No defects found ({time_desc}).",
            "data": None,
        }

    logger.info(
        "Most severe defect: type=%s severity=%s confidence=%.2f board=%s",
        defect.get("defect_type", "unknown"),
        defect.get("severity", "unknown"),
        defect.get("confidence", 0.0),
        defect.get("board_type", "unknown"),
    )

    return {
        "success": True,
        "message": f"Most severe defect: {defect.get('defect_type', 'unknown')} "
                   f"(severity: {defect.get('severity')}, confidence: {defect.get('confidence')}) "
                   f"on {defect.get('board_type', 'unknown')} at {defect.get('timestamp')}.",
        "data": defect,
    }


def tool_get_top_defect_sources(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Get which logs or sources (board types) produce the most defects.

    Parameters
    ----------
    hours : float, optional
        Time window in hours. Omit for all-time.
    """
    logger.info("tool_get_top_defect_sources invoked (hours=%s)", hours)

    from services.domains.pcb.defect_store import get_top_defect_sources

    sources = get_top_defect_sources(hours=hours)
    if not sources:
        logger.info("No defect sources recorded")
        return {
            "success": True,
            "message": "No defect sources recorded yet.",
            "data": {"sources": []},
        }

    time_desc = f"last {hours}h" if hours else "all time"
    summary_parts = [f"{s['board_type']}: {s['count']} defects" for s in sources[:5]]

    logger.info(
        "Top defect sources (%s): %d source(s), top=%s",
        time_desc, len(sources), summary_parts[:3],
    )

    return {
        "success": True,
        "message": f"Top defect sources ({time_desc}): {'; '.join(summary_parts)}.",
        "data": {"sources": sources, "time_window": time_desc},
    }


def tool_check_threshold_alerts(
    rate_threshold: Optional[float] = None,
    window_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Check whether any defects exceeded predefined thresholds or alerts.

    Parameters
    ----------
    rate_threshold : float, optional
        Max acceptable defects per window (default: 10).
    window_hours : float, optional
        Time window in hours (default: 24).
    """
    logger.info(
        "tool_check_threshold_alerts invoked (rate_threshold=%s, window_hours=%s)",
        rate_threshold, window_hours,
    )

    from services.domains.pcb.defect_store import check_threshold_alerts

    check = check_threshold_alerts(
        rate_threshold=rate_threshold or 10.0,
        window_hours=window_hours or 24.0,
    )

    if check["threshold_exceeded"]:
        alert_msgs = [a["message"] for a in check["alerts"]]
        msg = "THRESHOLD EXCEEDED: " + "; ".join(alert_msgs)
        logger.warning(
            "Threshold exceeded! total=%d, high=%d, window=%.1fh, alerts=%s",
            check["total_defects"], check["high_severity_count"],
            check["window_hours"], alert_msgs,
        )
    else:
        msg = f"All thresholds OK. {check['total_defects']} defect(s) in the last {check['window_hours']}h."
        logger.info(
            "Threshold check OK: total=%d, high=%d, window=%.1fh",
            check["total_defects"], check["high_severity_count"],
            check["window_hours"],
        )

    return {
        "success": True,
        "message": msg,
        "data": check,
    }


def tool_get_defect_insights(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Get recommendations and insights based on observed defect patterns.

    Parameters
    ----------
    hours : float, optional
        Analysis window in hours. Omit for all-time analysis.
    """
    logger.info("tool_get_defect_insights invoked (hours=%s)", hours)

    from services.domains.pcb.defect_store import get_defect_insights

    try:
        insights = get_defect_insights(hours=hours)
    except Exception as exc:
        logger.error("Failed to generate defect insights: %s", exc)
        return _safe_error("Failed to generate defect insights", exc=exc)

    recommendations = insights.get("recommendations", [])
    risk = insights.get("risk_level", "unknown")
    trend_direction = insights.get("trend_direction", "unknown")
    total_analyzed = insights.get("total_defects_analyzed", 0)

    logger.info(
        "Defect insights: risk=%s, trend=%s, total_analyzed=%d, "
        "recommendations=%d, threshold_exceeded=%s",
        risk, trend_direction, total_analyzed,
        len(recommendations), insights.get("threshold_exceeded", False),
    )

    return {
        "success": True,
        "message": f"Risk level: {risk.upper()}. "
                   f"{len(recommendations)} recommendation(s). "
                   f"Trend: {trend_direction}. "
                   f"Total analyzed: {total_analyzed}.",
        "data": insights,
    }


def tool_query_detection_logs(
    limit: int = 20,
    hours: Optional[float] = None,
    detected_only: bool = False,
    include_defects: bool = True,
) -> Dict[str, Any]:
    """Query detection logs enriched with defect information.

    Returns a unified view of detection events from ``detection_logs``
    combined with defect-specific data from ``pcb_defects``, giving a
    complete picture of what was inspected and what defects were found.

    Parameters
    ----------
    limit : int
        Maximum number of records to return (default 20).
    hours : float, optional
        Time window in hours. Omit for all records.
    detected_only : bool
        Only return detections with confidence > 0.
    include_defects : bool
        If True (default), also return recent defect records.
    """
    logger.info(
        "tool_query_detection_logs invoked (limit=%d, hours=%s, "
        "detected_only=%s, include_defects=%s)",
        limit, hours, detected_only, include_defects,
    )

    from app.database import DetectionLogRepository

    try:
        limit = max(1, min(100, int(limit)))
    except (TypeError, ValueError):
        return {"success": False, "message": "limit must be an integer"}

    try:
        logs, total_logs = DetectionLogRepository.get_paginated(
            page=1,
            per_page=limit,
            detected_only=detected_only,
        )
    except Exception as exc:
        logger.error("Failed to query detection logs: %s", exc)
        return _safe_error("Failed to query detection logs", exc=exc)

    log_entries = [log.to_dict() for log in logs]

    # Filter by time window if requested
    if hours is not None and hours > 0:
        from services.domains.pcb.defect_store import _time_window
        cutoff, _ = _time_window(hours)
        log_entries = [
            e for e in log_entries
            if e.get("timestamp", "") >= cutoff
        ]

    result: Dict[str, Any] = {
        "success": True,
        "message": f"Retrieved {len(log_entries)} detection log(s) (total: {total_logs}).",
        "data": {
            "detection_logs": log_entries,
            "total_detection_logs": total_logs,
            "showing": len(log_entries),
        },
    }

    if include_defects:
        try:
            from services.domains.pcb.defect_store import (
                get_defects_in_range,
                count_defects,
                get_latest_defect,
                get_defect_type_breakdown,
                _time_window,
            )

            start_time = None
            if hours is not None and hours > 0:
                start_time, _ = _time_window(hours)

            recent_defects = get_defects_in_range(
                start_time=start_time,
                limit=limit,
            )
            total_defects = count_defects(hours=hours)
            latest_defect = get_latest_defect()
            type_breakdown = get_defect_type_breakdown(hours=hours)

            # Severity breakdown from recent defects
            sev_counts: Dict[str, int] = {}
            for d in recent_defects:
                s = d.get("severity", "unknown")
                sev_counts[s] = sev_counts.get(s, 0) + 1

            result["data"]["defects"] = {
                "recent_defects": recent_defects,
                "total_defects": total_defects,
                "latest_defect": latest_defect,
                "severity_breakdown": sev_counts,
                "type_breakdown": type_breakdown,
            }

            logger.info(
                "Detection logs enriched: %d logs, %d defects, severities=%s",
                len(log_entries), total_defects, sev_counts,
            )
        except Exception as exc:
            logger.warning("Failed to enrich detection logs with defect data: %s", exc)
            result["data"]["defects"] = {"error": str(exc)}

    return result
