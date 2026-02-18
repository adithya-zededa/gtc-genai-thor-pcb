"""Defect analytics & notification preferences API endpoints.

Exposes the defect-store query functions and notification-preference
management that were previously only available via MCP tool calls.
These REST endpoints power the dashboard widgets, settings page,
and can be consumed by external integrations.
"""

from flask import jsonify, request

from core.logging import get_logger

from . import api_bp

logger = get_logger(__name__)


# =============================================================================
# DEFECT ANALYTICS ENDPOINTS
# =============================================================================


@api_bp.route("/defects/summary", methods=["GET"])
def defect_summary():
    """Return a defect summary for the given time window.

    Query params:
        hours: Time window in hours (default 24).
        board_type: Optional board type filter.
        severity: Optional severity filter (low/medium/high).
    """
    from services.domains.pcb.defect_store import get_defects_in_range, count_defects, _time_window

    hours = request.args.get("hours", 24, type=float)
    board_type = request.args.get("board_type", "").strip() or None
    severity = request.args.get("severity", "").strip() or None

    start, end = _time_window(hours)
    defects = get_defects_in_range(
        start_time=start,
        end_time=end,
        severity=severity,
        board_type=board_type,
        limit=50,
    )
    total = count_defects(hours=hours, severity=severity)

    return jsonify({
        "success": True,
        "total": total,
        "window_hours": hours,
        "defects": defects,
    })


@api_bp.route("/defects/count", methods=["GET"])
def defect_count():
    """Count defects within a time window.

    Query params:
        hours: Time window in hours (default 24).
        severity: Optional severity filter.
    """
    from services.domains.pcb.defect_store import count_defects

    hours = request.args.get("hours", 24, type=float)
    severity = request.args.get("severity", "").strip() or None

    total = count_defects(hours=hours, severity=severity)
    high = count_defects(hours=hours, severity="high")
    medium = count_defects(hours=hours, severity="medium")
    low = count_defects(hours=hours, severity="low")

    return jsonify({
        "success": True,
        "total": total,
        "breakdown": {"high": high, "medium": medium, "low": low},
        "window_hours": hours,
    })


@api_bp.route("/defects/latest", methods=["GET"])
def defect_latest():
    """Return the most recently recorded defect."""
    from services.domains.pcb.defect_store import get_latest_defect

    defect = get_latest_defect()
    return jsonify({
        "success": True,
        "defect": defect,
    })


@api_bp.route("/defects/trend", methods=["GET"])
def defect_trend():
    """Analyze defect rate trend over a window.

    Query params:
        hours: Window size in hours (default 168 = 7 days).
        buckets: Number of time buckets (default 7).
    """
    from services.domains.pcb.defect_store import get_defect_trend

    hours = request.args.get("hours", 168, type=float)
    buckets = request.args.get("buckets", 7, type=int)

    trend = get_defect_trend(window_hours=hours, bucket_count=buckets)
    return jsonify({"success": True, **trend})


@api_bp.route("/defects/types", methods=["GET"])
def defect_type_breakdown():
    """Return defect types ordered by frequency.

    Query params:
        hours: Time window in hours (optional, omit for all time).
    """
    from services.domains.pcb.defect_store import get_defect_type_breakdown

    hours = request.args.get("hours", type=float) or None
    breakdown = get_defect_type_breakdown(hours=hours)
    return jsonify({"success": True, "types": breakdown})


@api_bp.route("/defects/severity", methods=["GET"])
def defect_most_severe():
    """Return the highest-severity defect.

    Query params:
        hours: Time window in hours (optional).
    """
    from services.domains.pcb.defect_store import get_most_severe_defect

    hours = request.args.get("hours", type=float) or None
    defect = get_most_severe_defect(hours=hours)
    return jsonify({"success": True, "defect": defect})


@api_bp.route("/defects/sources", methods=["GET"])
def defect_top_sources():
    """Return board types that produce the most defects.

    Query params:
        hours: Time window in hours (optional).
        limit: Max results (default 10).
    """
    from services.domains.pcb.defect_store import get_top_defect_sources

    hours = request.args.get("hours", type=float) or None
    limit = request.args.get("limit", 10, type=int)
    sources = get_top_defect_sources(hours=hours, limit=limit)
    return jsonify({"success": True, "sources": sources})


@api_bp.route("/defects/thresholds", methods=["GET"])
def defect_threshold_check():
    """Check whether defect counts exceed thresholds.

    Query params:
        rate_threshold: Max defects per window (default 10).
        window_hours: Time window (default 24).
        high_severity_threshold: Max high-severity (default 5).
    """
    from services.domains.pcb.defect_store import check_threshold_alerts

    rate = request.args.get("rate_threshold", 10, type=float)
    window = request.args.get("window_hours", 24, type=float)
    high_sev = request.args.get("high_severity_threshold", 5, type=int)

    result = check_threshold_alerts(
        rate_threshold=rate,
        window_hours=window,
        high_severity_threshold=high_sev,
    )
    return jsonify({"success": True, **result})


@api_bp.route("/defects/report", methods=["GET"])
def defect_report():
    """Generate a comprehensive defect summary report.

    Query params:
        period: "daily", "weekly", or "all" (default "daily").
        board_type: Optional board type filter.
    """
    from services.domains.pcb.defect_store import generate_summary_report

    period = request.args.get("period", "daily").strip()
    board_type = request.args.get("board_type", "").strip() or None

    report = generate_summary_report(period=period, board_type=board_type)
    return jsonify(report)


@api_bp.route("/defects/insights", methods=["GET"])
def defect_insights():
    """Generate AI-driven recommendations and risk analysis.

    Query params:
        hours: Analysis window in hours (optional).
    """
    from services.domains.pcb.defect_store import get_defect_insights

    hours = request.args.get("hours", type=float) or None
    insights = get_defect_insights(hours=hours)
    return jsonify(insights)


# =============================================================================
# NOTIFICATION PREFERENCES ENDPOINTS
# =============================================================================


@api_bp.route("/notifications/preferences", methods=["GET", "PUT"])
def notification_preferences():
    """Get or update agent notification preferences.

    GET returns current preferences.
    PUT accepts a JSON body with fields to update:
        {
            "email_enabled": true,
            "min_severity": "medium",
            "email_recipients": ["a@b.com"],
            "quiet_hours_start": "22:00",
            "quiet_hours_end": "07:00",
            "daily_digest_enabled": false,
            "daily_digest_time": "08:00"
        }
    """
    from services.domains.pcb.notification_preferences import (
        get_notification_preferences,
        update_notification_preferences,
    )

    if request.method == "GET":
        prefs = get_notification_preferences()
        return jsonify({"success": True, "preferences": prefs.to_dict()})

    # PUT method
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "Invalid payload"}), 400

    # Whitelist accepted fields
    allowed = {
        "email_enabled", "email_recipients", "min_severity",
        "notify_on_threshold_exceeded", "quiet_hours_start",
        "quiet_hours_end", "daily_digest_enabled", "daily_digest_time",
    }
    updates = {k: v for k, v in data.items() if k in allowed}

    if not updates:
        return jsonify({"success": False, "error": "No valid fields provided"}), 400

    prefs = update_notification_preferences(**updates)
    return jsonify({
        "success": True,
        "message": "Notification preferences updated",
        "preferences": prefs.to_dict(),
    })


@api_bp.route("/monitoring/status", methods=["GET"])
def monitoring_comprehensive_status():
    """Enhanced monitoring status combining agent state, defect counts, and alerts.

    This powers the dashboard's monitoring overview widget.
    """
    from services.core.monitoring import get_monitoring_service
    from services.domains.pcb.defect_store import count_defects, check_threshold_alerts, get_defect_trend

    service = get_monitoring_service()

    # Agent state
    monitoring_active = False
    agent_state = "off"
    frames_processed = 0
    inspections = 0
    if service:
        monitoring_active = service.get_active_monitoring_mode() != "idle"
        snapshot = service.get_proactive_snapshot() or {}
        context = snapshot.get("context", {})
        frames_processed = context.get("frames_processed", 0)
        inspections = context.get("inspections_completed", 0)
        agent_state = snapshot.get("agent_state", "off")

    # Defect overview (last 24h)
    total_24h = count_defects(hours=24)
    high_24h = count_defects(hours=24, severity="high")
    threshold = check_threshold_alerts(window_hours=24)
    trend = get_defect_trend(window_hours=168, bucket_count=7)

    return jsonify({
        "success": True,
        "monitoring_active": monitoring_active,
        "agent_state": agent_state,
        "frames_processed": frames_processed,
        "inspections_completed": inspections,
        "defects_24h": total_24h,
        "high_severity_24h": high_24h,
        "threshold_exceeded": threshold.get("threshold_exceeded", False),
        "threshold_alerts": threshold.get("alerts", []),
        "trend_direction": trend.get("trend", "stable"),
        "trend_buckets": trend.get("buckets", []),
    })
