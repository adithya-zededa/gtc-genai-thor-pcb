"""PCB defect reporting tools."""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.logging import get_logger
from agents.tools.validation import safe_error as _safe_error

logger = get_logger(__name__)


def tool_generate_defect_report(
    board_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive summary report of all logged PCB defects.

    Includes severity breakdown, type breakdown, trend analysis, and
    recent defect list for complete reporting.
    """
    logger.info("tool_generate_defect_report invoked (board_type=%s)", board_type)

    from services.domains.pcb.service import generate_defect_report
    from services.domains.pcb.defect_store import (
        count_defects,
        get_defect_trend,
        get_defect_type_breakdown,
        get_defect_insights,
    )

    try:
        report = generate_defect_report(board_type=board_type)
    except Exception as exc:
        logger.error("Failed to generate defect report: %s", exc)
        return _safe_error("Failed to generate defect report", exc=exc)

    # Enrich with trend and insight data
    try:
        total_all = count_defects()
        total_24h = count_defects(hours=24)
        total_week = count_defects(hours=168)
        trend = get_defect_trend(window_hours=168.0)
        type_breakdown = get_defect_type_breakdown()
        insights = get_defect_insights()

        report_data = report.get("data", {}) if isinstance(report.get("data"), dict) else {}
        report_data["enriched"] = {
            "total_all_time": total_all,
            "total_last_24h": total_24h,
            "total_last_week": total_week,
            "trend": trend,
            "type_breakdown": type_breakdown,
            "risk_level": insights.get("risk_level", "unknown"),
            "recommendations": insights.get("recommendations", []),
        }
        report["data"] = report_data

        # Update message with enrichment summary
        trend_dir = trend.get("trend", "unknown")
        risk = insights.get("risk_level", "unknown")
        report["message"] = (
            f"Defect report: {total_all} total, {total_24h} in last 24h, "
            f"{total_week} in last week. Trend: {trend_dir}. Risk: {risk}."
        )

        logger.info(
            "Defect report generated: total=%d, 24h=%d, week=%d, trend=%s, risk=%s",
            total_all, total_24h, total_week, trend_dir, risk,
        )
    except Exception as exc:
        logger.warning("Failed to enrich defect report with analytics: %s", exc)

    return report


def tool_generate_summary_report(
    period: str = "daily",
    board_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a daily, weekly, or on-demand summary report of monitoring results.

    Parameters
    ----------
    period : str
        Report period: "daily" (24h), "weekly" (168h), or "all".
    board_type : str, optional
        Filter report to a specific board type.
    """
    logger.info("tool_generate_summary_report invoked (period=%s, board_type=%s)", period, board_type)

    from services.domains.pcb.defect_store import generate_summary_report

    try:
        report = generate_summary_report(period=period, board_type=board_type)
    except Exception as exc:
        logger.error("Failed to generate summary report: %s", exc)
        return _safe_error("Failed to generate summary report", exc=exc)

    total = report.get("total_defects", 0)
    trend = report.get("trend", {}).get("trend", "unknown")
    sev_breakdown = report.get("severity_breakdown", {})
    threshold_exceeded = report.get("threshold_check", {}).get("threshold_exceeded", False)

    logger.info(
        "Summary report (%s): total=%d, trend=%s, severities=%s, "
        "threshold_exceeded=%s, board_type=%s",
        period, total, trend, sev_breakdown, threshold_exceeded, board_type,
    )

    # Build a richer message for the LLM
    sev_parts = [f"{k}: {v}" for k, v in sev_breakdown.items() if v > 0]
    sev_summary = ", ".join(sev_parts) if sev_parts else "none"

    msg = (
        f"{period.capitalize()} report: {total} defect(s), "
        f"trend: {trend}, severity: [{sev_summary}]"
    )
    if threshold_exceeded:
        msg += " ⚠️ THRESHOLD EXCEEDED"

    return {
        "success": True,
        "message": msg,
        "data": report,
    }


def tool_query_pcb_inspections(
    limit: int = 10,
    result_filter: Optional[str] = None,
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Query past PCB inspection results from the database.

    This gives you access to historical inspection data so you can
    answer user questions about detected defects, pass rates, etc.

    Parameters
    ----------
    limit : int
        Max number of records to return (default 10).
    result_filter : str, optional
        Filter by result: ``"PASS"``, ``"FAIL"``, or ``None`` for all.
    hours : float, optional
        Time window in hours (e.g., 24 for last day). Omit for all time.
    """
    logger.info(
        "tool_query_pcb_inspections invoked (limit=%d, result_filter=%s, hours=%s)",
        limit, result_filter, hours,
    )

    from app.database import PCBInspectionRepository

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"success": False, "message": "limit must be an integer"}
    limit = max(1, min(50, limit))

    start_time = None
    if hours is not None:
        if hours <= 0:
            return {"success": False, "message": "hours must be > 0"}
        from services.domains.pcb.defect_store import _time_window
        start_time, _ = _time_window(hours)

    try:
        inspections, total = PCBInspectionRepository.get_paginated(
            page=1,
            per_page=limit,
            result=result_filter,
            start_time=start_time,
        )
    except Exception as exc:
        return _safe_error("Failed to query PCB inspections", exc=exc)

    records = [insp.to_dict() for insp in inspections]

    # Build a human-readable summary for the LLM
    pass_count = sum(1 for r in records if r.get("result") == "PASS")
    fail_count = sum(1 for r in records if r.get("result") == "FAIL")

    # Collect defect types from failed inspections
    fail_types: Dict[str, int] = {}
    for r in records:
        if r.get("result") == "FAIL" and r.get("defect_type"):
            dt = r["defect_type"]
            fail_types[dt] = fail_types.get(dt, 0) + 1

    time_desc = f"last {hours}h" if hours else "all time"
    summary = (
        f"Found {total} inspection(s) ({time_desc}). "
        f"Showing {len(records)} most recent. "
        f"Pass: {pass_count}, Fail: {fail_count}."
    )

    logger.info(
        "PCB inspections query: total=%d, showing=%d, pass=%d, fail=%d, "
        "fail_types=%s, filter=%s, hours=%s",
        total, len(records), pass_count, fail_count, fail_types, result_filter, hours,
    )

    return {
        "success": True,
        "message": summary,
        "data": {
            "inspections": records,
            "total": total,
            "showing": len(records),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "fail_defect_types": fail_types,
            "time_window": time_desc,
        },
    }
