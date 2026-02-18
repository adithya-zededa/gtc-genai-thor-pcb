"""PCB domain service."""

from .service import (
    record_defect, should_alert, generate_defect_report,
    classify_board_from_analysis, extract_defects_from_analysis,
)
from .defect_store import (
    get_defects_in_range,
    count_defects,
    get_latest_defect,
    get_most_severe_defect,
    get_defect_type_breakdown,
    get_defect_trend,
    get_top_defect_sources,
    check_threshold_alerts,
    generate_summary_report,
    get_defect_insights,
)
from .notification_preferences import (
    NotificationPreferences,
    get_notification_preferences,
    update_notification_preferences,
    load_preferences,
    save_preferences,
)

__all__ = [
    # Service
    "record_defect", "should_alert", "generate_defect_report",
    "classify_board_from_analysis", "extract_defects_from_analysis",
    # Defect store queries
    "get_defects_in_range", "count_defects", "get_latest_defect",
    "get_most_severe_defect", "get_defect_type_breakdown",
    "get_defect_trend", "get_top_defect_sources",
    "check_threshold_alerts", "generate_summary_report",
    "get_defect_insights",
    # Notification preferences
    "NotificationPreferences", "get_notification_preferences",
    "update_notification_preferences", "load_preferences", "save_preferences",
]
