"""Detection logs API endpoints."""

import csv
import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Response, jsonify, request, send_file

from app.database import DetectionLogRepository
from core.config import get_config
from core.logging import get_logger

from . import api_bp

logger = get_logger(__name__)


@api_bp.route("/logs", methods=["GET", "DELETE"])
def logs():
    """Get or clear detection logs."""
    if request.method == "DELETE":
        try:
            DetectionLogRepository.delete_all()
            return jsonify({"success": True, "message": "All logs cleared"})
        except sqlite3.Error as exc:
            return (
                jsonify({"success": False, "error": f"Failed to clear logs: {exc}"}),
                500,
            )

    # GET method with pagination
    try:
        page = max(1, request.args.get("page", 1, type=int))
        per_page = min(200, max(1, request.args.get("per_page", 50, type=int)))
        detected_only = request.args.get("detected_only", "").lower() == "true"

        logs, total_count = DetectionLogRepository.get_paginated(
            page=page,
            per_page=per_page,
            detected_only=detected_only,
        )

        total_pages = (total_count + per_page - 1) // per_page if per_page > 0 else 1

        return jsonify(
            {
                "success": True,
                "logs": [log.to_dict() for log in logs],
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_count": total_count,
                    "total_pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1,
                },
            }
        )
    except sqlite3.Error as exc:
        return jsonify({"success": False, "error": f"Database error: {exc}"}), 500


@api_bp.route("/logs/<int:log_id>", methods=["DELETE"])
def delete_log(log_id: int):
    """Delete a specific log entry."""
    try:
        DetectionLogRepository.delete(log_id)
        return jsonify({"success": True, "message": "Log deleted"})
    except sqlite3.Error as exc:
        return jsonify({"success": False, "error": f"Failed to delete log: {exc}"}), 500


@api_bp.route("/logs/export", methods=["POST"])
def export_logs():
    """Export logs as CSV or JSON."""
    try:
        data = request.get_json(silent=True) or {}
        format_type = data.get("format", "csv")

        logs = DetectionLogRepository.get_all_for_export()

        if format_type == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(
                [
                    "Timestamp",
                    "Confidence",
                    "Response",
                    "Image Path",
                    "Frame Number",
                    "Reason",
                    "Vision Description",
                    "Decision Details",
                ]
            )

            for log in logs:
                writer.writerow(
                    [
                        log.timestamp,
                        log.confidence,
                        log.response,
                        log.image_path,
                        log.frame_number,
                        log.reason,
                        log.vision_description,
                        json.dumps(log.decision_details),
                    ]
                )

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=detection_logs_{timestamp_str}.csv"
                },
            )
        else:
            logs_list = [log.to_dict() for log in logs]
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            return Response(
                json.dumps(logs_list, indent=2),
                mimetype="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=detection_logs_{timestamp_str}.json"
                },
            )
    except sqlite3.Error as exc:
        return (
            jsonify({"success": False, "error": f"Failed to export logs: {exc}"}),
            500,
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/image/<path:image_path>")
def serve_image(image_path: str):
    """Serve detection images."""
    config = get_config()

    try:
        candidate_path = Path(image_path)

        detected_dir = config.detected_images_dir.resolve()
        processed_dir = config.processed_frames_dir.resolve()
        data_dir = config.data_dir.resolve()

        paths_to_try = []

        if candidate_path.is_absolute():
            paths_to_try.append(candidate_path.resolve())
        else:
            paths_to_try.append((data_dir / candidate_path).resolve())
            paths_to_try.append(Path.cwd() / candidate_path)
            paths_to_try.append(detected_dir / candidate_path.name)
            paths_to_try.append(processed_dir / candidate_path.name)

            path_str = str(candidate_path)
            if path_str.startswith("detected_images/"):
                stripped = path_str[len("detected_images/") :]
                paths_to_try.append(detected_dir / stripped)
            if path_str.startswith("processed_frames/"):
                stripped = path_str[len("processed_frames/") :]
                paths_to_try.append(processed_dir / stripped)

        resolved_path = None
        for try_path in paths_to_try:
            try:
                resolved = try_path.resolve()
                if resolved.exists() and resolved.is_file():
                    if (
                        resolved.is_relative_to(detected_dir)
                        or resolved.is_relative_to(processed_dir)
                        or resolved.is_relative_to(data_dir)
                    ):
                        resolved_path = resolved
                        break
            except Exception:
                continue

        if resolved_path:
            return send_file(resolved_path, mimetype="image/jpeg")
        else:
            logger.warning(f"Image not found: {image_path}")
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        logger.error(f"Error serving image {image_path}: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/recent_images")
def recent_images():
    """Get recent detection images."""
    import base64
    from datetime import datetime as dt

    config = get_config()

    def collect_images(directory: Path, limit: int):
        if not directory.exists():
            return []

        images = []
        try:
            files = sorted(
                directory.glob("*.jpg"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )[:limit]

            for img_file in files:
                stat_info = img_file.stat()
                with img_file.open("rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                images.append(
                    {
                        "filename": img_file.name,
                        "timestamp": dt.fromtimestamp(stat_info.st_mtime).isoformat(),
                        "image_b64": encoded,
                    }
                )
        except Exception as exc:
            logger.error("Failed to collect images from %s: %s", directory, exc)

        return images

    try:
        detected_images = collect_images(config.detected_images_dir, 10)
        processed_images = collect_images(config.processed_frames_dir, 5)

        return jsonify(
            {
                "detected": detected_images,
                "processed": processed_images,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)})
