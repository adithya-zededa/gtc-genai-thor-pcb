"""User management API endpoints."""

import sqlite3

from flask import jsonify, request

from app.database import UserRepository
from core.logging import get_logger
from services.infrastructure.config import update_email_recipients

from . import api_bp

logger = get_logger(__name__)


@api_bp.route("/users", methods=["GET", "POST"])
def users():
    """User management API - list or create users."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        required_fields = [field for field in ("email", "name") if not data.get(field)]
        if required_fields:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Missing required fields: {', '.join(required_fields)}",
                    }
                ),
                400,
            )

        try:
            UserRepository.create(
                email=data["email"],
                name=data["name"],
                role=data.get("role", "user"),
            )
        except sqlite3.IntegrityError as exc:
            return (
                jsonify({"success": False, "error": f"Unable to add user: {exc}"}),
                400,
            )
        except sqlite3.Error as exc:
            return jsonify({"success": False, "error": f"Database error: {exc}"}), 500

        update_email_recipients()
        return jsonify({"success": True, "message": "User added successfully"})

    # GET method
    users = UserRepository.get_all(active_only=True)
    return jsonify([user.to_dict() for user in users])


@api_bp.route("/users/<int:user_id>", methods=["DELETE", "PUT"])
def user_modify(user_id: int):
    """Delete or update a user."""
    if request.method == "DELETE":
        try:
            UserRepository.deactivate(user_id)
        except sqlite3.Error as exc:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Failed to deactivate user: {exc}",
                    }
                ),
                500,
            )

        update_email_recipients()
        return jsonify({"success": True, "message": "User deactivated"})

    # PUT method
    data = request.get_json(silent=True) or {}
    required_fields = [field for field in ("email", "name") if not data.get(field)]
    if required_fields:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Missing required fields: {', '.join(required_fields)}",
                }
            ),
            400,
        )

    try:
        UserRepository.update(
            user_id=user_id,
            email=data["email"],
            name=data["name"],
            role=data.get("role", "user"),
        )
    except sqlite3.IntegrityError as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Unable to update user: {exc}",
                }
            ),
            400,
        )
    except sqlite3.Error as exc:
        return jsonify({"success": False, "error": f"Database error: {exc}"}), 500

    update_email_recipients()
    return jsonify({"success": True, "message": "User updated successfully"})
