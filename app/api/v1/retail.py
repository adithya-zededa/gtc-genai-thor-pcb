"""Retail API endpoints — catalog CRUD and invoice management.

Provides REST endpoints for:
- Retail catalog items (CRUD + bulk import)
- Invoice listing and detail
- Catalog search

All routes are registered under the ``/api`` blueprint prefix.
"""

# pylint: disable=broad-exception-caught,import-outside-toplevel,line-too-long

from flask import jsonify, request

from app.database.repositories import InvoiceRepository, RetailCatalogRepository
from core.logging import get_logger

from . import api_bp

logger = get_logger(__name__)


# =============================================================================
# CATALOG ENDPOINTS
# =============================================================================


@api_bp.route("/retail/catalog", methods=["GET"])
def list_catalog():
    """List catalog items with optional search.

    Query params:
        q: Search by item name (fuzzy LIKE)
        category: Filter by category
    """
    query = request.args.get("q", "").strip()
    category = request.args.get("category", "").strip()

    if query:
        items = RetailCatalogRepository.search_by_name(query)
    elif category:
        items = RetailCatalogRepository.search_by_category(category)
    else:
        items = RetailCatalogRepository.get_all()

    return jsonify(
        {
            "success": True,
            "items": [i.to_dict() for i in items],
            "count": len(items),
        }
    )


@api_bp.route("/retail/catalog/<int:item_id>", methods=["GET"])
def get_catalog_item(item_id: int):
    """Get a single catalog item by ID."""
    item = RetailCatalogRepository.get_by_id(item_id)
    if not item:
        return jsonify({"success": False, "error": "Item not found"}), 404
    return jsonify({"success": True, "item": item.to_dict()})


@api_bp.route("/retail/catalog", methods=["POST"])
def create_catalog_item():
    """Create a new catalog item.

    Request body:
        {
            "item_name": "Arduino Uno R3",
            "sku": "ARD-UNO-R3",
            "price": 799.00,
            "category": "Microcontroller"
        }
    """
    data = request.get_json() or {}

    required = ["item_name", "price"]
    missing = [f for f in required if f not in data]
    if missing:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Missing required fields: {', '.join(missing)}",
                }
            ),
            400,
        )

    try:
        item = RetailCatalogRepository.create(
            item_name=data["item_name"],
            sku=data.get("sku", ""),
            price=float(data["price"]),
            category=data.get("category", ""),
        )
        return jsonify({"success": True, "item": item.to_dict()}), 201
    except Exception as e:
        logger.error("Failed to create catalog item: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/retail/catalog/<int:item_id>", methods=["PUT"])
def update_catalog_item(item_id: int):
    """Update an existing catalog item.

    Request body — only include fields to update:
        {
            "price": 899.00,
            "category": "Boards"
        }
    """
    data = request.get_json() or {}
    if not data:
        return jsonify({"success": False, "error": "No update data provided"}), 400

    allowed_fields = {"item_name", "sku", "price", "category"}
    update_data = {k: v for k, v in data.items() if k in allowed_fields}

    if "price" in update_data:
        update_data["price"] = float(update_data["price"])

    try:
        item = RetailCatalogRepository.update(item_id, **update_data)
        if not item:
            return jsonify({"success": False, "error": "Item not found"}), 404
        return jsonify({"success": True, "item": item.to_dict()})
    except Exception as e:
        logger.error("Failed to update catalog item %d: %s", item_id, e)
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/retail/catalog/<int:item_id>", methods=["DELETE"])
def delete_catalog_item(item_id: int):
    """Delete a catalog item."""
    success = RetailCatalogRepository.delete(item_id)
    if not success:
        return jsonify({"success": False, "error": "Item not found"}), 404
    return jsonify({"success": True, "message": "Item deleted"})


@api_bp.route("/retail/catalog/bulk", methods=["POST"])
def bulk_import_catalog():
    """Bulk import catalog items.

    Request body:
        {
            "items": [
                {"item_name": "Arduino Uno R3", "sku": "ARD-UNO-R3", "price": 799.00, "category": "Microcontroller"},
                {"item_name": "Raspberry Pi 4B", "sku": "RPI-4B-4G", "price": 4999.00, "category": "SBC"}
            ]
        }
    """
    data = request.get_json() or {}
    items = data.get("items", [])

    if not items:
        return jsonify({"success": False, "error": "No items provided"}), 400

    # Validate each item has minimum required fields
    for i, item in enumerate(items):
        if "item_name" not in item or "price" not in item:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Item at index {i} missing 'item_name' or 'price'",
                    }
                ),
                400,
            )

    try:
        created = RetailCatalogRepository.bulk_create(items)
        return (
            jsonify(
                {
                    "success": True,
                    "created": len(created),
                    "items": [c.to_dict() for c in created],
                }
            ),
            201,
        )
    except Exception as e:
        logger.error("Bulk import failed: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# INVOICE ENDPOINTS
# =============================================================================


@api_bp.route("/retail/invoices", methods=["GET"])
def list_invoices():
    """List invoices with pagination.

    Query params:
        page: Page number (default 1)
        per_page: Items per page (default 20)
        status: Filter by status (draft / sent)
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    status = request.args.get("status", "").strip() or None

    invoices, total = InvoiceRepository.get_paginated(
        page=page,
        per_page=per_page,
        status=status,
    )

    return jsonify(
        {
            "success": True,
            "invoices": [inv.to_dict() for inv in invoices],
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page if per_page else 1,
        }
    )


@api_bp.route("/retail/invoices/<int:invoice_id>", methods=["GET"])
def get_invoice(invoice_id: int):
    """Get a single invoice by ID."""
    invoice = InvoiceRepository.get_by_id(invoice_id)
    if not invoice:
        return jsonify({"success": False, "error": "Invoice not found"}), 404
    return jsonify({"success": True, "invoice": invoice.to_dict()})


@api_bp.route("/retail/invoices/<int:invoice_id>/download", methods=["GET"])
def download_invoice_pdf(invoice_id: int):
    """Download the PDF for a specific invoice."""
    import os

    from flask import send_file

    invoice = InvoiceRepository.get_by_id(invoice_id)
    if not invoice:
        return jsonify({"success": False, "error": "Invoice not found"}), 404

    if not invoice.pdf_path or not os.path.exists(invoice.pdf_path):
        return jsonify({"success": False, "error": "PDF not available"}), 404

    try:
        return send_file(
            invoice.pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"invoice_{invoice_id}.pdf",
        )
    except Exception as e:
        logger.error("Failed to send PDF file: %s", e)
        return jsonify({"success": False, "error": "Failed to retrieve PDF"}), 500
