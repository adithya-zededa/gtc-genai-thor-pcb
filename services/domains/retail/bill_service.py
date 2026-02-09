"""Bill assembly and calculation service for retail domain.

Handles business logic for creating bills from scanned items or explicit item lists,
including catalog matching, price reconciliation, validation, and warning generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)


def validate_and_sanitize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize a single item for billing.

    Business rules:
    - Quantity must be positive integer, clamped to [1, 9999]
    - Price must be non-negative float
    - Name is required
    """
    from agents.tools.validation import clamp_quantity

    sanitized = item.copy()
    sanitized["quantity"] = clamp_quantity(item.get("quantity", 1))
    sanitized["price"] = max(0.0, float(item.get("price", 0.0)))

    if not sanitized.get("name"):
        sanitized["name"] = "Unknown Item"

    return sanitized


def reconcile_scanned_item(
    scanned: Dict[str, Any],
    catalog_match: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge a scanned item with its catalog match.

    Business rules:
    - Use catalog name if matched, else scanned name
    - Use catalog price if matched, else 0.0
    - Preserve scanned quantity
    - Include SKU and category from catalog if available
    """
    return {
        "name": catalog_match.get("item_name", scanned.get("name", "Unknown")),
        "quantity": scanned.get("quantity", 1),
        "price": float(catalog_match.get("price", 0.0)),
        "sku": catalog_match.get("sku", ""),
        "category": catalog_match.get("category", scanned.get("category", "")),
    }


def create_bill_from_scan(scan_result: Dict[str, Any]) -> Dict[str, Any]:
    """Create a bill from VLM scan results.

    Matches scanned items against the catalog, reconciles prices,
    calculates totals, and generates warnings for unmatched items.

    Parameters
    ----------
    scan_result : dict
        The scan data containing "items" list.

    Returns
    -------
    dict
        Bill with line_items, totals, and warnings.
    """
    from services.domains.retail.service import lookup_items, calculate_bill

    scanned_items = scan_result.get("items", [])
    if not scanned_items:
        return {
            "success": False,
            "message": "No items found in scan result",
        }

    # Lookup all items in catalog
    names = [it.get("name", "") for it in scanned_items]
    catalog_matches = lookup_items(names)

    # Reconcile and track unmatched
    items = []
    unmatched = []

    for scanned, matched in zip(scanned_items, catalog_matches):
        reconciled = reconcile_scanned_item(scanned, matched)
        items.append(reconciled)

        if not matched.get("matched"):
            unmatched.append(scanned.get("name", "Unknown"))

    # Calculate bill
    bill = calculate_bill(items)

    # Add warnings for unmatched items
    if unmatched:
        bill["warnings"] = [
            f"Price not found in catalog for: {', '.join(unmatched)}. "
            "These items are listed with price 0.00."
        ]

    bill["success"] = True
    bill["message"] = f"Bill created with {bill['unique_items']} item(s), total: {bill['total']}"

    return bill


def create_bill_from_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a bill from an explicit list of items.

    Validates and sanitizes each item, then calculates totals.

    Parameters
    ----------
    items : list of dict
        Each item should have name, quantity, price.

    Returns
    -------
    dict
        Bill with line_items and totals.
    """
    from services.domains.retail.service import calculate_bill

    if not items:
        return {
            "success": False,
            "message": "No items provided",
        }

    # Validate and sanitize all items
    sanitized_items = [validate_and_sanitize_item(item) for item in items]

    # Calculate bill
    bill = calculate_bill(sanitized_items)
    bill["success"] = True
    bill["message"] = f"Bill created with {bill['unique_items']} item(s), total: {bill['total']}"

    return bill
