"""Retail billing service for item lookup, billing, and invoice generation.

Provides business logic for:
- Catalog item lookup (fuzzy name matching, SKU, category)
- Bill calculation with configurable tax rate
- HTML invoice generation via Jinja2 template
- Invoice persistence and email delivery
"""

from __future__ import annotations

import html
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)

_service_lock = threading.Lock()

# Default tax rate (can be overridden via env var RETAIL_TAX_RATE)
DEFAULT_TAX_RATE = 0.18  # 18% GST

# Quantity bounds for sanity checking
MIN_QUANTITY = 1
MAX_QUANTITY = 9999
MAX_UNIT_PRICE = 10_000_000.0  # 1 crore


def get_tax_rate() -> float:
    """Get the configured tax rate."""
    try:
        return float(os.getenv("RETAIL_TAX_RATE", str(DEFAULT_TAX_RATE)))
    except (ValueError, TypeError):
        return DEFAULT_TAX_RATE


def lookup_items(item_names: List[str]) -> List[Dict[str, Any]]:
    """Look up catalog items by name, returning matches with prices.

    Performs case-insensitive fuzzy search for each name.
    Returns a list of dicts with item info and a ``matched`` flag.

    Args:
        item_names: List of product name strings to search for.

    Returns:
        List of dicts, one per input name, with catalog match info.
    """
    from app.database.repositories import RetailCatalogRepository

    results: List[Dict[str, Any]] = []
    for name in item_names:
        matches = RetailCatalogRepository.search_by_name(name.strip())
        if matches:
            best = matches[0]  # first match (alphabetical, closest)
            results.append({
                "query": name,
                "matched": True,
                "item_id": best.id,
                "item_name": best.item_name,
                "sku": best.sku,
                "price": best.price,
                "category": best.category,
            })
        else:
            results.append({
                "query": name,
                "matched": False,
                "item_name": name,
                "price": 0.0,
                "reason": "Item not found in catalog",
            })
    return results


def lookup_item_by_sku(sku: str) -> Optional[Dict[str, Any]]:
    """Look up a single catalog item by SKU.

    Returns:
        Item dict or None.
    """
    from app.database.repositories import RetailCatalogRepository

    item = RetailCatalogRepository.get_by_sku(sku)
    if item:
        return item.to_dict()
    return None


def calculate_bill(
    items: List[Dict[str, Any]],
    tax_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate a bill from a list of line items.

    Each item dict should have at minimum:
        - ``name`` (str)
        - ``quantity`` (int)
        - ``price`` (float) — unit price

    Args:
        items: Line item dicts.
        tax_rate: Override tax rate (0‒1). Defaults to configured rate.

    Returns:
        Bill dict with ``line_items``, ``subtotal``, ``tax``, ``total``.
    """
    if tax_rate is None:
        tax_rate = get_tax_rate()
    tax_rate = max(0.0, min(1.0, float(tax_rate)))

    line_items: List[Dict[str, Any]] = []
    subtotal = 0.0

    for item in items:
        name = str(item.get("name", item.get("item_name", "Unknown Item")))[:200]
        try:
            quantity = int(item.get("quantity", 1))
        except (ValueError, TypeError):
            quantity = 1
        quantity = max(MIN_QUANTITY, min(MAX_QUANTITY, quantity))

        try:
            unit_price = float(item.get("price", 0.0))
        except (ValueError, TypeError):
            unit_price = 0.0
        unit_price = max(0.0, min(MAX_UNIT_PRICE, unit_price))

        line_total = round(unit_price * quantity, 2)

        line_items.append({
            "name": name,
            "quantity": quantity,
            "unit_price": unit_price,
            "line_total": line_total,
            "sku": str(item.get("sku", ""))[:50],
            "category": str(item.get("category", ""))[:100],
        })
        subtotal += line_total

    subtotal = round(subtotal, 2)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    return {
        "line_items": line_items,
        "subtotal": subtotal,
        "tax_rate": tax_rate,
        "tax": tax,
        "total": total,
        "item_count": sum(li["quantity"] for li in line_items),
        "unique_items": len(line_items),
        "currency": os.getenv("RETAIL_CURRENCY", "INR"),
        "generated_at": datetime.now().isoformat(),
    }


def generate_invoice_html(
    bill: Dict[str, Any],
    recipient: str = "",
    invoice_id: Optional[int] = None,
) -> str:
    """Render an HTML invoice from a bill dict.

    Uses the Jinja2 template at ``templates/invoice.html``.
    Falls back to a simple HTML string if the template is missing.

    Args:
        bill: Bill dict from :func:`calculate_bill`.
        recipient: Recipient email address.
        invoice_id: Optional invoice DB id.

    Returns:
        Rendered HTML string.
    """
    try:
        from jinja2 import Environment, FileSystemLoader

        template_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "templates",
        )
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("invoice.html")
        return template.render(
            bill=bill,
            recipient=recipient,
            invoice_id=invoice_id,
            generated_at=bill.get("generated_at", datetime.now().isoformat()),
        )
    except Exception as e:
        logger.warning("Template rendering failed, using fallback: %s", e)
        return _generate_invoice_html_fallback(bill, recipient, invoice_id)


def _generate_invoice_html_fallback(
    bill: Dict[str, Any],
    recipient: str,
    invoice_id: Optional[int],
) -> str:
    """Simple fallback HTML invoice when Jinja2 template is unavailable."""
    currency = html.escape(str(bill.get("currency", "INR")))
    rows = ""
    for li in bill.get("line_items", []):
        safe_name = html.escape(str(li.get("name", "")))
        safe_sku = html.escape(str(li.get("sku", "")))
        rows += (
            f"<tr>"
            f"<td>{safe_name}</td>"
            f"<td>{safe_sku}</td>"
            f"<td style='text-align:center'>{int(li.get('quantity', 0))}</td>"
            f"<td style='text-align:right'>{currency} {float(li.get('unit_price', 0)):.2f}</td>"
            f"<td style='text-align:right'>{currency} {float(li.get('line_total', 0)):.2f}</td>"
            f"</tr>"
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Invoice {invoice_id or ''}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; }}
th {{ background: #f5f5f5; }}
.totals td {{ font-weight: bold; }}
</style></head>
<body>
<h1>Invoice{f' #{invoice_id}' if invoice_id else ''}</h1>
<p><strong>Date:</strong> {bill.get('generated_at', '')}</p>
<p><strong>Recipient:</strong> {html.escape(recipient) if recipient else 'N/A'}</p>
<table>
<thead><tr><th>Item</th><th>SKU</th><th>Qty</th><th>Unit Price</th><th>Total</th></tr></thead>
<tbody>{rows}</tbody>
<tfoot>
<tr class="totals"><td colspan="4" style="text-align:right">Subtotal</td><td style="text-align:right">{currency} {bill['subtotal']:.2f}</td></tr>
<tr class="totals"><td colspan="4" style="text-align:right">Tax ({bill.get('tax_rate', 0) * 100:.0f}%)</td><td style="text-align:right">{currency} {bill['tax']:.2f}</td></tr>
<tr class="totals"><td colspan="4" style="text-align:right">Total</td><td style="text-align:right">{currency} {bill['total']:.2f}</td></tr>
</tfoot>
</table>
</body></html>"""


def save_invoice(
    bill: Dict[str, Any],
    recipient_email: str = "",
) -> Dict[str, Any]:
    """Persist an invoice to the database.

    Args:
        bill: Bill dict from :func:`calculate_bill`.
        recipient_email: Recipient email address.

    Returns:
        Dict with ``invoice_id`` and success status.
    """
    from app.database.repositories import InvoiceRepository

    try:
        invoice_id = InvoiceRepository.create(
            recipient_email=recipient_email,
            items_json=json.dumps(bill.get("line_items", [])),
            subtotal=bill["subtotal"],
            tax=bill["tax"],
            total=bill["total"],
            status="draft",
        )
        return {"success": True, "invoice_id": invoice_id}
    except Exception as e:
        logger.error("Failed to save invoice: %s", e)
        return {"success": False, "error": str(e)}


def save_and_send_invoice(
    bill: Dict[str, Any],
    recipient_email: str,
) -> Dict[str, Any]:
    """Save invoice to DB, render HTML, and email it.

    Args:
        bill: Bill dict from :func:`calculate_bill`.
        recipient_email: Email address to send the invoice to.

    Returns:
        Dict with success status, invoice_id, and email result.
    """
    from agents.email_tools import send_email

    # 1. Save to DB
    save_result = save_invoice(bill, recipient_email)
    if not save_result.get("success"):
        return save_result

    invoice_id = save_result["invoice_id"]

    # 2. Render HTML
    html = generate_invoice_html(bill, recipient_email, invoice_id)

    # 3. Send email
    try:
        currency = bill.get("currency", "INR")
        email_result = send_email({
            "to": [recipient_email],
            "subject": f"Invoice #{invoice_id} — {currency} {bill['total']:.2f}",
            "body": html,
        })

        # Mark as sent
        from app.database.repositories import InvoiceRepository
        InvoiceRepository.update_status(invoice_id, "sent")

        return {
            "success": True,
            "invoice_id": invoice_id,
            "email_result": email_result,
            "total": bill["total"],
        }
    except Exception as e:
        logger.error("Invoice email failed: %s", e)
        return {
            "success": True,
            "invoice_id": invoice_id,
            "email_error": str(e),
            "message": "Invoice saved but email delivery failed",
        }
