"""Retail billing domain tools for the MCP agent.

Provides thin-adapter tool handler functions used by the Retail MCP executor.
Each function validates its inputs, delegates to the service layer,
and returns a sanitized result dict.

Tools:
- scan_tray_items: Analyze current frame to identify items on a tray
- lookup_item_price: Look up item prices from the retail catalog DB
- create_bill: Assemble a bill from detected/looked-up items
- generate_invoice: Generate complete invoice (HTML + PDF) with automatic TTS
- send_invoice_email: Email the invoice to a recipient
"""

from __future__ import annotations

import json
import tempfile
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from agents.tools.validation import (
    validate_email as _validate_email,
    safe_error as _safe_error,
    clamp_quantity as _clamp_quantity,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _format_price_announcement(total: float, currency: str = "USD") -> str:
    """Format a price total as a natural spoken announcement.
    
    Parameters
    ----------
    total : float
        The total price to announce.
    currency : str
        Currency code (default: USD).
    
    Returns
    -------
    str
        A natural language announcement string.
    
    Examples
    --------
    >>> _format_price_announcement(13.81)
    'Your total is thirteen dollars and eighty-one cents.'
    >>> _format_price_announcement(100.00)
    'Your total is one hundred dollars.'
    """
    dollars = int(total)
    cents = round((total - dollars) * 100)
    
    # Convert numbers to words
    dollar_words = _number_to_words(dollars)
    
    if cents > 0:
        cent_words = _number_to_words(cents)
        return f"Your total is {dollar_words} dollars and {cent_words} cents."
    else:
        return f"Your total is {dollar_words} dollars."


def _number_to_words(n: int) -> str:
    """Convert a number (0-9999) to words.
    
    Parameters
    ----------
    n : int
        Number to convert.
    
    Returns
    -------
    str
        Number in words.
    """
    if n == 0:
        return "zero"
    
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
             "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    if n < 10:
        return ones[n]
    elif n < 20:
        return teens[n - 10]
    elif n < 100:
        return tens[n // 10] + ("-" + ones[n % 10] if n % 10 != 0 else "")
    elif n < 1000:
        hundreds = ones[n // 100] + " hundred"
        remainder = n % 100
        if remainder > 0:
            return hundreds + " " + _number_to_words(remainder)
        return hundreds
    elif n < 10000:
        thousands = ones[n // 1000] + " thousand"
        remainder = n % 1000
        if remainder > 0:
            return thousands + " " + _number_to_words(remainder)
        return thousands
    else:
        # For very large numbers, just use digits
        return str(n)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def tool_scan_tray_items(
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze the current camera frame to identify and count retail items.

    Sends a custom prompt to the VLM describing the retail billing task.
    The detailed scan prompt is always used; the user's query is appended
    as additional context so the VLM always returns structured item data.

    This tool handles structure extraction from VLM output but delegates
    validation to the service layer.
    """
    logger.info("tool_scan_tray_items invoked (query=%s)", query and query[:80])

    from services.core.monitoring import get_monitoring_service
    from agents.vlm.task_types import TaskType

    _scan_prompt = (
        "You are a retail item scanner. Look at this image carefully and identify EVERY "
        "distinct item visible on the tray, counter, or surface for retail billing.\n\n"
        "For each item provide: name, category "
        "(electronics|food|beverage|household|clothing|stationery|other), quantity, brand "
        "(if visible), variant (size/color if distinguishable), visible_price (if any), "
        "and barcode_visible (boolean).\n\n"
        "CRITICAL: You MUST return item data in the details.items array. Every visible "
        "product, package, snack, drink, or object that could be sold MUST be listed. "
        "Do NOT return an empty items list if you can see products in the image.\n\n"
        "Return your response with details structured EXACTLY like this:\n"
        '"details": {\n'
        '  "items": [\n'
        '    {"name": "item name", "category": "food", "quantity": 1, "brand": "brand or unknown", '
        '"variant": "", "visible_price": null, "barcode_visible": false}\n'
        '  ],\n'
        '  "total_item_count": <number>,\n'
        '  "total_unique_items": <number>\n'
        '}'
    )

    # Append user context if provided, but always keep the structured scan prompt
    if query:
        effective_prompt = f"{_scan_prompt}\n\nAdditional context from user: {query}"
    else:
        effective_prompt = _scan_prompt

    service = get_monitoring_service()
    if not service:
        return {"success": False, "message": "Monitoring service not available"}

    try:
        event = service.analyze_single_frame(
            task_type=TaskType.CUSTOM,
            custom_prompt=effective_prompt,
        )
    except Exception as exc:
        return _safe_error("VLM tray scan failed", exc=exc)

    if not event:
        return {
            "success": False,
            "message": service.last_error or "Tray scan failed — no frame available",
        }

    # Extract structured items from VLM response
    # This is adapter logic: transform external format (VLM JSON) to internal format
    items: List[Dict[str, Any]] = []
    total_count = 0

    # Try 1: parse from full_response (complete VLM output)
    parsed: Optional[Dict[str, Any]] = None
    if event.full_response and isinstance(event.full_response, str):
        try:
            parsed = json.loads(event.full_response)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("VLM full_response was not valid JSON: %s", exc)
            # Try to extract JSON from mixed text (thinking tokens, markdown, etc.)
            parsed = _extract_json_from_text(event.full_response)

    # Try 2: fall back to already-parsed details in decision_trace
    if not isinstance(parsed, dict):
        dt = getattr(event, "decision_trace", None) or {}
        if isinstance(dt.get("details"), dict):
            parsed = dt["details"]
            logger.info("Using decision_trace.details as fallback for item extraction")

    if isinstance(parsed, dict):
        raw_items = _extract_items_from_parsed(parsed)

        # Basic quantity normalization (not validation — just type safety)
        for it in raw_items:
            if "quantity" in it:
                it["quantity"] = _clamp_quantity(it.get("quantity", 1))

        items = raw_items
        total_count = _extract_total_count(parsed, len(items))

    # Last resort: if the VLM described items in reasoning but returned no
    # structured items, try to parse item names from the description text
    if not items and event.vision_description:
        items = _extract_items_from_description(event.vision_description)
        total_count = len(items)
        if items:
            logger.info(
                "Extracted %d item(s) from VLM description text as fallback",
                len(items),
            )

    return {
        "success": True,
        "message": f"Tray scanned: {total_count} item(s) detected",
        "data": {
            "detected": event.detected,
            "confidence": event.confidence,
            "description": event.vision_description,
            "items": items,
            "total_item_count": total_count,
            "full_response": event.full_response,
        },
    }


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from mixed text (thinking tokens, markdown, etc.)."""
    import re
    # Strip thinking tokens and markdown fences
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'```json\s*', '', cleaned)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()

    start = cleaned.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(cleaned[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _extract_items_from_parsed(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Search for items list in various locations within parsed JSON.

    The VLM may place items at different nesting levels depending on how
    it interprets the CUSTOM_QUERY_TEMPLATE:
    - parsed["items"]
    - parsed["details"]["items"]
    - parsed["details"]["details"]["items"]  (double-nested via client.py stripping)
    - parsed["detected_items"]
    - parsed["products"]
    - parsed["objects"]
    """
    # Direct top-level
    for key in ("items", "detected_items", "products", "objects", "scanned_items"):
        candidate = parsed.get(key)
        if isinstance(candidate, list) and candidate:
            return candidate

    # One level deep under "details"
    details = parsed.get("details")
    if isinstance(details, dict):
        for key in ("items", "detected_items", "products", "objects", "scanned_items"):
            candidate = details.get(key)
            if isinstance(candidate, list) and candidate:
                return candidate

        # Double-nested (client.py strips top-level keys into details)
        inner_details = details.get("details")
        if isinstance(inner_details, dict):
            for key in ("items", "detected_items", "products", "objects", "scanned_items"):
                candidate = inner_details.get(key)
                if isinstance(candidate, list) and candidate:
                    return candidate

    # Walk all dict values looking for a list of dicts with 'name' keys
    for value in parsed.values():
        if isinstance(value, list) and value and isinstance(value[0], dict) and "name" in value[0]:
            return value
        if isinstance(value, dict):
            for inner_value in value.values():
                if isinstance(inner_value, list) and inner_value and isinstance(inner_value[0], dict) and "name" in inner_value[0]:
                    return inner_value

    return []


def _extract_total_count(parsed: Dict[str, Any], fallback: int) -> int:
    """Extract total_item_count from various locations in parsed response."""
    for key in ("total_item_count", "total_items", "count"):
        val = parsed.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)

    details = parsed.get("details")
    if isinstance(details, dict):
        for key in ("total_item_count", "total_items", "count"):
            val = details.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return int(val)

    return fallback


def _extract_items_from_description(description: str) -> List[Dict[str, Any]]:
    """Last-resort: extract item names from VLM reasoning/description text.

    When the VLM describes items in natural language but fails to return
    structured data, this attempts to salvage item names.
    """
    import re

    items: List[Dict[str, Any]] = []
    # Common patterns: "a chocolate bar", "snack packet", etc.
    # Look for noun phrases after common article/quantity patterns
    # This is deliberately generous — false positives are preferable to missing items
    patterns = [
        r'(?:a|an|one|two|three|four|five|\d+)\s+([a-zA-Z][a-zA-Z\s]{2,30}?)(?:\.|,|;|\s+and\s|$)',
    ]

    # Also split on comma-separated lists of items
    # e.g. "including a snack packet, a chocolate bar, a marker, and a granola bar"
    # or   "several items: an orange packet of YOGGIES candy, a chocolate bar, ..."
    including_match = re.search(
        r'(?:including|contains?|shows?|with|items[:\s]|are|visible:?|has|have|found|identified|see|sees)\s+(.+?)(?:\.\s|$)',
        description,
        re.IGNORECASE,
    )
    if including_match:
        items_text = including_match.group(1)
        # Split by commas and "and"
        parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', items_text)
        for part in parts:
            part = part.strip()
            # Remove leading articles and adjective-heavy prefixes
            part = re.sub(r'^(?:a|an|the|one|two|three|four|five|\d+)\s+', '', part, flags=re.IGNORECASE)
            # Remove common visual adjective prefixes (e.g. "partially visible", "orange packet of")
            part = re.sub(r'^(?:partially\s+visible\s+)', '', part, flags=re.IGNORECASE)
            # Strip colour / packaging adjectives before "of <product>" patterns
            of_match = re.search(r'\bof\s+(.+)', part, re.IGNORECASE)
            if of_match and len(of_match.group(1).strip()) > 2:
                part = of_match.group(1).strip()
            part = part.strip()
            if part and len(part) > 1 and len(part) < 60:
                items.append({
                    "name": part,
                    "category": "other",
                    "quantity": 1,
                    "brand": "unknown",
                    "source": "description_extraction",
                })

    # Deduplicate by name
    seen = set()
    deduped = []
    for it in items:
        key = it["name"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(it)

    return deduped


def tool_lookup_item_price(
    item_name: str = "",
    sku: str = "",
) -> Dict[str, Any]:
    """Look up an item's price from the retail catalog database.

    Thin adapter over service lookup methods with input sanitization.
    """
    logger.info("tool_lookup_item_price invoked (item_name=%s, sku=%s)", item_name, sku)

    from services.domains.retail.service import lookup_items, lookup_item_by_sku

    try:
        if sku:
            result = lookup_item_by_sku(sku.strip()[:50])
            if result:
                return {
                    "success": True,
                    "message": f"Found: {result['item_name']} — {result['price']}",
                    "data": result,
                }
            return {"success": False, "message": f"No item found with SKU: {sku}"}

        if item_name:
            results = lookup_items([item_name.strip()[:200]])
            if results and results[0].get("matched"):
                match = results[0]
                return {
                    "success": True,
                    "message": f"Found: {match['item_name']} — {match['price']}",
                    "data": match,
                }
            return {"success": False, "message": f"No catalog match for: {item_name}"}

        return {"success": False, "message": "Provide either item_name or sku to look up"}

    except Exception as exc:
        return _safe_error("Item lookup failed", exc=exc)


def tool_create_bill(
    items: Optional[List[Dict[str, Any]]] = None,
    scan_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a bill from a list of items or a previous scan result.

    Thin adapter that delegates all business logic to the service layer.

    Parameters
    ----------
    items : list of dict, optional
        Explicit items with name, quantity, price.
    scan_result : dict, optional
        Scan data from tool_scan_tray_items.

    Returns
    -------
    dict
        Bill with line_items, totals, warnings (if any).
    """
    logger.info(
        "tool_create_bill invoked (explicit_items=%s, has_scan=%s)",
        bool(items), bool(scan_result),
    )

    from services.domains.retail.bill_service import (
        create_bill_from_scan,
        create_bill_from_items,
    )

    try:
        # Path A: Create from scan result
        if scan_result:
            return create_bill_from_scan(scan_result)

        # Path B: Create from explicit items
        if items:
            return create_bill_from_items(items)

        # No input provided
        return {
            "success": False,
            "message": "No items provided. Run scan_tray_items first or provide items list.",
        }

    except Exception as exc:
        return _safe_error("Bill creation failed", exc=exc)


def tool_generate_invoice(
    bill: Optional[Dict[str, Any]] = None,
    recipient_email: str = "",
    output_path: str = "",
) -> Dict[str, Any]:
    """Generate a complete invoice (HTML + PDF) with automatic TTS announcement.

    This tool automatically:
    1. Saves the invoice to the database
    2. Generates an HTML version
    3. Generates a PDF version
    4. Announces the total via text-to-speech

    Parameters
    ----------
    bill : dict
        Bill dict containing line items, totals, tax, etc.
    recipient_email : str, optional
        Recipient email or name to display on invoice.
    output_path : str, optional
        Custom output path for the PDF. If not provided, a timestamped
        filename will be generated.

    Returns
    -------
    dict
        Result with invoice_id, html, pdf_path, total, and tts status.
    """
    logger.info("tool_generate_invoice invoked (has_bill=%s, email=%s)", bool(bill), recipient_email)

    from services.domains.retail.service import (
        generate_invoice_html,
        generate_invoice_pdf,
        save_invoice,
    )

    if not bill:
        return {"success": False, "message": "No bill data provided. Run create_bill first."}

    # Validate email if provided
    if recipient_email:
        err = _validate_email(recipient_email)
        if err:
            return {"success": False, "message": err}

    try:
        # Save invoice to database
        save_result = save_invoice(bill, recipient_email)
        if not save_result.get("success"):
            return {"success": False, "message": "Failed to save invoice"}

        invoice_id = save_result["invoice_id"]

        # Generate HTML version
        html = generate_invoice_html(bill, recipient_email, invoice_id)

        # Generate PDF version
        pdf_result = generate_invoice_pdf(
            bill=bill,
            recipient=recipient_email,
            invoice_id=invoice_id,
            output_path=output_path if output_path else None,
        )

        pdf_path = pdf_result.get("data", {}).get("pdf_path", "") if pdf_result.get("success") else None

        # Update invoice record with PDF path
        if pdf_path and invoice_id:
            try:
                from app.database.repositories import InvoiceRepository
                InvoiceRepository.update_pdf_path(invoice_id, pdf_path)
                logger.info("Updated invoice #%d with PDF path: %s", invoice_id, pdf_path)
            except Exception as e:
                logger.warning("Failed to update invoice PDF path: %s", e)

        # Announce the total price via TTS
        total = bill.get("total", 0)
        currency = bill.get("currency", "USD")
        announcement_text = _format_price_announcement(total, currency)
        
        logger.info("Announcing invoice total: %.2f %s", total, currency)
        tts_result = _speak_invoice(announcement_text)
        
        if tts_result.get("success"):
            logger.info("TTS announcement completed successfully")
        else:
            logger.warning("TTS announcement failed: %s", tts_result.get("message"))

        # Build detailed status message
        status_parts = [f"✅ Invoice #{invoice_id} generated successfully."]
        status_parts.append(f"**Total: {currency} {total:.2f}**")
        
        if pdf_path:
            status_parts.append(f"📄 PDF saved to: `{pdf_path}`")
        else:
            status_parts.append("⚠️ PDF generation failed")
        
        if tts_result.get("success"):
            status_parts.append("🔊 Price announced via speaker")
        else:
            status_parts.append(f"⚠️ TTS announcement failed: {tts_result.get('message', 'Unknown error')}")
        
        if recipient_email:
            status_parts.append(f"📧 For: {recipient_email}")

        return {
            "success": True,
            "message": "\\n\\n".join(status_parts),
            "data": {
                "invoice_id": invoice_id,
                "html": html,
                "pdf_path": pdf_path,
                "total": total,
                "recipient": recipient_email,
                "tts_announced": tts_result.get("success", False),
            },
        }

    except Exception as exc:
        return _safe_error("Invoice generation failed", exc=exc)


def tool_send_invoice_email(
    recipient_email: str = "",
    bill: Optional[Dict[str, Any]] = None,
    invoice_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Send an invoice via email.

    Thin adapter that delegates to service layer workflows.

    Parameters
    ----------
    recipient_email : str
        Email address to send to.
    bill : dict, optional
        If provided, a new invoice is created, saved, and emailed.
    invoice_id : int, optional
        If provided, the existing invoice is fetched and emailed.

    Returns
    -------
    dict
        Result with success, message, email_result.
    """
    logger.info(
        "tool_send_invoice_email invoked (email=%s, has_bill=%s, invoice_id=%s)",
        recipient_email, bool(bill), invoice_id,
    )

    # Validate recipient
    err = _validate_email(recipient_email)
    if err:
        return {"success": False, "message": err}

    from services.domains.retail.service import save_and_send_invoice
    from services.domains.retail.invoice_service import resend_invoice

    try:
        # Path A: Create new invoice from bill and send
        if bill:
            return save_and_send_invoice(bill, recipient_email)

        # Path B: Resend existing invoice
        if invoice_id:
            return resend_invoice(invoice_id, recipient_email)

        return {"success": False, "message": "Provide either a bill or an invoice_id to send"}

    except Exception as exc:
        return _safe_error("Invoice email delivery failed", exc=exc)


def _speak_invoice(text: str = "") -> Dict[str, Any]:
    """Convert text to spoken audio and play it back (internal helper).

    Uses Google Text-to-Speech (gTTS) and the infrastructure audio player.
    This is an internal helper function used by invoice generation.

    Parameters
    ----------
    text : str
        A spoken-friendly invoice summary.

    Returns
    -------
    dict
        Standard result dict with audio_file path if successful.
    """
    logger.info("_speak_invoice invoked (text_len=%d)", len(text))

    if not text or not text.strip():
        return {"success": False, "message": "No text provided for speech synthesis."}

    # Enforce max length
    max_length = 2000
    if len(text) > max_length:
        text = text[:max_length - 3] + "..."

    try:
        from gtts import gTTS
    except ImportError:
        return {
            "success": False,
            "message": "TTS dependency 'gtts' is not installed. Run: pip install gtts",
        }

    audio_path: Optional[str] = None
    try:
        tts = gTTS(text=text, lang="en", slow=False)

        # Write to a temporary mp3 file
        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp3", prefix="invoice_tts_", delete=False,
        )
        audio_path = tmp.name
        tmp.close()
        tts.save(audio_path)
        logger.info("TTS audio saved to %s", audio_path)

        # Use infrastructure service for playback
        import os
        from services.infrastructure.audio import AudioPlayer
        
        speaker_device = os.getenv("SPEAKER_DEVICE", "default")
        logger.info("Playing TTS audio on speaker device: %s", speaker_device)
        
        audio_player = AudioPlayer()
        playback_started = audio_player.play_file(audio_path)
        
        if not playback_started:
            logger.warning("Audio playback could not be started, but file saved to: %s", audio_path)

        return {
            "success": True,
            "message": "Invoice audio summary played successfully.",
            "data": {
                "audio_file": audio_path,
                "text_length": len(text),
                "speaker_device": speaker_device,
                "playback_started": playback_started,
            },
        }

    except Exception as exc:
        logger.error("TTS synthesis/playback failed: %s", exc, exc_info=True)
        return _safe_error("Text-to-speech failed", exc=exc)


# Deprecated: tool_generate_invoice_pdf has been merged into tool_generate_invoice
# The tool_generate_invoice now automatically generates both HTML and PDF invoices
# with TTS announcement. This function is kept for backward compatibility only.
def tool_generate_invoice_pdf(
    bill: Optional[Dict[str, Any]] = None,
    recipient_email: str = "",
    output_path: str = "",
) -> Dict[str, Any]:
    """[DEPRECATED] Generate a PDF invoice - use tool_generate_invoice instead.

    This function has been deprecated. Use tool_generate_invoice() which now
    automatically generates both HTML and PDF invoices with TTS announcement.
    """
    logger.warning(
        "tool_generate_invoice_pdf is deprecated. Use tool_generate_invoice instead."
    )
    return tool_generate_invoice(bill=bill, recipient_email=recipient_email, output_path=output_path)
