#!/usr/bin/env python3
"""Test script to generate a professional PDF invoice.

This script demonstrates the complete invoice generation workflow:
1. Create a sample bill with items, quantities, prices
2. Calculate totals with tax
3. Generate a professional PDF invoice

Run with: source venv/bin/activate && python test_invoice_generation.py
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.domains.retail.service import calculate_bill, generate_invoice_pdf


def create_sample_bill():
    """Create a sample bill with typical retail items."""
    items = [
        {
            "name": "Organic Bananas",
            "quantity": 3,
            "price": 1.99,
            "sku": "PRD-001",
            "category": "produce",
        },
        {
            "name": "Whole Wheat Bread",
            "quantity": 2,
            "price": 3.49,
            "sku": "BKY-042",
            "category": "bakery",
        },
        {
            "name": "Greek Yogurt (32oz)",
            "quantity": 1,
            "price": 5.99,
            "sku": "DRY-103",
            "category": "dairy",
        },
        {
            "name": "Mixed Salad Greens",
            "quantity": 2,
            "price": 4.29,
            "sku": "PRD-087",
            "category": "produce",
        },
        {
            "name": "Orange Juice (64oz)",
            "quantity": 1,
            "price": 6.49,
            "sku": "BEV-201",
            "category": "beverage",
        },
    ]
    
    # Calculate with 8% tax
    bill = calculate_bill(items, tax_rate=0.08)
    return bill


def main():
    """Generate a professional PDF invoice."""
    print("=" * 70)
    print("INVOICE GENERATION TEST")
    print("=" * 70)
    print()
    
    # Create sample bill
    print("1. Creating sample bill...")
    bill = create_sample_bill()
    
    print(f"   ✓ {bill['unique_items']} unique items")
    print(f"   ✓ {bill['item_count']} total items")
    print(f"   ✓ Subtotal: ${bill['subtotal']:.2f} {bill['currency']}")
    print(f"   ✓ Tax ({bill['tax_rate']*100:.0f}%): ${bill['tax']:.2f} {bill['currency']}")
    print(f"   ✓ Total: ${bill['total']:.2f} {bill['currency']}")
    print()
    
    # Generate PDF
    print("2. Generating professional PDF invoice...")
    recipient = "customer@example.com"
    invoice_id = 12345  # Sample invoice ID
    
    result = generate_invoice_pdf(
        bill=bill,
        recipient=recipient,
        invoice_id=invoice_id,
    )
    
    if result.get("success"):
        data = result["data"]
        print(f"   ✓ PDF generated successfully!")
        print()
        print("   Invoice Details:")
        print(f"   - File path: {data['pdf_path']}")
        print(f"   - File size: {data['file_size']:,} bytes")
        print(f"   - Invoice ID: {data['invoice_id']}")
        print(f"   - Total: ${data['total']:.2f} {data['currency']}")
        print(f"   - Items: {data['item_count']}")
        print(f"   - Recipient: {data['recipient']}")
        print()
        print("=" * 70)
        print(f"✓ SUCCESS: Invoice PDF ready at:")
        print(f"  {data['pdf_path']}")
        print("=" * 70)
        return 0
    else:
        print(f"   ✗ PDF generation failed: {result.get('message')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
