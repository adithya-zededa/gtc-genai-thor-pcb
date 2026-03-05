"""Save-evidence tool implementation."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from core.logging import get_logger

logger = get_logger(__name__)


def tool_save_evidence(
    image_data: bytes,
    label: str = "detection",
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Save detection image as evidence."""
    evidence_dir = Path(os.getenv("DETECTED_IMAGES_DIR", "detected_images"))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
    filename = f"{safe_label}_{timestamp}.jpg"
    filepath = evidence_dir / filename

    try:
        with open(filepath, "wb") as f:
            f.write(image_data)

        if metadata:
            meta_path = filepath.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "label": label,
                    **metadata
                }, f, indent=2)

        return {
            "success": True,
            "message": f"Evidence saved as {filename}",
            "data": {
                "filepath": str(filepath),
                "filename": filename,
            },
        }
    except Exception as e:
        logger.error("Failed to save evidence: %s", e)
        return {"success": False, "message": f"Failed to save evidence: {e}"}
