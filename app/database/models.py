"""Database model dataclasses representing table structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class User:
    """User model representing a system user."""
    id: Optional[int] = None
    email: str = ""
    name: str = ""
    role: str = "user"
    active: bool = True
    created_at: Optional[str] = None
    
    @classmethod
    def from_row(cls, row) -> "User":
        """Create User from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            email=row["email"],
            name=row["name"],
            role=row["role"],
            active=bool(row["active"]),
            created_at=row["created_at"],
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "active": self.active,
            "created_at": self.created_at,
        }


@dataclass
class DetectionLog:
    """Detection log model representing a detection event."""
    id: Optional[int] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None
    response: str = ""
    image_path: str = ""
    frame_number: Optional[int] = None
    reason: str = ""
    vision_description: str = ""
    decision_details: Dict[str, Any] = field(default_factory=dict)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def from_row(cls, row) -> "DetectionLog":
        """Create DetectionLog from database row."""
        if row is None:
            return None
        
        import json
        
        decision_details = {}
        if row["decision_details"]:
            try:
                decision_details = json.loads(row["decision_details"])
            except json.JSONDecodeError:
                decision_details = {"raw": row["decision_details"]}
        
        tool_trace = []
        try:
            tool_trace_raw = row["tool_trace"] if "tool_trace" in row.keys() else None
            if tool_trace_raw:
                tool_trace = json.loads(tool_trace_raw)
        except (json.JSONDecodeError, KeyError):
            tool_trace = []
        
        return cls(
            id=row["id"],
            timestamp=row["timestamp"],
            confidence=row["confidence"],
            response=row["response"],
            image_path=row["image_path"],
            frame_number=row["frame_number"],
            reason=row["reason"],
            vision_description=row["vision_description"],
            decision_details=decision_details,
            tool_trace=tool_trace,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        is_agentic = self.decision_details.get("classification") == "AGENTIC_ANALYSIS"
        
        # Determine detected status
        if "detected" in self.decision_details:
            detected = self.decision_details.get("detected", False)
        else:
            detected = self.confidence is not None and self.confidence > 0
        
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "response": self.response,
            "image_path": self.image_path,
            "frame_number": self.frame_number,
            "reason": self.reason,
            "vision_description": self.vision_description,
            "decision_details": self.decision_details,
            "tool_trace": self.tool_trace,
            "agentic_mode": is_agentic,
            "detected": detected,
        }


@dataclass
class ConfigHistory:
    """Configuration history model for tracking config changes."""
    id: Optional[int] = None
    timestamp: Optional[str] = None
    config_type: str = ""
    changes: str = ""
    user_email: str = ""
    
    @classmethod
    def from_row(cls, row) -> "ConfigHistory":
        """Create ConfigHistory from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            timestamp=row["timestamp"],
            config_type=row["config_type"],
            changes=row["changes"],
            user_email=row["user_email"],
        )


@dataclass
class LogSettings:
    """Log settings model for logging configuration."""
    log_level: str = "INFO"
    log_retention: int = 30
    max_log_size: int = 100
    log_to_file: bool = True
    log_to_console: bool = True
    log_database: bool = False
    
    @classmethod
    def from_row(cls, row) -> "LogSettings":
        """Create LogSettings from database row."""
        if row is None:
            return cls()  # Return defaults
        return cls(
            log_level=(row["log_level"] or "INFO").upper(),
            log_retention=int(row["log_retention"]) if row["log_retention"] is not None else 30,
            max_log_size=int(row["max_log_size"]) if row["max_log_size"] is not None else 100,
            log_to_file=bool(row["log_to_file"]),
            log_to_console=bool(row["log_to_console"]),
            log_database=bool(row["log_database"]),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "log_level": self.log_level,
            "log_retention": self.log_retention,
            "max_log_size": self.max_log_size,
            "log_to_file": self.log_to_file,
            "log_to_console": self.log_to_console,
            "log_database": self.log_database,
        }


@dataclass
class RetailCatalogItem:
    """Retail catalog item representing a product in the store catalog."""
    id: Optional[int] = None
    item_name: str = ""
    sku: str = ""
    price: float = 0.0
    category: str = "other"
    created_at: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> Optional["RetailCatalogItem"]:
        """Create RetailCatalogItem from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            item_name=row["item_name"],
            sku=row["sku"],
            price=float(row["price"]),
            category=row["category"],
            created_at=row["created_at"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "item_name": self.item_name,
            "sku": self.sku,
            "price": self.price,
            "category": self.category,
            "created_at": self.created_at,
        }


@dataclass
class Invoice:
    """Invoice model representing a generated retail invoice."""
    id: Optional[int] = None
    timestamp: Optional[str] = None
    recipient_email: str = ""
    items_json: str = "[]"
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    status: str = "draft"
    pdf_path: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> Optional["Invoice"]:
        """Create Invoice from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            timestamp=row["timestamp"],
            recipient_email=row["recipient_email"],
            items_json=row["items_json"],
            subtotal=float(row["subtotal"]),
            tax=float(row["tax"]),
            total=float(row["total"]),
            status=row["status"],
            pdf_path=row.get("pdf_path"),
            created_at=row["created_at"],
        )

    @property
    def items(self) -> List[Dict[str, Any]]:
        """Parse items from JSON string."""
        import json
        try:
            return json.loads(self.items_json)
        except (json.JSONDecodeError, TypeError):
            return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "recipient_email": self.recipient_email,
            "items": self.items,
            "subtotal": self.subtotal,
            "tax": self.tax,
            "total": self.total,
            "status": self.status,
            "pdf_path": self.pdf_path,
            "created_at": self.created_at,
        }


@dataclass
class PCBDefect:
    """PCB defect record from inspection analysis."""
    id: Optional[int] = None
    timestamp: Optional[str] = None
    board_type: str = "unknown"
    defect_type: str = ""
    severity: str = "low"
    confidence: float = 0.0
    image_path: str = ""
    description: str = ""
    created_at: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> Optional["PCBDefect"]:
        """Create PCBDefect from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            timestamp=row["timestamp"],
            board_type=row["board_type"],
            defect_type=row["defect_type"],
            severity=row["severity"],
            confidence=float(row["confidence"]),
            image_path=row["image_path"],
            description=row["description"],
            created_at=row["created_at"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "board_type": self.board_type,
            "defect_type": self.defect_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "image_path": self.image_path,
            "description": self.description,
            "created_at": self.created_at,
        }
