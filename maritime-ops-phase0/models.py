"""
Pydantic models for maritime email classification.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class UrgencyLevel(str, Enum):
    """Email urgency classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


class ClassificationSource(str, Enum):
    """Source of classification decision."""
    RULES = "RULES"
    LLM = "LLM"


class AttachmentInfo(BaseModel):
    """Metadata about an email attachment."""
    filename: str
    content_type: str
    size_bytes: int

    @field_validator('size_bytes')
    @classmethod
    def validate_size(cls, v):
        if v < 0:
            raise ValueError('size_bytes must be non-negative')
        return v


class ParsedEmail(BaseModel):
    """Parsed email data."""
    message_id: Optional[str] = None
    subject: str
    sender: str
    recipients: List[str] = Field(default_factory=list)
    received_at: Optional[datetime] = None
    body_text: Optional[str] = None
    body_html: Optional[str] = None
    attachments: List[AttachmentInfo] = Field(default_factory=list)
    raw_path: Path

    class Config:
        arbitrary_types_allowed = True

    @field_validator('subject', 'sender')
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()


class ClassificationResult(BaseModel):
    """Result of email classification."""
    vessel_name: Optional[str] = None
    category: str
    subcategory: Optional[str] = None
    port: Optional[str] = None
    dates_mentioned: List[str] = Field(default_factory=list)
    urgency: UrgencyLevel
    summary: str
    action_required: bool
    confidence: float = Field(ge=0.0, le=1.0)
    source: ClassificationSource

    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        if not v or not v.strip():
            raise ValueError('Category cannot be empty')
        return v.strip().upper()

    @field_validator('subcategory', mode='before')
    @classmethod
    def validate_subcategory(cls, v):
        if v is not None and isinstance(v, str):
            v = v.strip().upper()
            if not v:
                return None
        return v

    @field_validator('summary')
    @classmethod
    def validate_summary(cls, v):
        if not v or not v.strip():
            raise ValueError('Summary cannot be empty')
        return v.strip()


class ProcessedEmail(BaseModel):
    """Complete processed email with classification."""
    parsed_email: ParsedEmail
    classification: Optional[ClassificationResult] = None
    errors: List[str] = Field(default_factory=list)
    processing_time_seconds: Optional[float] = None

    @property
    def success(self) -> bool:
        """Check if processing was successful."""
        return self.classification is not None and len(self.errors) == 0

    def to_summary_dict(self) -> dict:
        """Generate a summary dictionary for reporting."""
        return {
            'file': self.parsed_email.raw_path.name,
            'subject': self.parsed_email.subject,
            'category': self.classification.category if self.classification else 'ERROR',
            'subcategory': self.classification.subcategory if self.classification else None,
            'vessel': self.classification.vessel_name if self.classification else None,
            'urgency': self.classification.urgency.value if self.classification else None,
            'source': self.classification.source.value if self.classification else None,
            'confidence': self.classification.confidence if self.classification else 0.0,
            'success': self.success,
            'errors': self.errors
        }
