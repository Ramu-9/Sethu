"""
PII redaction utilities for persistent audit records (Section 21.3 & 22).

Redacts Indian phone numbers, Aadhaar numbers, PAN numbers, and email addresses
before writing any metadata to the append-only audit log.
"""

from collections.abc import Mapping
import re
from typing import Any


_AADHAAR_PATTERN = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_INDIAN_PHONE_PATTERN = re.compile(r"(?:\+91[\s-]?)?[6-9]\d{9}\b")
_PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def redact_pii(text: str) -> str:
    """Redact Aadhaar, Indian phone numbers, PAN numbers, and emails from text."""
    result = _AADHAAR_PATTERN.sub("[REDACTED_AADHAAR]", text)
    result = _INDIAN_PHONE_PATTERN.sub("[REDACTED_PHONE]", result)
    result = _PAN_PATTERN.sub("[REDACTED_PAN]", result)
    result = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", result)
    return result


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively redact PII strings inside a dictionary."""
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            cleaned[key] = redact_pii(value)
        elif isinstance(value, Mapping):
            cleaned[key] = redact_mapping(value)
        elif isinstance(value, (list, tuple)):
            cleaned[key] = [
                redact_pii(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned
