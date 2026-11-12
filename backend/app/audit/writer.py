"""
Append-only audit writer and immutable audit event models (Section 21.2 & 21.3).

Enforces append-only semantics: historical audit entries can never be updated,
deleted, or truncated. Corrections must append a new corrective event.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import threading
from typing import Any
from uuid import UUID, uuid4

from backend.app.audit.redaction import redact_mapping


class AuditSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AppendOnlyAuditError(PermissionError):
    """Raised when any code attempts to UPDATE, DELETE, or TRUNCATE audit records (Section 21.2)."""


@dataclass(frozen=True)
class AuditEvent:
    """Immutable append-only audit event (Section 21.3)."""
    event_id: UUID
    session_id: UUID
    incident_id: UUID
    event_type: str
    severity: AuditSeverity
    timestamp: datetime
    outcome: str
    error_code: str | None
    reason_code: str | None
    provenance_refs: tuple[str, ...]
    details: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        *,
        session_id: UUID,
        incident_id: UUID,
        event_type: str,
        severity: AuditSeverity,
        outcome: str,
        now: datetime,
        error_code: str | None = None,
        reason_code: str | None = None,
        provenance_refs: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> "AuditEvent":
        """Create an immutable AuditEvent with automatic PII redaction on details."""
        safe_details = redact_mapping(details) if details else {}
        return cls(
            event_id=uuid4(),
            session_id=session_id,
            incident_id=incident_id,
            event_type=event_type,
            severity=severity,
            timestamp=now,
            outcome=outcome,
            error_code=error_code,
            reason_code=reason_code,
            provenance_refs=tuple(provenance_refs),
            details=safe_details,
        )


class InMemoryAuditWriter:
    """
    Append-only audit store enforcing immutable audit semantics (Section 21.2).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> AuditEvent:
        """Append a single immutable audit event."""
        with self._lock:
            self._events.append(event)
            return event

    def list_events(self, *, incident_id: UUID | None = None) -> list[AuditEvent]:
        """List audit events in chronological append order."""
        with self._lock:
            if incident_id is None:
                return list(self._events)
            return [e for e in self._events if e.incident_id == incident_id]

    def update_event(self, event_id: UUID, **_: Any) -> None:
        """Section 21.2: Rejects UPDATE on audit_events."""
        raise AppendOnlyAuditError(
            f"Append-only violation: UPDATE on audit event {event_id} is strictly prohibited."
        )

    def delete_event(self, event_id: UUID) -> None:
        """Section 21.2: Rejects DELETE on audit_events."""
        raise AppendOnlyAuditError(
            f"Append-only violation: DELETE on audit event {event_id} is strictly prohibited."
        )

    def truncate(self) -> None:
        """Section 21.2: Rejects TRUNCATE on audit_events."""
        raise AppendOnlyAuditError(
            "Append-only violation: TRUNCATE on audit_events is strictly prohibited."
        )
