"""
Possible duplicate-incident detection (Section 14).

Flags a new incident as a possible duplicate when ALL conditions hold:
1. Same normalised emergency type (excluding 'unknown').
2. Confirmed incident coordinates within 200 metres.
3. Incident creation timestamps within 5 minutes (300 seconds).

Runs synchronously only when BOTH a supported category above threshold and a
confirmed location exist, and re-runs idempotently on category or location updates.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from backend.app.adapters.base import EmergencyCategory
from backend.app.orchestrator.dispatch import geodesic_distance_meters


DUPLICATE_RADIUS_METERS: float = 200.0
DUPLICATE_TIME_WINDOW_SECONDS: float = 300.0


@dataclass(frozen=True)
class IncidentSummaryForDuplicateCheck:
    """Minimal incident projection required for duplicate evaluation (Section 14)."""
    incident_id: UUID
    category: EmergencyCategory
    category_above_threshold: bool
    confirmed_latitude: float | None
    confirmed_longitude: float | None
    is_location_confirmed: bool
    created_at: datetime

    def is_eligible_for_duplicate_check(self) -> bool:
        """
        Section 14: Incidents whose category is 'unknown' or lack a confirmed location
        are not compared for duplicate detection.
        """
        return (
            self.category != EmergencyCategory.UNKNOWN
            and self.category_above_threshold
            and self.is_location_confirmed
            and self.confirmed_latitude is not None
            and self.confirmed_longitude is not None
        )


@dataclass(frozen=True)
class DuplicateMatchResult:
    """Result when a possible duplicate incident is detected (Section 14)."""
    possible_duplicate: bool
    incident_id: UUID
    matched_incident_id: UUID
    distance_meters: float
    time_difference_seconds: float
    match_rule: str = "same_category_within_200m_and_5min"
    control_room_review_required: bool = True


class DuplicateDetector:
    """Deterministic duplicate-incident detector (Section 14)."""

    def check_for_duplicate(
        self,
        target: IncidentSummaryForDuplicateCheck,
        candidates: Sequence[IncidentSummaryForDuplicateCheck],
    ) -> DuplicateMatchResult | None:
        """
        Compare target incident against candidate incidents.
        Returns a DuplicateMatchResult if a match is found, else None.
        Never merges or cancels assignments.
        """
        if not target.is_eligible_for_duplicate_check():
            return None

        assert target.confirmed_latitude is not None
        assert target.confirmed_longitude is not None

        for other in candidates:
            if other.incident_id == target.incident_id:
                continue
            if not other.is_eligible_for_duplicate_check():
                continue
            if other.category != target.category:
                continue

            time_diff = abs((target.created_at - other.created_at).total_seconds())
            if time_diff > DUPLICATE_TIME_WINDOW_SECONDS:
                continue

            assert other.confirmed_latitude is not None
            assert other.confirmed_longitude is not None
            dist_m = geodesic_distance_meters(
                target.confirmed_latitude,
                target.confirmed_longitude,
                other.confirmed_latitude,
                other.confirmed_longitude,
            )
            if dist_m <= DUPLICATE_RADIUS_METERS:
                return DuplicateMatchResult(
                    possible_duplicate=True,
                    incident_id=target.incident_id,
                    matched_incident_id=other.incident_id,
                    distance_meters=round(dist_m, 2),
                    time_difference_seconds=round(time_diff, 1),
                )

        return None
