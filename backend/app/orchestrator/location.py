"""
Location resolution with explicit confidence rules (Section 12).

Enforces:
- Section 12.1: Browser GPS accuracy threshold of <= 500 metres.
- Section 12.2 & Gate 0.7b: Spoken landmark matching against a clearly labelled
  synthetic landmark dataset with native-script and romanised names.
- Section 12.3: GPS within 500m is a location candidate, not a confirmed location,
  until confirmed by the caller.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from difflib import SequenceMatcher
from uuid import UUID

from backend.app.adapters.base import BackendProvenance, BackendSourceType
from backend.app.orchestrator.dispatch import LocationCandidateInput


GPS_ACCURACY_THRESHOLD_METERS: float = 500.0


class GPSPermissionStatus(str, Enum):
    GRANTED = "granted"
    DENIED = "denied"
    PROMPT = "prompt"
    UNAVAILABLE = "unavailable"


class LocationConfirmationStatus(str, Enum):
    UNRESOLVED = "unresolved"
    CANDIDATE_AWAITING_CONFIRMATION = "candidate_awaiting_confirmation"
    CONFIRMED = "confirmed"


@dataclass(frozen=True)
class LandmarkRecord:
    """
    Landmark record retaining native script, romanised form, coordinates,
    source metadata, and explicit synthetic labelling per Gate 0.7b.
    """
    landmark_id: str
    name_romanised: str
    name_native: str
    language: str
    city: str
    latitude: float
    longitude: float
    source_identifier: str
    source_licence: str
    ingestion_date: date
    is_synthetic: bool = True


# Clearly labelled synthetic landmark dataset per Gate 0.7b interim default
SYNTHETIC_LANDMARKS: tuple[LandmarkRecord, ...] = (
    LandmarkRecord(
        landmark_id="SYN-LM-HYD-01",
        name_romanised="Secunderabad Railway Station",
        name_native="సికింద్రాబాద్ రైల్వే స్టేషన్",
        language="te",
        city="Hyderabad",
        latitude=17.4337,
        longitude=78.5016,
        source_identifier="setu-synthetic-landmarks-v1",
        source_licence="Synthetic Demo Dataset (Not Real Map Data)",
        ingestion_date=date(2026, 11, 1),
        is_synthetic=True,
    ),
    LandmarkRecord(
        landmark_id="SYN-LM-HYD-02",
        name_romanised="Hitech City Cyber Towers",
        name_native="హైటెక్ సిటీ సైబర్ టవర్స్",
        language="te",
        city="Hyderabad",
        latitude=17.4504,
        longitude=78.3808,
        source_identifier="setu-synthetic-landmarks-v1",
        source_licence="Synthetic Demo Dataset (Not Real Map Data)",
        ingestion_date=date(2026, 11, 1),
        is_synthetic=True,
    ),
    LandmarkRecord(
        landmark_id="SYN-LM-HYD-03",
        name_romanised="Charminar Bus Stand",
        name_native="చార్మినార్ బస్ స్టాండ్",
        language="te",
        city="Hyderabad",
        latitude=17.3616,
        longitude=78.4747,
        source_identifier="setu-synthetic-landmarks-v1",
        source_licence="Synthetic Demo Dataset (Not Real Map Data)",
        ingestion_date=date(2026, 11, 1),
        is_synthetic=True,
    ),
    LandmarkRecord(
        landmark_id="SYN-LM-DEL-01",
        name_romanised="Kashmere Gate ISBT Metro Station",
        name_native="कश्मीरी गेट बस अड्डा मेट्रो स्टेशन",
        language="hi",
        city="New Delhi",
        latitude=28.6675,
        longitude=77.2282,
        source_identifier="setu-synthetic-landmarks-v1",
        source_licence="Synthetic Demo Dataset (Not Real Map Data)",
        ingestion_date=date(2026, 11, 1),
        is_synthetic=True,
    ),
)


@dataclass(frozen=True)
class ResolvedLocationState:
    """Authoritative location state with typed provenance (Section 12.3)."""
    latitude: float | None
    longitude: float | None
    accuracy_m: float | None
    source: str  # 'browser_gps' | 'landmark' | 'caller_spoken_text' | 'unresolved'
    permission_status: GPSPermissionStatus
    confirmation_status: LocationConfirmationStatus
    is_usable_candidate: bool
    requires_spoken_landmark: bool
    match_confidence: float | None
    resolved_label: str | None
    is_synthetic_landmark: bool
    caller_location_text: str | None
    source_utterance_id: UUID | None
    provenance: BackendProvenance
    broad_area_text: str | None = None
    caller_landmark_text: str | None = None
    confirmed_at: datetime | None = None

    def to_dispatch_candidate(self) -> LocationCandidateInput | None:
        """Convert to a dispatch LocationCandidateInput for coordinate or confirmed-text candidates."""
        if (
            self.latitude is not None
            and self.longitude is not None
            and self.is_usable_candidate
        ):
            return LocationCandidateInput(
                latitude=self.latitude,
                longitude=self.longitude,
                accuracy_m=self.accuracy_m,
                source=self.source,
                is_confirmed=(self.confirmation_status == LocationConfirmationStatus.CONFIRMED),
            )
        if (
            self.confirmation_status == LocationConfirmationStatus.CONFIRMED
            and (self.caller_location_text or self.broad_area_text or self.caller_landmark_text)
        ):
            return LocationCandidateInput(
                latitude=None,
                longitude=None,
                accuracy_m=None,
                source="caller_spoken_text",
                is_confirmed=True,
            )
        return None


@dataclass(frozen=True)
class LandmarkCandidateMatch:
    """Ranked landmark candidate with trigram/token similarity score."""
    landmark: LandmarkRecord
    similarity_score: float
    caller_location_text: str
    source_utterance_id: UUID
    confirmation_status: LocationConfirmationStatus
    provenance: BackendProvenance


class LocationResolver:
    """Resolves caller location via browser GPS threshold rules and fuzzy landmark search."""

    def __init__(self, landmarks: Sequence[LandmarkRecord] = SYNTHETIC_LANDMARKS) -> None:
        self._landmarks = tuple(landmarks)

    def evaluate_browser_gps(
        self,
        *,
        latitude: float | None,
        longitude: float | None,
        accuracy_m: float | None,
        permission_status: GPSPermissionStatus,
        timestamp: datetime,
    ) -> ResolvedLocationState:
        """
        Evaluate browser Geolocation API coordinates against the strict 500m threshold (Section 12.1).
        """
        if (
            permission_status != GPSPermissionStatus.GRANTED
            or latitude is None
            or longitude is None
            or accuracy_m is None
            or accuracy_m <= 0.0
            or accuracy_m > GPS_ACCURACY_THRESHOLD_METERS
        ):
            reason = (
                "permission_not_granted"
                if permission_status != GPSPermissionStatus.GRANTED
                else "accuracy_exceeds_500m_or_missing"
            )
            return ResolvedLocationState(
                latitude=latitude,
                longitude=longitude,
                accuracy_m=accuracy_m,
                source="browser_gps",
                permission_status=permission_status,
                confirmation_status=LocationConfirmationStatus.UNRESOLVED,
                is_usable_candidate=False,
                requires_spoken_landmark=True,
                match_confidence=None,
                resolved_label=None,
                is_synthetic_landmark=False,
                caller_location_text=None,
                source_utterance_id=None,
                provenance=BackendProvenance(
                    source_type=BackendSourceType.BROWSER_GPS,
                    source_id="browser-geolocation-api",
                    source_timestamp=timestamp,
                    source_metadata={
                        "accuracy_m": accuracy_m,
                        "permission_status": permission_status.value,
                        "rejection_reason": reason,
                    },
                ),
            )

        return ResolvedLocationState(
            latitude=latitude,
            longitude=longitude,
            accuracy_m=accuracy_m,
            source="browser_gps",
            permission_status=permission_status,
            confirmation_status=LocationConfirmationStatus.CANDIDATE_AWAITING_CONFIRMATION,
            is_usable_candidate=True,
            requires_spoken_landmark=False,
            match_confidence=None,
            resolved_label=f"Browser GPS ({accuracy_m:.0f}m accuracy)",
            is_synthetic_landmark=False,
            caller_location_text=None,
            source_utterance_id=None,
            provenance=BackendProvenance(
                source_type=BackendSourceType.BROWSER_GPS,
                source_id="browser-geolocation-api",
                source_timestamp=timestamp,
                source_metadata={
                    "accuracy_m": accuracy_m,
                    "permission_status": permission_status.value,
                },
            ),
            broad_area_text=None,
            caller_landmark_text=None,
            confirmed_at=None,
        )

    def search_landmarks(
        self,
        *,
        caller_location_text: str,
        source_utterance_id: UUID,
        now: datetime,
        min_similarity: float = 0.30,
    ) -> list[LandmarkCandidateMatch]:
        """
        Rank synthetic landmark candidates against caller spoken landmark text (Section 12.2).
        """
        query = caller_location_text.strip().lower()
        if not query:
            return []

        results: list[LandmarkCandidateMatch] = []
        for lm in self._landmarks:
            roman_lower = lm.name_romanised.lower()
            ratio_roman = SequenceMatcher(None, query, roman_lower).ratio()
            ratio_native = SequenceMatcher(None, query, lm.name_native).ratio()
            # Boost token overlap for multi-word spoken phrases like "near railway station"
            query_tokens = set(query.replace(",", " ").split())
            lm_tokens = set(roman_lower.split())
            overlap = (
                len(query_tokens & lm_tokens) / max(1, len(lm_tokens))
                if query_tokens
                else 0.0
            )
            score = min(1.0, max(ratio_roman, ratio_native, (ratio_roman * 0.5) + (overlap * 0.6)))
            if score >= min_similarity:
                results.append(
                    LandmarkCandidateMatch(
                        landmark=lm,
                        similarity_score=round(score, 3),
                        caller_location_text=caller_location_text,
                        source_utterance_id=source_utterance_id,
                        confirmation_status=LocationConfirmationStatus.CANDIDATE_AWAITING_CONFIRMATION,
                        provenance=BackendProvenance(
                            source_type=BackendSourceType.DATABASE_RECORD,
                            source_id=lm.landmark_id,
                            source_timestamp=now,
                            source_metadata={
                                "source_utterance_id": str(source_utterance_id),
                                "is_synthetic": lm.is_synthetic,
                                "similarity_score": round(score, 3),
                            },
                        ),
                    )
                )

        results.sort(key=lambda item: (-item.similarity_score, item.landmark.landmark_id))
        return results

    def select_landmark_candidate(
        self,
        match: LandmarkCandidateMatch,
        *,
        confirmed_by_caller: bool,
        now: datetime,
    ) -> ResolvedLocationState:
        """
        Create a ResolvedLocationState from a landmark match.
        Allows provisional allocation on an unconfirmed landmark candidate (flagged for review)
        and records match_confidence rather than a GPS accuracy radius.
        """
        status = (
            LocationConfirmationStatus.CONFIRMED
            if confirmed_by_caller
            else LocationConfirmationStatus.CANDIDATE_AWAITING_CONFIRMATION
        )
        return ResolvedLocationState(
            latitude=match.landmark.latitude,
            longitude=match.landmark.longitude,
            accuracy_m=None,
            source="landmark",
            permission_status=GPSPermissionStatus.UNAVAILABLE,
            confirmation_status=status,
            is_usable_candidate=True,
            requires_spoken_landmark=False,
            match_confidence=match.similarity_score,
            resolved_label=f"{match.landmark.name_romanised} (Synthetic Landmark Data)",
            is_synthetic_landmark=match.landmark.is_synthetic,
            caller_location_text=match.caller_location_text,
            source_utterance_id=match.source_utterance_id,
            provenance=BackendProvenance(
                source_type=BackendSourceType.DATABASE_RECORD,
                source_id=match.landmark.landmark_id,
                source_timestamp=now,
                source_metadata={
                    "source_utterance_id": str(match.source_utterance_id),
                    "confirmed_by_caller": confirmed_by_caller,
                    "is_synthetic": match.landmark.is_synthetic,
                    "match_confidence": match.similarity_score,
                },
            ),
            broad_area_text=match.landmark.city,
            caller_landmark_text=match.caller_location_text,
            confirmed_at=(now if confirmed_by_caller else None),
        )

    @staticmethod
    def apply_spoken_location_detail(
        location: ResolvedLocationState | None,
        *,
        broad_area_text: str | None = None,
        caller_landmark_text: str | None = None,
        source_utterance_id: UUID,
        invalidate_confirmation: bool = False,
        now: datetime,
    ) -> ResolvedLocationState:
        """
        Record caller-spoken broad area and/or landmark text cleanly without fabricating
        coordinates or confidence scores. Preserves existing GPS coordinates when available,
        and resets prior confirmation if invalidate_confirmation is True.
        """
        merged_broad = broad_area_text or (location.broad_area_text if location else None)
        merged_landmark = caller_landmark_text or (location.caller_landmark_text if location else None)
        parts = [p for p in (merged_broad, merged_landmark) if p and p.strip()]
        combined_text = " — ".join(dict.fromkeys(parts)) if parts else (location.caller_location_text if location else None)

        if location is not None:
            has_coords = location.latitude is not None and location.longitude is not None
            next_confirm = location.confirmation_status
            next_confirmed_at = location.confirmed_at
            if invalidate_confirmation and next_confirm == LocationConfirmationStatus.CONFIRMED:
                next_confirm = (
                    LocationConfirmationStatus.CANDIDATE_AWAITING_CONFIRMATION
                    if has_coords
                    else LocationConfirmationStatus.UNRESOLVED
                )
                next_confirmed_at = None
            return ResolvedLocationState(
                latitude=location.latitude,
                longitude=location.longitude,
                accuracy_m=location.accuracy_m,
                source=location.source,
                permission_status=location.permission_status,
                confirmation_status=next_confirm,
                is_usable_candidate=location.is_usable_candidate,
                requires_spoken_landmark=location.requires_spoken_landmark,
                match_confidence=location.match_confidence,
                resolved_label=location.resolved_label or combined_text,
                is_synthetic_landmark=location.is_synthetic_landmark,
                caller_location_text=combined_text,
                source_utterance_id=source_utterance_id,
                provenance=BackendProvenance(
                    source_type=(
                        location.provenance.source_type
                        if location.source in ("browser_gps", "landmark")
                        else BackendSourceType.CALLER_UTTERANCE
                    ),
                    source_id=location.provenance.source_id,
                    source_timestamp=now,
                    source_metadata={
                        **dict(location.provenance.source_metadata),
                        "caller_utterance_id": str(source_utterance_id),
                        "broad_area_text": merged_broad,
                        "caller_landmark_text": merged_landmark,
                    },
                ),
                broad_area_text=merged_broad,
                caller_landmark_text=merged_landmark,
                confirmed_at=next_confirmed_at,
            )

        return ResolvedLocationState(
            latitude=None,
            longitude=None,
            accuracy_m=None,
            source="caller_spoken_text",
            permission_status=GPSPermissionStatus.UNAVAILABLE,
            confirmation_status=LocationConfirmationStatus.UNRESOLVED,
            is_usable_candidate=False,
            requires_spoken_landmark=True,
            match_confidence=None,
            resolved_label=combined_text,
            is_synthetic_landmark=False,
            caller_location_text=combined_text,
            source_utterance_id=source_utterance_id,
            provenance=BackendProvenance(
                source_type=BackendSourceType.CALLER_UTTERANCE,
                source_id=str(source_utterance_id),
                source_timestamp=now,
                source_metadata={
                    "broad_area_text": merged_broad,
                    "caller_landmark_text": merged_landmark,
                    "unverified_coordinates": True,
                },
            ),
            broad_area_text=merged_broad,
            caller_landmark_text=merged_landmark,
            confirmed_at=None,
        )

    @staticmethod
    def confirm_location(
        location: ResolvedLocationState,
        *,
        confirmed_at: datetime,
        allow_text_only: bool = False,
    ) -> ResolvedLocationState:
        """Mark a candidate location as confirmed by the caller (Section 12.3)."""
        has_coords = location.latitude is not None and location.longitude is not None
        has_spoken_text = bool(
            location.caller_location_text or location.broad_area_text or location.caller_landmark_text
        )
        if not has_coords and not (allow_text_only and has_spoken_text):
            raise ValueError("Cannot confirm a location without coordinates")
        return ResolvedLocationState(
            latitude=location.latitude,
            longitude=location.longitude,
            accuracy_m=location.accuracy_m,
            source=location.source,
            permission_status=location.permission_status,
            confirmation_status=LocationConfirmationStatus.CONFIRMED,
            is_usable_candidate=has_coords or allow_text_only,
            requires_spoken_landmark=False,
            match_confidence=location.match_confidence,
            resolved_label=location.resolved_label,
            is_synthetic_landmark=location.is_synthetic_landmark,
            caller_location_text=location.caller_location_text,
            source_utterance_id=location.source_utterance_id,
            provenance=BackendProvenance(
                source_type=location.provenance.source_type,
                source_id=location.provenance.source_id,
                source_timestamp=confirmed_at,
                source_metadata={
                    **dict(location.provenance.source_metadata),
                    "confirmed_by_caller": True,
                    "confirmed_at": confirmed_at.isoformat(),
                },
            ),
            broad_area_text=location.broad_area_text,
            caller_landmark_text=location.caller_landmark_text,
            confirmed_at=confirmed_at,
        )
