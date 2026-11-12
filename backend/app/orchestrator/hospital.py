"""
Hospital directory filtering and deterministic weighted ranking (Section 13).

Enforces:
- Hard eligibility filtering before ranking (Section 13.1).
- Deterministic weighted scoring with math.isclose float tolerance validation (Section 13.2).
- Weight re-normalisation across distance and capability fit when routing ETA is unavailable (Gate 0.8).
- Full score breakdown and provenance exposure for UI transparency (Section 13.3).
- Never fabricates bed availability, ICU availability, or live hospital readiness.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import math

from backend.app.adapters.base import (
    BackendProvenance,
    BackendSourceType,
    EmergencyCategory,
)
from backend.app.orchestrator.dispatch import geodesic_distance_meters


WEIGHT_SUM_TOLERANCE: float = 1e-6


class HospitalCapability(str, Enum):
    EMERGENCY_DEPARTMENT = "emergency_department"
    TRAUMA_CARE = "trauma_care"
    ICU = "icu"
    BURN_UNIT = "burn_unit"
    CARDIAC_CARE = "cardiac_care"


@dataclass(frozen=True)
class HospitalDirectoryRecord:
    """
    Hospital directory record retaining capabilities, source identifier,
    verification date, and explicit synthetic development dataset labelling (Gate 0.7).
    """
    hospital_id: str
    name: str
    latitude: float
    longitude: float
    address: str
    capabilities: frozenset[HospitalCapability]
    source_identifier: str
    verification_date: date
    is_synthetic_directory: bool = True


SYNTHETIC_HOSPITAL_DIRECTORY: tuple[HospitalDirectoryRecord, ...] = (
    HospitalDirectoryRecord(
        hospital_id="SYN-HOSP-HYD-01",
        name="NIMS Emergency & Trauma Centre (Synthetic Directory)",
        latitude=17.4226,
        longitude=78.4525,
        address="Punjagutta, Hyderabad, Telangana",
        capabilities=frozenset(
            {
                HospitalCapability.EMERGENCY_DEPARTMENT,
                HospitalCapability.TRAUMA_CARE,
                HospitalCapability.ICU,
                HospitalCapability.CARDIAC_CARE,
                HospitalCapability.BURN_UNIT,
            }
        ),
        source_identifier="setu-synthetic-hospital-directory-v1",
        verification_date=date(2026, 11, 1),
        is_synthetic_directory=True,
    ),
    HospitalDirectoryRecord(
        hospital_id="SYN-HOSP-HYD-02",
        name="Hitech Metro Care Hospital (Synthetic Directory)",
        latitude=17.4474,
        longitude=78.3762,
        address="Madhapur Main Road, Hyderabad, Telangana",
        capabilities=frozenset(
            {
                HospitalCapability.EMERGENCY_DEPARTMENT,
                HospitalCapability.TRAUMA_CARE,
                HospitalCapability.ICU,
            }
        ),
        source_identifier="setu-synthetic-hospital-directory-v1",
        verification_date=date(2026, 11, 1),
        is_synthetic_directory=True,
    ),
    HospitalDirectoryRecord(
        hospital_id="SYN-HOSP-HYD-03",
        name="Secunderabad General Emergency Facility (Synthetic Directory)",
        latitude=17.4399,
        longitude=78.4983,
        address="Secunderabad Station Road, Hyderabad, Telangana",
        capabilities=frozenset(
            {
                HospitalCapability.EMERGENCY_DEPARTMENT,
                HospitalCapability.CARDIAC_CARE,
            }
        ),
        source_identifier="setu-synthetic-hospital-directory-v1",
        verification_date=date(2026, 11, 1),
        is_synthetic_directory=True,
    ),
)


@dataclass(frozen=True)
class HospitalScoringWeights:
    """
    Configuration-driven scoring weights validated with math.isclose (Section 13.2).
    """
    distance_weight: float = 0.50
    eta_weight: float = 0.20
    capability_weight: float = 0.30

    def __post_init__(self) -> None:
        for name, val in (
            ("distance_weight", self.distance_weight),
            ("eta_weight", self.eta_weight),
            ("capability_weight", self.capability_weight),
        ):
            if math.isnan(val) or math.isinf(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {val}")

        total = self.distance_weight + self.eta_weight + self.capability_weight
        if not math.isclose(total, 1.0, abs_tol=WEIGHT_SUM_TOLERANCE):
            raise ValueError(
                f"Configured hospital scoring weights must sum to 1.0 within {WEIGHT_SUM_TOLERANCE}, got {total}"
            )


DESIRED_CAPABILITIES_BY_CATEGORY: dict[EmergencyCategory, tuple[HospitalCapability, ...]] = {
    EmergencyCategory.MEDICAL_EMERGENCY: (
        HospitalCapability.EMERGENCY_DEPARTMENT,
        HospitalCapability.ICU,
        HospitalCapability.CARDIAC_CARE,
    ),
    EmergencyCategory.FIRE: (
        HospitalCapability.EMERGENCY_DEPARTMENT,
        HospitalCapability.BURN_UNIT,
    ),
    EmergencyCategory.FIRE_WITH_INJURIES: (
        HospitalCapability.EMERGENCY_DEPARTMENT,
        HospitalCapability.BURN_UNIT,
        HospitalCapability.TRAUMA_CARE,
        HospitalCapability.ICU,
    ),
    EmergencyCategory.ROAD_ACCIDENT: (
        HospitalCapability.EMERGENCY_DEPARTMENT,
        HospitalCapability.TRAUMA_CARE,
        HospitalCapability.ICU,
    ),
    EmergencyCategory.BUILDING_COLLAPSE: (
        HospitalCapability.EMERGENCY_DEPARTMENT,
        HospitalCapability.TRAUMA_CARE,
        HospitalCapability.ICU,
    ),
    EmergencyCategory.PHYSICAL_ATTACK: (
        HospitalCapability.EMERGENCY_DEPARTMENT,
        HospitalCapability.TRAUMA_CARE,
    ),
    EmergencyCategory.GAS_LEAK: (
        HospitalCapability.EMERGENCY_DEPARTMENT,
        HospitalCapability.ICU,
    ),
    EmergencyCategory.EXPLOSION: (
        HospitalCapability.EMERGENCY_DEPARTMENT,
        HospitalCapability.TRAUMA_CARE,
        HospitalCapability.BURN_UNIT,
        HospitalCapability.ICU,
    ),
    EmergencyCategory.UNKNOWN: (
        HospitalCapability.EMERGENCY_DEPARTMENT,
    ),
}


@dataclass(frozen=True)
class HospitalScoreBreakdown:
    """Complete hospital score breakdown for UI transparency and handoff (Section 13.3 & 25.4)."""
    hospital: HospitalDirectoryRecord
    distance_km: float
    eta_minutes: float | None
    eta_source_label: str
    matched_capabilities: tuple[str, ...]
    distance_component_score: float
    eta_component_score: float | None
    capability_component_score: float
    applied_distance_weight: float
    applied_eta_weight: float
    applied_capability_weight: float
    weighted_distance_contribution: float
    weighted_eta_contribution: float
    weighted_capability_contribution: float
    final_score: float
    methodology_note: str
    provenance: BackendProvenance


class HospitalRanker:
    """Deterministic hospital eligibility filter and weighted ranker (Section 13)."""

    def __init__(
        self,
        directory: Sequence[HospitalDirectoryRecord] = SYNTHETIC_HOSPITAL_DIRECTORY,
        *,
        weights: HospitalScoringWeights | None = None,
        verified_eta_provider: Callable[[float, float, float, float], float | None] | None = None,
    ) -> None:
        self._directory = tuple(directory)
        self._weights = weights or HospitalScoringWeights()
        self._verified_eta_provider = verified_eta_provider

    def filter_eligible(self) -> list[HospitalDirectoryRecord]:
        """
        Hard eligibility filter (Section 13.1): requires verified emergency department capability.
        """
        return [
            h
            for h in self._directory
            if HospitalCapability.EMERGENCY_DEPARTMENT in h.capabilities
        ]

    def rank_hospitals(
        self,
        *,
        incident_category: EmergencyCategory,
        caller_latitude: float,
        caller_longitude: float,
        now: datetime,
    ) -> list[HospitalScoreBreakdown]:
        """
        Rank eligible hospitals deterministically with transparent component scores (Section 13.2).
        """
        eligible = self.filter_eligible()
        desired_caps = DESIRED_CAPABILITIES_BY_CATEGORY.get(
            incident_category,
            (HospitalCapability.EMERGENCY_DEPARTMENT,),
        )

        breakdowns: list[HospitalScoreBreakdown] = []
        for hosp in eligible:
            dist_m = geodesic_distance_meters(
                caller_latitude,
                caller_longitude,
                hosp.latitude,
                hosp.longitude,
            )
            dist_km = dist_m / 1000.0
            # Smooth decaying distance score in [0, 1] over a 25 km reference scale
            dist_score = 1.0 / (1.0 + (dist_km / 10.0))

            matched = [c.value for c in desired_caps if c in hosp.capabilities]
            cap_score = len(matched) / max(1, len(desired_caps))

            eta_min = (
                self._verified_eta_provider(
                    caller_latitude,
                    caller_longitude,
                    hosp.latitude,
                    hosp.longitude,
                )
                if self._verified_eta_provider is not None
                else None
            )

            if eta_min is not None:
                eta_score: float | None = 1.0 / (1.0 + (eta_min / 20.0))
                w_dist = self._weights.distance_weight
                w_eta = self._weights.eta_weight
                w_cap = self._weights.capability_weight
                eta_label = "simulated_routing_eta"
                methodology = "Standard 3-component weighted score (distance + ETA + capability)"
            else:
                eta_score = None
                active_sum = self._weights.distance_weight + self._weights.capability_weight
                if active_sum <= 0.0:
                    w_dist, w_cap = 0.5, 0.5
                else:
                    w_dist = self._weights.distance_weight / active_sum
                    w_cap = self._weights.capability_weight / active_sum
                w_eta = 0.0
                eta_label = "unavailable (re-normalised distance + capability)"
                methodology = (
                    "Routing ETA unavailable (Gate 0.8); weights re-normalised across "
                    f"distance ({w_dist:.2f}) and capability fit ({w_cap:.2f})."
                )

            c_dist = dist_score * w_dist
            c_eta = (eta_score * w_eta) if eta_score is not None else 0.0
            c_cap = cap_score * w_cap
            final_score = c_dist + c_eta + c_cap

            breakdowns.append(
                HospitalScoreBreakdown(
                    hospital=hosp,
                    distance_km=round(dist_km, 2),
                    eta_minutes=round(eta_min, 1) if eta_min is not None else None,
                    eta_source_label=eta_label,
                    matched_capabilities=tuple(matched),
                    distance_component_score=round(dist_score, 4),
                    eta_component_score=round(eta_score, 4) if eta_score is not None else None,
                    capability_component_score=round(cap_score, 4),
                    applied_distance_weight=round(w_dist, 4),
                    applied_eta_weight=round(w_eta, 4),
                    applied_capability_weight=round(w_cap, 4),
                    weighted_distance_contribution=round(c_dist, 4),
                    weighted_eta_contribution=round(c_eta, 4),
                    weighted_capability_contribution=round(c_cap, 4),
                    final_score=round(final_score, 4),
                    methodology_note=methodology,
                    provenance=BackendProvenance(
                        source_type=BackendSourceType.DATABASE_RECORD,
                        source_id=hosp.hospital_id,
                        source_timestamp=now,
                        source_metadata={
                            "source_identifier": hosp.source_identifier,
                            "verification_date": hosp.verification_date.isoformat(),
                            "is_synthetic_directory": hosp.is_synthetic_directory,
                        },
                    ),
                )
            )

        breakdowns.sort(key=lambda b: (-b.final_score, b.distance_km, b.hospital.hospital_id))
        return breakdowns
