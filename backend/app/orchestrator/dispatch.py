"""
Deterministic emergency-to-resource dispatch and allocation (Section 10).

Rules enforced:
- Rule 1: Emergency-to-resource mapping belongs exclusively to Python.
- Rule 2: Only backend code generates resource IDs, assignment IDs, statuses, and ETAs.
- Gate 0.8: Simulated ETA defaults to None unless an approved backend ETA formula is configured.
- Gate 0.9: Only the exact base mapping from Section 10.1 is active until additional rules are approved.
- Section 10.3: Location candidate precondition (GPS <= 500m or confirmed landmark).
- Section 10.4: Additive-only reclassification; never cancels or releases existing assignments.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import math
import threading
from typing import Protocol
from uuid import UUID, uuid4

from backend.app.adapters.base import (
    BackendProvenance,
    BackendSourceType,
    EmergencyCategory,
)


class ResourceCategory(str, Enum):
    """Supported synthetic emergency resource categories (Section 10.1)."""
    AMBULANCE = "ambulance"
    FIRE_ENGINE = "fire_engine"
    FIRE_AND_RESCUE = "fire_and_rescue"
    POLICE_PATROL = "police_patrol"


class AssignmentStatus(str, Enum):
    """Simulated assignment statuses (Section 10.3)."""
    SIMULATED_ASSIGNED = "simulated_assigned"
    PENDING_LOCATION = "pending_location"
    SIMULATED_UNAVAILABLE = "simulated_unavailable"


class DemoLifecycleStatus(str, Enum):
    """Synthetic demo vehicle lifecycle statuses for simulated tracking."""
    ASSIGNED = "ASSIGNED"
    EN_ROUTE = "EN_ROUTE"
    ARRIVED = "ARRIVED"
    UNAVAILABLE = "UNAVAILABLE"


DEMO_SIMULATED_ETA_MINUTES: dict[ResourceCategory, int] = {
    ResourceCategory.AMBULANCE: 6,
    ResourceCategory.FIRE_ENGINE: 7,
    ResourceCategory.FIRE_AND_RESCUE: 8,
    ResourceCategory.POLICE_PATROL: 9,
}

DEFAULT_DEMO_INCIDENT_COORDS: tuple[float, float] = (17.4435, 78.3772)


def build_demo_fleet_resources() -> list["SyntheticResourceRecord"]:
    """Return the fixed 9-vehicle synthetic demo fleet (AMB-01..03, FIR-01..03, POL-01..03)."""
    return [
        SyntheticResourceRecord(
            resource_id="AMB-01",
            category=ResourceCategory.AMBULANCE,
            callsign="DEMO-Ambulance AMB-01 (Madhapur)",
            latitude=17.4455,
            longitude=78.3795,
            is_available=True,
        ),
        SyntheticResourceRecord(
            resource_id="AMB-02",
            category=ResourceCategory.AMBULANCE,
            callsign="DEMO-Ambulance AMB-02 (Secunderabad)",
            latitude=17.4350,
            longitude=78.5000,
            is_available=True,
        ),
        SyntheticResourceRecord(
            resource_id="AMB-03",
            category=ResourceCategory.AMBULANCE,
            callsign="DEMO-Ambulance AMB-03 (Banjara Hills)",
            latitude=17.4195,
            longitude=78.4480,
            is_available=True,
        ),
        SyntheticResourceRecord(
            resource_id="FIR-01",
            category=ResourceCategory.FIRE_ENGINE,
            callsign="DEMO-Fire-Engine FIR-01 (Hitech City)",
            latitude=17.4480,
            longitude=78.3820,
            is_available=True,
        ),
        SyntheticResourceRecord(
            resource_id="FIR-02",
            category=ResourceCategory.FIRE_ENGINE,
            callsign="DEMO-Fire-Engine FIR-02 (Jubilee Hills)",
            latitude=17.4320,
            longitude=78.4070,
            is_available=True,
        ),
        SyntheticResourceRecord(
            resource_id="FIR-03",
            category=ResourceCategory.FIRE_ENGINE,
            callsign="DEMO-Fire-Engine FIR-03 (Charminar)",
            latitude=17.3750,
            longitude=78.4760,
            is_available=True,
        ),
        SyntheticResourceRecord(
            resource_id="POL-01",
            category=ResourceCategory.POLICE_PATROL,
            callsign="DEMO-Police-Patrol POL-01 (Cyberabad)",
            latitude=17.4440,
            longitude=78.3780,
            is_available=True,
        ),
        SyntheticResourceRecord(
            resource_id="POL-02",
            category=ResourceCategory.POLICE_PATROL,
            callsign="DEMO-Police-Patrol POL-02 (Begumpet)",
            latitude=17.4290,
            longitude=78.4610,
            is_available=True,
        ),
        SyntheticResourceRecord(
            resource_id="POL-03",
            category=ResourceCategory.POLICE_PATROL,
            callsign="DEMO-Police-Patrol POL-03 (Abids)",
            latitude=17.3920,
            longitude=78.4890,
            is_available=True,
        ),
    ]


# Exact starting approved base dispatch table per Section 10.1
BASE_DISPATCH_MAPPING: dict[EmergencyCategory, tuple[ResourceCategory, ...]] = {
    EmergencyCategory.MEDICAL_EMERGENCY: (ResourceCategory.AMBULANCE,),
    EmergencyCategory.FIRE: (ResourceCategory.FIRE_ENGINE,),
    EmergencyCategory.FIRE_WITH_INJURIES: (
        ResourceCategory.FIRE_ENGINE,
        ResourceCategory.AMBULANCE,
    ),
    EmergencyCategory.ROAD_ACCIDENT: (
        ResourceCategory.AMBULANCE,
        ResourceCategory.POLICE_PATROL,
    ),
    EmergencyCategory.BUILDING_COLLAPSE: (
        ResourceCategory.FIRE_AND_RESCUE,
        ResourceCategory.AMBULANCE,
        ResourceCategory.POLICE_PATROL,
    ),
    EmergencyCategory.PHYSICAL_ATTACK: (ResourceCategory.POLICE_PATROL,),
    EmergencyCategory.GAS_LEAK: (
        ResourceCategory.FIRE_ENGINE,
        ResourceCategory.POLICE_PATROL,
    ),
    EmergencyCategory.EXPLOSION: (
        ResourceCategory.FIRE_ENGINE,
        ResourceCategory.AMBULANCE,
        ResourceCategory.POLICE_PATROL,
    ),
    EmergencyCategory.UNKNOWN: (ResourceCategory.AMBULANCE,),
}


@dataclass(frozen=True)
class LocationCandidateInput:
    """Validated location candidate passed into dispatch allocation (Section 10.3 & 12)."""
    latitude: float | None
    longitude: float | None
    accuracy_m: float | None
    source: str  # 'browser_gps', 'landmark', or 'caller_spoken_text'
    is_confirmed: bool

    def is_eligible_for_allocation(self) -> bool:
        """
        GPS requires coordinates and accuracy <= 500m.
        Landmark candidate requires resolved synthetic landmark coordinates.
        Verbal text-only location requires explicit caller confirmation in DEMO mode.
        """
        if self.source == "browser_gps":
            return (
                self.latitude is not None
                and self.longitude is not None
                and self.accuracy_m is not None
                and 0.0 <= self.accuracy_m <= 500.0
            )
        if self.source == "landmark":
            return self.latitude is not None and self.longitude is not None
        if self.source == "caller_spoken_text":
            return self.is_confirmed
        return False


@dataclass(frozen=True)
class SyntheticResourceRecord:
    """Synthetic database record representing a simulated vehicle or team."""
    resource_id: str
    category: ResourceCategory
    callsign: str
    latitude: float
    longitude: float
    is_available: bool
    is_simulated: bool = True


@dataclass(frozen=True)
class SimulatedAssignment:
    """Backend-generated link between an incident and a simulated resource (Section 10.3)."""
    assignment_id: UUID
    incident_id: UUID
    idempotency_key: str
    category: ResourceCategory
    resource_id: str | None
    resource_callsign: str | None
    status: AssignmentStatus
    assigned_at: datetime
    simulated_eta_minutes: int | None
    distance_meters: float | None
    provenance: BackendProvenance
    is_simulated: bool = True
    lifecycle_status: str = DemoLifecycleStatus.ASSIGNED.value
    en_route_at: datetime | None = None
    arrived_at: datetime | None = None
    current_latitude: float | None = None
    current_longitude: float | None = None
    target_latitude: float | None = None
    target_longitude: float | None = None
    remaining_eta_minutes: int | None = None
    allocation_mode: str = "STRICT"
    estimated_arrival_at: datetime | None = None
    depot_latitude: float | None = None
    depot_longitude: float | None = None
    remaining_eta_seconds: int | None = None


@dataclass(frozen=True)
class DispatchOutcome:
    """Result of a deterministic dispatch or reclassification operation."""
    incident_id: UUID
    category: EmergencyCategory
    required_categories: tuple[ResourceCategory, ...]
    assignments: tuple[SimulatedAssignment, ...]
    control_room_review_required: bool
    review_reasons: tuple[str, ...]
    no_longer_required_categories: tuple[ResourceCategory, ...] = ()


def geodesic_distance_meters(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Compute WGS84 Haversine geodesic distance in metres."""
    earth_radius_m = 6_371_008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_m * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


class ResourceRepository(Protocol):
    """Transactional synthetic resource store interface."""

    def claim_nearest_available(
        self,
        *,
        category: ResourceCategory,
        latitude: float,
        longitude: float,
    ) -> tuple[SyntheticResourceRecord, float] | None:
        """
        Atomically select and claim the nearest available synthetic resource
        by geodesic distance, tie-broken by resource_id.
        """
        ...


class InMemoryResourceRepository:
    """
    Thread-safe, row-locked synthetic resource store for tests and local runtime.
    Enforces atomic claim semantics equivalent to SELECT ... FOR UPDATE SKIP LOCKED.
    """

    def __init__(self, initial_resources: Sequence[SyntheticResourceRecord]) -> None:
        self._lock = threading.Lock()
        self._resources: dict[str, SyntheticResourceRecord] = {
            r.resource_id: r for r in initial_resources
        }
        self._depot_coords: dict[str, tuple[float, float]] = {
            r.resource_id: (r.latitude, r.longitude) for r in initial_resources
        }
        self._claimed_by_incident: dict[str, UUID] = {}

    def list_all(self) -> list[SyntheticResourceRecord]:
        """Return a snapshot of all synthetic resources."""
        with self._lock:
            return [
                self._resources[rid]
                for rid in sorted(self._resources.keys())
            ]

    def reset_all_available(self) -> None:
        """Restore all demo fleet vehicles to AVAILABLE at their depot coordinates."""
        with self._lock:
            self._claimed_by_incident.clear()
            for rid, res in list(self._resources.items()):
                depot_lat, depot_lon = self._depot_coords.get(rid, (res.latitude, res.longitude))
                self._resources[rid] = SyntheticResourceRecord(
                    resource_id=res.resource_id,
                    category=res.category,
                    callsign=res.callsign,
                    latitude=depot_lat,
                    longitude=depot_lon,
                    is_available=True,
                    is_simulated=True,
                )

    def release_incident_resources(self, incident_id: UUID) -> list[str]:
        """Release all vehicles currently held by incident_id back to AVAILABLE."""
        released: list[str] = []
        with self._lock:
            for rid, owner_id in list(self._claimed_by_incident.items()):
                if owner_id == incident_id:
                    res = self._resources.get(rid)
                    if res is not None:
                        depot_lat, depot_lon = self._depot_coords.get(rid, (res.latitude, res.longitude))
                        self._resources[rid] = SyntheticResourceRecord(
                            resource_id=res.resource_id,
                            category=res.category,
                            callsign=res.callsign,
                            latitude=depot_lat,
                            longitude=depot_lon,
                            is_available=True,
                            is_simulated=True,
                        )
                    self._claimed_by_incident.pop(rid, None)
                    released.append(rid)
        return released

    def claim_next_demo_vehicle(
        self,
        *,
        category: ResourceCategory,
        incident_id: UUID,
        target_latitude: float,
        target_longitude: float,
    ) -> tuple[SyntheticResourceRecord, float] | None:
        """
        Deterministically claim the first available synthetic demo vehicle for `category`
        (ordered by code: e.g. AMB-01, AMB-02, AMB-03), preventing duplicate active incident claims.
        For FIRE_AND_RESCUE, falls back to FIRE_ENGINE if no dedicated rescue unit is seeded.
        """
        acceptable_categories = [category]
        if category == ResourceCategory.FIRE_AND_RESCUE:
            acceptable_categories.append(ResourceCategory.FIRE_ENGINE)

        with self._lock:
            candidates: list[tuple[int, str, float, SyntheticResourceRecord]] = []
            for resource in self._resources.values():
                if not resource.is_available:
                    continue
                if resource.resource_id in self._claimed_by_incident:
                    continue
                if resource.category not in acceptable_categories:
                    continue
                pref_rank = 0 if resource.category == category else 1
                dist_m = geodesic_distance_meters(
                    target_latitude,
                    target_longitude,
                    resource.latitude,
                    resource.longitude,
                )
                candidates.append((pref_rank, resource.resource_id, dist_m, resource))

            if not candidates:
                return None

            candidates.sort(key=lambda item: (item[0], item[1]))
            _, best_id, dist_m, best_resource = candidates[0]
            self._claimed_by_incident[best_id] = incident_id
            claimed = SyntheticResourceRecord(
                resource_id=best_resource.resource_id,
                category=best_resource.category,
                callsign=best_resource.callsign,
                latitude=best_resource.latitude,
                longitude=best_resource.longitude,
                is_available=False,
                is_simulated=True,
            )
            self._resources[best_id] = claimed
            return claimed, dist_m

    def claim_nearest_available(
        self,
        *,
        category: ResourceCategory,
        latitude: float,
        longitude: float,
    ) -> tuple[SyntheticResourceRecord, float] | None:
        """
        Atomically find and mark unavailable the nearest available resource
        for the requested category, tie-broken by resource_id.
        """
        with self._lock:
            candidates: list[tuple[float, str, SyntheticResourceRecord]] = []
            for resource in self._resources.values():
                if resource.category == category and resource.is_available:
                    dist_m = geodesic_distance_meters(
                        latitude,
                        longitude,
                        resource.latitude,
                        resource.longitude,
                    )
                    candidates.append((dist_m, resource.resource_id, resource))

            if not candidates:
                return None

            candidates.sort(key=lambda item: (item[0], item[1]))
            best_dist_m, best_id, best_resource = candidates[0]
            claimed = SyntheticResourceRecord(
                resource_id=best_resource.resource_id,
                category=best_resource.category,
                callsign=best_resource.callsign,
                latitude=best_resource.latitude,
                longitude=best_resource.longitude,
                is_available=False,
                is_simulated=True,
            )
            self._resources[best_id] = claimed
            return claimed, best_dist_m


class DispatchEngine:
    """
    Deterministic emergency-to-resource dispatcher (Section 10).
    """

    def __init__(
        self,
        repository: ResourceRepository,
        *,
        approved_eta_calculator: Callable[[float], int] | None = None,
        demo_mode: bool = False,
        require_explicit_location_confirmation: bool | None = None,
        demo_journey_duration_seconds: float = 12.0,
    ) -> None:
        """
        Initialize the deterministic dispatch engine.

        Inputs:
            repository: Transactional synthetic resource repository.
            approved_eta_calculator: Optional approved ETA function mapping distance_m -> minutes.
                                     Defaults to None per Gate 0.8 (no ad-hoc ETA invention).
            demo_mode: When True, allocates deterministically from the DEMO fleet with fixed
                       simulated demo ETAs and simulated depot-to-target EN_ROUTE lifecycle.
            require_explicit_location_confirmation: When True (default in DEMO mode), holds
                       new assignments as PENDING_LOCATION until caller confirms location.
            demo_journey_duration_seconds: Simulated transit duration in seconds before EN_ROUTE -> ARRIVED.
        """
        self._repository = repository
        self._approved_eta_calculator = approved_eta_calculator
        self._demo_mode = demo_mode
        self._require_explicit_location_confirmation = (
            demo_mode
            if require_explicit_location_confirmation is None
            else require_explicit_location_confirmation
        )
        self._demo_journey_duration_seconds = max(3.0, float(demo_journey_duration_seconds))

    @property
    def is_demo_mode(self) -> bool:
        return self._demo_mode

    def determine_required_categories(
        self,
        category: EmergencyCategory,
    ) -> tuple[ResourceCategory, ...]:
        """
        Return the exact required resource categories from BASE_DISPATCH_MAPPING (Section 10.1).
        Gate 0.9: No unapproved additional-resource rules are inferred.
        """
        if category not in BASE_DISPATCH_MAPPING:
            raise ValueError(f"Unsupported emergency category: {category}")
        return BASE_DISPATCH_MAPPING[category]

    def determine_multi_hazard_categories(
        self,
        category: EmergencyCategory,
        *,
        identified_hazards: Sequence[str] = (),
        context_text: str = "",
    ) -> tuple[ResourceCategory, ...]:
        """
        Deterministically evaluate compound multi-hazard resource requirements while
        preserving single-category BASE_DISPATCH_MAPPING contracts (Gate 0.9).
        """
        base = list(self.determine_required_categories(category))
        normalized_hazards = {
            str(h).strip().lower().replace("-", "_").replace(" ", "_")
            for h in identified_hazards
            if str(h).strip()
        }
        lower_ctx = context_text.lower()

        # Infer compound hazard tags from caller utterance context when present together
        if any(kw in lower_ctx for kw in ("collid", "collision", "crash", "accident", "bus", "truck", "car")):
            normalized_hazards.add("vehicle_collision")
        if any(kw in lower_ctx for kw in ("fire", "flames", "burning", "smoke", "आग", "మంట")):
            normalized_hazards.add("fire")
        if any(kw in lower_ctx for kw in ("cylinder", "lpg", "gas", "सिलेंडर", "సిలిండర్")):
            normalized_hazards.add("gas_cylinder")
        if any(kw in lower_ctx for kw in ("injur", "bleed", "burn", "hurt", "casualt", "घायल", "గాయ")):
            normalized_hazards.add("injuries")
        if any(kw in lower_ctx for kw in ("trapped", "collapse", "debris", "collapsed")):
            normalized_hazards.add("trapped_persons")

        # Preserve strict single-category mapping when there is at most one hazard domain
        extra: list[ResourceCategory] = list(base)
        if len(normalized_hazards) >= 2:
            if "fire" in normalized_hazards or "gas_cylinder" in normalized_hazards or "gas_leak" in normalized_hazards:
                extra.append(ResourceCategory.FIRE_ENGINE)
            if "injuries" in normalized_hazards or "medical_trauma" in normalized_hazards or "vehicle_collision" in normalized_hazards or "road_accident" in normalized_hazards:
                extra.append(ResourceCategory.AMBULANCE)
            if (
                "vehicle_collision" in normalized_hazards
                or "road_accident" in normalized_hazards
                or "explosion" in normalized_hazards
                or "assault" in normalized_hazards
            ):
                extra.append(ResourceCategory.POLICE_PATROL)
            if "trapped_persons" in normalized_hazards or "building_collapse" in normalized_hazards:
                extra.append(ResourceCategory.FIRE_AND_RESCUE)

        priority_order = (
            ResourceCategory.FIRE_ENGINE,
            ResourceCategory.AMBULANCE,
            ResourceCategory.POLICE_PATROL,
            ResourceCategory.FIRE_AND_RESCUE,
        )
        present = set(extra)
        return tuple(cat for cat in priority_order if cat in present)

    def check_fleet_availability(
        self,
        required_categories: Sequence[ResourceCategory],
    ) -> dict[str, int]:
        """Thread-safely inspect current synthetic vehicle availability counts without mutating fleet state."""
        repo = self._repository
        if not hasattr(repo, "list_all"):
            return {cat.value: 0 for cat in required_categories}
        snapshot = repo.list_all()
        counts: dict[str, int] = {cat.value: 0 for cat in required_categories}
        for rec in snapshot:
            if rec.is_available and rec.category.value in counts:
                counts[rec.category.value] += 1
        return counts

    @staticmethod
    def build_idempotency_key(
        incident_id: UUID,
        category: ResourceCategory,
    ) -> str:
        """Build an incident-and-category-scoped idempotency key (Section 10.3)."""
        return f"{incident_id}:{category.value}"

    def allocate_for_incident(
        self,
        *,
        incident_id: UUID,
        category: EmergencyCategory,
        location_candidate: LocationCandidateInput | None,
        existing_assignments: Sequence[SimulatedAssignment],
        now: datetime,
        required_categories_override: Sequence[ResourceCategory] | None = None,
    ) -> DispatchOutcome:
        """
        Allocate simulated resources for an incident according to Section 10.3.

        Inputs:
            incident_id: Authoritative incident UUID.
            category: Validated EmergencyCategory.
            location_candidate: Optional location candidate (GPS <= 500m or confirmed landmark).
            existing_assignments: Existing assignments for this incident.
            now: Current UTC timestamp.
        Outputs:
            DispatchOutcome with updated assignments and review flags.
        """
        required = (
            tuple(dict.fromkeys(required_categories_override))
            if required_categories_override
            else self.determine_required_categories(category)
        )
        by_category: dict[ResourceCategory, SimulatedAssignment] = {
            a.category: a for a in existing_assignments
        }
        updated_list: list[SimulatedAssignment] = list(existing_assignments)
        review_reasons: list[str] = []

        has_valid_location = (
            location_candidate is not None
            and location_candidate.is_eligible_for_allocation()
        )
        has_confirmed_location = (
            location_candidate is not None
            and location_candidate.is_confirmed
        )
        demo_target_lat = (
            location_candidate.latitude
            if (location_candidate is not None and location_candidate.latitude is not None)
            else DEFAULT_DEMO_INCIDENT_COORDS[0]
        )
        demo_target_lon = (
            location_candidate.longitude
            if (location_candidate is not None and location_candidate.longitude is not None)
            else DEFAULT_DEMO_INCIDENT_COORDS[1]
        )

        for req_cat in required:
            idem_key = self.build_idempotency_key(incident_id, req_cat)
            existing = by_category.get(req_cat)

            # Already assigned a simulated resource -> preserve idempotently
            if (
                existing is not None
                and existing.status == AssignmentStatus.SIMULATED_ASSIGNED
            ):
                continue

            if (self._require_explicit_location_confirmation and not has_confirmed_location) or (
                not self._demo_mode
                and (
                    not has_valid_location
                    or location_candidate is None
                    or location_candidate.latitude is None
                    or location_candidate.longitude is None
                )
            ):
                gate_reason = (
                    "awaiting_location_confirmation"
                    if self._require_explicit_location_confirmation and has_valid_location
                    else "pending_location"
                )
                review_reasons.append(gate_reason)
                if existing is None:
                    pending_assignment = SimulatedAssignment(
                        assignment_id=uuid4(),
                        incident_id=incident_id,
                        idempotency_key=idem_key,
                        category=req_cat,
                        resource_id=None,
                        resource_callsign=None,
                        status=AssignmentStatus.PENDING_LOCATION,
                        assigned_at=now,
                        simulated_eta_minutes=None,
                        distance_meters=None,
                        lifecycle_status="PENDING_LOCATION",
                        allocation_mode="DEMO" if self._demo_mode else "STRICT",
                        provenance=BackendProvenance(
                            source_type=BackendSourceType.BACKEND_RULE,
                            source_id=f"dispatch:{category.value}:{req_cat.value}",
                            source_timestamp=now,
                            source_metadata={"reason": gate_reason},
                        ),
                    )
                    by_category[req_cat] = pending_assignment
                    updated_list.append(pending_assignment)
                continue

            if self._demo_mode:
                if hasattr(self._repository, "claim_next_demo_vehicle"):
                    claimed = self._repository.claim_next_demo_vehicle(
                        category=req_cat,
                        incident_id=incident_id,
                        target_latitude=demo_target_lat,
                        target_longitude=demo_target_lon,
                    )
                else:
                    claimed = self._repository.claim_nearest_available(
                        category=req_cat,
                        latitude=demo_target_lat,
                        longitude=demo_target_lon,
                    )
            else:
                # Valid location candidate exists -> claim nearest available resource
                assert (
                    location_candidate is not None
                    and location_candidate.latitude is not None
                    and location_candidate.longitude is not None
                )
                claimed = self._repository.claim_nearest_available(
                    category=req_cat,
                    latitude=location_candidate.latitude,
                    longitude=location_candidate.longitude,
                )

            if claimed is None:
                review_reasons.append(f"resource_unavailable:{req_cat.value}")
                unavail_assignment = SimulatedAssignment(
                    assignment_id=existing.assignment_id if existing else uuid4(),
                    incident_id=incident_id,
                    idempotency_key=idem_key,
                    category=req_cat,
                    resource_id=None,
                    resource_callsign=None,
                    status=AssignmentStatus.SIMULATED_UNAVAILABLE,
                    assigned_at=now,
                    simulated_eta_minutes=None,
                    distance_meters=None,
                    lifecycle_status=DemoLifecycleStatus.UNAVAILABLE.value,
                    target_latitude=demo_target_lat if self._demo_mode else None,
                    target_longitude=demo_target_lon if self._demo_mode else None,
                    allocation_mode="DEMO" if self._demo_mode else "STRICT",
                    provenance=BackendProvenance(
                        source_type=BackendSourceType.BACKEND_RULE,
                        source_id=f"dispatch:{category.value}:{req_cat.value}",
                        source_timestamp=now,
                        source_metadata={"reason": "no_available_synthetic_resource"},
                    ),
                )
                self._replace_or_append(updated_list, by_category, req_cat, unavail_assignment)
            else:
                resource_rec, dist_m = claimed
                if self._demo_mode:
                    eta_min: int | None = DEMO_SIMULATED_ETA_MINUTES.get(req_cat, 6)
                elif self._approved_eta_calculator is not None:
                    eta_min = self._approved_eta_calculator(dist_m)
                else:
                    eta_min = None

                # In explicit-confirmation demo mode, confirmed dispatch enters EN_ROUTE immediately
                # with initial depot coordinates and realistic journey ETA schedule.
                immediate_en_route = bool(
                    self._demo_mode and self._require_explicit_location_confirmation and has_confirmed_location
                )
                est_arrival = (
                    now + timedelta(seconds=self._demo_journey_duration_seconds)
                    if immediate_en_route
                    else None
                )
                assigned = SimulatedAssignment(
                    assignment_id=existing.assignment_id if existing else uuid4(),
                    incident_id=incident_id,
                    idempotency_key=idem_key,
                    category=req_cat,
                    resource_id=resource_rec.resource_id,
                    resource_callsign=resource_rec.callsign,
                    status=AssignmentStatus.SIMULATED_ASSIGNED,
                    assigned_at=now,
                    simulated_eta_minutes=eta_min,
                    remaining_eta_minutes=eta_min,
                    remaining_eta_seconds=int(round(self._demo_journey_duration_seconds)) if immediate_en_route else (eta_min * 60 if eta_min is not None else None),
                    distance_meters=round(dist_m, 2),
                    lifecycle_status=(
                        DemoLifecycleStatus.EN_ROUTE.value
                        if immediate_en_route
                        else DemoLifecycleStatus.ASSIGNED.value
                    ),
                    en_route_at=(now if immediate_en_route else None),
                    estimated_arrival_at=est_arrival,
                    depot_latitude=resource_rec.latitude,
                    depot_longitude=resource_rec.longitude,
                    current_latitude=resource_rec.latitude,
                    current_longitude=resource_rec.longitude,
                    target_latitude=(
                        demo_target_lat
                        if self._demo_mode
                        else (location_candidate.latitude if location_candidate else None)
                    ),
                    target_longitude=(
                        demo_target_lon
                        if self._demo_mode
                        else (location_candidate.longitude if location_candidate else None)
                    ),
                    allocation_mode="DEMO" if self._demo_mode else "STRICT",
                    provenance=BackendProvenance(
                        source_type=BackendSourceType.DATABASE_RECORD,
                        source_id=resource_rec.resource_id,
                        source_timestamp=now,
                        source_metadata={
                            "rule": f"dispatch:{category.value}",
                            "allocation_mode": "DEMO" if self._demo_mode else "STRICT",
                            "distance_m": round(dist_m, 2),
                        },
                    ),
                )
                self._replace_or_append(updated_list, by_category, req_cat, assigned)

        return DispatchOutcome(
            incident_id=incident_id,
            category=category,
            required_categories=required,
            assignments=tuple(updated_list),
            control_room_review_required=len(review_reasons) > 0,
            review_reasons=tuple(dict.fromkeys(review_reasons)),
        )

    def advance_assignment_lifecycle(
        self,
        assignments: Sequence[SimulatedAssignment],
        *,
        target_status: DemoLifecycleStatus,
        now: datetime,
        eligible_assignment_ids: set[UUID] | None = None,
    ) -> tuple[list[SimulatedAssignment], list[SimulatedAssignment]]:
        """
        Deterministically transition active simulated assignments along ASSIGNED -> EN_ROUTE -> ARRIVED,
        interpolating simulated coordinates toward incident location and updating remaining demo ETA.
        """
        updated_all: list[SimulatedAssignment] = []
        changed: list[SimulatedAssignment] = []

        for asgn in assignments:
            if (
                asgn.status != AssignmentStatus.SIMULATED_ASSIGNED
                or asgn.resource_id is None
                or (eligible_assignment_ids is not None and asgn.assignment_id not in eligible_assignment_ids)
            ):
                updated_all.append(asgn)
                continue

            cur = asgn.lifecycle_status
            if target_status == DemoLifecycleStatus.EN_ROUTE and cur in (
                DemoLifecycleStatus.ASSIGNED.value,
                DemoLifecycleStatus.EN_ROUTE.value,
            ):
                en_route_start = asgn.en_route_at or now
                est_arrival = asgn.estimated_arrival_at or (
                    en_route_start + timedelta(seconds=self._demo_journey_duration_seconds)
                )
                total_secs = max(1.0, (est_arrival - en_route_start).total_seconds())
                elapsed_secs = max(0.0, (now - en_route_start).total_seconds())
                progress = min(0.95, max(0.15 if cur == DemoLifecycleStatus.ASSIGNED.value else 0.0, elapsed_secs / total_secs))

                depot_lat = asgn.depot_latitude if asgn.depot_latitude is not None else asgn.current_latitude
                depot_lon = asgn.depot_longitude if asgn.depot_longitude is not None else asgn.current_longitude
                tgt_lat = asgn.target_latitude
                tgt_lon = asgn.target_longitude
                if depot_lat is not None and tgt_lat is not None:
                    next_lat = round(depot_lat + (tgt_lat - depot_lat) * progress, 6)
                else:
                    next_lat = asgn.current_latitude
                if depot_lon is not None and tgt_lon is not None:
                    next_lon = round(depot_lon + (tgt_lon - depot_lon) * progress, 6)
                else:
                    next_lon = asgn.current_longitude

                base_eta = asgn.simulated_eta_minutes or DEMO_SIMULATED_ETA_MINUTES.get(asgn.category, 6)
                rem_eta = max(1, math.ceil(base_eta * (1.0 - progress)))
                rem_secs = max(1, int(round(max(0.0, total_secs - elapsed_secs))))
                updated = SimulatedAssignment(
                    assignment_id=asgn.assignment_id,
                    incident_id=asgn.incident_id,
                    idempotency_key=asgn.idempotency_key,
                    category=asgn.category,
                    resource_id=asgn.resource_id,
                    resource_callsign=asgn.resource_callsign,
                    status=asgn.status,
                    assigned_at=asgn.assigned_at,
                    simulated_eta_minutes=asgn.simulated_eta_minutes,
                    distance_meters=asgn.distance_meters,
                    provenance=asgn.provenance,
                    is_simulated=asgn.is_simulated,
                    lifecycle_status=DemoLifecycleStatus.EN_ROUTE.value,
                    en_route_at=en_route_start,
                    arrived_at=asgn.arrived_at,
                    current_latitude=next_lat,
                    current_longitude=next_lon,
                    target_latitude=tgt_lat,
                    target_longitude=tgt_lon,
                    remaining_eta_minutes=rem_eta,
                    allocation_mode=asgn.allocation_mode,
                    estimated_arrival_at=est_arrival,
                    depot_latitude=depot_lat,
                    depot_longitude=depot_lon,
                    remaining_eta_seconds=rem_secs,
                )
                updated_all.append(updated)
                if cur == DemoLifecycleStatus.ASSIGNED.value or (
                    asgn.remaining_eta_minutes != rem_eta
                    or asgn.remaining_eta_seconds != rem_secs
                    or asgn.current_latitude != next_lat
                ):
                    changed.append(updated)
            elif target_status == DemoLifecycleStatus.ARRIVED and cur == DemoLifecycleStatus.EN_ROUTE.value:
                updated = SimulatedAssignment(
                    assignment_id=asgn.assignment_id,
                    incident_id=asgn.incident_id,
                    idempotency_key=asgn.idempotency_key,
                    category=asgn.category,
                    resource_id=asgn.resource_id,
                    resource_callsign=asgn.resource_callsign,
                    status=asgn.status,
                    assigned_at=asgn.assigned_at,
                    simulated_eta_minutes=asgn.simulated_eta_minutes,
                    distance_meters=0.0 if asgn.target_latitude is not None else asgn.distance_meters,
                    provenance=asgn.provenance,
                    is_simulated=asgn.is_simulated,
                    lifecycle_status=DemoLifecycleStatus.ARRIVED.value,
                    en_route_at=asgn.en_route_at or now,
                    arrived_at=now,
                    current_latitude=asgn.target_latitude if asgn.target_latitude is not None else asgn.current_latitude,
                    current_longitude=asgn.target_longitude if asgn.target_longitude is not None else asgn.current_longitude,
                    target_latitude=asgn.target_latitude,
                    target_longitude=asgn.target_longitude,
                    remaining_eta_minutes=0,
                    allocation_mode=asgn.allocation_mode,
                    estimated_arrival_at=asgn.estimated_arrival_at or now,
                    depot_latitude=asgn.depot_latitude,
                    depot_longitude=asgn.depot_longitude,
                    remaining_eta_seconds=0,
                )
                updated_all.append(updated)
                changed.append(updated)
            else:
                updated_all.append(asgn)

        return updated_all, changed

    def reclassify_incident(
        self,
        *,
        incident_id: UUID,
        previous_category: EmergencyCategory,
        new_category: EmergencyCategory,
        location_candidate: LocationCandidateInput | None,
        existing_assignments: Sequence[SimulatedAssignment],
        now: datetime,
    ) -> DispatchOutcome:
        """
        Recompute required resources on category reclassification (Section 10.4).

        Never cancels, releases, or modifies existing assignments. If a previously
        assigned category is no longer required by the new category, it remains
        assigned and flags the incident for simulated control-room review.
        """
        outcome = self.allocate_for_incident(
            incident_id=incident_id,
            category=new_category,
            location_candidate=location_candidate,
            existing_assignments=existing_assignments,
            now=now,
        )
        new_required_set = set(outcome.required_categories)
        no_longer_required: list[ResourceCategory] = []
        for assignment in outcome.assignments:
            if assignment.category not in new_required_set:
                no_longer_required.append(assignment.category)

        reasons = list(outcome.review_reasons)
        if no_longer_required:
            reasons.append(
                f"reclassified_retained_resources:{previous_category.value}->{new_category.value}"
            )

        return DispatchOutcome(
            incident_id=incident_id,
            category=new_category,
            required_categories=outcome.required_categories,
            assignments=outcome.assignments,
            control_room_review_required=len(reasons) > 0,
            review_reasons=tuple(dict.fromkeys(reasons)),
            no_longer_required_categories=tuple(no_longer_required),
        )

    @staticmethod
    def _replace_or_append(
        assignments: list[SimulatedAssignment],
        by_category: dict[ResourceCategory, SimulatedAssignment],
        category: ResourceCategory,
        new_assignment: SimulatedAssignment,
    ) -> None:
        by_category[category] = new_assignment
        for idx, item in enumerate(assignments):
            if item.category == category:
                assignments[idx] = new_assignment
                return
        assignments.append(new_assignment)
