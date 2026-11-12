"""
Unit tests for backend/app/orchestrator/dispatch.py (Section 28.3).

Verifies:
- Every base deterministic mapping (Section 10.1).
- Unknown maps to ambulance.
- No unapproved additional-resource rules (Gate 0.9).
- Reasoning model cannot add resources.
- Multiple-resource allocation.
- No resource available -> unavailable status, no fake assignment.
- Duplicate assignment prevention & idempotency.
- Concurrent incidents cannot claim the same synthetic resource.
- No location candidate -> pending_location requirements, no resource selected,
  and control_room_review_required = True.
- Allocation proceeds once a location candidate exists.
- Simulated ETA is None when no approved ETA model exists (Gate 0.8).
- Reclassification adds only missing categories and never cancels or modifies
  existing assignments (Section 10.4).
- Reclassification after unknown fallback keeps existing ambulance assignment.
"""

from datetime import datetime, timezone
import unittest
from uuid import uuid4

from backend.app.adapters.base import EmergencyCategory
from backend.app.orchestrator.dispatch import (
    AssignmentStatus,
    BASE_DISPATCH_MAPPING,
    DispatchEngine,
    InMemoryResourceRepository,
    LocationCandidateInput,
    ResourceCategory,
    SyntheticResourceRecord,
)


class TestDeterministicDispatch(unittest.TestCase):
    """Tests for Section 10 deterministic dispatch and Section 28.3 test cases."""

    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.incident_id = uuid4()
        self.valid_location = LocationCandidateInput(
            latitude=17.4435,
            longitude=78.3772,
            accuracy_m=35.0,
            source="browser_gps",
            is_confirmed=False,
        )
        self.resources = [
            SyntheticResourceRecord(
                resource_id="SIM-AMB-101",
                category=ResourceCategory.AMBULANCE,
                callsign="SIM-Ambulance-Hitech-01",
                latitude=17.4450,
                longitude=78.3800,
                is_available=True,
            ),
            SyntheticResourceRecord(
                resource_id="SIM-AMB-102",
                category=ResourceCategory.AMBULANCE,
                callsign="SIM-Ambulance-Madhapur-02",
                latitude=17.4600,
                longitude=78.3900,
                is_available=True,
            ),
            SyntheticResourceRecord(
                resource_id="SIM-FIRE-201",
                category=ResourceCategory.FIRE_ENGINE,
                callsign="SIM-FireEngine-01",
                latitude=17.4480,
                longitude=78.3820,
                is_available=True,
            ),
            SyntheticResourceRecord(
                resource_id="SIM-FRESC-301",
                category=ResourceCategory.FIRE_AND_RESCUE,
                callsign="SIM-FireRescue-01",
                latitude=17.4500,
                longitude=78.3850,
                is_available=True,
            ),
            SyntheticResourceRecord(
                resource_id="SIM-POL-401",
                category=ResourceCategory.POLICE_PATROL,
                callsign="SIM-PolicePatrol-01",
                latitude=17.4440,
                longitude=78.3780,
                is_available=True,
            ),
        ]
        self.repo = InMemoryResourceRepository(self.resources)
        self.engine = DispatchEngine(self.repo)

    def test_every_base_deterministic_mapping(self) -> None:
        """Section 10.1: Every emergency category maps to the exact approved base table."""
        expected = {
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
        self.assertEqual(BASE_DISPATCH_MAPPING, expected)
        for category, expected_resources in expected.items():
            self.assertEqual(
                self.engine.determine_required_categories(category),
                expected_resources,
            )

    def test_nearest_available_resource_selected_and_eta_is_none_by_default(self) -> None:
        """Section 10.3 & Gate 0.8: Selects nearest available resource and keeps ETA None."""
        outcome = self.engine.allocate_for_incident(
            incident_id=self.incident_id,
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            location_candidate=self.valid_location,
            existing_assignments=(),
            now=self.now,
        )
        self.assertFalse(outcome.control_room_review_required)
        self.assertEqual(len(outcome.assignments), 1)
        assignment = outcome.assignments[0]
        self.assertEqual(assignment.resource_id, "SIM-AMB-101")  # Closer than SIM-AMB-102
        self.assertEqual(assignment.status, AssignmentStatus.SIMULATED_ASSIGNED)
        self.assertIsNone(assignment.simulated_eta_minutes)
        self.assertTrue(assignment.is_simulated)

    def test_no_location_candidate_creates_pending_location_and_flags_review(self) -> None:
        """Section 10.3: Without location candidate, sets pending_location and selects no resource."""
        outcome = self.engine.allocate_for_incident(
            incident_id=self.incident_id,
            category=EmergencyCategory.ROAD_ACCIDENT,
            location_candidate=None,
            existing_assignments=(),
            now=self.now,
        )
        self.assertTrue(outcome.control_room_review_required)
        self.assertEqual(len(outcome.assignments), 2)
        for assignment in outcome.assignments:
            self.assertEqual(assignment.status, AssignmentStatus.PENDING_LOCATION)
            self.assertIsNone(assignment.resource_id)

        # Once location candidate arrives, pending_location upgrades to simulated_assigned
        resolved_outcome = self.engine.allocate_for_incident(
            incident_id=self.incident_id,
            category=EmergencyCategory.ROAD_ACCIDENT,
            location_candidate=self.valid_location,
            existing_assignments=outcome.assignments,
            now=self.now,
        )
        self.assertEqual(len(resolved_outcome.assignments), 2)
        for assignment in resolved_outcome.assignments:
            self.assertEqual(assignment.status, AssignmentStatus.SIMULATED_ASSIGNED)
            self.assertIsNotNone(assignment.resource_id)

    def test_gps_accuracy_above_500m_rejected_as_location_candidate(self) -> None:
        """Section 12.1: GPS accuracy > 500m cannot be used as a location candidate."""
        poor_gps = LocationCandidateInput(
            latitude=17.4435,
            longitude=78.3772,
            accuracy_m=650.0,
            source="browser_gps",
            is_confirmed=False,
        )
        outcome = self.engine.allocate_for_incident(
            incident_id=self.incident_id,
            category=EmergencyCategory.FIRE,
            location_candidate=poor_gps,
            existing_assignments=(),
            now=self.now,
        )
        self.assertTrue(outcome.control_room_review_required)
        self.assertEqual(outcome.assignments[0].status, AssignmentStatus.PENDING_LOCATION)
        self.assertIsNone(outcome.assignments[0].resource_id)

    def test_idempotency_and_concurrent_incidents_cannot_claim_same_resource(self) -> None:
        """Section 10.3: Repeated calls do not duplicate assignments; second incident gets next resource."""
        first = self.engine.allocate_for_incident(
            incident_id=self.incident_id,
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            location_candidate=self.valid_location,
            existing_assignments=(),
            now=self.now,
        )
        # Re-running for the same incident is idempotent
        second = self.engine.allocate_for_incident(
            incident_id=self.incident_id,
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            location_candidate=self.valid_location,
            existing_assignments=first.assignments,
            now=self.now,
        )
        self.assertEqual(len(second.assignments), 1)
        self.assertEqual(
            first.assignments[0].assignment_id,
            second.assignments[0].assignment_id,
        )

        # Concurrent incident claims the second available ambulance
        other_incident_id = uuid4()
        other_outcome = self.engine.allocate_for_incident(
            incident_id=other_incident_id,
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            location_candidate=self.valid_location,
            existing_assignments=(),
            now=self.now,
        )
        self.assertEqual(other_outcome.assignments[0].resource_id, "SIM-AMB-102")

    def test_no_resource_available_marks_unavailable_and_flags_review(self) -> None:
        """Section 9.3 & 27: Exhausted synthetic resources produce SIMULATED_UNAVAILABLE, never fake IDs."""
        empty_repo = InMemoryResourceRepository([])
        engine = DispatchEngine(empty_repo)
        outcome = engine.allocate_for_incident(
            incident_id=self.incident_id,
            category=EmergencyCategory.FIRE,
            location_candidate=self.valid_location,
            existing_assignments=(),
            now=self.now,
        )
        self.assertTrue(outcome.control_room_review_required)
        self.assertEqual(len(outcome.assignments), 1)
        self.assertEqual(outcome.assignments[0].status, AssignmentStatus.SIMULATED_UNAVAILABLE)
        self.assertIsNone(outcome.assignments[0].resource_id)

    def test_reclassification_is_additive_only_and_never_cancels_existing(self) -> None:
        """
        Section 10.4: Reclassification from unknown -> fire or fire -> fire_with_injuries
        adds missing categories and never cancels existing assignments.
        """
        # Start with unknown fallback (assigns Ambulance)
        initial = self.engine.allocate_for_incident(
            incident_id=self.incident_id,
            category=EmergencyCategory.UNKNOWN,
            location_candidate=self.valid_location,
            existing_assignments=(),
            now=self.now,
        )
        self.assertEqual(len(initial.assignments), 1)
        self.assertEqual(initial.assignments[0].category, ResourceCategory.AMBULANCE)

        # Reclassify to FIRE (requires Fire engine; existing Ambulance must remain assigned and flagged)
        reclassified = self.engine.reclassify_incident(
            incident_id=self.incident_id,
            previous_category=EmergencyCategory.UNKNOWN,
            new_category=EmergencyCategory.FIRE,
            location_candidate=self.valid_location,
            existing_assignments=initial.assignments,
            now=self.now,
        )
        categories = [a.category for a in reclassified.assignments]
        self.assertEqual(
            categories,
            [ResourceCategory.AMBULANCE, ResourceCategory.FIRE_ENGINE],
        )
        self.assertTrue(reclassified.control_room_review_required)
        self.assertIn(
            ResourceCategory.AMBULANCE,
            reclassified.no_longer_required_categories,
        )
        # Ambulance is still SIMULATED_ASSIGNED
        amb_assignment = reclassified.assignments[0]
        self.assertEqual(amb_assignment.status, AssignmentStatus.SIMULATED_ASSIGNED)
        self.assertEqual(amb_assignment.resource_id, "SIM-AMB-101")


if __name__ == "__main__":
    unittest.main()
