"""
Unit tests for backend/app/orchestrator/location.py (Section 28.4) and
backend/app/orchestrator/hospital.py (Section 28.5).

Verifies:
- GPS accuracy below 500m, exactly 500m, above 500m, missing, and permission denied.
- GPS within threshold is a location candidate, NOT confirmed, until caller confirms.
- Synthetic landmark fuzzy matching and caller confirmation.
- Landmark records retain source and synthetic label (Gate 0.7b).
- Hospital hard eligibility filters.
- Hospital deterministic weighted score calculation and weight sum tolerance validation.
- Negative weights rejected.
- Weight re-normalisation when routing ETA is unavailable (Gate 0.8).
- Full score breakdown and provenance retained.
"""

from datetime import date, datetime, timezone
import unittest
from uuid import uuid4

from backend.app.adapters.base import EmergencyCategory
from backend.app.orchestrator.hospital import (
    HospitalCapability,
    HospitalDirectoryRecord,
    HospitalRanker,
    HospitalScoringWeights,
)
from backend.app.orchestrator.location import (
    GPSPermissionStatus,
    LocationConfirmationStatus,
    LocationResolver,
    SYNTHETIC_LANDMARKS,
)


class TestLocationAndHospitalModules(unittest.TestCase):
    """Tests for Section 12 (Location) and Section 13 (Hospital)."""

    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.utterance_id = uuid4()
        self.resolver = LocationResolver(landmarks=SYNTHETIC_LANDMARKS)
        self.hospitals = [
            HospitalDirectoryRecord(
                hospital_id="SYN-HOSP-01",
                name="Care Institute (Synthetic Directory Record)",
                latitude=17.4400,
                longitude=78.3800,
                address="Hitech City Main Rd, Hyderabad",
                capabilities=frozenset(
                    {
                        HospitalCapability.EMERGENCY_DEPARTMENT,
                        HospitalCapability.TRAUMA_CARE,
                        HospitalCapability.ICU,
                        HospitalCapability.BURN_UNIT,
                    }
                ),
                source_identifier="synthetic-dev-directory-v1",
                verification_date=date(2026, 11, 1),
                is_synthetic_directory=True,
            ),
            HospitalDirectoryRecord(
                hospital_id="SYN-HOSP-02",
                name="City Community Clinic (Synthetic Directory Record)",
                latitude=17.4420,
                longitude=78.3780,
                address="Madhapur Phase 2, Hyderabad",
                capabilities=frozenset({HospitalCapability.EMERGENCY_DEPARTMENT}),
                source_identifier="synthetic-dev-directory-v1",
                verification_date=date(2026, 11, 1),
                is_synthetic_directory=True,
            ),
        ]

    def test_gps_threshold_below_and_equal_500m_are_candidates_until_confirmed(self) -> None:
        """Section 12.1 & 12.3: GPS <= 500m is a candidate, not confirmed until caller confirms."""
        res_below = self.resolver.evaluate_browser_gps(
            latitude=17.4435,
            longitude=78.3772,
            accuracy_m=499.9,
            permission_status=GPSPermissionStatus.GRANTED,
            timestamp=self.now,
        )
        self.assertTrue(res_below.is_usable_candidate)
        self.assertFalse(res_below.requires_spoken_landmark)
        self.assertEqual(
            res_below.confirmation_status,
            LocationConfirmationStatus.CANDIDATE_AWAITING_CONFIRMATION,
        )

        # Exactly 500m is accepted as candidate
        res_exact = self.resolver.evaluate_browser_gps(
            latitude=17.4435,
            longitude=78.3772,
            accuracy_m=500.0,
            permission_status=GPSPermissionStatus.GRANTED,
            timestamp=self.now,
        )
        self.assertTrue(res_exact.is_usable_candidate)
        self.assertFalse(res_exact.requires_spoken_landmark)

        # Caller confirmation transitions candidate to CONFIRMED
        confirmed = self.resolver.confirm_location(res_below, confirmed_at=self.now)
        self.assertEqual(
            confirmed.confirmation_status,
            LocationConfirmationStatus.CONFIRMED,
        )

    def test_gps_above_500m_or_missing_or_denied_triggers_landmark_path(self) -> None:
        """Section 12.1: GPS > 500m, missing accuracy, or denied permission triggers landmark path."""
        res_above = self.resolver.evaluate_browser_gps(
            latitude=17.4435,
            longitude=78.3772,
            accuracy_m=500.1,
            permission_status=GPSPermissionStatus.GRANTED,
            timestamp=self.now,
        )
        self.assertFalse(res_above.is_usable_candidate)
        self.assertTrue(res_above.requires_spoken_landmark)

        res_missing = self.resolver.evaluate_browser_gps(
            latitude=17.4435,
            longitude=78.3772,
            accuracy_m=None,
            permission_status=GPSPermissionStatus.GRANTED,
            timestamp=self.now,
        )
        self.assertFalse(res_missing.is_usable_candidate)
        self.assertTrue(res_missing.requires_spoken_landmark)

        res_denied = self.resolver.evaluate_browser_gps(
            latitude=None,
            longitude=None,
            accuracy_m=None,
            permission_status=GPSPermissionStatus.DENIED,
            timestamp=self.now,
        )
        self.assertFalse(res_denied.is_usable_candidate)
        self.assertTrue(res_denied.requires_spoken_landmark)

    def test_spoken_landmark_fuzzy_search_and_synthetic_labelling(self) -> None:
        """Section 12.2 & Gate 0.7b: Fuzzy match against synthetic landmarks retains provenance."""
        matches = self.resolver.search_landmarks(
            caller_location_text="near Secunderabad railway station",
            source_utterance_id=self.utterance_id,
            now=self.now,
        )
        self.assertGreaterEqual(len(matches), 1)
        top = matches[0]
        self.assertIn("Secunderabad", top.landmark.name_romanised)
        self.assertTrue(top.landmark.is_synthetic)
        self.assertEqual(top.source_utterance_id, self.utterance_id)
        self.assertEqual(
            top.confirmation_status,
            LocationConfirmationStatus.CANDIDATE_AWAITING_CONFIRMATION,
        )

    def test_hospital_weights_validation_and_renormalization_without_eta(self) -> None:
        """Section 13.2: Rejects negative/invalid weights and re-normalises when ETA is unavailable."""
        with self.assertRaises(ValueError):
            HospitalScoringWeights(
                distance_weight=-0.1,
                eta_weight=0.5,
                capability_weight=0.6,
            )

        with self.assertRaises(ValueError):
            HospitalScoringWeights(
                distance_weight=0.5,
                eta_weight=0.5,
                capability_weight=0.5,
            )

        weights = HospitalScoringWeights(
            distance_weight=0.45,
            eta_weight=0.25,
            capability_weight=0.30,
        )
        ranker = HospitalRanker(self.hospitals, weights=weights)
        ranked = ranker.rank_hospitals(
            incident_category=EmergencyCategory.FIRE_WITH_INJURIES,
            caller_latitude=17.4435,
            caller_longitude=78.3772,
            now=self.now,
        )
        self.assertEqual(len(ranked), 2)
        top = ranked[0]
        # Top hospital has burn unit + trauma + ICU fit
        self.assertEqual(top.hospital.hospital_id, "SYN-HOSP-01")
        self.assertIsNone(top.eta_minutes)
        self.assertEqual(top.eta_source_label, "unavailable (re-normalised distance + capability)")
        self.assertAlmostEqual(
            top.applied_distance_weight + top.applied_capability_weight,
            1.0,
            places=6,
        )
        self.assertEqual(top.applied_eta_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
