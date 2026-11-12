"""
Unit tests for:
- backend/app/orchestrator/instructions.py (Section 28.7)
- backend/app/orchestrator/duplicates.py (Section 28.6)
- backend/app/orchestrator/policy.py (Section 28.2 & 28.6)
- backend/app/audit/redaction.py & writer.py (Section 28.9)
"""

from datetime import datetime, timedelta, timezone
import unittest
from uuid import uuid4

from backend.app.adapters.base import EmergencyCategory
from backend.app.audit.redaction import redact_pii
from backend.app.audit.writer import AppendOnlyAuditError, AuditEvent, AuditSeverity, InMemoryAuditWriter
from backend.app.orchestrator.duplicates import (
    DuplicateDetector,
    IncidentSummaryForDuplicateCheck,
)
from backend.app.orchestrator.instructions import InstructionLibrary
from backend.app.orchestrator.policy import (
    PolicyValidator,
    PrankFlagReviewStatus,
    create_possible_prank_flag,
    update_prank_flag_review,
)


class TestInstructionsDuplicatesPolicyAudit(unittest.TestCase):
    """Tests for instructions, duplicate detection, policy validation, and append-only audit."""

    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.library = InstructionLibrary()
        self.validator = PolicyValidator(self.library)
        self.audit = InMemoryAuditWriter()

    def test_instruction_library_and_spoken_disclosure_in_all_languages(self) -> None:
        """Section 2.1, 15.1 & 28.7: Spoken disclosure and outage/block phrases exist for en, te, hi."""
        for lang in ("en", "te", "hi"):
            disclosure = self.library.get_spoken_disclosure(lang)
            self.assertIn("112", disclosure.approved_text)
            self.assertEqual(disclosure.language, lang)

            outage = self.library.get_outage_reassurance_sequence(lang)
            self.assertGreaterEqual(len(outage), 2)

            block_reask = self.library.get_provider_blocked_reask_prompt(lang)
            self.assertIsNotNone(block_reask.approved_text)

            block_fallback = self.library.get_provider_blocked_fallback_statement(lang)
            self.assertIsNotNone(block_fallback.approved_text)

            didnt_catch = self.library.get_didnt_catch_that_phrase(lang)
            self.assertIsNotNone(didnt_catch.approved_text)

    def test_policy_rejects_dynamic_text_claiming_real_dispatch(self) -> None:
        """Section 3.2 & 28.2: Dynamic text claiming real dispatch is rejected and replaced."""
        unsafe_text = "I have called 112 and a real ambulance is on the way to your house."
        safe_text, was_replaced = self.validator.sanitize_caller_response_text(
            unsafe_text,
            language="en",
            has_simulated_assignment=True,
        )
        self.assertTrue(was_replaced)
        self.assertIn("simulated", safe_text.lower())

        valid_demo_text = "In this demo, a simulated ambulance assignment has been created."
        result_text, was_replaced_2 = self.validator.sanitize_caller_response_text(
            valid_demo_text,
            language="en",
            has_simulated_assignment=True,
        )
        self.assertFalse(was_replaced_2)
        self.assertEqual(result_text, valid_demo_text)

    def test_duplicate_detection_200m_5min_and_confirmed_location_rules(self) -> None:
        """Section 14 & 28.6: Duplicate detection requires same category, <=200m, <=5min, confirmed location."""
        detector = DuplicateDetector()
        inc_1 = IncidentSummaryForDuplicateCheck(
            incident_id=uuid4(),
            category=EmergencyCategory.ROAD_ACCIDENT,
            category_above_threshold=True,
            confirmed_latitude=17.4435,
            confirmed_longitude=78.3772,
            is_location_confirmed=True,
            created_at=self.now - timedelta(minutes=2),
        )
        # Same category, 50m away, 2 minutes apart -> duplicate!
        inc_2 = IncidentSummaryForDuplicateCheck(
            incident_id=uuid4(),
            category=EmergencyCategory.ROAD_ACCIDENT,
            category_above_threshold=True,
            confirmed_latitude=17.4438,
            confirmed_longitude=78.3774,
            is_location_confirmed=True,
            created_at=self.now,
        )
        match = detector.check_for_duplicate(inc_2, [inc_1])
        self.assertIsNotNone(match)
        assert match is not None
        self.assertTrue(match.possible_duplicate)
        self.assertEqual(match.matched_incident_id, inc_1.incident_id)
        self.assertLessEqual(match.distance_meters, 200.0)

        # Unknown category is excluded from duplicate detection
        inc_unknown = IncidentSummaryForDuplicateCheck(
            incident_id=uuid4(),
            category=EmergencyCategory.UNKNOWN,
            category_above_threshold=False,
            confirmed_latitude=17.4435,
            confirmed_longitude=78.3772,
            is_location_confirmed=True,
            created_at=self.now,
        )
        self.assertIsNone(detector.check_for_duplicate(inc_unknown, [inc_1]))

        # Unconfirmed location does not run duplicate detection
        inc_unconfirmed = IncidentSummaryForDuplicateCheck(
            incident_id=uuid4(),
            category=EmergencyCategory.ROAD_ACCIDENT,
            category_above_threshold=True,
            confirmed_latitude=17.4435,
            confirmed_longitude=78.3772,
            is_location_confirmed=False,
            created_at=self.now,
        )
        self.assertIsNone(detector.check_for_duplicate(inc_unconfirmed, [inc_1]))

    def test_false_or_prank_flag_rules(self) -> None:
        """Section 11 & 28.6: Silence, language difficulty, or provider block cannot trigger prank flag."""
        with self.assertRaises(ValueError):
            create_possible_prank_flag(
                flag_reason="caller_silence",
                flag_source="automated",
                supporting_utterance_ids=[uuid4()],
                now=self.now,
            )
        with self.assertRaises(ValueError):
            create_possible_prank_flag(
                flag_reason="provider_blocked_emergency_content",
                flag_source="automated",
                supporting_utterance_ids=[uuid4()],
                now=self.now,
            )

        flag = create_possible_prank_flag(
            flag_reason="explicit_contradictory_test_call_statement",
            flag_source="reviewer_queue_rule",
            supporting_utterance_ids=[uuid4()],
            now=self.now,
        )
        self.assertEqual(flag.review_status, PrankFlagReviewStatus.FLAGGED_FOR_REVIEW)

        reviewed = update_prank_flag_review(
            flag,
            new_status=PrankFlagReviewStatus.NOT_FALSE_OR_PRANK,
            reviewer_id="sim-reviewer-01",
            reviewed_at=self.now,
        )
        self.assertEqual(reviewed.review_status, PrankFlagReviewStatus.NOT_FALSE_OR_PRANK)
        self.assertEqual(reviewed.reviewer_id, "sim-reviewer-01")

    def test_pii_redaction_and_append_only_audit_immutability(self) -> None:
        """Section 21.2, 21.3 & 22: PII is redacted and audit log rejects mutation/deletion."""
        raw = "Caller phone 9876543210 Aadhaar 1234 5678 9012 PAN ABCDE1234F"
        redacted = redact_pii(raw)
        self.assertNotIn("9876543210", redacted)
        self.assertNotIn("1234 5678 9012", redacted)
        self.assertNotIn("ABCDE1234F", redacted)

        evt = AuditEvent.create(
            session_id=uuid4(),
            incident_id=uuid4(),
            event_type="session_created",
            severity=AuditSeverity.INFO,
            outcome="success",
            details={"note": raw},
            now=self.now,
        )
        self.audit.append(evt)
        self.assertEqual(len(self.audit.list_events()), 1)
        self.assertNotIn("9876543210", str(self.audit.list_events()[0].details))

        with self.assertRaises(AppendOnlyAuditError):
            self.audit.update_event(evt.event_id)
        with self.assertRaises(AppendOnlyAuditError):
            self.audit.delete_event(evt.event_id)
        with self.assertRaises(AppendOnlyAuditError):
            self.audit.truncate()


if __name__ == "__main__":
    unittest.main()
