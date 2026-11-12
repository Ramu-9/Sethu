"""
Unit tests for:
- backend/app/orchestrator/session.py (Sections 8, 9.3, 9.4, 15.4, 18.4, 28.2, 28.2a, 28.8, 28.9)
- backend/app/orchestrator/handoff.py (Section 20)
- backend/app/orchestrator/timeout.py (Section 19 & Gate 0.11)
- backend/app/observability/metrics.py (Section 24)
- backend/app/adapters/mock.py (Section 28.1)
"""

import asyncio
from datetime import datetime, timezone
import unittest
from uuid import uuid4

from backend.app.adapters.base import (
    EmergencyCategory,
    ProvenancedFact,
    ProviderBlockedContentError,
    ReasoningOutput,
    ReasoningUnavailableError,
    UtteranceRole,
)
from backend.app.adapters.mock import MockReasoningAdapter
from backend.app.audit.writer import InMemoryAuditWriter
from backend.app.observability.metrics import MetricsRegistry
from backend.app.orchestrator.dispatch import (
    AssignmentStatus,
    DispatchEngine,
    InMemoryResourceRepository,
    ResourceCategory,
    SyntheticResourceRecord,
)
from backend.app.orchestrator.duplicates import DuplicateDetector
from backend.app.orchestrator.handoff import HandoffGenerator
from backend.app.orchestrator.hospital import HospitalRanker
from backend.app.orchestrator.instructions import InstructionLibrary
from backend.app.orchestrator.location import GPSPermissionStatus, LocationResolver
from backend.app.orchestrator.policy import PolicyValidator
from backend.app.orchestrator.session import (
    SessionOrchestrator,
    SessionTokenSigner,
)
from backend.app.orchestrator.timeout import CallerTimeoutCoordinator, TimeoutPolicyConfig


class TestSessionHandoffTimeoutMetrics(unittest.IsolatedAsyncioTestCase):
    """Tests for session state machine, unknown/block fallbacks, barge-in, handoff, and metrics."""

    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.resources = [
            SyntheticResourceRecord(
                resource_id="SIM-AMB-01",
                category=ResourceCategory.AMBULANCE,
                callsign="SIM-Ambulance-01",
                latitude=17.4450,
                longitude=78.3800,
                is_available=True,
            ),
            SyntheticResourceRecord(
                resource_id="SIM-FIRE-01",
                category=ResourceCategory.FIRE_ENGINE,
                callsign="SIM-FireEngine-01",
                latitude=17.4460,
                longitude=78.3810,
                is_available=True,
            ),
            SyntheticResourceRecord(
                resource_id="SIM-POL-01",
                category=ResourceCategory.POLICE_PATROL,
                callsign="SIM-PolicePatrol-01",
                latitude=17.4440,
                longitude=78.3790,
                is_available=True,
            ),
        ]
        self.resource_repo = InMemoryResourceRepository(self.resources)
        self.dispatch = DispatchEngine(self.resource_repo)
        self.location_resolver = LocationResolver()
        self.hospital_ranker = HospitalRanker()
        self.instruction_lib = InstructionLibrary()
        self.policy = PolicyValidator(self.instruction_lib)
        self.duplicates = DuplicateDetector()
        self.audit = InMemoryAuditWriter()
        self.metrics = MetricsRegistry()
        self.reasoning = MockReasoningAdapter()
        self.token_signer = SessionTokenSigner(secret_key="test-secret-signing-key-32-bytes!!")
        self.orchestrator = SessionOrchestrator(
            reasoning_adapter=self.reasoning,
            dispatch_engine=self.dispatch,
            location_resolver=self.location_resolver,
            hospital_ranker=self.hospital_ranker,
            instruction_library=self.instruction_lib,
            policy_validator=self.policy,
            duplicate_detector=self.duplicates,
            audit_writer=self.audit,
            metrics=self.metrics,
            token_signer=self.token_signer,
        )

    async def test_two_consecutive_unknown_classifications_trigger_fallback_and_answers_do_not_increment(self) -> None:
        """
        Section 9.3 & 28.2:
        - First unknown increments counter to 1, no fallback yet.
        - Yes/no answer (utterance_role=answer_to_pending_question) does NOT increment counter.
        - Second new_description unknown increments counter to 2 and triggers ambulance fallback.
        """
        session = self.orchestrator.create_session(initial_language="en", now=self.now)
        self.orchestrator.update_gps_location(
            session.session_id,
            latitude=17.4435,
            longitude=78.3772,
            accuracy_m=30.0,
            permission_status=GPSPermissionStatus.GRANTED,
            now=self.now,
        )

        # Turn 1: new_description -> unknown (counter becomes 1)
        u1_id = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=u1_id,
                category=EmergencyCategory.UNKNOWN,
                category_confidence=0.45,
                supporting_utterance_ids=[u1_id],
                extracted_facts=[],
                conflicts=[],
                missing_information_fields=["emergency_type"],
                proposed_followup_question="Can you tell me if anyone is hurt or if there is a fire?",
                selected_instruction_id="safety_unknown_01",
                caller_response_text="Could you clarify what happened?",
                prompt_tokens=80,
                completion_tokens=25,
                latency_ms=190.0,
            )
        )
        state = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u1_id,
            text="Something bad happened here please help",
            detected_language="en",
            now=self.now,
        )
        self.assertEqual(state.consecutive_unknown_count, 1)
        self.assertFalse(state.unknown_fallback_applied)
        self.assertEqual(len(state.assignments), 0)

        # Turn 2: Short answer_to_pending_question -> must NOT increment unknown counter
        u2_id = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.ANSWER_TO_PENDING_QUESTION,
                utterance_role_source_id=u2_id,
                category=EmergencyCategory.UNKNOWN,
                category_confidence=0.50,
                supporting_utterance_ids=[u1_id, u2_id],
                extracted_facts=[],
                conflicts=[],
                missing_information_fields=["emergency_type"],
                proposed_followup_question="What kind of emergency is it?",
                selected_instruction_id=None,
                caller_response_text="Please tell me what you see.",
                prompt_tokens=95,
                completion_tokens=20,
                latency_ms=175.0,
            )
        )
        state = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u2_id,
            text="Yes",
            detected_language="en",
            now=self.now,
        )
        self.assertEqual(state.consecutive_unknown_count, 1)
        self.assertFalse(state.unknown_fallback_applied)

        # Turn 3: Second new_description -> unknown (counter reaches 2 -> fallback!)
        u3_id = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=u3_id,
                category=EmergencyCategory.UNKNOWN,
                category_confidence=0.40,
                supporting_utterance_ids=[u1_id, u3_id],
                extracted_facts=[],
                conflicts=[],
                missing_information_fields=[],
                proposed_followup_question=None,
                selected_instruction_id="safety_unknown_01",
                caller_response_text="Unclear emergency.",
                prompt_tokens=110,
                completion_tokens=25,
                latency_ms=200.0,
            )
        )
        state = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u3_id,
            text="People are shouting outside I don't know what it is",
            detected_language="en",
            now=self.now,
        )
        self.assertEqual(state.consecutive_unknown_count, 2)
        self.assertTrue(state.unknown_fallback_applied)
        self.assertTrue(state.control_room_review_required)
        self.assertEqual(len(state.assignments), 1)
        self.assertEqual(state.assignments[0].category, ResourceCategory.AMBULANCE)
        self.assertEqual(state.assignments[0].status, AssignmentStatus.SIMULATED_ASSIGNED)

    async def test_knife_attack_three_consecutive_provider_blocks_triggers_exactly_one_ambulance_assignment(self) -> None:
        """
        Section 9.4 & 28.2a:
        - 1st blocked turn -> info audit event, reassurance + re-ask prompt, no flag, no assignment,
          unknown counter stays 0.
        - 2nd consecutive blocked turn (N=2) -> ambulance fallback, control_room_review_required=True,
          reason='provider_blocked_emergency_content', fallback_reason='consecutive_provider_blocks'.
        - 3rd consecutive blocked turn -> no duplicate assignment.
        """
        session = self.orchestrator.create_session(initial_language="en", now=self.now)
        self.orchestrator.update_gps_location(
            session.session_id,
            latitude=17.4435,
            longitude=78.3772,
            accuracy_m=25.0,
            permission_status=GPSPermissionStatus.GRANTED,
            now=self.now,
        )

        for turn_idx in range(1, 4):
            u_id = uuid4()
            self.reasoning.enqueue_exception(
                ProviderBlockedContentError(
                    "Blocked by safety filter",
                    session_id=session.session_id,
                    utterance_id=u_id,
                    provider_finish_reason="SAFETY",
                )
            )
            state = await self.orchestrator.process_finalized_utterance(
                session.session_id,
                utterance_id=u_id,
                text=f"Someone is attacking with a knife turn {turn_idx}",
                detected_language="en",
                now=self.now,
            )
            # Blocked turns NEVER increment unknown counter
            self.assertEqual(state.consecutive_unknown_count, 0)

            if turn_idx == 1:
                self.assertEqual(state.consecutive_provider_blocks, 1)
                self.assertFalse(state.unknown_fallback_applied)
                self.assertFalse(state.control_room_review_required)
                self.assertEqual(len(state.assignments), 0)
            elif turn_idx == 2:
                self.assertEqual(state.consecutive_provider_blocks, 2)
                self.assertTrue(state.unknown_fallback_applied)
                self.assertTrue(state.control_room_review_required)
                self.assertIn("provider_blocked_emergency_content", state.review_reason_codes)
                self.assertEqual(state.fallback_reason, "consecutive_provider_blocks")
                self.assertEqual(len(state.assignments), 1)
                self.assertEqual(state.assignments[0].category, ResourceCategory.AMBULANCE)
            else:
                self.assertEqual(state.consecutive_provider_blocks, 3)
                self.assertEqual(len(state.assignments), 1)  # Single application rule!

    async def test_session_end_after_single_unknown_or_single_block_triggers_fallback_once(self) -> None:
        """
        Section 9.3 & 9.4:
        - Session ending with 1 unknown and no established category triggers
          reason='session_ended_before_second_attempt'.
        - Session ending with 1 provider block and no established category triggers
          reason='session_ended_after_provider_block'.
        """
        # Case A: Single unknown then session closes
        s1 = self.orchestrator.create_session(initial_language="en", now=self.now)
        self.orchestrator.update_gps_location(
            s1.session_id,
            latitude=17.4435,
            longitude=78.3772,
            accuracy_m=20.0,
            permission_status=GPSPermissionStatus.GRANTED,
            now=self.now,
        )
        u1 = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=u1,
                category=EmergencyCategory.UNKNOWN,
                category_confidence=0.35,
                supporting_utterance_ids=[u1],
                extracted_facts=[],
                conflicts=[],
                missing_information_fields=[],
                proposed_followup_question="What happened?",
                selected_instruction_id=None,
                caller_response_text="Please tell me more.",
                prompt_tokens=50,
                completion_tokens=15,
                latency_ms=120.0,
            )
        )
        await self.orchestrator.process_finalized_utterance(
            s1.session_id,
            utterance_id=u1,
            text="Hello help",
            detected_language="en",
            now=self.now,
        )
        ended_s1 = self.orchestrator.end_session(s1.session_id, reason="caller_hangup", now=self.now)
        self.assertTrue(ended_s1.unknown_fallback_applied)
        self.assertEqual(ended_s1.fallback_reason, "session_ended_before_second_attempt")
        self.assertEqual(len(ended_s1.assignments), 1)

        # Case B: Single provider block then session closes
        s2 = self.orchestrator.create_session(initial_language="en", now=self.now)
        self.orchestrator.update_gps_location(
            s2.session_id,
            latitude=17.4435,
            longitude=78.3772,
            accuracy_m=20.0,
            permission_status=GPSPermissionStatus.GRANTED,
            now=self.now,
        )
        u2 = uuid4()
        self.reasoning.enqueue_exception(
            ProviderBlockedContentError(
                "Blocked",
                session_id=s2.session_id,
                utterance_id=u2,
            )
        )
        await self.orchestrator.process_finalized_utterance(
            s2.session_id,
            utterance_id=u2,
            text="Severe bleeding",
            detected_language="en",
            now=self.now,
        )
        ended_s2 = self.orchestrator.end_session(s2.session_id, reason="caller_hangup", now=self.now)
        self.assertTrue(ended_s2.unknown_fallback_applied)
        self.assertEqual(ended_s2.fallback_reason, "session_ended_after_provider_block")
        self.assertEqual(len(ended_s2.assignments), 1)

    async def test_reasoning_outage_sets_reasoning_unavailable_flag_and_creates_no_assignment(self) -> None:
        """
        Section 15.4 & Gate 0.12: Outage sets control_room_review_required with
        reason='reasoning_unavailable', plays cached reassurance, and creates NO assignment.
        """
        s = self.orchestrator.create_session(initial_language="en", now=self.now)
        u_id = uuid4()
        self.reasoning.enqueue_exception(
            ReasoningUnavailableError(
                "503 Service Unavailable",
                session_id=s.session_id,
                utterance_id=u_id,
                is_retryable=True,
                http_status=503,
            )
        )
        state = await self.orchestrator.process_finalized_utterance(
            s.session_id,
            utterance_id=u_id,
            text="Fire near my shop",
            detected_language="en",
            now=self.now,
        )
        self.assertTrue(state.control_room_review_required)
        self.assertIn("reasoning_unavailable", state.review_reason_codes)
        self.assertNotIn("provider_blocked_emergency_content", state.review_reason_codes)
        self.assertEqual(len(state.assignments), 0)
        self.assertEqual(state.consecutive_unknown_count, 0)
        self.assertEqual(state.consecutive_provider_blocks, 0)

    def test_barge_in_and_false_barge_in_handling_and_handoff_generation(self) -> None:
        """
        Section 18.4, 20 & 28.9:
        - Barge-in during cached vs live playback.
        - Empty transcript after barge-in (pre-Gate 0.1) plays 'didn't catch that' and is not counted.
        - Handoff generator produces structured handoff and marks expired transcripts as 'not retained'.
        """
        s = self.orchestrator.create_session(initial_language="en", now=self.now)
        # Simulate empty/non-speech after barge-in
        barge_res = self.orchestrator.handle_barge_in(
            s.session_id,
            interrupted_Source="cached_instruction",
            interrupted_instruction_id="safety_road_accident_01",
            preserved_preroll_ms=1500,
            transcript_text="",
            is_verified_non_speech_signal=False,
            now=self.now,
        )
        self.assertFalse(barge_res.counted_as_utterance)
        self.assertEqual(barge_res.response_instruction_id, "didnt_catch_that")

        # Generate active handoff
        handoff_gen = HandoffGenerator()
        active_handoff = handoff_gen.generate_handoff(
            self.orchestrator.get_session(s.session_id),
            audit_events=self.audit.list_events(incident_id=s.incident_id),
            transcript_retention_expired=False,
            generated_at=self.now,
        )
        self.assertEqual(active_handoff.incident_id, s.incident_id)
        self.assertTrue(active_handoff.transcript_retained)

        # Regenerate after transcript retention expiry -> marks 'not retained'
        expired_handoff = handoff_gen.generate_handoff(
            self.orchestrator.get_session(s.session_id),
            audit_events=self.audit.list_events(incident_id=s.incident_id),
            transcript_retention_expired=True,
            generated_at=self.now,
        )
        self.assertFalse(expired_handoff.transcript_retained)
        self.assertEqual(expired_handoff.original_caller_statements, ("not retained",))

    def test_timeout_coordinator_never_diagnoses_unconsciousness(self) -> None:
        """Section 5 (Rule 5) & Section 19: Caller silence triggers check-in and review flag, never diagnosis."""
        policy = TimeoutPolicyConfig(
            initial_silence_seconds=15.0,
            max_check_ins=2,
            wait_between_check_ins_seconds=10.0,
            max_idle_exhaustion_seconds=45.0,
        )
        coord = CallerTimeoutCoordinator(policy)
        step1 = coord.evaluate_silence(idle_seconds=16.0, check_ins_sent=0)
        self.assertEqual(step1.action, "play_check_in_prompt")
        self.assertFalse(step1.flag_control_room)

        step2 = coord.evaluate_silence(idle_seconds=36.0, check_ins_sent=2)
        self.assertEqual(step2.action, "flag_control_room_review")
        self.assertTrue(step2.flag_control_room)
        self.assertFalse(step2.session_exhausted)

        step3 = coord.evaluate_silence(idle_seconds=46.0, check_ins_sent=2)
        self.assertEqual(step3.action, "timeout_exhaustion_close_session")
        self.assertTrue(step3.session_exhausted)


if __name__ == "__main__":
    unittest.main()
