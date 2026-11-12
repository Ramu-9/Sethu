"""
Unit tests for backend/app/adapters/base.py.

Verifies application-owned domain models, strict provenance validation,
utterance_role validation, and distinct error reason codes for
provider-blocked emergency content vs. reasoning unavailability.
"""

from datetime import datetime, timezone
import unittest
from uuid import uuid4

from backend.app.adapters.base import (
    AudioChunk,
    BackendProvenance,
    BackendSourceType,
    DialogueContext,
    EmergencyCategory,
    FactConflict,
    FinalizedUtteranceRecord,
    ProvenancedFact,
    ProviderBlockedContentError,
    ReasoningOutput,
    ReasoningSchemaValidationError,
    ReasoningUnavailableError,
    STTSignalType,
    STTProviderError,
    SynthesisAudioChunk,
    TranscriptEvent,
    TTSProviderError,
    UtteranceRole,
)


class TestAdapterBaseDomainModels(unittest.TestCase):
    """Tests for application-owned adapter interfaces and domain models."""

    def setUp(self) -> None:
        self.session_id = uuid4()
        self.incident_id = uuid4()
        self.utterance_id_1 = uuid4()
        self.utterance_id_2 = uuid4()
        self.now = datetime.now(timezone.utc)

    def test_emergency_category_taxonomy_complete(self) -> None:
        """All 9 required categories from Section 9 must exist."""
        expected = {
            "medical_emergency",
            "fire",
            "fire_with_injuries",
            "road_accident",
            "building_collapse",
            "physical_attack",
            "gas_leak",
            "explosion",
            "unknown",
        }
        actual = {cat.value for cat in EmergencyCategory}
        self.assertEqual(actual, expected)

    def test_utterance_role_enum_values(self) -> None:
        """Section 9.2 utterance_role enum values must match specification."""
        expected = {
            "new_description",
            "answer_to_pending_question",
            "acknowledgement",
            "other",
        }
        actual = {role.value for role in UtteranceRole}
        self.assertEqual(actual, expected)

    def test_distinct_error_reason_codes(self) -> None:
        """
        Section 9.4 & 15.4: provider_blocked_emergency_content and
        reasoning_unavailable must never share reason codes.
        """
        blocked_err = ProviderBlockedContentError(
            "Safety block triggered",
            session_id=self.session_id,
            utterance_id=self.utterance_id_1,
            provider_finish_reason="SAFETY",
            safety_ratings={"HARM_CATEGORY_DANGEROUS_CONTENT": "HIGH"},
        )
        outage_err = ReasoningUnavailableError(
            "HTTP 503 Service Unavailable",
            session_id=self.session_id,
            utterance_id=self.utterance_id_1,
            is_retryable=True,
            http_status=503,
        )
        schema_err = ReasoningSchemaValidationError(
            "Missing utterance_role",
            session_id=self.session_id,
            utterance_id=self.utterance_id_1,
        )
        stt_err = STTProviderError("Prisma stream disconnected")
        tts_err = TTSProviderError("Timbre stream timeout")

        self.assertEqual(blocked_err.reason_code, "provider_blocked_emergency_content")
        self.assertEqual(outage_err.reason_code, "reasoning_unavailable")
        self.assertNotEqual(blocked_err.reason_code, outage_err.reason_code)
        self.assertEqual(schema_err.reason_code, "reasoning_schema_invalid")
        self.assertEqual(stt_err.reason_code, "stt_provider_unavailable")
        self.assertEqual(tts_err.reason_code, "tts_provider_unavailable")

    def test_provenanced_fact_requires_non_empty_evidence_and_valid_uuid(self) -> None:
        """Rule 3: Every LLM-extracted fact must reference a valid source_utterance_id."""
        fact = ProvenancedFact(
            field_name="trapped_persons",
            value=True,
            source_utterance_id=self.utterance_id_1,
            evidence_text="Oka person lopala irukupoyaru",
        )
        self.assertEqual(fact.source_utterance_id, self.utterance_id_1)
        self.assertTrue(fact.value)

        with self.assertRaises(ValueError):
            ProvenancedFact(
                field_name="trapped_persons",
                value=True,
                source_utterance_id=self.utterance_id_1,
                evidence_text="   ",
            )

    def test_backend_provenance_structure(self) -> None:
        """Rule 3: Backend-generated facts use separate typed provenance."""
        prov = BackendProvenance(
            source_type=BackendSourceType.BROWSER_GPS,
            source_id="gps-fix-01",
            source_timestamp=self.now,
            source_metadata={"accuracy_m": 42.5},
        )
        self.assertEqual(prov.source_type, BackendSourceType.BROWSER_GPS)
        self.assertEqual(prov.source_metadata["accuracy_m"], 42.5)

    def test_audio_chunk_enforces_16khz_pcm(self) -> None:
        """Section 18.1: AudioChunk must validate non-empty PCM and positive sample rate."""
        chunk = AudioChunk(
            pcm_bytes=b"\x00\x01" * 160,
            sample_rate_hz=16000,
            sequence_number=0,
            captured_at=self.now,
        )
        self.assertEqual(chunk.sample_rate_hz, 16000)

        with self.assertRaises(ValueError):
            AudioChunk(
                pcm_bytes=b"",
                sample_rate_hz=16000,
                sequence_number=0,
                captured_at=self.now,
            )

    def test_dialogue_context_and_reasoning_output_validation(self) -> None:
        """Section 9.2: ReasoningOutput validates provenance against known utterances."""
        u1 = FinalizedUtteranceRecord(
            utterance_id=self.utterance_id_1,
            sequence_number=1,
            text="Road accident near railway station",
            language="en",
            utterance_role=UtteranceRole.NEW_DESCRIPTION,
            finalized_at=self.now,
        )
        u2 = FinalizedUtteranceRecord(
            utterance_id=self.utterance_id_2,
            sequence_number=2,
            text="Two people are injured",
            language="en",
            utterance_role=None,
            finalized_at=self.now,
        )
        context = DialogueContext(
            session_id=self.session_id,
            incident_id=self.incident_id,
            active_language="en",
            prior_utterances=[u1],
            validated_facts=[],
            current_utterance=u2,
            pending_followup_question="Is anyone injured?",
            eligible_instruction_ids=["inst_road_01"],
            backend_status_summary={"assigned_ambulance": "SIM-AMB-01"},
        )

        output = ReasoningOutput(
            utterance_role=UtteranceRole.ANSWER_TO_PENDING_QUESTION,
            utterance_role_source_id=self.utterance_id_2,
            category=EmergencyCategory.ROAD_ACCIDENT,
            category_confidence=0.92,
            supporting_utterance_ids=[self.utterance_id_1, self.utterance_id_2],
            extracted_facts=[
                ProvenancedFact(
                    field_name="injured_persons",
                    value=True,
                    source_utterance_id=self.utterance_id_2,
                    evidence_text="Two people are injured",
                ),
                ProvenancedFact(
                    field_name="affected_person_count",
                    value=2,
                    source_utterance_id=self.utterance_id_2,
                    evidence_text="Two people are injured",
                ),
            ],
            conflicts=[],
            missing_information_fields=[],
            proposed_followup_question=None,
            selected_instruction_id="inst_road_01",
            caller_response_text="In this demo, a simulated ambulance is assigned.",
            prompt_tokens=150,
            completion_tokens=45,
            latency_ms=310.5,
        )

        # Validate provenance against context
        output.validate_against_context(context)

        # Fabricated utterance ID must fail validation
        bad_output = ReasoningOutput(
            utterance_role=UtteranceRole.NEW_DESCRIPTION,
            utterance_role_source_id=self.utterance_id_2,
            category=EmergencyCategory.ROAD_ACCIDENT,
            category_confidence=0.85,
            supporting_utterance_ids=[self.utterance_id_2],
            extracted_facts=[
                ProvenancedFact(
                    field_name="trapped_persons",
                    value=True,
                    source_utterance_id=uuid4(),  # Unknown utterance ID!
                    evidence_text="fabricated",
                )
            ],
            conflicts=[],
            missing_information_fields=[],
            proposed_followup_question=None,
            selected_instruction_id=None,
            caller_response_text="Simulated response.",
            prompt_tokens=100,
            completion_tokens=30,
            latency_ms=200.0,
        )
        with self.assertRaises(ReasoningSchemaValidationError):
            bad_output.validate_against_context(context)


if __name__ == "__main__":
    unittest.main()
