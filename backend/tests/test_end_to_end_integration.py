"""
End-to-end scripted integration tests for SETU (Section 28.10) and
adapter failure-mode & retry-policy tests (Section 28.1 & 23.2).

Verifies:
1. GeminiReasoningAdapter bounded 1-retry policy (Section 23.2):
   - Retries once on transient HTTP 503 / timeout and succeeds if 2nd attempt works.
   - Never retries ProviderBlockedContentError or ReasoningSchemaValidationError.
   - Honours Retry-After only if it fits within the remaining deadline.
2. GnaniPrismaAdapter & GnaniTimbreAdapter v2.5 tag sanitization and Gate 0.1/0.2 guards.
3. End-to-end multi-turn session journeys (Section 28.10):
   - Journey A: Code-mixed Tanglish -> English -> Hindi language switching mid-session,
     spoken landmark resolution, caller confirmation, additive reclassification
     (road_accident -> fire_with_injuries), and duplicate incident detection across
     two sessions within 200m and 5 minutes.
   - Journey B: WebSocket disconnect, signed token verification, reconnect, and
     complete structured handoff verification matching authoritative state.
"""

from datetime import datetime, timedelta, timezone
import unittest
from uuid import uuid4

from backend.app.adapters.base import (
    AudioChunk,
    DialogueContext,
    EmergencyCategory,
    FinalizedUtteranceRecord,
    ProvenancedFact,
    ProviderBlockedContentError,
    ReasoningOutput,
    ReasoningSchemaValidationError,
    ReasoningUnavailableError,
    STTSignalType,
    STTProviderError,
    TTSProviderError,
    UtteranceRole,
)
from backend.app.adapters.prisma import GnaniPrismaAdapter
from backend.app.adapters.reasoning import GeminiReasoningAdapter
from backend.app.adapters.timbre import GnaniTimbreAdapter, strip_unverified_v3_expressive_tags
from backend.app.main import (
    LiveSessionBridge,
    build_default_orchestrator,
    serialize_session_snapshot,
)
from backend.app.orchestrator.dispatch import AssignmentStatus, ResourceCategory
from backend.app.orchestrator.location import GPSPermissionStatus, LocationConfirmationStatus
from backend.app.orchestrator.session import SessionTokenError


class TestEndToEndIntegrationAndAdapters(unittest.IsolatedAsyncioTestCase):
    """Tests for Section 28.1 (Adapters) and Section 28.10 (End-to-End Integration)."""

    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.session_id = uuid4()
        self.incident_id = uuid4()
        self.utterance_id = uuid4()
        self.context = DialogueContext(
            session_id=self.session_id,
            incident_id=self.incident_id,
            active_language="en",
            prior_utterances=(),
            validated_facts=(),
            current_utterance=FinalizedUtteranceRecord(
                utterance_id=self.utterance_id,
                sequence_number=1,
                text="There is a fire in the shop",
                language="en",
                utterance_role=None,
                finalized_at=self.now,
            ),
            pending_followup_question=None,
            eligible_instruction_ids=("safety_fire_01",),
            backend_status_summary={},
        )

    async def test_gemini_reasoning_adapter_retries_once_on_transient_503_and_never_retries_blocks(self) -> None:
        """Section 23.2 & 28.1: Retries transient 503 once; never retries safety blocks."""
        call_count = 0

        async def flaky_transport(ctx: DialogueContext, _rem: float) -> ReasoningOutput:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ReasoningUnavailableError(
                    "HTTP 503 transient failure",
                    session_id=ctx.session_id,
                    utterance_id=ctx.current_utterance.utterance_id,
                    is_retryable=True,
                    http_status=503,
                    retry_after_seconds=0.01,
                )
            return ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=ctx.current_utterance.utterance_id,
                category=EmergencyCategory.FIRE,
                category_confidence=0.91,
                supporting_utterance_ids=(ctx.current_utterance.utterance_id,),
                extracted_facts=(
                    ProvenancedFact(
                        field_name="fire_reported",
                        value=True,
                        source_utterance_id=ctx.current_utterance.utterance_id,
                        evidence_text="fire in the shop",
                    ),
                ),
                conflicts=(),
                missing_information_fields=(),
                proposed_followup_question=None,
                selected_instruction_id="safety_fire_01",
                caller_response_text="In this demo, a simulated fire engine is being checked.",
                prompt_tokens=60,
                completion_tokens=20,
                latency_ms=110.0,
            )

        adapter = GeminiReasoningAdapter(
            model_id="gemini-test",
            transport_callable=flaky_transport,
        )
        out = await adapter.analyze_utterance(self.context, timeout_seconds=2.0)
        self.assertEqual(call_count, 2)
        self.assertEqual(out.category, EmergencyCategory.FIRE)

        # Safety-filter block must NEVER be retried
        block_calls = 0

        async def blocked_transport(ctx: DialogueContext, _rem: float) -> ReasoningOutput:
            nonlocal block_calls
            block_calls += 1
            raise ProviderBlockedContentError(
                "Blocked by safety filter",
                session_id=ctx.session_id,
                utterance_id=ctx.current_utterance.utterance_id,
                provider_finish_reason="SAFETY",
            )

        block_adapter = GeminiReasoningAdapter(
            model_id="gemini-test",
            transport_callable=blocked_transport,
        )
        with self.assertRaises(ProviderBlockedContentError):
            await block_adapter.analyze_utterance(self.context, timeout_seconds=2.0)
        self.assertEqual(block_calls, 1)

        # Retry-After exceeding remaining deadline must fail immediately without sleeping/retrying
        retry_after_calls = 0

        async def long_retry_after_transport(ctx: DialogueContext, _rem: float) -> ReasoningOutput:
            nonlocal retry_after_calls
            retry_after_calls += 1
            raise ReasoningUnavailableError(
                "HTTP 429 Rate Limit",
                session_id=ctx.session_id,
                utterance_id=ctx.current_utterance.utterance_id,
                is_retryable=True,
                http_status=429,
                retry_after_seconds=30.0,
            )

        rate_limit_adapter = GeminiReasoningAdapter(
            model_id="gemini-test",
            transport_callable=long_retry_after_transport,
        )
        with self.assertRaises(ReasoningUnavailableError):
            await rate_limit_adapter.analyze_utterance(self.context, timeout_seconds=0.5)
        self.assertEqual(retry_after_calls, 1)

    async def test_prisma_and_timbre_v25_adapters_and_tag_sanitization(self) -> None:
        """Section 4.1, 4.2 & 28.1: Strips v3 expressive tags and enforces Gate 0.1/0.2 guards."""
        raw = "<slow>Please stay calm.</slow> <pauses> Help in this simulation is being reviewed. <sigh>"
        cleaned = strip_unverified_v3_expressive_tags(raw)
        self.assertNotIn("<slow>", cleaned)
        self.assertNotIn("<pauses>", cleaned)
        self.assertNotIn("<sigh>", cleaned)
        self.assertIn("Please stay calm.", cleaned)

        prisma = GnaniPrismaAdapter()
        with self.assertRaises(STTProviderError):
            await prisma.open_stream(
                session_id=self.session_id,
                utterance_id=self.utterance_id,
                language_hint="te",
                timeout_seconds=1.0,
            )

        timbre = GnaniTimbreAdapter()
        with self.assertRaises(TTSProviderError):
            await timbre.start_synthesis(
                session_id=self.session_id,
                synthesis_id=uuid4(),
                text="Simulated status update.",
                language="en",
                voice_id="timbre-v2.5-en-in-std",
                timeout_seconds=1.0,
            )

        # Verify configured Prisma v2.5 streaming lifecycle (16 kHz mono PCM -> FINAL_TRANSCRIPT)
        async def _mock_prisma_finalize(chunks, lang, _timeout):
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].sample_rate_hz, 16000)
            return {
                "signal_type": "final_transcript",
                "text": "Anna, ikkada road accident jarigindi",
                "detected_language": lang,
                "language_confidence": 0.94,
                "transcript_confidence": 0.92,
            }

        configured_prisma = GnaniPrismaAdapter(
            endpoint_url="https://prisma.example.invalid/v2.5/stream",
            api_key="test-prisma-key",
            finalize_transport=_mock_prisma_finalize,
        )
        stt_stream = await configured_prisma.open_stream(
            session_id=self.session_id,
            utterance_id=self.utterance_id,
            language_hint="te",
            timeout_seconds=1.0,
        )
        await stt_stream.send_audio_chunk(
            AudioChunk(
                pcm_bytes=b"\x01\x00" * 160,
                sample_rate_hz=16000,
                sequence_number=1,
                captured_at=self.now,
            ),
            timeout_seconds=1.0,
        )
        final_evt = await stt_stream.finalize_utterance(timeout_seconds=1.0)
        self.assertEqual(final_evt.signal_type, STTSignalType.FINAL_TRANSCRIPT)
        self.assertEqual(final_evt.text, "Anna, ikkada road accident jarigindi")
        await stt_stream.close(timeout_seconds=1.0)
        self.assertFalse(stt_stream.is_active)

        # Verify configured Timbre v2.5 streaming lifecycle (v3 tag stripping -> 16 kHz PCM chunks -> cancel)
        async def _mock_timbre_chunks(clean_text, _lang, _voice, _timeout):
            self.assertNotIn("<slow>", clean_text)
            return [b"\x02\x00" * 160, b"\x03\x00" * 160]

        configured_timbre = GnaniTimbreAdapter(
            endpoint_url="https://timbre.example.invalid/v2.5/synthesize",
            api_key="test-timbre-key",
            chunk_transport=_mock_timbre_chunks,
        )
        tts_handle = await configured_timbre.start_synthesis(
            session_id=self.session_id,
            synthesis_id=uuid4(),
            text="<slow>In this simulation, stay calm.</slow>",
            language="en",
            voice_id="timbre-v2.5-en-in-std",
            timeout_seconds=1.0,
        )
        emitted = [
            c
            async for c in tts_handle.iter_audio_chunks(
                first_byte_timeout_seconds=1.0,
                chunk_timeout_seconds=1.0,
            )
        ]
        self.assertEqual(len(emitted), 2)
        self.assertTrue(emitted[0].is_first_byte)
        self.assertTrue(emitted[1].is_final_chunk)
        self.assertEqual(emitted[0].sample_rate_hz, 16000)

    async def test_end_to_end_multilingual_reclassification_duplicate_and_reconnect_journey(self) -> None:
        """
        Section 28.10: Complete scripted multi-session integration test verifying:
        1. Session 1: Poor GPS (>500m) -> pending_location -> spoken landmark match ->
           caller confirmation -> allocation -> mid-session language switching (te -> en -> hi) ->
           additive reclassification (road_accident -> fire_with_injuries) -> WebSocket reconnect ->
           serialized handoff state.
        2. Session 2: Created 2 minutes later at same confirmed landmark with same category ->
           flagged as possible duplicate without merging or cancelling assignments.
        """
        orchestrator, handoff_gen, audit, _metrics, _lib = build_default_orchestrator(demo_mode=False)
        reasoning = orchestrator._reasoning  # MockReasoningAdapter

        # --- SESSION 1 ---
        s1 = orchestrator.create_session(initial_language="te", now=self.now)
        # Step 1: Browser GPS reports 650m accuracy (> 500m threshold) -> rejected
        orchestrator.update_gps_location(
            s1.session_id,
            latitude=17.4337,
            longitude=78.5016,
            accuracy_m=650.0,
            permission_status=GPSPermissionStatus.GRANTED,
            now=self.now,
        )
        self.assertFalse(s1.location_state.is_usable_candidate)

        # Step 2: Caller speaks in Tanglish describing road accident near Secunderabad Railway Station
        u1_id = uuid4()
        reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=u1_id,
                category=EmergencyCategory.ROAD_ACCIDENT,
                category_confidence=0.93,
                supporting_utterance_ids=(u1_id,),
                extracted_facts=(
                    ProvenancedFact(
                        field_name="vehicle_collision_reported",
                        value=True,
                        source_utterance_id=u1_id,
                        evidence_text="road accident jarigindi",
                    ),
                    ProvenancedFact(
                        field_name="caller_location_text",
                        value="Near Secunderabad Railway Station",
                        source_utterance_id=u1_id,
                        evidence_text="Near Secunderabad Railway Station",
                    ),
                ),
                conflicts=(),
                missing_information_fields=(),
                proposed_followup_question="Do you confirm Secunderabad Railway Station as your location?",
                selected_instruction_id="safety_road_accident_01",
                caller_response_text="In this demo, please confirm if you are at Secunderabad Railway Station.",
                prompt_tokens=90,
                completion_tokens=25,
                latency_ms=180.0,
            )
        )
        await orchestrator.process_finalized_utterance(
            s1.session_id,
            utterance_id=u1_id,
            text="Anna ikkada road accident jarigindi near Secunderabad Railway Station",
            detected_language="te",
            now=self.now,
        )
        # Step 2: Unconfirmed landmark candidate allows provisional allocation with provisional_location_unconfirmed review flag
        self.assertEqual(len(s1.assignments), 2)
        for a in s1.assignments:
            self.assertEqual(a.status, AssignmentStatus.SIMULATED_ASSIGNED)
            self.assertIsNotNone(a.resource_id)
        self.assertIn("provisional_location_unconfirmed", s1.review_reason_codes)
        self.assertNotIn("pending_location", s1.review_reason_codes)

        # Step 3: Caller confirms location in English -> provisional_location_unconfirmed review flag cleared
        orchestrator.confirm_current_location(s1.session_id, now=self.now)
        for a in s1.assignments:
            self.assertEqual(a.status, AssignmentStatus.SIMULATED_ASSIGNED)
            self.assertIsNotNone(a.resource_id)
        self.assertNotIn("provisional_location_unconfirmed", s1.review_reason_codes)

        # Step 4: Caller switches to Hindi and reports the vehicle caught fire with injuries
        # Reclassification: road_accident (ambulance, police_patrol) -> fire_with_injuries (fire_engine, ambulance)
        # Existing police_patrol MUST remain assigned (additive-only reclassification)!
        u2_id = uuid4()
        reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=u2_id,
                category=EmergencyCategory.FIRE_WITH_INJURIES,
                category_confidence=0.95,
                supporting_utterance_ids=(u1_id, u2_id),
                extracted_facts=(
                    ProvenancedFact(
                        field_name="fire_reported",
                        value=True,
                        source_utterance_id=u2_id,
                        evidence_text="gaadi mein aag lag gayi hai",
                    ),
                ),
                conflicts=(),
                missing_information_fields=(),
                proposed_followup_question=None,
                selected_instruction_id="safety_fire_injuries_01",
                caller_response_text="In this demo, a simulated fire engine has been added.",
                prompt_tokens=120,
                completion_tokens=30,
                latency_ms=210.0,
            )
        )
        await orchestrator.process_finalized_utterance(
            s1.session_id,
            utterance_id=u2_id,
            text="Bhaiya abhi gaadi mein aag lag gayi hai aur log jal rahe hain",
            detected_language="hi",
            now=self.now,
        )
        self.assertEqual(s1.active_language, "hi")
        self.assertEqual(s1.category, EmergencyCategory.FIRE_WITH_INJURIES)
        assigned_cats = {a.category for a in s1.assignments}
        self.assertEqual(
            assigned_cats,
            {
                ResourceCategory.AMBULANCE,
                ResourceCategory.POLICE_PATROL,
                ResourceCategory.FIRE_ENGINE,
            },
        )

        # Step 5: Test WebSocket reconnect token validation
        with self.assertRaises(SessionTokenError):
            orchestrator.reconnect_session(
                s1.session_id,
                session_token="invalid:token:sig",
                last_received_sequence=1,
                now=self.now,
            )
        reconnected = orchestrator.reconnect_session(
            s1.session_id,
            session_token=s1.session_token,
            last_received_sequence=2,
            now=self.now,
        )
        self.assertEqual(reconnected.session_state, "active")

        # --- SESSION 2 (Duplicate Incident within 200m and 5 minutes) ---
        s2_time = self.now + timedelta(minutes=2)
        s2 = orchestrator.create_session(initial_language="en", now=s2_time)
        orchestrator.update_gps_location(
            s2.session_id,
            latitude=17.4338,  # ~15m from Session 1's confirmed Secunderabad station coordinates
            longitude=78.5017,
            accuracy_m=25.0,
            permission_status=GPSPermissionStatus.GRANTED,
            confirmed_by_caller=True,
            now=s2_time,
        )
        u_s2 = uuid4()
        reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=u_s2,
                category=EmergencyCategory.FIRE_WITH_INJURIES,
                category_confidence=0.90,
                supporting_utterance_ids=(u_s2,),
                extracted_facts=(
                    ProvenancedFact(
                        field_name="fire_reported",
                        value=True,
                        source_utterance_id=u_s2,
                        evidence_text="vehicle fire with injuries",
                    ),
                ),
                conflicts=(),
                missing_information_fields=(),
                proposed_followup_question=None,
                selected_instruction_id="safety_fire_injuries_01",
                caller_response_text="In this demo, simulated assignments are being checked.",
                prompt_tokens=80,
                completion_tokens=20,
                latency_ms=160.0,
            )
        )
        await orchestrator.process_finalized_utterance(
            s2.session_id,
            utterance_id=u_s2,
            text="There is a vehicle fire with injuries outside Secunderabad station",
            detected_language="en",
            now=s2_time,
        )
        self.assertIsNotNone(s2.duplicate_match)
        self.assertEqual(s2.duplicate_match.matched_incident_id, s1.incident_id)
        self.assertIn("possible_duplicate_incident", s2.review_reason_codes)

        # Verify serialized snapshot matches authoritative backend state
        snap = serialize_session_snapshot(s2.session_id, orchestrator, handoff_gen, audit)
        self.assertEqual(snap["category"], "fire_with_injuries")
        self.assertTrue(snap["handoff"]["possible_duplicate_flag"])
        self.assertEqual(snap["handoff"]["matched_duplicate_incident_id"], str(s1.incident_id))

    async def test_live_session_ws_bridge_explosion_pcm_and_silence_evaluation(self) -> None:
        """
        Verify that LiveSessionBridge (/ws/session protocol) executes:
        - session_init with placeholder disclosure asset metadata (development_placeholder_pending_gate_0_2)
        - audio_chunk_pcm16 through GnaniPrismaAdapter returning stt_gate_blocked while Gate 0.1 is open
        - scripted_finalized_utterance for 'explosion' mapping to fire_engine + ambulance + police_patrol (§10.1)
        - evaluate_caller_silence through CallerTimeoutCoordinator (Rule 5 & §19)
        """
        bridge = LiveSessionBridge()
        init_res = await bridge.handle_ws_message(
            {
                "type": "session_init",
                "initial_language": "en",
                "latitude": 17.4435,
                "longitude": 78.3772,
                "accuracy_m": 42.0,
            }
        )
        self.assertEqual(init_res["type"], "session_state")
        self.assertEqual(
            init_res["cached_instruction_audio"]["generation_status"],
            "development_placeholder_pending_gate_0_2",
        )

        # 16 kHz mono PCM frame ingested -> Prisma Gate 0.1 guard
        pcm_res = await bridge.handle_ws_message(
            {
                "type": "audio_chunk_pcm16",
                "pcm_base64": "AAAA",
                "sample_rate_hz": 16000,
                "sequence_number": 1,
            }
        )
        self.assertEqual(pcm_res["type"], "stt_gate_blocked")
        self.assertEqual(pcm_res["pcm_chunks_ingested"], 1)

        # Explosion utterance -> §10.1 mapping (fire_engine + ambulance + police_patrol)
        exp_res = await bridge.handle_ws_message(
            {
                "type": "scripted_finalized_utterance",
                "text": "Loud explosion at Secunderabad godown, two injured.",
                "language": "en",
                "utterance_role": "new_description",
                "category": "explosion",
                "confidence": 0.95,
                "extracted_facts": [
                    {
                        "field": "explosion_reported",
                        "value": True,
                        "evidence_text": "Loud explosion",
                    }
                ],
            }
        )
        self.assertEqual(exp_res["type"], "session_state")
        self.assertEqual(exp_res["snapshot"]["category"], "explosion")
        assigned_cats = {a["resource_category"] for a in exp_res["snapshot"]["assignments"]}
        self.assertEqual(assigned_cats, {"fire_engine", "ambulance", "police_patrol"})

        # Silence evaluation -> Rule 5 check-in prompt without diagnosing unconsciousness
        sil_res = await bridge.handle_ws_message(
            {
                "type": "evaluate_caller_silence",
                "idle_seconds": 16.0,
                "check_ins_sent": 0,
            }
        )
        self.assertEqual(sil_res["type"], "session_state")
        self.assertEqual(sil_res["timeout_evaluation"]["action"], "play_check_in_prompt")
        self.assertFalse(sil_res["timeout_evaluation"]["flag_control_room"])


if __name__ == "__main__":
    unittest.main()

