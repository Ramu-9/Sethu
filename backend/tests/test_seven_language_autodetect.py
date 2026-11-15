from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

from backend.app.adapters.base import (
    ALL_SUPPORTED_LANGUAGES,
    CORE_MANUAL_LANGUAGES,
    EmergencyCategory,
    ReasoningOutput,
    TranscriptEvent,
)
from backend.app.adapters.mock import MockReasoningAdapter
from backend.app.adapters.prisma import GnaniPrismaAdapter, LANGUAGE_CODE_MAP
from backend.app.adapters.reasoning import (
    build_compact_dialogue_prompt,
    select_adaptive_thinking_level,
)
from backend.app.adapters.timbre import (
    DEFAULT_TIMBRE_V25_VOICE_BY_LANGUAGE,
    GnaniTimbreAdapter,
    TIMBRE_LANGUAGE_CODE_MAP,
)
from backend.app.audit.writer import InMemoryAuditWriter
from backend.app.config import load_settings
from backend.app.main import LiveSessionBridge
from backend.app.observability.metrics import MetricsRegistry
from backend.app.orchestrator.announcements import (
    build_demo_dispatch_announcement,
    build_demo_lifecycle_announcement,
    build_pending_location_dispatch_message,
    deterministic_demo_classify_and_extract,
)
from backend.app.orchestrator.dispatch import (
    DemoLifecycleStatus,
    DispatchEngine,
    InMemoryResourceRepository,
    ResourceCategory,
    SimulatedAssignment,
    SyntheticResourceRecord,
)
from backend.app.orchestrator.duplicates import DuplicateDetector
from backend.app.orchestrator.handoff import HandoffGenerator
from backend.app.orchestrator.hospital import HospitalRanker
from backend.app.orchestrator.instructions import InstructionLibrary
from backend.app.orchestrator.location import GPSPermissionStatus, LocationResolver
from backend.app.orchestrator.policy import PolicyValidator
from backend.app.orchestrator.session import SessionOrchestrator, SessionTokenSigner


class SevenLanguageAutoDetectTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)

    def _build_orchestrator(self) -> tuple[SessionOrchestrator, MockReasoningAdapter, InMemoryAuditWriter]:
        resources = [
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
        repo = InMemoryResourceRepository(resources)
        dispatch = DispatchEngine(repo)
        location_resolver = LocationResolver()
        hospital_ranker = HospitalRanker()
        instruction_lib = InstructionLibrary(timbre_gate_0_2_verified=True)
        policy = PolicyValidator(instruction_lib)
        duplicates = DuplicateDetector()
        audit = InMemoryAuditWriter()
        metrics = MetricsRegistry()
        reasoning = MockReasoningAdapter()
        token_signer = SessionTokenSigner(secret_key="test-secret-signing-key-32-bytes!!")
        orchestrator = SessionOrchestrator(
            reasoning_adapter=reasoning,
            dispatch_engine=dispatch,
            location_resolver=location_resolver,
            hospital_ranker=hospital_ranker,
            instruction_library=instruction_lib,
            policy_validator=policy,
            duplicate_detector=duplicates,
            audit_writer=audit,
            metrics=metrics,
            token_signer=token_signer,
        )
        return orchestrator, reasoning, audit

    def test_reconciled_126_wav_manifest_integrity(self) -> None:
        """Verify 126 WAV assets (18 instructions x 7 languages) exist and match SHA-256."""
        repo_root = Path(__file__).resolve().parents[2]
        manifest_path = repo_root / "backend-media" / "instructions" / "manifest.json"
        self.assertTrue(manifest_path.exists(), "manifest.json must exist")

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest_data.get("entries", []) if isinstance(manifest_data, dict) else manifest_data
        self.assertEqual(len(rows), 126, "Manifest must contain 126 rows (18 instructions x 7 languages)")

        by_lang: dict[str, int] = {}
        for row in rows:
            lang = row["language"]
            by_lang[lang] = by_lang.get(lang, 0) + 1
            self.assertEqual(row["generation_status"], "ready")

            wav_path = repo_root / row["file_path"]
            self.assertTrue(wav_path.exists(), f"Missing WAV file: {wav_path}")
            wav_bytes = wav_path.read_bytes()
            self.assertGreater(len(wav_bytes), 16000, f"WAV too small: {wav_path}")
            digest = hashlib.sha256(wav_bytes).hexdigest()
            self.assertEqual(digest, row["checksum_sha256"], f"SHA-256 mismatch for {wav_path}")

        for lang in ALL_SUPPORTED_LANGUAGES:
            self.assertEqual(by_lang.get(lang), 18, f"Expected 18 instructions for {lang}")

        lib = InstructionLibrary(timbre_gate_0_2_verified=True)
        for lang in ALL_SUPPORTED_LANGUAGES:
            inst, asset = lib.resolve_cached_asset("disclosure_session_start", lang)
            self.assertEqual(inst.language, lang)
            self.assertIsNotNone(asset)

    def test_multilingual_feature_flag_defaults_false_in_code(self) -> None:
        """Verify MULTILINGUAL_AUTO_DETECT_ENABLED defaults to False when unset and enables when true."""
        default_settings = load_settings(env_override={})
        self.assertFalse(default_settings.multilingual_auto_detect_enabled)
        enabled_settings = load_settings(env_override={"MULTILINGUAL_AUTO_DETECT_ENABLED": "true"})
        self.assertTrue(enabled_settings.multilingual_auto_detect_enabled)
        self.assertEqual(CORE_MANUAL_LANGUAGES, ("en", "te", "hi"))
        self.assertEqual(ALL_SUPPORTED_LANGUAGES, ("en", "te", "hi", "kn", "ta", "ml", "bn"))

    async def test_all_seven_languages_end_to_end_classification_dispatch_and_handoff(self) -> None:
        """Test each of the 7 languages individually for classification, extraction, safety WAV, and handoff."""
        utterances_by_lang = {
            "en": (
                "Ambulance needed urgently near HITEC City Cyber Towers, road accident and two people are bleeding.",
                EmergencyCategory.ROAD_ACCIDENT,
            ),
            "te": (
                "హైటెక్ సిటీ సైబర్ టవర్స్ దగ్గర రోడ్డు ప్రమాదం జరిగింది, ఇద్దరికి రక్తస్రావం అవుతోంది, వెంటనే అంబులెన్స్ పంపండి.",
                EmergencyCategory.ROAD_ACCIDENT,
            ),
            "hi": (
                "हाईटेक सिटी साइबर टावर्स के पास सड़क हादसा हुआ है, दो लोग घायल हैं और खून बह रहा है, तुरंत एम्बुलेंस भेजिए।",
                EmergencyCategory.ROAD_ACCIDENT,
            ),
            "kn": (
                "ಹೈಟೆಕ್ ಸಿಟಿ ಸೈಬರ್ ಟವರ್ಸ್ ಹತ್ತಿರ ರಸ್ತೆ ಅಪಘಾತವಾಗಿದೆ, ತೀವ್ರ ರಕ್ತಸ್ರಾವವಾಗುತ್ತಿದೆ, ತಕ್ಷಣ ಅಂಬ್ಯುಲೆನ್ಸ್ ಕಳುಹಿಸಿ.",
                EmergencyCategory.ROAD_ACCIDENT,
            ),
            "ta": (
                "ஹைடெக் சிட்டி சைபர் டவர்ஸ் அருகே சாலை விபத்து நடந்துள்ளது, ரத்தப்போக்கு அதிகமாக உள்ளது, உடனே ஆம்புலன்ஸ் அனுப்புங்கள்.",
                EmergencyCategory.ROAD_ACCIDENT,
            ),
            "ml": (
                "ഹൈടെക് സിറ്റി സൈബർ ടവേഴ്സ് സമീപം റോഡ് അപകടം ഉണ്ടായി, കടുത്ത രക്തസ്രാവമുണ്ട്, ഉടൻ ആംബുലൻസ് അയക്കൂ.",
                EmergencyCategory.ROAD_ACCIDENT,
            ),
            "bn": (
                "হাইটেক সিটি সাইবার টাওয়ার্সের কাছে সড়ক দুর্ঘটনা ঘটেছে, প্রচণ্ড রক্তপাত হচ্ছে, অবিলম্বে অ্যাম্বুলেন্স পাঠান।",
                EmergencyCategory.ROAD_ACCIDENT,
            ),
        }

        lib = InstructionLibrary(timbre_gate_0_2_verified=True)
        handoff_gen = HandoffGenerator()

        for lang, (caller_text, expected_category) in utterances_by_lang.items():
            with self.subTest(language=lang):
                self.assertIn(lang, LANGUAGE_CODE_MAP)
                self.assertIn(lang, TIMBRE_LANGUAGE_CODE_MAP)
                self.assertEqual(DEFAULT_TIMBRE_V25_VOICE_BY_LANGUAGE[lang], "Pranav")

                u_id = uuid4()
                cat, role, facts, followup, caller_resp = deterministic_demo_classify_and_extract(
                    text=caller_text,
                    utterance_id=u_id,
                    active_language=lang,  # type: ignore[arg-type]
                    existing_category=EmergencyCategory.UNKNOWN,
                    existing_established=False,
                    existing_facts=[],
                    pending_followup_question=None,
                )
                self.assertEqual(cat, expected_category)
                self.assertTrue(caller_resp)

                orchestrator, reasoning, audit = self._build_orchestrator()
                session = orchestrator.create_session(
                    initial_language=lang,  # type: ignore[arg-type]
                    language_mode="auto",
                    now=self.now,
                )
                orchestrator.update_gps_location(
                    session.session_id,
                    latitude=17.4504,
                    longitude=78.3808,
                    accuracy_m=25.0,
                    permission_status=GPSPermissionStatus.GRANTED,
                    confirmed_by_caller=True,
                    now=self.now,
                )

                reasoning.enqueue_output(
                    ReasoningOutput(
                        utterance_role=role,
                        utterance_role_source_id=u_id,
                        category=cat,
                        category_confidence=0.92,
                        supporting_utterance_ids=[u_id],
                        extracted_facts=facts,
                        conflicts=[],
                        missing_information_fields=[],
                        proposed_followup_question=followup,
                        selected_instruction_id="safety_road_accident_01",
                        caller_response_text=caller_resp,
                        prompt_tokens=90,
                        completion_tokens=30,
                        latency_ms=140.0,
                    )
                )
                state = await orchestrator.process_finalized_utterance(
                    session.session_id,
                    utterance_id=u_id,
                    text=caller_text,
                    detected_language=lang,  # type: ignore[arg-type]
                    now=self.now,
                )
                self.assertEqual(state.active_language, lang)
                self.assertEqual(state.category, expected_category)
                self.assertGreaterEqual(len(state.assignments), 1)

                safety_inst = lib.get_safety_guidance_for_category(
                    expected_category,
                    lang,  # type: ignore[arg-type]
                    context_text=caller_text,
                )
                self.assertIsNotNone(safety_inst)
                assert safety_inst is not None
                self.assertEqual(safety_inst.language, lang)

                dispatch_msg = build_demo_dispatch_announcement(
                    state.assignments,
                    lang,  # type: ignore[arg-type]
                )
                self.assertTrue(len(dispatch_msg) > 10)

                from dataclasses import replace as dc_replace
                arrived_assignment = dc_replace(
                    state.assignments[0],
                    lifecycle_status=DemoLifecycleStatus.ARRIVED.value,
                )
                lifecycle_msg = build_demo_lifecycle_announcement(
                    [arrived_assignment],
                    lang,  # type: ignore[arg-type]
                )
                self.assertIn("SIM-AMB-01", lifecycle_msg)

                pending_loc_msg = build_pending_location_dispatch_message(
                    [ResourceCategory.AMBULANCE],
                    lang,  # type: ignore[arg-type]
                )
                self.assertTrue(len(pending_loc_msg) > 10)

                handoff = handoff_gen.generate_handoff(
                    state,
                    audit_events=audit.list_events(),
                    generated_at=self.now,
                )
                self.assertEqual(handoff.initial_language, lang)
                self.assertEqual(handoff.active_language, lang)
                self.assertEqual(handoff.language_mode, "auto")
                self.assertIn(lang, handoff.detected_languages_seen)

    async def test_five_required_mid_call_language_switches_preserve_context(self) -> None:
        """Verify mid-call switches (te->en, hi->ta, kn->en, ml->hi, bn->en) preserve session state and dispatch."""
        switch_pairs = [
            (
                "te",
                "మాదాపూర్ మెట్రో దగ్గర రోడ్డు ప్రమాదం జరిగింది, అంబులెన్స్ పంపండి.",
                "en",
                "There are 2 injured people and one person is bleeding heavily.",
            ),
            (
                "hi",
                "माधापुर मेट्रो के पास आग लगी है, तुरंत दमकल भेजिए।",
                "ta",
                "இங்கே புகை அதிகமாக உள்ளது, இரண்டு பேர் உள்ளே சிக்கியுள்ளனர்.",
            ),
            (
                "kn",
                "ಮಾಧಾಪುರ ಮೆಟ್ರೋ ಹತ್ತಿರ ರಸ್ತೆ ಅಪಘಾತವಾಗಿದೆ, ತಕ್ಷಣ ಅಂಬ್ಯುಲೆನ್ಸ್ ಕಳುಹಿಸಿ.",
                "en",
                "Please hurry, the patient is unconscious near Gate 2.",
            ),
            (
                "ml",
                "മാധാപൂർ മെട്രോ സമീപം റോഡ് അപകടം ഉണ്ടായി, ആംബുലൻസ് വേണം.",
                "hi",
                "यहाँ दो लोग घायल हैं और खून बह रहा है।",
            ),
            (
                "bn",
                "মাধাপুর মেট্রোর কাছে সড়ক দুর্ঘটনা ঘটেছে, অ্যাম্বুলেন্স পাঠান।",
                "en",
                "One victim has severe bleeding and we are waiting at the main road.",
            ),
        ]

        handoff_gen = HandoffGenerator()
        for lang_a, text_a, lang_b, text_b in switch_pairs:
            with self.subTest(switch=f"{lang_a}->{lang_b}"):
                orchestrator, reasoning, audit = self._build_orchestrator()
                session = orchestrator.create_session(
                    initial_language=lang_a,  # type: ignore[arg-type]
                    language_mode="auto",
                    now=self.now,
                )
                orchestrator.update_gps_location(
                    session.session_id,
                    latitude=17.4435,
                    longitude=78.3772,
                    accuracy_m=30.0,
                    permission_status=GPSPermissionStatus.GRANTED,
                    confirmed_by_caller=True,
                    now=self.now,
                )
                sid_before = session.session_id
                iid_before = session.incident_id

                # Turn 1 in lang_a
                u1_id = uuid4()
                cat1, role1, facts1, followup1, resp1 = deterministic_demo_classify_and_extract(
                    text=text_a,
                    utterance_id=u1_id,
                    active_language=lang_a,  # type: ignore[arg-type]
                    existing_category=EmergencyCategory.UNKNOWN,
                    existing_established=False,
                    existing_facts=[],
                    pending_followup_question=None,
                )
                reasoning.enqueue_output(
                    ReasoningOutput(
                        utterance_role=role1,
                        utterance_role_source_id=u1_id,
                        category=cat1,
                        category_confidence=0.91,
                        supporting_utterance_ids=[u1_id],
                        extracted_facts=facts1,
                        conflicts=[],
                        missing_information_fields=[],
                        proposed_followup_question=followup1,
                        selected_instruction_id="safety_road_accident_01",
                        caller_response_text=resp1,
                        prompt_tokens=80,
                        completion_tokens=25,
                        latency_ms=130.0,
                    )
                )
                state1 = await orchestrator.process_finalized_utterance(
                    session.session_id,
                    utterance_id=u1_id,
                    text=text_a,
                    detected_language=lang_a,  # type: ignore[arg-type]
                    now=self.now,
                )
                self.assertEqual(state1.active_language, lang_a)
                assignments_after_turn1 = list(state1.assignments)
                self.assertGreaterEqual(len(assignments_after_turn1), 1)

                # Turn 2 mid-call switch to lang_b without restarting session
                u2_id = uuid4()
                cat2, role2, facts2, followup2, resp2 = deterministic_demo_classify_and_extract(
                    text=text_b,
                    utterance_id=u2_id,
                    active_language=lang_b,  # type: ignore[arg-type]
                    existing_category=state1.category,
                    existing_established=state1.category_established_above_threshold,
                    existing_facts=state1.validated_facts,
                    pending_followup_question=state1.pending_followup_question,
                )
                reasoning.enqueue_output(
                    ReasoningOutput(
                        utterance_role=role2,
                        utterance_role_source_id=u2_id,
                        category=cat2,
                        category_confidence=0.93,
                        supporting_utterance_ids=[u1_id, u2_id],
                        extracted_facts=facts2,
                        conflicts=[],
                        missing_information_fields=[],
                        proposed_followup_question=followup2,
                        selected_instruction_id="safety_road_accident_01",
                        caller_response_text=resp2,
                        prompt_tokens=90,
                        completion_tokens=30,
                        latency_ms=135.0,
                    )
                )
                state2 = await orchestrator.process_finalized_utterance(
                    session.session_id,
                    utterance_id=u2_id,
                    text=text_b,
                    detected_language=lang_b,  # type: ignore[arg-type]
                    now=self.now,
                )

                self.assertEqual(state2.session_id, sid_before)
                self.assertEqual(state2.incident_id, iid_before)
                self.assertEqual(state2.active_language, lang_b)
                self.assertEqual(state2.detected_languages_seen, [lang_a, lang_b])
                self.assertGreaterEqual(len(state2.assignments), len(assignments_after_turn1))

                switched_events = [
                    ev for ev in audit.list_events() if ev.event_type == "language_switched"
                ]
                self.assertEqual(len(switched_events), 1)
                self.assertEqual(switched_events[0].details["previous_language"], lang_a)
                self.assertEqual(switched_events[0].details["new_language"], lang_b)

                handoff = handoff_gen.generate_handoff(
                    state2,
                    audit_events=audit.list_events(),
                    generated_at=self.now,
                )
                self.assertEqual(handoff.initial_language, lang_a)
                self.assertEqual(handoff.active_language, lang_b)
                self.assertEqual(list(handoff.detected_languages_seen), [lang_a, lang_b])

    def test_edge_cases_out_of_scope_language_low_confidence_and_provider_verification(self) -> None:
        """Verify out-of-scope language (mr-IN), low confidence, and strict gate_0_3_gemini_verified."""
        from backend.app.adapters.base import DialogueContext, FinalizedUtteranceRecord

        orchestrator, _, audit = self._build_orchestrator()
        session = orchestrator.create_session(
            initial_language="kn",
            language_mode="auto",
            now=self.now,
        )

        # Out-of-scope language (Marathi mr-IN) returned by indic-auto must NOT overwrite active_language
        state_kn = orchestrator.record_uncertain_language_detection(
            session.session_id,
            raw_detected_language_code="mr-IN",
            language_confidence=0.88,
            request_clarification_if_needed=True,
            now=self.now,
        )
        self.assertEqual(state_kn.active_language, "kn")
        self.assertEqual(state_kn.last_selected_instruction_id, "didnt_catch_that")
        self.assertTrue(len(state_kn.last_spoken_response_text) > 10)

        # Low confidence detection (< 0.65) preserves active_language
        state_low = orchestrator.record_uncertain_language_detection(
            session.session_id,
            raw_detected_language_code="ta-IN",
            language_confidence=0.41,
            request_clarification_if_needed=True,
            now=self.now,
        )
        self.assertEqual(state_low.active_language, "kn")
        self.assertEqual(state_low.last_selected_instruction_id, "didnt_catch_that")

        uncertain_events = [
            ev for ev in audit.list_events() if ev.event_type == "language_detection_uncertain"
        ]
        self.assertEqual(len(uncertain_events), 2)

        # Verify LiveSessionBridge keeps gate_0_3_gemini_verified False until a real Gemini call succeeds
        bridge = LiveSessionBridge()
        self.assertFalse(bridge.gate_0_3_gemini_verified)

        # Verify adaptive thinking and compact dialogue prompt support all 7 languages
        for lang in ALL_SUPPORTED_LANGUAGES:
            ctx = DialogueContext(
                session_id=session.session_id,
                incident_id=session.incident_id,
                active_language=lang,  # type: ignore[arg-type]
                current_utterance=FinalizedUtteranceRecord(
                    utterance_id=uuid4(),
                    sequence_number=1,
                    text="Heavy bleeding and unconscious patient",
                    language=lang,  # type: ignore[arg-type]
                    utterance_role=None,
                    finalized_at=self.now,
                ),
                prior_utterances=[],
                validated_facts=[],
                pending_followup_question=None,
                eligible_instruction_ids=["safety_medical_01"],
                backend_status_summary={"current_category": "unknown", "location_confirmed": 1},
            )
            level = select_adaptive_thinking_level(ctx)
            self.assertEqual(level, "MEDIUM")
            sys_inst, prompt = build_compact_dialogue_prompt(ctx)
            self.assertTrue(len(sys_inst) > 20)
            self.assertTrue(len(prompt) > 20)

    async def test_live_prisma_indic_auto_and_timbre_seven_languages_roundtrip(self) -> None:
        """Verify live Timbre v2.5 synthesis + live Prisma v3.0 indic-auto STT across all 7 languages."""
        from backend.app.adapters.base import AudioChunk

        settings = load_settings()
        if not settings.prisma_auth_token or not settings.timbre_auth_token:
            self.skipTest("Gnani credentials not configured in environment")

        timbre = GnaniTimbreAdapter(
            endpoint_url=settings.timbre_endpoint_url,
            api_key=settings.timbre_auth_token,
        )
        prisma = GnaniPrismaAdapter(
            endpoint_url=settings.prisma_endpoint_url,
            api_key=settings.prisma_auth_token,
        )

        # Verify live Timbre v2.5 synthesis on one new language (Kannada)
        kn_pcm = timbre.synthesize_pcm_sync(
            text="ಮಾಧಾಪುರ ಮೆಟ್ರೋ ಹತ್ತಿರ ತುರ್ತು ಪರಿಸ್ಥಿತಿ ಇದೆ, ತಕ್ಷಣ ಅಂಬ್ಯುಲೆನ್ಸ್ ಕಳುಹಿಸಿ.",
            language="kn",
            voice_id="Pranav",
            timeout_seconds=15.0,
        )
        self.assertGreater(len(kn_pcm), 8000)

        # Verify live Prisma v3.0 indic-auto across all 7 languages using the verified 16 kHz Timbre WAVs
        repo_root = Path(__file__).resolve().parents[2]
        for lang in ALL_SUPPORTED_LANGUAGES:
            wav_path = repo_root / "backend-media" / "instructions" / lang / "safety_road_accident_01" / "v1.0.wav"
            pcm_bytes = wav_path.read_bytes()[44:]
            stream = await prisma.open_stream(
                session_id=uuid4(),
                utterance_id=uuid4(),
                language_hint="en",
                timeout_seconds=15.0,
                auto_detect_language=True,
            )
            await stream.send_audio_chunk(
                AudioChunk(
                    pcm_bytes=pcm_bytes,
                    sample_rate_hz=16000,
                    sequence_number=0,
                    captured_at=self.now,
                ),
                timeout_seconds=5.0,
            )
            event: TranscriptEvent = await stream.finalize_utterance(timeout_seconds=15.0)
            await stream.close(timeout_seconds=2.0)
            self.assertIsNotNone(event.text, f"Empty transcript for {lang}")
            assert event.text is not None
            self.assertTrue(len(event.text.strip()) > 0, f"Blank transcript for {lang}")
            self.assertEqual(
                event.detected_language,
                lang,
                f"Expected indic-auto to detect {lang}, got {event.detected_language} (raw={event.raw_detected_language_code})",
            )

    async def test_direct_seven_language_selection_and_mid_session_switching(self) -> None:
        """
        Verify direct selection across all 7 languages:
        1. Starting a session directly in any of the 7 languages sets active_language and manual_language.
        2. Emergency utterances stay in the selected language even if a stale client payload passes 'en'.
        3. Switching languages mid-session updates active_language without resetting incident_id, utterances, or dispatch.
        """
        from backend.app.adapters.base import ProvenancedFact, UtteranceRole
        from backend.app.main import build_default_orchestrator
        from backend.app.orchestrator.session import LANGUAGE_DISPLAY_NAMES_BY_CODE

        utterances_by_lang = {
            "en": "My father has severe chest pain near Madhapur Metro.",
            "te": "మా నాన్నకు మాదాపూర్ మెట్రో దగ్గర తీవ్రమైన ఛాతీ నొప్పిగా ఉంది.",
            "hi": "मेरे पिताजी को माधापुर मेट्रो के पास तेज सीने में दर्द है।",
            "kn": "ನನ್ನ ತಂದೆಗೆ ಮಾಧಾಪುರ ಮೆಟ್ರೋ ಹತ್ತಿರ ತೀವ್ರ ಎದೆ ನೋವು ಇದೆ.",
            "ta": "மாதாப்பூர் மெட்ரோ அருகே என் தந்தைக்கு கடுமையான நெஞ்சு வலி உள்ளது.",
            "ml": "മാധാപൂർ മെട്രോയ്ക്ക് സമീപം എന്റെ അച്ഛന് കഠിനമായ നെഞ്ചുവേദനയുണ്ട്.",
            "bn": "মাধাপুর মেট্রোর কাছে আমার বাবার তীব্র বুকে ব্যথা হচ্ছে।",
        }

        for lang_code in ALL_SUPPORTED_LANGUAGES:
            self.assertIn(lang_code, LANGUAGE_DISPLAY_NAMES_BY_CODE)
            orchestrator, handoff_gen, audit, _, _ = build_default_orchestrator(
                timbre_gate_0_2_verified=True,
                demo_mode=True,
            )
            session = orchestrator.create_session(
                initial_language=lang_code,  # type: ignore[arg-type]
                language_mode="manual",
                now=self.now,
            )
            self.assertEqual(session.active_language, lang_code)
            self.assertEqual(session.initial_language, lang_code)
            self.assertEqual(session.language_mode, "manual")

            bridge = LiveSessionBridge()
            bridge.orchestrator = orchestrator
            bridge.handoff_gen = handoff_gen
            bridge.audit = audit
            bridge._active_session_id = session.session_id

            utt_id = uuid4()
            orchestrator._reasoning.enqueue_output(
                ReasoningOutput(
                    utterance_role=UtteranceRole.NEW_DESCRIPTION,
                    utterance_role_source_id=utt_id,
                    category=EmergencyCategory.MEDICAL_EMERGENCY,
                    category_confidence=0.95,
                    supporting_utterance_ids=[utt_id],
                    extracted_facts=[
                        ProvenancedFact(
                            field_name="caller_landmark",
                            value="Madhapur Metro",
                            source_utterance_id=utt_id,
                            evidence_text="Madhapur Metro",
                        )
                    ],
                    conflicts=[],
                    missing_information_fields=[],
                    proposed_followup_question=None,
                    selected_instruction_id="safety_medical_01",
                    caller_response_text="Emergency help is being coordinated.",
                    prompt_tokens=110,
                    completion_tokens=40,
                    latency_ms=95.0,
                )
            )
            ws_resp = await bridge.handle_ws_message(
                {
                    "type": "live_gemini_utterance",
                    "session_id": str(session.session_id),
                    "utterance_id": str(utt_id),
                    "text": utterances_by_lang[lang_code],
                    "language": "en",  # Stale client default must not override active manual language
                }
            )
            self.assertEqual(ws_resp["event"], "live_gemini_utterance_processed")
            snap = ws_resp["snapshot"]
            self.assertEqual(snap["active_language"], lang_code)
            self.assertEqual(snap["utterances"][-1]["language"], lang_code)
            self.assertEqual(snap["handoff"]["active_language"], lang_code)

            # Switch language mid-session directly via set_language_mode and verify incident state is preserved
            next_lang = "kn" if lang_code != "kn" else "ta"
            orig_incident_id = snap["incident_id"]
            switch_resp = await bridge.handle_ws_message(
                {
                    "type": "set_language_mode",
                    "session_id": str(session.session_id),
                    "language_mode": "manual",
                    "manual_language": next_lang,
                }
            )
            self.assertEqual(ws_resp["event"], "live_gemini_utterance_processed")
            self.assertEqual(switch_resp["event"], "language_mode_updated")
            switched_snap = switch_resp["snapshot"]
            self.assertEqual(switched_snap["active_language"], next_lang)
            self.assertEqual(switched_snap["language_mode"], "manual")
            self.assertEqual(switched_snap["incident_id"], orig_incident_id)
            self.assertEqual(len(switched_snap["utterances"]), 1)

    async def test_gemini_diagnostic_classification_and_single_attempt_timeout(self) -> None:
        """
        Verify structured Gemini diagnostic failure classification and that a timeout
        does not trigger a duplicate live retry call.
        """
        from backend.app.adapters.base import (
            DialogueContext,
            FinalizedUtteranceRecord,
            ReasoningUnavailableError,
        )
        from backend.app.adapters.reasoning import (
            GeminiReasoningAdapter,
            classify_gemini_diagnostic_failure,
        )

        sid = uuid4()
        uid = uuid4()
        cat_408 = classify_gemini_diagnostic_failure(
            ReasoningUnavailableError(
                "Gemini network/timeout failure: The read operation timed out",
                session_id=sid,
                utterance_id=uid,
                http_status=408,
                is_retryable=False,
            )
        )
        self.assertEqual(cat_408["category"], "timeout_exceeded")

        cat_404 = classify_gemini_diagnostic_failure(
            ReasoningUnavailableError(
                "Gemini HTTP 404: models/gemini-3.8-flash is not found",
                session_id=sid,
                utterance_id=uid,
                http_status=404,
                is_retryable=False,
            )
        )
        self.assertEqual(cat_404["category"], "http_404_model_not_found")

        cat_429 = classify_gemini_diagnostic_failure(
            ReasoningUnavailableError(
                "Gemini HTTP 429: Resource has been exhausted",
                session_id=sid,
                utterance_id=uid,
                http_status=429,
                is_retryable=True,
            )
        )
        self.assertEqual(cat_429["category"], "http_429_rate_limit_or_quota")

        calls = 0

        async def slow_transport(ctx_arg: DialogueContext, timeout_s: float) -> ReasoningOutput:
            nonlocal calls
            calls += 1
            await asyncio.sleep(timeout_s + 0.2)
            raise RuntimeError("unreachable")

        adapter = GeminiReasoningAdapter(transport_callable=slow_transport)
        ctx = DialogueContext(
            session_id=uuid4(),
            incident_id=uuid4(),
            active_language="en",
            current_utterance=FinalizedUtteranceRecord(
                utterance_id=uuid4(),
                sequence_number=1,
                text="Road accident near Madhapur Metro",
                language="en",
                utterance_role=None,
                finalized_at=self.now,
            ),
            prior_utterances=[],
            validated_facts=[],
            pending_followup_question=None,
            eligible_instruction_ids=["safety_road_accident_01"],
            backend_status_summary={"current_category": "unknown", "location_confirmed": 1},
        )
        with self.assertRaises(ReasoningUnavailableError):
            await adapter.analyze_utterance(ctx, timeout_seconds=0.5)

        self.assertEqual(calls, 1, "Timeout must not trigger a second Gemini attempt")
        self.assertEqual(adapter.last_diagnostic_category, "timeout_exceeded")
        self.assertEqual(adapter.last_http_status, 408)


if __name__ == "__main__":
    unittest.main()



