"""
Targeted unit and integration tests for SETU upgrades:
1. Session-based conversational memory & non-redundant follow-up filtering.
2. SQL-backed 9-vehicle synthetic demo fleet schema & idempotent seed definitions.
3. Deterministic demo allocation regardless of caller GPS/location status.
4. Multi-incident isolation, additive reclassification, and honest unavailability reporting.
5. Simulated ETA & ASSIGNED -> EN_ROUTE -> ARRIVED lifecycle with coordinate movement.
6. Multilingual (en/te/hi) spoken dispatch & arrival announcements complying with PolicyValidator.
7. Gemini HTTP 503 outage resilience in DEMO mode without blocking valid demo allocations.
8. LiveSessionBridge end-to-end WS message sequence including session_end memory clearing.
"""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import threading
import time
import unittest
from uuid import UUID, uuid4

from backend.app.adapters.base import (
    EmergencyCategory,
    ProvenancedFact,
    ReasoningOutput,
    ReasoningUnavailableError,
    UtteranceRole,
)
from backend.app.adapters.mock import MockReasoningAdapter
from backend.app.adapters.timbre import GnaniTimbreAdapter
from backend.app.audit.writer import InMemoryAuditWriter
import base64
import unittest.mock as mock
from backend.app.db.pool import DEMO_VEHICLES_SEED, NeonPostgresStore
from backend.app.main import LiveSessionBridge, build_default_orchestrator, serialize_session_snapshot
from backend.app.observability.metrics import MetricsRegistry
from backend.app.orchestrator.announcements import (
    build_demo_dispatch_announcement,
    build_demo_lifecycle_announcement,
)
from backend.app.orchestrator.dispatch import (
    AssignmentStatus,
    DEMO_SIMULATED_ETA_MINUTES,
    DemoLifecycleStatus,
    DispatchEngine,
    InMemoryResourceRepository,
    LocationCandidateInput,
    ResourceCategory,
    build_demo_fleet_resources,
)
from backend.app.orchestrator.duplicates import DuplicateDetector
from backend.app.orchestrator.handoff import HandoffGenerator
from backend.app.orchestrator.hospital import HospitalRanker
from backend.app.orchestrator.instructions import InstructionLibrary
from backend.app.orchestrator.location import GPSPermissionStatus, LocationConfirmationStatus, LocationResolver
from backend.app.orchestrator.policy import PolicyValidator
from backend.app.orchestrator.session import SessionOrchestrator, SessionTokenSigner


class TestDemoFleetMemoryAndSpokenDispatch(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.reasoning = MockReasoningAdapter()
        self.repo = InMemoryResourceRepository(build_demo_fleet_resources())
        self.dispatch = DispatchEngine(self.repo, demo_mode=True)
        self.instructions = InstructionLibrary()
        self.policy = PolicyValidator(self.instructions)
        self.audit = InMemoryAuditWriter()
        self.metrics = MetricsRegistry()
        self.orchestrator = SessionOrchestrator(
            reasoning_adapter=self.reasoning,
            dispatch_engine=self.dispatch,
            location_resolver=LocationResolver(),
            hospital_ranker=HospitalRanker(),
            instruction_library=self.instructions,
            policy_validator=self.policy,
            duplicate_detector=DuplicateDetector(),
            audit_writer=self.audit,
            metrics=self.metrics,
            token_signer=SessionTokenSigner(secret_key="test-demo-signing-secret-key-2026!!"),
            demo_mode=True,
        )

    def test_migration_0002_seed_contains_exact_nine_vehicles_initially_available(self) -> None:
        migration = importlib.import_module("backend.app.db.migrations.0002_demo_fleet_and_lifecycle")
        ddl = migration.get_upgrade_sql()
        self.assertIn("CREATE TABLE IF NOT EXISTS demo_vehicles", ddl)
        self.assertIn("CREATE TABLE IF NOT EXISTS demo_assignments", ddl)
        self.assertIn("uq_demo_assignments_inc_veh_active", ddl)
        self.assertIn("uq_demo_assignments_vehicle_active", ddl)

        codes = [row[1] for row in DEMO_VEHICLES_SEED]
        self.assertEqual(
            codes,
            [
                "AMB-01", "AMB-02", "AMB-03",
                "FIR-01", "FIR-02", "FIR-03",
                "POL-01", "POL-02", "POL-03",
            ],
        )
        fleet = build_demo_fleet_resources()
        self.assertEqual(len(fleet), 9)
        self.assertTrue(all(v.is_available for v in fleet))

    async def test_multi_turn_conversational_memory_additive_reclassification_and_session_clear(self) -> None:
        session = self.orchestrator.create_session(initial_language="en", now=self.now)
        # Intentionally leave GPS missing/denied to prove DEMO mode allocates regardless of location!
        self.orchestrator.update_gps_location(
            session.session_id,
            latitude=None,
            longitude=None,
            accuracy_m=None,
            permission_status=GPSPermissionStatus.DENIED,
            now=self.now,
        )

        # Turn 1: Fire at Madhapur -> allocates FIR-01 (7 min), asks if anyone injured
        u1_id = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=u1_id,
                category=EmergencyCategory.FIRE,
                category_confidence=0.94,
                supporting_utterance_ids=[u1_id],
                extracted_facts=[
                    ProvenancedFact("caller_location_text", "Hitech City", u1_id, "at Hitech City"),
                    ProvenancedFact("emergency_type", "fire", u1_id, "huge fire"),
                ],
                conflicts=[],
                missing_information_fields=["injuries_reported", "caller_location_text"],
                proposed_followup_question="Is anyone injured at Hitech City?",
                selected_instruction_id="safety_fire_01",
                caller_response_text="Stay clear of smoke in this SETU demo.",
                prompt_tokens=100,
                completion_tokens=40,
                latency_ms=150.0,
            )
        )
        state = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u1_id,
            text="There is a huge fire at Hitech City",
            detected_language="en",
            now=self.now,
        )
        self.assertEqual(state.category, EmergencyCategory.FIRE)
        self.assertEqual(len(state.assignments), 1)
        # Explicit Location Confirmation Dispatch Gate: held in PENDING_LOCATION before confirm
        self.assertIsNone(state.assignments[0].resource_id)
        self.assertEqual(state.assignments[0].status, AssignmentStatus.PENDING_LOCATION)
        self.assertEqual(state.assignments[0].lifecycle_status, "PENDING_LOCATION")
        # Location was already extracted in Turn 1 -> filtered out of missing_information_fields
        self.assertNotIn("caller_location_text", state.missing_information_fields)

        # Turn 2: Caller answers follow-up ("Two people injured") -> reclassifies fire -> fire_with_injuries on SAME incident!
        u2_id = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.ANSWER_TO_PENDING_QUESTION,
                utterance_role_source_id=u2_id,
                category=EmergencyCategory.FIRE_WITH_INJURIES,
                category_confidence=0.95,
                supporting_utterance_ids=[u1_id, u2_id],
                extracted_facts=[
                    ProvenancedFact("injuries_reported", True, u2_id, "two people are injured"),
                    ProvenancedFact("affected_person_count", 2, u2_id, "two people"),
                ],
                conflicts=[],
                missing_information_fields=["caller_location_text", "affected_person_count"],
                # Even if LLM redundantly asks "Where is the location?", backend filters it out!
                proposed_followup_question="Where is your location?",
                selected_instruction_id="safety_fire_injuries_01",
                caller_response_text="Keep injured persons in fresh air in this demo.",
                prompt_tokens=140,
                completion_tokens=45,
                latency_ms=160.0,
            )
        )
        state = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u2_id,
            text="Yes, two people are injured",
            detected_language="en",
            now=self.now + timedelta(seconds=2),
        )
        # Assert DialogueContext on Turn 2 received Turn 1 history + facts + pending question
        turn2_ctx = self.reasoning.received_contexts[1]
        self.assertGreaterEqual(len(turn2_ctx.conversation_history), 2)
        self.assertEqual(turn2_ctx.pending_followup_question, "Is anyone injured at Hitech City?")
        self.assertTrue(any(f.field_name == "caller_location_text" for f in turn2_ctx.validated_facts))

        # Still pending explicit confirmation before committing units
        self.assertEqual(state.category, EmergencyCategory.FIRE_WITH_INJURIES)
        self.assertEqual(
            [a.category for a in state.assignments],
            [ResourceCategory.FIRE_ENGINE, ResourceCategory.AMBULANCE],
        )
        self.assertTrue(all(a.status == AssignmentStatus.PENDING_LOCATION for a in state.assignments))
        # Redundant question about already-confirmed location was suppressed!
        self.assertIsNone(state.pending_followup_question)
        # Provenance preserved on updated facts
        fact_map = {f.field_name: f for f in state.validated_facts}
        self.assertEqual(fact_map["caller_location_text"].source_utterance_id, u1_id)
        self.assertEqual(fact_map["affected_person_count"].value, 2)
        self.assertEqual(fact_map["affected_person_count"].source_utterance_id, u2_id)

        # Explicit caller confirmation commits FIR-01 & AMB-01 directly to EN_ROUTE with ETAs & speech
        state = self.orchestrator.confirm_current_location(
            session.session_id,
            now=self.now + timedelta(seconds=5),
        )
        unit_ids = [a.resource_id for a in state.assignments]
        self.assertEqual(unit_ids, ["FIR-01", "AMB-01"])
        self.assertTrue(all(a.lifecycle_status == "EN_ROUTE" for a in state.assignments))
        self.assertEqual(state.assignments[0].simulated_eta_minutes, DEMO_SIMULATED_ETA_MINUTES[ResourceCategory.FIRE_ENGINE])
        self.assertEqual(state.assignments[1].simulated_eta_minutes, DEMO_SIMULATED_ETA_MINUTES[ResourceCategory.AMBULANCE])
        self.assertIn("FIR-01", state.last_spoken_response_text)
        self.assertIn("AMB-01", state.last_spoken_response_text)
        self.assertIn("7 minutes", state.last_spoken_response_text)
        self.assertIn("6 minutes", state.last_spoken_response_text)

        state, arrived_changed, arrived_speech = self.orchestrator.advance_demo_lifecycle(
            session.session_id,
            target_status=DemoLifecycleStatus.ARRIVED,
            now=self.now + timedelta(seconds=12),
        )
        self.assertEqual(len(arrived_changed), 2)
        self.assertIsNotNone(arrived_speech)
        self.assertIn("arrived", arrived_speech.lower())
        self.assertIn("FIR-01", arrived_speech)
        self.assertIn("AMB-01", arrived_speech)
        self.assertTrue(all(a.lifecycle_status == "ARRIVED" and a.remaining_eta_minutes == 0 for a in state.assignments))

        # PolicyValidator must accept all generated spoken announcements without fallback replacement
        for phrase in (
            build_demo_dispatch_announcement(state.assignments, "en"),
            build_demo_dispatch_announcement(state.assignments, "te"),
            build_demo_dispatch_announcement(state.assignments, "hi"),
            build_demo_lifecycle_announcement(arrived_changed, "en"),
            build_demo_lifecycle_announcement(arrived_changed, "te"),
            build_demo_lifecycle_announcement(arrived_changed, "hi"),
        ):
            _, was_replaced = self.policy.sanitize_caller_response_text(
                phrase, language="en", has_simulated_assignment=True
            )
            self.assertFalse(was_replaced, f"Sanitizer rejected valid demo phrase: {phrase}")

        # End session clears conversational memory and frees FIR-01 & AMB-01
        ended = self.orchestrator.end_session(session.session_id, reason="caller_hangup", now=self.now + timedelta(seconds=15))
        self.assertTrue(ended.conversational_memory_cleared)
        self.assertEqual(len(ended.conversation_history), 0)
        self.assertIsNone(ended.pending_followup_question)
        self.assertTrue(all(v.is_available for v in self.repo.list_all()))

    def test_multi_incident_isolation_and_exhaustion_reporting(self) -> None:
        # Unconfirmed candidate holds PENDING_LOCATION in explicit-confirmation DEMO mode
        pending_out = self.dispatch.allocate_for_incident(
            incident_id=uuid4(),
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            location_candidate=None,
            existing_assignments=(),
            now=self.now,
        )
        self.assertEqual(pending_out.assignments[0].status, AssignmentStatus.PENDING_LOCATION)

        # Exhaust all 3 ambulances across 3 concurrent confirmed incidents
        confirmed_loc = LocationCandidateInput(
            latitude=17.4435,
            longitude=78.3772,
            accuracy_m=50.0,
            source="browser_gps",
            is_confirmed=True,
        )
        inc_ids = [uuid4() for _ in range(4)]
        assigned_codes: list[str] = []
        for inc_id in inc_ids[:3]:
            out = self.dispatch.allocate_for_incident(
                incident_id=inc_id,
                category=EmergencyCategory.MEDICAL_EMERGENCY,
                location_candidate=confirmed_loc,
                existing_assignments=(),
                now=self.now,
            )
            self.assertEqual(len(out.assignments), 1)
            self.assertEqual(out.assignments[0].status, AssignmentStatus.SIMULATED_ASSIGNED)
            assigned_codes.append(out.assignments[0].resource_id or "")

        self.assertEqual(assigned_codes, ["AMB-01", "AMB-02", "AMB-03"])

        # 4th concurrent incident must NOT steal AMB-01..03 and must report unavailable accurately
        fourth = self.dispatch.allocate_for_incident(
            incident_id=inc_ids[3],
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            location_candidate=confirmed_loc,
            existing_assignments=(),
            now=self.now,
        )
        self.assertEqual(fourth.assignments[0].status, AssignmentStatus.SIMULATED_UNAVAILABLE)
        self.assertEqual(fourth.assignments[0].lifecycle_status, "UNAVAILABLE")
        self.assertIsNone(fourth.assignments[0].resource_id)
        self.assertTrue(fourth.control_room_review_required)
        ann = build_demo_dispatch_announcement(fourth.assignments, "en")
        self.assertIn("no synthetic ambulance is currently available", ann.lower())

    async def test_gemini_outage_does_not_block_demo_allocation_and_speaks_multilingual_template(self) -> None:
        session = self.orchestrator.create_session(initial_language="te", now=self.now)
        u_id = uuid4()
        self.reasoning.enqueue_exception(
            ReasoningUnavailableError(
                "503 Service Unavailable",
                session_id=session.session_id,
                utterance_id=u_id,
                is_retryable=True,
                http_status=503,
            )
        )
        state = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u_id,
            text="రోడ్డు ప్రమాదం జరిగింది హెచ్ఐటెక్ సిటీ వద్ద ఇద్దరు గాయపడ్డారు",
            detected_language="te",
            now=self.now,
        )
        self.assertIn("reasoning_unavailable", state.review_reason_codes)
        self.assertEqual(state.category, EmergencyCategory.ROAD_ACCIDENT)
        # Held awaiting explicit confirmation until caller confirms
        self.assertTrue(all(a.status == AssignmentStatus.PENDING_LOCATION for a in state.assignments))

        state = self.orchestrator.confirm_current_location(
            session.session_id,
            now=self.now + timedelta(seconds=1),
        )
        self.assertEqual([a.resource_id for a in state.assignments], ["AMB-01", "POL-01"])
        self.assertTrue(all(a.lifecycle_status == "EN_ROUTE" for a in state.assignments))
        self.assertIn("AMB-01", state.last_spoken_response_text)
        self.assertIn("POL-01", state.last_spoken_response_text)
        self.assertIn("డెమో", state.last_spoken_response_text)

    async def test_live_session_bridge_lifecycle_and_session_end_flow(self) -> None:
        bridge = LiveSessionBridge(use_live_providers=False)
        init_res = await bridge.handle_ws_message({
            "type": "session_init",
            "initial_language": "en",
            "fresh_scenario": True,
        })
        self.assertEqual(init_res["event"], "session_created")
        sid = init_res["snapshot"]["session_id"]

        # Turn 1: Road accident via scripted_finalized_utterance holds PENDING_LOCATION until location_confirm
        turn_res = await bridge.handle_ws_message({
            "type": "scripted_finalized_utterance",
            "session_id": sid,
            "utterance_id": str(uuid4()),
            "text": "Road accident near Cyber Towers",
            "language": "en",
            "category": "road_accident",
            "confidence": 0.93,
            "extracted_facts": [
                {"field": "caller_location_text", "value": "Cyber Towers", "evidence_text": "near Cyber Towers"},
            ],
            "response_text": "SETU demo simulation recorded this road accident.",
        })
        snap = turn_res["snapshot"]
        self.assertEqual([a["lifecycle_status"] for a in snap["assignments"]], ["PENDING_LOCATION", "PENDING_LOCATION"])
        self.assertEqual(len(snap["demo_vehicles"]), 9)

        # Explicit location confirmation dispatches AMB-01 & POL-01 straight to EN_ROUTE with TTS/announcement
        conf_res = await bridge.handle_ws_message({
            "type": "location_confirm",
            "session_id": sid,
        })
        conf_snap = conf_res["snapshot"]
        self.assertEqual(
            [a["vehicle_code"] for a in conf_snap["assignments"]],
            ["AMB-01", "POL-01"],
        )
        self.assertEqual([a["lifecycle_status"] for a in conf_snap["assignments"]], ["EN_ROUTE", "EN_ROUTE"])
        self.assertIn("AMB-01", conf_snap["last_spoken_response_text"])

        # Advance ARRIVED via demo_lifecycle_tick
        tick2 = await bridge.handle_ws_message({
            "type": "demo_lifecycle_tick",
            "session_id": sid,
            "force_stage": "ARRIVED",
        })
        self.assertEqual(tick2["event"], "demo_lifecycle_updated")
        self.assertEqual([a["lifecycle_status"] for a in tick2["snapshot"]["assignments"]], ["ARRIVED", "ARRIVED"])
        self.assertIn("arrived", tick2["spoken_announcement_text"].lower())

        # session_end clears conversational memory and releases demo fleet back to AVAILABLE
        end_res = await bridge.handle_ws_message({
            "type": "session_end",
            "session_id": sid,
        })
        self.assertEqual(end_res["event"], "session_completed")
        self.assertTrue(end_res["snapshot"]["conversational_memory"]["cleared"])
        self.assertEqual(end_res["snapshot"]["conversational_memory"]["turn_count"], 0)
        fleet_res = await bridge.handle_ws_message({"type": "get_demo_fleet"})
        self.assertEqual(len(fleet_res["vehicles"]), 9)
        self.assertTrue(all(v["is_available"] for v in fleet_res["vehicles"]))

    async def test_demo_outage_followup_injury_qa_location_split_and_question_stripping(self) -> None:
        # 1. Multi-turn outage Q&A: SETU asks about injuries -> caller answers "No" -> no repeated question
        session_no = self.orchestrator.create_session(initial_language="en", now=self.now)
        u1 = uuid4()
        self.reasoning.enqueue_exception(
            ReasoningUnavailableError(
                "429 Quota exceeded",
                session_id=session_no.session_id,
                utterance_id=u1,
                is_retryable=False,
                http_status=429,
            )
        )
        s1 = await self.orchestrator.process_finalized_utterance(
            session_no.session_id,
            utterance_id=u1,
            text="There is a fire near Charminar",
            detected_language="en",
            now=self.now,
        )
        self.assertEqual(s1.category, EmergencyCategory.FIRE)
        self.assertEqual([a.lifecycle_status for a in s1.assignments], ["PENDING_LOCATION"])
        self.assertTrue(
            any(
                f.field_name in ("location", "caller_location_text")
                and "Charminar" in str(f.value)
                for f in s1.validated_facts
            )
        )
        self.assertIsNotNone(s1.pending_followup_question)
        pq_lower = str(s1.pending_followup_question).lower()
        self.assertTrue("injured" in pq_lower or "hurt" in pq_lower)

        s1_confirmed = self.orchestrator.confirm_current_location(
            session_no.session_id,
            now=self.now + timedelta(seconds=1),
        )
        self.assertEqual([a.resource_id for a in s1_confirmed.assignments], ["FIR-01"])

        u2 = uuid4()
        self.reasoning.enqueue_exception(
            ReasoningUnavailableError(
                "429 Quota exceeded",
                session_id=session_no.session_id,
                utterance_id=u2,
                is_retryable=False,
                http_status=429,
            )
        )
        s2 = await self.orchestrator.process_finalized_utterance(
            session_no.session_id,
            utterance_id=u2,
            text="No",
            detected_language="en",
            now=self.now + timedelta(seconds=3),
        )
        self.assertEqual(s2.utterances[-1].utterance_role, UtteranceRole.ANSWER_TO_PENDING_QUESTION)
        self.assertTrue(
            any(f.field_name == "injuries_reported" and f.value is False for f in s2.validated_facts)
        )
        self.assertIsNone(s2.pending_followup_question)
        self.assertNotIn("injured or trapped", s2.last_spoken_response_text.lower())

        # 2. Affirmative reply upgrades FIRE -> FIRE_WITH_INJURIES additively & captures count
        session_yes = self.orchestrator.create_session(initial_language="en", now=self.now)
        u3 = uuid4()
        self.reasoning.enqueue_exception(
            ReasoningUnavailableError(
                "429 Quota exceeded",
                session_id=session_yes.session_id,
                utterance_id=u3,
                is_retryable=False,
                http_status=429,
            )
        )
        s3 = await self.orchestrator.process_finalized_utterance(
            session_yes.session_id,
            utterance_id=u3,
            text="There is a fire near Charminar and two people are injured",
            detected_language="en",
            now=self.now,
        )
        self.assertEqual(s3.category, EmergencyCategory.FIRE_WITH_INJURIES)
        loc_facts = [f for f in s3.validated_facts if f.field_name in ("location", "caller_location_text")]
        self.assertTrue(loc_facts)
        self.assertEqual(str(loc_facts[-1].value), "Charminar")
        self.assertNotIn("injured", str(loc_facts[-1].value).lower())

        # 3. Redundant location follow-up emitted by reasoning is stripped from spoken text too
        session_strip = self.orchestrator.create_session(initial_language="en", now=self.now)
        u4 = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=u4,
                category=EmergencyCategory.FIRE,
                category_confidence=0.92,
                supporting_utterance_ids=[u4],
                extracted_facts=[
                    ProvenancedFact("location", "Madhapur", u4, "near Madhapur"),
                    ProvenancedFact("location_confirmed", True, u4, "near Madhapur"),
                ],
                conflicts=[],
                missing_information_fields=["injuries_reported"],
                proposed_followup_question="Is anyone injured?",
                selected_instruction_id=None,
                caller_response_text="Recorded the fire near Madhapur. Is anyone injured?",
                prompt_tokens=10,
                completion_tokens=10,
                latency_ms=10.0,
            )
        )
        await self.orchestrator.process_finalized_utterance(
            session_strip.session_id,
            utterance_id=u4,
            text="Fire near Madhapur",
            detected_language="en",
            now=self.now,
        )
        u5 = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.ANSWER_TO_PENDING_QUESTION,
                utterance_role_source_id=u5,
                category=EmergencyCategory.FIRE,
                category_confidence=0.94,
                supporting_utterance_ids=[u5],
                extracted_facts=[
                    ProvenancedFact("injuries_reported", "no", u5, "No injuries"),
                ],
                conflicts=[],
                missing_information_fields=[],
                proposed_followup_question="Please provide the exact location.",
                selected_instruction_id=None,
                caller_response_text="Understood, no injuries reported. Please provide the exact location?",
                prompt_tokens=10,
                completion_tokens=10,
                latency_ms=10.0,
            )
        )
        s5 = await self.orchestrator.process_finalized_utterance(
            session_strip.session_id,
            utterance_id=u5,
            text="No injuries",
            detected_language="en",
            now=self.now + timedelta(seconds=2),
        )
        self.assertIsNone(s5.pending_followup_question)
        self.assertNotIn("exact location", s5.last_spoken_response_text.lower())

    async def test_per_assignment_demo_lifecycle_tick_does_not_skip_en_route(self) -> None:
        # 1. Verify DispatchEngine strictly enforces ASSIGNED -> EN_ROUTE -> ARRIVED without skipping EN_ROUTE
        repo = InMemoryResourceRepository(build_demo_fleet_resources())
        assign_dispatch = DispatchEngine(
            repo,
            demo_mode=True,
            require_explicit_location_confirmation=False,
        )
        inc_id = uuid4()
        out = assign_dispatch.allocate_for_incident(
            incident_id=inc_id,
            category=EmergencyCategory.FIRE,
            location_candidate=None,
            existing_assignments=(),
            now=self.now,
        )
        self.assertEqual(out.assignments[0].lifecycle_status, "ASSIGNED")
        updated_same, chg_skip = assign_dispatch.advance_assignment_lifecycle(
            out.assignments,
            target_status=DemoLifecycleStatus.ARRIVED,
            now=self.now + timedelta(seconds=5),
        )
        self.assertEqual(len(chg_skip), 0)
        self.assertEqual(updated_same[0].lifecycle_status, "ASSIGNED")
        updated_en_route, chg_er = assign_dispatch.advance_assignment_lifecycle(
            updated_same,
            target_status=DemoLifecycleStatus.EN_ROUTE,
            now=self.now + timedelta(seconds=6),
        )
        self.assertEqual(len(chg_er), 1)
        self.assertEqual(updated_en_route[0].lifecycle_status, "EN_ROUTE")

        # 2. Verify LiveSessionBridge ages FIR-01 past estimated_arrival_at independently from newly added AMB-01
        bridge = LiveSessionBridge(use_live_providers=False)
        init_res = await bridge.handle_ws_message({
            "type": "session_init",
            "initial_language": "en",
            "fresh_scenario": True,
        })
        sid = init_res["snapshot"]["session_id"]

        await bridge.handle_ws_message({
            "type": "scripted_finalized_utterance",
            "session_id": sid,
            "utterance_id": str(uuid4()),
            "text": "There is a fire near Charminar",
            "language": "en",
            "category": "fire",
            "confidence": 0.94,
            "extracted_facts": [
                {"field": "caller_location_text", "value": "Charminar", "evidence_text": "near Charminar"},
            ],
            "response_text": "Please confirm Charminar.",
        })
        conf1 = await bridge.handle_ws_message({"type": "location_confirm", "session_id": sid})
        self.assertEqual([a["vehicle_code"] for a in conf1["snapshot"]["assignments"]], ["FIR-01"])
        self.assertEqual([a["lifecycle_status"] for a in conf1["snapshot"]["assignments"]], ["EN_ROUTE"])

        session_state = bridge.orchestrator.get_session(UUID(sid))
        now_utc = datetime.now(timezone.utc)
        session_state.assignments[0] = replace(
            session_state.assignments[0],
            assigned_at=now_utc - timedelta(seconds=20),
            en_route_at=now_utc - timedelta(seconds=15),
            estimated_arrival_at=now_utc - timedelta(seconds=1),
        )

        # Turn 2: Upgrade to fire_with_injuries adds AMB-01 at EN_ROUTE with fresh future ETA
        t2 = await bridge.handle_ws_message({
            "type": "scripted_finalized_utterance",
            "session_id": sid,
            "utterance_id": str(uuid4()),
            "text": "Two people are injured inside",
            "language": "en",
            "category": "fire_with_injuries",
            "confidence": 0.96,
            "response_text": "SETU demo added an ambulance.",
        })
        by_code = {a["vehicle_code"]: a for a in t2["snapshot"]["assignments"]}
        self.assertEqual(by_code["FIR-01"]["lifecycle_status"], "EN_ROUTE")
        self.assertEqual(by_code["AMB-01"]["lifecycle_status"], "EN_ROUTE")

        # Natural demo_lifecycle_tick advances ONLY FIR-01 -> ARRIVED, keeping AMB-01 EN_ROUTE
        tick_res = await bridge.handle_ws_message({
            "type": "demo_lifecycle_tick",
            "session_id": sid,
        })
        by_code_after = {a["vehicle_code"]: a for a in tick_res["snapshot"]["assignments"]}
        self.assertEqual(by_code_after["FIR-01"]["lifecycle_status"], "ARRIVED")
        self.assertEqual(by_code_after["AMB-01"]["lifecycle_status"], "EN_ROUTE")

    async def test_pending_question_capture_no_repeat_and_ambiguous_reask(self) -> None:
        session = self.orchestrator.create_session(initial_language="en", now=self.now)
        u1 = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=u1,
                category=EmergencyCategory.FIRE,
                category_confidence=0.93,
                supporting_utterance_ids=[u1],
                extracted_facts=[
                    ProvenancedFact("caller_location_text", "Hitech City", u1, "Hitech City"),
                ],
                conflicts=[],
                missing_information_fields=["injuries_reported"],
                proposed_followup_question="Is anyone injured?",
                proposed_followup_target_field="injuries_reported",
                selected_instruction_id="safety_fire_01",
                caller_response_text="Stay clear of the smoke. Is anyone injured?",
                prompt_tokens=60,
                completion_tokens=20,
                latency_ms=90.0,
            )
        )
        s1 = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u1,
            text="Fire at Hitech City",
            detected_language="en",
            now=self.now,
        )
        self.assertIsNotNone(s1.pending_question_record)
        self.assertEqual(s1.pending_question_record.target_field, "injuries_reported")
        self.assertEqual(s1.pending_question_record.status, "pending")

        # Caller replies simply "Yes" while model fails to echo injuries_reported and asks again
        u2 = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.ANSWER_TO_PENDING_QUESTION,
                utterance_role_source_id=u2,
                category=EmergencyCategory.FIRE,
                category_confidence=0.92,
                supporting_utterance_ids=[u2],
                extracted_facts=[],
                conflicts=[],
                missing_information_fields=["affected_person_count"],
                proposed_followup_question="Is anyone injured?",
                proposed_followup_target_field="injuries_reported",
                selected_instruction_id="safety_fire_injuries_01",
                caller_response_text="Are people hurt? How many people are injured?",
                prompt_tokens=60,
                completion_tokens=25,
                latency_ms=95.0,
            )
        )
        s2 = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u2,
            text="Yes",
            detected_language="en",
            now=self.now + timedelta(seconds=2),
        )
        facts = {f.field_name: f for f in s2.validated_facts}
        self.assertEqual(facts["injuries_reported"].value, True)
        self.assertEqual(facts["injuries_reported"].source_utterance_id, u2)
        self.assertEqual(s2.category, EmergencyCategory.FIRE_WITH_INJURIES)
        self.assertTrue(any(q.status == "answered" and q.answered_by_utterance_id == u2 for q in s2.question_history))
        self.assertNotIn("anyone injured", s2.last_spoken_response_text.lower())

        # Ask affected_person_count -> ambiguous "Not sure" must stay pending without inventing count
        u3 = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=u3,
                category=EmergencyCategory.FIRE_WITH_INJURIES,
                category_confidence=0.94,
                supporting_utterance_ids=[u3],
                extracted_facts=[],
                conflicts=[],
                missing_information_fields=["affected_person_count"],
                proposed_followup_question="Roughly how many people are injured?",
                proposed_followup_target_field="affected_person_count",
                selected_instruction_id=None,
                caller_response_text="Roughly how many people are injured?",
                prompt_tokens=40,
                completion_tokens=15,
                latency_ms=70.0,
            )
        )
        await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u3,
            text="People are inside",
            detected_language="en",
            now=self.now + timedelta(seconds=4),
        )
        u4 = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.ANSWER_TO_PENDING_QUESTION,
                utterance_role_source_id=u4,
                category=EmergencyCategory.FIRE_WITH_INJURIES,
                category_confidence=0.94,
                supporting_utterance_ids=[u4],
                extracted_facts=[],
                conflicts=[],
                missing_information_fields=["affected_person_count"],
                proposed_followup_question="Roughly how many people are hurt if you can see?",
                proposed_followup_target_field="affected_person_count",
                pending_question_answered=False,
                selected_instruction_id=None,
                caller_response_text="That's okay. Roughly how many people are hurt if you can see?",
                prompt_tokens=40,
                completion_tokens=15,
                latency_ms=75.0,
            )
        )
        s4 = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u4,
            text="Maybe, not sure",
            detected_language="en",
            now=self.now + timedelta(seconds=6),
        )
        self.assertNotIn("affected_person_count", {f.field_name for f in s4.validated_facts})
        self.assertIsNotNone(s4.pending_question_record)
        self.assertEqual(s4.pending_question_record.status, "pending")

        # Next clear reply "Two" resolves affected_person_count cleanly
        u5 = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.ANSWER_TO_PENDING_QUESTION,
                utterance_role_source_id=u5,
                category=EmergencyCategory.FIRE_WITH_INJURIES,
                category_confidence=0.95,
                supporting_utterance_ids=[u5],
                extracted_facts=[],
                conflicts=[],
                missing_information_fields=[],
                proposed_followup_question=None,
                selected_instruction_id=None,
                caller_response_text="Thank you, I have recorded 2 injured people.",
                prompt_tokens=40,
                completion_tokens=15,
                latency_ms=75.0,
            )
        )
        s5 = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u5,
            text="Two",
            detected_language="en",
            now=self.now + timedelta(seconds=8),
        )
        facts5 = {f.field_name: f for f in s5.validated_facts}
        self.assertEqual(facts5["affected_person_count"].value, 2)
        self.assertIsNone(s5.pending_question_record)

    async def test_broad_area_landmark_preservation_text_confirm_invalidation_and_zero_fake_coords(self) -> None:
        session = self.orchestrator.create_session(initial_language="en", now=self.now)
        self.orchestrator.update_gps_location(
            session.session_id,
            latitude=None,
            longitude=None,
            accuracy_m=None,
            permission_status=GPSPermissionStatus.DENIED,
            now=self.now,
        )
        u1 = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=u1,
                category=EmergencyCategory.FIRE,
                category_confidence=0.93,
                supporting_utterance_ids=[u1],
                extracted_facts=[
                    ProvenancedFact("broad_location_text", "Madhapur", u1, "at Madhapur"),
                ],
                conflicts=[],
                missing_information_fields=["injuries_reported"],
                proposed_followup_question="Near which road or landmark in Madhapur?",
                proposed_followup_target_field="caller_landmark_text",
                selected_instruction_id="safety_fire_01",
                caller_response_text="Stay calm. Near which landmark in Madhapur?",
                prompt_tokens=50,
                completion_tokens=20,
                latency_ms=80.0,
            )
        )
        s1 = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u1,
            text="Fire at Madhapur",
            detected_language="en",
            now=self.now,
        )
        self.assertIsNotNone(s1.location_state)
        self.assertIsNone(s1.location_state.latitude)
        self.assertIsNone(s1.location_state.match_confidence)
        self.assertEqual(s1.location_state.broad_area_text, "Madhapur")

        # Turn 2 adds street/landmark detail -> broad area preserved without fake coordinates
        u2 = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.ANSWER_TO_PENDING_QUESTION,
                utterance_role_source_id=u2,
                category=EmergencyCategory.FIRE,
                category_confidence=0.94,
                supporting_utterance_ids=[u2],
                extracted_facts=[
                    ProvenancedFact("caller_landmark_text", "ABC building, near main road", u2, "ABC building, near main road"),
                ],
                conflicts=[],
                missing_information_fields=[],
                proposed_followup_question="Please confirm Madhapur near ABC building.",
                proposed_followup_target_field="location_confirmed",
                selected_instruction_id="safety_fire_01",
                caller_response_text="Got it, Madhapur near ABC building. Please confirm your location.",
                prompt_tokens=50,
                completion_tokens=20,
                latency_ms=80.0,
            )
        )
        s2 = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u2,
            text="Near ABC building, near main road",
            detected_language="en",
            now=self.now + timedelta(seconds=2),
        )
        self.assertEqual(s2.location_state.broad_area_text, "Madhapur")
        self.assertIn("ABC building", s2.location_state.caller_landmark_text or "")
        self.assertIsNone(s2.location_state.latitude)
        self.assertIsNone(s2.location_state.longitude)
        self.assertIsNone(s2.location_state.match_confidence)
        self.assertEqual([a.lifecycle_status for a in s2.assignments], ["PENDING_LOCATION"])

        # Turn 3: Verbal spoken confirmation ("Yes, that's my location") commits dispatch without fake coordinates
        u3 = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.ANSWER_TO_PENDING_QUESTION,
                utterance_role_source_id=u3,
                category=EmergencyCategory.FIRE,
                category_confidence=0.94,
                supporting_utterance_ids=[u3],
                extracted_facts=[],
                conflicts=[],
                missing_information_fields=[],
                proposed_followup_question=None,
                selected_instruction_id="safety_fire_01",
                caller_response_text="Thank you.",
                prompt_tokens=30,
                completion_tokens=10,
                latency_ms=50.0,
            )
        )
        s3 = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u3,
            text="Yes, that's my location",
            detected_language="en",
            now=self.now + timedelta(seconds=4),
        )
        self.assertEqual(s3.location_state.confirmation_status, LocationConfirmationStatus.CONFIRMED)
        self.assertIsNotNone(s3.location_state.confirmed_at)
        self.assertIsNone(s3.location_state.latitude)
        self.assertEqual([a.resource_id for a in s3.assignments], ["FIR-01"])
        self.assertEqual([a.lifecycle_status for a in s3.assignments], ["EN_ROUTE"])
        self.assertIn("FIR-01", s3.last_spoken_response_text)

        # Turn 4: Caller updates location later -> invalidates prior confirmation without silently dropping FIR-01
        u4 = uuid4()
        self.reasoning.enqueue_output(
            ReasoningOutput(
                utterance_role=UtteranceRole.NEW_DESCRIPTION,
                utterance_role_source_id=u4,
                category=EmergencyCategory.FIRE,
                category_confidence=0.94,
                supporting_utterance_ids=[u4],
                extracted_facts=[
                    ProvenancedFact("caller_location_text", "Banjara Hills Road 12", u4, "Banjara Hills Road 12"),
                ],
                conflicts=[],
                missing_information_fields=[],
                proposed_followup_question=None,
                selected_instruction_id="safety_fire_01",
                caller_response_text="Please reconfirm the updated Banjara Hills location.",
                prompt_tokens=40,
                completion_tokens=15,
                latency_ms=60.0,
            )
        )
        s4 = await self.orchestrator.process_finalized_utterance(
            session.session_id,
            utterance_id=u4,
            text="Actually it is Banjara Hills Road 12",
            detected_language="en",
            now=self.now + timedelta(seconds=6),
        )
        self.assertNotEqual(s4.location_state.confirmation_status, LocationConfirmationStatus.CONFIRMED)
        self.assertIsNone(s4.location_state.confirmed_at)
        self.assertEqual([a.resource_id for a in s4.assignments], ["FIR-01"])

    def test_localized_safety_guidance_gas_cylinder_and_ready_manifest(self) -> None:
        lib = InstructionLibrary()
        for lang in ("en", "te", "hi"):
            gas_inst = lib.get_safety_guidance_for_category(
                EmergencyCategory.FIRE,
                lang,
                context_text="Kitchen LPG gas cylinder leak and fire",
            )
            self.assertIsNotNone(gas_inst)
            self.assertEqual(gas_inst.instruction_id, "safety_fire_gas_cylinder_01")

        manifest_path = Path("backend-media/instructions/manifest.json")
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_list = manifest.get("entries", []) if isinstance(manifest, dict) else manifest
        entries = {(e["instruction_id"], e["language"]): e for e in manifest_list}
        self.assertGreater(len(entries), 15)
        for lang in ("en", "te", "hi"):
            entry = entries[("safety_fire_gas_cylinder_01", lang)]
            self.assertEqual(entry["generation_status"], "ready")
            self.assertEqual(len(entry["checksum_sha256"]), 64)
            self.assertTrue(Path(entry["file_path"]).exists())

    async def test_timbre_incremental_sse_chunk_streaming_no_full_buffering(self) -> None:
        release_second = threading.Event()
        chunk1_pcm = b"\x01\x02" * 400
        chunk2_pcm = b"\x03\x04" * 400
        line1 = f'data: {json.dumps({"chunk_index": 0, "audio": base64.b64encode(b"RIFF" + b"0" * 40 + chunk1_pcm).decode("ascii")})}\n'.encode("utf-8")
        line2 = f'data: {json.dumps({"chunk_index": 1, "audio": base64.b64encode(chunk2_pcm).decode("ascii")})}\n'.encode("utf-8")

        class _FakeSSEResp:
            def __init__(self) -> None:
                self.headers = {"Content-Type": "text/event-stream"}

            def __enter__(self) -> "_FakeSSEResp":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def __iter__(self):
                yield b"event: chunk\n"
                yield line1
                release_second.wait(timeout=2.0)
                yield b"event: chunk\n"
                yield line2
                yield b"event: done\ndata: {}\n"

        adapter = GnaniTimbreAdapter(endpoint_url="https://example.invalid/sse", api_key="test-key")
        with mock.patch("urllib.request.urlopen", return_value=_FakeSSEResp()):
            handle = await adapter.start_synthesis(
                session_id=uuid4(),
                synthesis_id=uuid4(),
                text="Stay calm.",
                language="en",
                voice_id="Pranav",
                timeout_seconds=5.0,
            )
            aiter = handle.iter_audio_chunks(first_byte_timeout_seconds=2.0, chunk_timeout_seconds=2.0).__aiter__()
            first = await aiter.__anext__()
            self.assertTrue(first.is_first_byte)
            self.assertEqual(first.audio_bytes, chunk1_pcm)
            # Chunk 1 arrived BEFORE release_second was set -> zero full-stream buffering!
            release_second.set()
            second = await aiter.__anext__()
            self.assertEqual(second.audio_bytes, chunk2_pcm)
            with self.assertRaises(StopAsyncIteration):
                await aiter.__anext__()

    async def test_handoff_synchronization_fields_and_pending_location_persistence_guard(self) -> None:
        bridge = LiveSessionBridge(use_live_providers=False)
        init_res = await bridge.handle_ws_message({"type": "session_init", "initial_language": "en", "fresh_scenario": True})
        sid = init_res["snapshot"]["session_id"]
        turn_before = await bridge.handle_ws_message({
            "type": "scripted_finalized_utterance",
            "session_id": sid,
            "utterance_id": str(uuid4()),
            "text": "Gas cylinder fire near Cyber Towers",
            "language": "en",
            "category": "fire",
            "confidence": 0.95,
            "extracted_facts": [
                {"field": "caller_location_text", "value": "Cyber Towers", "evidence_text": "near Cyber Towers"},
            ],
            "selected_instruction_id": "safety_fire_gas_cylinder_01",
            "response_text": "Turn off the regulator safely.",
        })
        executed_sql: list[tuple[str, object]] = []

        class _FakeCursor:
            def __enter__(self) -> "_FakeCursor":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def execute(self, sql: str, params: object = None) -> None:
                executed_sql.append((sql, params))

        class _FakeConn:
            def __enter__(self) -> "_FakeConn":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def cursor(self) -> _FakeCursor:
                return _FakeCursor()

        # persist_demo_fleet_state must skip PENDING_LOCATION placeholder rows with no vehicle_id
        store = NeonPostgresStore.__new__(NeonPostgresStore)
        store._is_remote_neon = True
        store._database_url = "postgresql://localhost/test"
        store._connect = lambda: _FakeConn()  # type: ignore[method-assign]
        store.persist_demo_fleet_state(turn_before["snapshot"])
        assignment_inserts = [s for s, _ in executed_sql if "INSERT INTO demo_assignments" in s]
        self.assertEqual(len(assignment_inserts), 0)

        conf_res = await bridge.handle_ws_message({"type": "location_confirm", "session_id": sid})
        handoff_res = await bridge.handle_ws_message({"type": "get_handoff", "session_id": sid})
        h = handoff_res["handoff"]
        self.assertEqual(h["active_language"], "en")
        self.assertIsNotNone(h["safety_guidance_attribution"])
        self.assertIsNotNone(h["location_confirmed_at"])
        self.assertTrue(len(h["utterance_records"]) >= 1)
        self.assertEqual(h["simulated_assignments"][0]["vehicle_code"], "FIR-01")
        self.assertEqual(h["simulated_assignments"][0]["lifecycle_status"], "EN_ROUTE")
        self.assertEqual(conf_res["snapshot"]["handoff"]["location_confirmed_at"], h["location_confirmed_at"])

    async def test_parallel_orchestration_real_async_overlap_and_telemetry(self) -> None:
        bridge = LiveSessionBridge(use_live_providers=False)
        orch = bridge.orchestrator
        init_res = await bridge.handle_ws_message({"type": "session_init", "initial_language": "en", "fresh_scenario": True})
        sid = UUID(init_res["snapshot"]["session_id"])

        orig_facts = orch._branch_resolve_facts_and_question
        orig_haz = orch._branch_evaluate_hazards_and_fleet
        orig_guid = orch._branch_prepare_safety_guidance
        orig_loc = orch._branch_prepare_location_and_hospitals

        async def _slow_facts(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.04)
            return await orig_facts(*args, **kwargs)

        async def _slow_haz(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.04)
            return await orig_haz(*args, **kwargs)

        async def _slow_guid(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.04)
            return await orig_guid(*args, **kwargs)

        async def _slow_loc(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.04)
            return await orig_loc(*args, **kwargs)

        orch._branch_resolve_facts_and_question = _slow_facts  # type: ignore[method-assign]
        orch._branch_evaluate_hazards_and_fleet = _slow_haz  # type: ignore[method-assign]
        orch._branch_prepare_safety_guidance = _slow_guid  # type: ignore[method-assign]
        orch._branch_prepare_location_and_hospitals = _slow_loc  # type: ignore[method-assign]

        t0 = time.perf_counter()
        turn_res = await bridge.handle_ws_message({
            "type": "scripted_finalized_utterance",
            "session_id": sid,
            "utterance_id": str(uuid4()),
            "text": "Gas cylinder fire near Cyber Towers and people injured",
            "language": "en",
            "category": "fire_with_injuries",
            "confidence": 0.96,
            "extracted_facts": [
                {"field": "caller_location_text", "value": "Cyber Towers", "evidence_text": "near Cyber Towers"},
                {"field": "injuries_reported", "value": True, "evidence_text": "people injured"},
            ],
            "selected_instruction_id": "safety_fire_gas_cylinder_01",
            "response_text": "Please move away from the cylinder.",
        })
        wall_ms = (time.perf_counter() - t0) * 1000.0
        tel = turn_res["snapshot"]["task_execution_telemetry"]
        self.assertEqual(tel["mode"], "parallel_asyncio")
        self.assertGreaterEqual(tel["sum_branch_ms"], 150.0)
        self.assertLess(tel["parallel_wall_ms"], 115.0)
        self.assertLess(wall_ms, 150.0)
        self.assertGreater(tel["sequential_saved_ms"], 40.0)

    async def test_multi_hazard_compound_collision_fire_gas_cylinder_and_injuries_with_location_gate(self) -> None:
        bridge = LiveSessionBridge(use_live_providers=False)
        init_res = await bridge.handle_ws_message({
            "type": "session_init",
            "initial_language": "en",
            "fresh_scenario": True,
            "latitude": 17.4435,
            "longitude": 78.3772,
            "accuracy_m": 78.0,
        })
        sid = init_res["snapshot"]["session_id"]

        res = await bridge.handle_ws_message({
            "type": "scripted_finalized_utterance",
            "session_id": sid,
            "utterance_id": str(uuid4()),
            "text": "A vehicle carrying gas cylinders collided with a bus. There is a fire and several people are badly injured.",
            "language": "en",
            "category": "fire_with_injuries",
            "confidence": 0.95,
            "extracted_facts": [
                {"field": "injuries_reported", "value": True, "evidence_text": "several people are badly injured"},
                {"field": "hazard_details", "value": "bus collision and burning gas cylinders", "evidence_text": "gas cylinders collided with a bus"},
            ],
            "selected_instruction_id": "safety_fire_gas_cylinder_01",
            "response_text": "Stay clear of the burning vehicle and cylinders. Can you tell me where this happened?",
        })
        snap = res["snapshot"]
        self.assertIn("fire", snap["identified_hazards"])
        self.assertIn("gas_cylinder", snap["identified_hazards"])
        self.assertIn("vehicle_collision", snap["identified_hazards"])
        self.assertIn("injuries", snap["identified_hazards"])
        # Under Requirement 7 policy: Fire engine + Ambulance required; Police only when an explicit condition requires it
        self.assertEqual(
            set(snap["required_resource_categories"]),
            {"fire_engine", "ambulance"},
        )
        self.assertEqual(snap["active_safety_guidance"]["instruction_id"], "safety_fire_gas_cylinder_01")
        # Mandatory Location Confirmation Gate must hold both units at PENDING_LOCATION
        self.assertEqual(len(snap["assignments"]), 2)
        self.assertTrue(all(a["status"] == "pending_location" for a in snap["assignments"]))
        self.assertTrue(all(a["vehicle_code"] is None for a in snap["assignments"]))

        # Now confirm location -> Fire engine + Ambulance allocate deterministically without duplicates
        conf = await bridge.handle_ws_message({"type": "location_confirm", "session_id": sid})
        conf_snap = conf["snapshot"]
        codes = {a["vehicle_code"] for a in conf_snap["assignments"]}
        self.assertEqual(codes, {"FIR-01", "AMB-01"})
        self.assertTrue(all(a["lifecycle_status"] == "EN_ROUTE" for a in conf_snap["assignments"]))

        # Second confirmation / follow-up must NOT duplicate allocations
        conf2 = await bridge.handle_ws_message({"type": "location_confirm", "session_id": sid})
        self.assertEqual(len(conf2["snapshot"]["assignments"]), 2)

        # If an explicit police condition (e.g. explosion or violence/need police) is reported, POL-01 is added additively
        res_pol = await bridge.handle_ws_message({
            "type": "scripted_finalized_utterance",
            "session_id": sid,
            "utterance_id": str(uuid4()),
            "text": "One of the cylinders just caused a blast and we need police for crowd control!",
            "language": "en",
            "category": "explosion",
            "confidence": 0.96,
            "extracted_facts": [
                {"field": "explosion_reported", "value": True, "evidence_text": "caused a blast"},
            ],
            "selected_instruction_id": "safety_fire_gas_cylinder_01",
            "response_text": "Move further back immediately.",
        })
        codes_after_pol = {a["vehicle_code"] for a in res_pol["snapshot"]["assignments"]}
        self.assertEqual(codes_after_pol, {"FIR-01", "AMB-01", "POL-01"})

    async def test_parallel_branch_fault_isolation_preserves_critical_turn_state(self) -> None:
        bridge = LiveSessionBridge(use_live_providers=False)
        orch = bridge.orchestrator
        init_res = await bridge.handle_ws_message({"type": "session_init", "initial_language": "en", "fresh_scenario": True})
        sid = init_res["snapshot"]["session_id"]

        async def _boom_loc(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated hospital directory hiccup")

        orch._branch_prepare_location_and_hospitals = _boom_loc  # type: ignore[method-assign]
        res = await bridge.handle_ws_message({
            "type": "scripted_finalized_utterance",
            "session_id": sid,
            "utterance_id": str(uuid4()),
            "text": "Gas cylinder fire with injuries in Madhapur",
            "language": "en",
            "category": "fire_with_injuries",
            "confidence": 0.94,
            "extracted_facts": [
                {"field": "broad_location_text", "value": "Madhapur", "evidence_text": "in Madhapur"},
                {"field": "injuries_reported", "value": True, "evidence_text": "with injuries"},
            ],
            "selected_instruction_id": "safety_fire_gas_cylinder_01",
            "response_text": "Please stay back safely.",
        })
        snap = res["snapshot"]
        self.assertEqual(snap["category"], "fire_with_injuries")
        self.assertEqual(snap["active_safety_guidance"]["instruction_id"], "safety_fire_gas_cylinder_01")
        tel = snap["task_execution_telemetry"]
        self.assertEqual(tel["degraded_branches"], ["location_and_hospitals_prep"])
        self.assertEqual(tel["branches"]["location_and_hospitals_prep"]["status"], "error")
        self.assertEqual(tel["branches"]["fact_merge_and_question"]["status"], "ok")
        self.assertEqual(tel["branches"]["hazard_and_fleet_eval"]["status"], "ok")

    async def test_pre_tts_orchestration_state_updated_event_and_overlap_telemetry(self) -> None:
        bridge = LiveSessionBridge(use_live_providers=False)
        init_res = await bridge.handle_ws_message({"type": "session_init", "initial_language": "en", "fresh_scenario": True})
        sid = init_res["snapshot"]["session_id"]
        emitted: list[dict[str, object]] = []

        async def _delayed_tts(**_kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.06)
            return None

        async def _delayed_persist(_sid, _snap):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.05)

        bridge._synthesize_timbre_payload = _delayed_tts  # type: ignore[method-assign]
        bridge._persist_to_postgres_if_configured = _delayed_persist  # type: ignore[method-assign]

        t0 = time.perf_counter()
        res = await bridge.handle_ws_message(
            {
                "type": "scripted_finalized_utterance",
                "session_id": sid,
                "utterance_id": str(uuid4()),
                "text": "Road accident near Shilparamam",
                "language": "en",
                "category": "road_accident",
                "confidence": 0.93,
                "extracted_facts": [
                    {"field": "caller_location_text", "value": "Shilparamam", "evidence_text": "near Shilparamam"},
                ],
                "selected_instruction_id": "safety_road_accident_01",
                "response_text": "Help is being prepared. Please confirm your location.",
            },
            emit_event=emitted.append,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        pre_tts_events = [e for e in emitted if e.get("type") == "orchestration_state_updated"]
        self.assertEqual(len(pre_tts_events), 1)
        self.assertEqual(pre_tts_events[0]["snapshot"]["category"], "road_accident")  # type: ignore[index]
        post_overlap = res["stage_latencies"]["task_execution_telemetry"]["post_orchestration_overlap"]
        self.assertIsNotNone(post_overlap["postgres_persist_ms"])
        self.assertLess(post_overlap["parallel_wall_ms"], 105.0)
        self.assertLess(total_ms, 130.0)

    async def test_stale_revision_persistence_guard_and_concurrent_turns_no_duplicate_units(self) -> None:
        # 1. Verify NeonPostgresStore.persist_session_snapshot rejects stale revision writes
        store = NeonPostgresStore.__new__(NeonPostgresStore)
        store._is_remote_neon = True
        calls: list[tuple[str, list[object]]] = []
        current_db_rev = {"rev": 0}

        def _fake_execute_sql(query: str, params=None, **_kwargs):  # type: ignore[no-untyped-def]
            p = list(params or [])
            calls.append((query.strip(), p))
            if "INSERT INTO sessions" in query:
                incoming_rev = int(p[4])
                if incoming_rev >= current_db_rev["rev"]:
                    current_db_rev["rev"] = incoming_rev
                    return {"rows": [{"revision": incoming_rev}]}
                return {"rows": []}
            return {"rows": []}

        store.execute_sql = _fake_execute_sql  # type: ignore[method-assign]
        sid = str(uuid4())
        inc_id = str(uuid4())
        snap_rev4 = {
            "session_id": sid,
            "incident_id": inc_id,
            "session_state": "active",
            "incident_state": "open",
            "revision": 4,
            "last_event_sequence": 4,
            "initial_language": "en",
            "active_language": "en",
            "consecutive_unknown_count": 0,
            "consecutive_provider_blocks": 0,
            "unknown_fallback_applied": False,
            "control_room_review_required": False,
            "category": "fire",
            "category_established_above_threshold": True,
            "review_reason_codes": [],
            "utterances": [],
            "validated_facts": [],
            "assignments": [],
        }
        snap_rev2 = dict(snap_rev4, revision=2, last_event_sequence=2, category="unknown")
        self.assertTrue(store.persist_session_snapshot(snap_rev4, []))
        calls.clear()
        self.assertFalse(store.persist_session_snapshot(snap_rev2, []))
        self.assertFalse(any("INSERT INTO incidents" in q for q, _ in calls))
        self.assertEqual(current_db_rev["rev"], 4)

        # 2. Verify concurrent finalized utterances + location_confirm produce zero duplicate allocations
        bridge = LiveSessionBridge(use_live_providers=False)
        init_res = await bridge.handle_ws_message({"type": "session_init", "initial_language": "en", "fresh_scenario": True})
        sess_id = init_res["snapshot"]["session_id"]
        res_a, res_b, res_c = await asyncio.gather(
            bridge.handle_ws_message({
                "type": "scripted_finalized_utterance",
                "session_id": sess_id,
                "utterance_id": str(uuid4()),
                "text": "Gas cylinder fire near Hitech City",
                "language": "en",
                "category": "fire",
                "confidence": 0.92,
                "extracted_facts": [
                    {"field": "caller_location_text", "value": "Hitech City", "evidence_text": "near Hitech City"},
                ],
                "selected_instruction_id": "safety_fire_gas_cylinder_01",
                "response_text": "Move away from the cylinder.",
            }),
            bridge.handle_ws_message({
                "type": "scripted_finalized_utterance",
                "session_id": sess_id,
                "utterance_id": str(uuid4()),
                "text": "Two people have burn injuries",
                "language": "en",
                "category": "fire_with_injuries",
                "confidence": 0.94,
                "extracted_facts": [
                    {"field": "injuries_reported", "value": True, "evidence_text": "burn injuries"},
                    {"field": "affected_person_count", "value": 2, "evidence_text": "Two people"},
                ],
                "selected_instruction_id": "safety_fire_gas_cylinder_01",
                "response_text": "Do not put ice on burns.",
            }),
            bridge.handle_ws_message({"type": "location_confirm", "session_id": sess_id}),
        )
        final_conf = await bridge.handle_ws_message({"type": "location_confirm", "session_id": sess_id})
        asgns = final_conf["snapshot"]["assignments"]
        by_cat = [a["resource_category"] for a in asgns]
        self.assertEqual(len(by_cat), len(set(by_cat)))
        self.assertEqual(set(by_cat), {"fire_engine", "ambulance"})
        self.assertTrue(all(a["vehicle_code"] in ("FIR-01", "AMB-01") for a in asgns))
        _ = (res_a, res_b, res_c)


if __name__ == "__main__":
    unittest.main()
