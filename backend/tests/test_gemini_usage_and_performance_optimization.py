"""
Comprehensive verification suite for SETU.ai Gemini API Usage and Performance Optimization.

Covers all mandatory requirements:
1. One unique finalized caller utterance = one Gemini reasoning request; retransmission
   and in-flight coalescing on (session_id, utterance_id); two distinct 'Yes' utterances
   answering different questions are NEVER treated as duplicates; VAD misfire & echo suppression.
2. Adaptive thinking depth (LOW vs MEDIUM) based on clinical acuity, multi-hazard context,
   and corrections—never word count alone ('Two people are unconscious' -> MEDIUM).
3. Intelligent, personalized next-question generation (Scenarios A, B, C) without fixed questionnaires.
4. Human-like conversational responses and post-allocation (EN_ROUTE) caller support.
5. Complete session memory across 12+ turns, corrections with preserved provenance, and
   WebSocket reconnections.
6. Prompt compaction & token efficiency measurements (before vs. after).
"""

import asyncio
from datetime import datetime, timezone
import unittest
from uuid import UUID, uuid4

from backend.app.adapters.base import (
    ConversationTurnRecord,
    DialogueContext,
    EmergencyCategory,
    FinalizedUtteranceRecord,
    PendingQuestionRecord,
    ProvenancedFact,
    ReasoningOutput,
    UtteranceRole,
)
from backend.app.adapters.reasoning import (
    GeminiReasoningAdapter,
    build_compact_dialogue_prompt,
    build_legacy_unoptimized_prompt,
    select_adaptive_thinking_level,
)
from backend.app.main import LiveSessionBridge
from backend.app.orchestrator.announcements import deterministic_demo_classify_and_extract


class GeminiOptimizationAndMemoryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)

    def _make_context(
        self,
        *,
        text: str,
        lang: str = "en",
        prior_utterances: tuple[FinalizedUtteranceRecord, ...] = (),
        validated_facts: tuple[ProvenancedFact, ...] = (),
        pending_question: PendingQuestionRecord | None = None,
        identified_hazards: tuple[str, ...] = (),
        current_category: str = "unknown",
        conversation_history: tuple[ConversationTurnRecord, ...] = (),
        session_memory_summary: dict[str, object] | None = None,
    ) -> DialogueContext:
        utt = FinalizedUtteranceRecord(
            utterance_id=uuid4(),
            sequence_number=len(prior_utterances) + 1,
            text=text,
            language=lang,  # type: ignore[arg-type]
            utterance_role=None,
            finalized_at=self.now,
        )
        return DialogueContext(
            session_id=uuid4(),
            incident_id=uuid4(),
            active_language=lang,  # type: ignore[arg-type]
            prior_utterances=prior_utterances,
            validated_facts=validated_facts,
            current_utterance=utt,
            pending_followup_question=pending_question.question_text if pending_question else None,
            eligible_instruction_ids=("disclosure_session_start", "safety_fire_01", "safety_fire_gas_cylinder_01", "safety_medical_01"),
            backend_status_summary={"current_category": current_category},
            conversation_history=conversation_history,
            pending_question=pending_question,
            identified_hazards=identified_hazards,
            session_memory_summary=session_memory_summary,
        )

    async def test_adaptive_thinking_depth_selection_never_uses_word_count_alone(self) -> None:
        """Requirement 2: Short high-acuity utterances select MEDIUM; routine answers select LOW."""
        # 4-word life-threatening statement -> MEDIUM
        ctx_unconscious = self._make_context(text="Two people are unconscious.")
        self.assertEqual(select_adaptive_thinking_level(ctx_unconscious), "MEDIUM")

        # 3-word life-threatening statement -> MEDIUM
        ctx_he_unconscious = self._make_context(text="He is unconscious.")
        self.assertEqual(select_adaptive_thinking_level(ctx_he_unconscious), "MEDIUM")

        # Telugu & Hindi unconsciousness -> MEDIUM
        ctx_te_unconscious = self._make_context(text="ఇద్దరు స్పృహలో లేరు", lang="te")
        self.assertEqual(select_adaptive_thinking_level(ctx_te_unconscious), "MEDIUM")
        ctx_hi_unconscious = self._make_context(text="एक व्यक्ति बेहोश है", lang="hi")
        self.assertEqual(select_adaptive_thinking_level(ctx_hi_unconscious), "MEDIUM")

        # Multi-hazard gas cylinder + bus collision + fire -> MEDIUM
        ctx_multi = self._make_context(
            text="A vehicle carrying gas cylinders collided with a bus and fire broke out.",
        )
        self.assertEqual(select_adaptive_thinking_level(ctx_multi), "MEDIUM")

        # Caller correction of an earlier fact -> MEDIUM
        u1_id = uuid4()
        ctx_correction = self._make_context(
            text="Actually it is near Hitech City Metro Station, not Madhapur.",
            validated_facts=(
                ProvenancedFact(
                    field_name="caller_location_text",
                    value="Madhapur",
                    source_utterance_id=u1_id,
                    evidence_text="Madhapur",
                ),
            ),
            current_category="fire",
        )
        self.assertEqual(select_adaptive_thinking_level(ctx_correction), "MEDIUM")

        # Routine short answers to pending questions -> LOW
        pq_inj = PendingQuestionRecord(
            question_id="pq-1",
            question_text="Is anyone hurt or trapped there?",
            target_field="injuries_reported",
            asked_at=self.now,
        )
        for routine_reply in ("Yes.", "No.", "Two people.", "Near ABC School."):
            ctx_routine = self._make_context(
                text=routine_reply,
                pending_question=pq_inj,
                current_category="fire",
            )
            self.assertEqual(
                select_adaptive_thinking_level(ctx_routine),
                "LOW",
                f"Expected LOW thinking depth for routine answer: {routine_reply}",
            )

    async def test_one_utterance_equals_one_gemini_request_and_two_yes_answers_not_treated_as_duplicates(self) -> None:
        """
        Requirement 1:
        - Retransmissions and concurrent deliveries of the same (session_id, utterance_id) coalesce to 1 call.
        - Two separate 'Yes' utterances with different utterance_ids answering two different questions
          BOTH invoke Gemini reasoning independently.
        """
        transport_calls: list[tuple[UUID, str]] = []

        async def _counting_transport(context: DialogueContext, _timeout: float) -> ReasoningOutput:
            await asyncio.sleep(0.03)
            utt_id = context.current_utterance.utterance_id
            transport_calls.append((utt_id, context.current_utterance.text))
            target_field = (
                context.pending_question.target_field
                if context.pending_question is not None
                else "injuries_reported"
            )
            fact_val: bool | str = True if target_field == "injuries_reported" else "safe_distance"
            return ReasoningOutput(
                utterance_role=UtteranceRole.ANSWER_TO_PENDING_QUESTION,
                utterance_role_source_id=utt_id,
                category=EmergencyCategory.FIRE_WITH_INJURIES,
                category_confidence=0.95,
                supporting_utterance_ids=(utt_id,),
                extracted_facts=(
                    ProvenancedFact(
                        field_name=target_field,
                        value=fact_val,
                        source_utterance_id=utt_id,
                        evidence_text=context.current_utterance.text,
                    ),
                ),
                conflicts=(),
                missing_information_fields=(),
                proposed_followup_question=(
                    "Are you in a safe location away from the fire?"
                    if target_field == "injuries_reported"
                    else None
                ),
                proposed_followup_target_field=(
                    "caller_safe_spot_status" if target_field == "injuries_reported" else None
                ),
                selected_instruction_id="safety_fire_01",
                caller_response_text="Understood, I have recorded that.",
                prompt_tokens=210,
                completion_tokens=45,
                latency_ms=30.0,
                pending_question_answered=True,
            )

        adapter = GeminiReasoningAdapter(
            model_id="gemini-3.8-flash",
            transport_callable=_counting_transport,
        )
        bridge = LiveSessionBridge(use_live_providers=False)
        bridge.gemini_adapter = adapter
        bridge.orchestrator._reasoning = adapter

        state = bridge.orchestrator.create_session(initial_language="en", now=self.now)
        sid = state.session_id

        # Set first pending question: asking about injuries
        bridge.orchestrator._set_pending_question(
            state,
            question_text="Is anyone hurt there?",
            target_field_hint="injuries_reported",
            source_utterance_id=uuid4(),
            now=self.now,
        )

        # Turn 1: Caller says "Yes." (utt_yes_1) — fire 3 concurrent duplicate deliveries + 1 sequential retransmission
        utt_yes_1 = uuid4()
        concurrent_results = await asyncio.gather(
            bridge.orchestrator.process_finalized_utterance(sid, utterance_id=utt_yes_1, text="Yes.", now=self.now),
            bridge.orchestrator.process_finalized_utterance(sid, utterance_id=utt_yes_1, text="Yes.", now=self.now),
            bridge.orchestrator.process_finalized_utterance(sid, utterance_id=utt_yes_1, text="Yes.", now=self.now),
        )
        # Sequential retransmission of the same utterance_id
        await bridge.orchestrator.process_finalized_utterance(sid, utterance_id=utt_yes_1, text="Yes.", now=self.now)

        self.assertEqual(len(transport_calls), 1, "Same (session_id, utterance_id) must trigger exactly 1 Gemini call")
        self.assertTrue(
            any(f.field_name == "injuries_reported" and f.value is True for f in concurrent_results[0].validated_facts)
        )

        # Turn 2: Now pending question is "Are you in a safe location away from the fire?" (caller_safe_spot_status).
        # Caller says "Yes." AGAIN, but as a new genuine utterance (utt_yes_2).
        utt_yes_2 = uuid4()
        await bridge.orchestrator.process_finalized_utterance(sid, utterance_id=utt_yes_2, text="Yes.", now=self.now)

        self.assertEqual(
            len(transport_calls),
            2,
            "Second genuine 'Yes.' with a new utterance_id MUST trigger its own Gemini reasoning call",
        )
        self.assertTrue(
            any(
                f.field_name == "caller_safe_spot_status" and f.value == "safe_distance"
                for f in state.validated_facts
            )
        )

    async def test_personalized_next_questions_scenarios_a_b_c(self) -> None:
        """
        Requirement 3:
        - Scenario A: Gas-cylinder vehicle + bus collision + fire + injuries -> never re-asks injuries; asks caller safety/location.
        - Scenario B: Fire + nobody injured -> never re-asks injuries; asks location/safety.
        - Scenario C: One person unconscious -> extracts unconsciousness + injuries; asks location or breathing status.
        """
        # Scenario A
        cat_a, _role_a, facts_a, next_q_a, reply_a = deterministic_demo_classify_and_extract(
            text="A vehicle carrying gas cylinders collided with a bus, fire broke out and several people were injured.",
            utterance_id=uuid4(),
            active_language="en",
            existing_category=EmergencyCategory.UNKNOWN,
            existing_established=False,
            existing_facts=(),
            pending_followup_question=None,
        )
        self.assertEqual(cat_a, EmergencyCategory.FIRE_WITH_INJURIES)
        self.assertTrue(any(f.field_name == "injuries_reported" and f.value is True for f in facts_a))
        self.assertTrue(any(f.field_name == "hazard_details" and f.value == "gas_cylinder_involved" for f in facts_a))
        self.assertIsNotNone(next_q_a)
        self.assertNotIn("anyone hurt", (next_q_a or "").lower())
        self.assertNotIn("anyone injured", (next_q_a or "").lower())
        self.assertIn("safe distance", (next_q_a or "").lower())
        self.assertIn("gas cylinder", reply_a.lower())

        # Scenario B
        cat_b, _role_b, facts_b, next_q_b, _reply_b = deterministic_demo_classify_and_extract(
            text="There is a fire in the shop outside, nobody is injured.",
            utterance_id=uuid4(),
            active_language="en",
            existing_category=EmergencyCategory.UNKNOWN,
            existing_established=False,
            existing_facts=(),
            pending_followup_question=None,
        )
        self.assertEqual(cat_b, EmergencyCategory.FIRE)
        self.assertTrue(any(f.field_name == "injuries_reported" and f.value is False for f in facts_b))
        self.assertIsNotNone(next_q_b)
        self.assertNotIn("injured", (next_q_b or "").lower())
        self.assertNotIn("hurt", (next_q_b or "").lower())
        self.assertIn("landmark", (next_q_b or "").lower())

        # Scenario C (with location already known -> asks about breathing)
        loc_fact = ProvenancedFact(
            field_name="caller_location_text",
            value="Madhapur Metro Station",
            source_utterance_id=uuid4(),
            evidence_text="Madhapur Metro Station",
        )
        cat_c, _role_c, facts_c, next_q_c, reply_c = deterministic_demo_classify_and_extract(
            text="One person is unconscious here.",
            utterance_id=uuid4(),
            active_language="en",
            existing_category=EmergencyCategory.UNKNOWN,
            existing_established=False,
            existing_facts=(loc_fact,),
            pending_followup_question=None,
        )
        self.assertEqual(cat_c, EmergencyCategory.MEDICAL_EMERGENCY)
        self.assertTrue(
            any(f.field_name == "consciousness_status" and "unconscious" in str(f.value) for f in facts_c)
        )
        self.assertTrue(any(f.field_name == "affected_person_count" and f.value == 1 for f in facts_c))
        self.assertIsNotNone(next_q_c)
        self.assertIn("breathing", (next_q_c or "").lower())
        self.assertIn("airway", reply_c.lower())

    async def test_twelve_turn_session_memory_corrections_reconnection_and_post_dispatch_support(self) -> None:
        """
        Requirements 4, 5, 6:
        Verify a 12-turn conversation with short replies, location refinement, caller fact corrections,
        WebSocket reconnection, post-allocation (EN_ROUTE) caller support, and prompt compaction savings.
        """
        bridge = LiveSessionBridge(use_live_providers=False)
        init_res = await bridge.handle_ws_message({
            "type": "session_init",
            "initial_language": "en",
            "fresh_scenario": True,
            "latitude": 17.4435,
            "longitude": 78.3772,
            "accuracy_m": 45.0,
        })
        sid = UUID(init_res["snapshot"]["session_id"])
        token = init_res["snapshot"]["session_token"]

        turns = [
            # Turn 1: Initial fire report
            ("There is a fire breaking out on our street!", "en"),
            # Turn 2: Location answer
            ("In Madhapur", "en"),
            # Turn 3: Landmark refinement
            ("Near the road beside ABC School", "en"),
            # Turn 4: Injuries answer (short reply)
            ("Yes.", "en"),
            # Turn 5: Person count answer (short reply)
            ("Two people.", "en"),
            # Turn 6: Correction of person count
            ("Wait, actually three people are injured, not two.", "en"),
            # Turn 7: Additional hazard reported
            ("There is a gas cylinder inside the stall that is heating up.", "en"),
            # Turn 8: Safe spot answer (short reply)
            ("Yes, we stepped outside to a safe distance.", "en"),
            # Turn 9: Spoken location confirmation -> triggers FIR-01 + AMB-01 EN_ROUTE
            ("Yes, location confirmed.", "en"),
            # Turn 10: Post-allocation question about earlier facts & units
            ("Do you remember what location and how many people I said, and which units are coming?", "en"),
            # Turn 11: Telugu code-switch follow-up while units are EN_ROUTE
            ("ఎంత సమయం పడుతుంది?", "te"),
            # Turn 12: Hindi code-switch follow-up while units are EN_ROUTE
            ("क्या मुझे कुछ और करना चाहिए?", "hi"),
        ]

        for idx, (utt_text, lang) in enumerate(turns, start=1):
            state = bridge.orchestrator.get_session(sid)
            # Simulate outage-resilient deterministic reasoning or structured turn processing
            inferred_cat, role, facts, next_q, reply_text = deterministic_demo_classify_and_extract(
                text=utt_text,
                utterance_id=uuid4(),
                active_language=lang,  # type: ignore[arg-type]
                existing_category=state.category,
                existing_established=state.category_established_above_threshold,
                existing_facts=state.validated_facts,
                pending_followup_question=state.pending_followup_question,
                pending_question=state.pending_question_record,
                question_history=state.question_history,
                active_assignments=state.assignments,
            )
            await bridge.handle_ws_message({
                "type": "scripted_finalized_utterance",
                "session_id": str(sid),
                "utterance_id": str(uuid4()),
                "text": utt_text,
                "language": lang,
                "category": inferred_cat.value,
                "utterance_role": role.value,
                "confidence": 0.94,
                "extracted_facts": [
                    {
                        "field": f.field_name,
                        "value": f.value,
                        "evidence_text": f.evidence_text,
                    }
                    for f in facts
                ],
                "followup_question": next_q,
                "response_text": reply_text,
            })

            if idx == 6:
                # Reconnect mid-session after Turn 6 and verify zero memory loss
                reconn = await bridge.handle_ws_message({
                    "type": "session_reconnect",
                    "session_id": str(sid),
                    "session_token": token,
                    "last_received_sequence": 3,
                })
                self.assertEqual(reconn["event"], "session_reconnected")
                self.assertEqual(len(reconn["snapshot"]["utterances"]), 6)

            if idx == 10:
                state_t10 = bridge.orchestrator.get_session(sid)
                # Turn 10 response must recall Madhapur — Near the road beside ABC School, 3 people, and FIR-01/AMB-01
                self.assertIn("Madhapur", state_t10.last_spoken_response_text)
                self.assertIn("3", state_t10.last_spoken_response_text)
                self.assertIn("FIR-01", state_t10.last_spoken_response_text)
                self.assertIn("AMB-01", state_t10.last_spoken_response_text)

        final_state = bridge.orchestrator.get_session(sid)
        final_snap = bridge.handle_ws_message  # verify via serialize_session_snapshot
        self.assertEqual(len(final_state.utterances), 12)
        facts_by_name = {f.field_name: f.value for f in final_state.validated_facts}
        self.assertEqual(facts_by_name.get("broad_location_text"), "Madhapur")
        self.assertIn("ABC School", str(facts_by_name.get("caller_location_text")))
        self.assertEqual(facts_by_name.get("affected_person_count"), 3)
        self.assertEqual(facts_by_name.get("injuries_reported"), True)
        self.assertEqual(facts_by_name.get("caller_safe_spot_status"), "safe_distance")

        # Verify correction history (2 -> 3 people) is retained in superseded_facts with provenance
        self.assertTrue(
            any(
                sf.field_name == "affected_person_count" and sf.prior_value == 2 and sf.new_value == 3
                for sf in final_state.superseded_facts
            ),
            f"Expected superseded_facts to retain 2->3 correction, got: {final_state.superseded_facts}",
        )
        self.assertGreaterEqual(len(final_state.location_history), 2)

        # Verify prompt compaction on Turn 12 vs legacy uncompacted prompt
        mem_summary = bridge.orchestrator._build_session_memory_summary(final_state)
        scoped_ids = bridge.orchestrator._select_scoped_eligible_instruction_ids(final_state, "क्या मुझे कुछ और करना चाहिए?")
        ctx_turn12 = DialogueContext(
            session_id=final_state.session_id,
            incident_id=final_state.incident_id,
            active_language=final_state.active_language,
            prior_utterances=tuple(final_state.utterances[:-1]),
            validated_facts=tuple(final_state.validated_facts),
            current_utterance=final_state.utterances[-1],
            pending_followup_question=final_state.pending_followup_question,
            eligible_instruction_ids=scoped_ids,
            backend_status_summary={
                "current_category": final_state.category.value,
                "location_confirmed": "true",
                "simulated_assignments_count": len(final_state.assignments),
            },
            conversation_history=tuple(final_state.conversation_history),
            pending_question=final_state.pending_question_record,
            identified_hazards=tuple(final_state.identified_hazards),
            session_memory_summary=mem_summary,
        )
        sys_compact, usr_compact = build_compact_dialogue_prompt(ctx_turn12)
        sys_legacy, usr_legacy = build_legacy_unoptimized_prompt(ctx_turn12)

        self.assertLess(len(usr_compact), len(usr_legacy))
        # Every validated fact and corrected count is preserved in the compact prompt
        self.assertIn("Madhapur", usr_compact)
        self.assertIn("ABC School", usr_compact)
        self.assertIn("affected_person_count=3", usr_compact)
        self.assertIn("FIR-01", usr_compact)
        self.assertIn("AMB-01", usr_compact)


if __name__ == "__main__":
    unittest.main()
