"""
Structured incident handoff generation and persistence (Section 20).

Includes all required fields from Section 20 and marks transcript content as
'not retained' if regenerated after the transcript retention period expires.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from backend.app.audit.writer import AuditEvent
from backend.app.orchestrator.hospital import HospitalScoreBreakdown
from backend.app.orchestrator.instructions import InstructionLibrary
from backend.app.orchestrator.session import IncidentSessionState


@dataclass(frozen=True)
class StructuredIncidentHandoff:
    """Complete structured handoff for an incident (Section 20)."""
    incident_id: UUID
    session_id: UUID
    generated_at: datetime
    incident_category: str
    classification_attempt_count: int
    classification_confidences: tuple[float, ...]
    provider_blocked_turn_count: int
    fallback_reason: str | None
    transcript_retained: bool
    original_caller_statements: tuple[str, ...]
    confirmed_facts: tuple[dict[str, object], ...]
    source_utterance_ids: tuple[str, ...]
    backend_provenance_references: tuple[dict[str, object], ...]
    unanswered_questions: tuple[str, ...]
    conflicting_information: tuple[dict[str, object], ...]
    confirmed_location_label: str | None
    location_source: str | None
    accuracy_radius_meters: float | None
    location_confidence: float | None
    location_confirmation_status: str
    affected_person_count: int | None
    required_resource_categories: tuple[str, ...]
    simulated_assignments: tuple[dict[str, object], ...]
    hospital_directory_suggestions: tuple[HospitalScoreBreakdown, ...]
    possible_duplicate_flag: bool
    matched_duplicate_incident_id: str | None
    possible_false_or_prank_flag: bool
    control_room_review_required: bool
    control_room_review_reason_codes: tuple[str, ...]
    event_timeline: tuple[dict[str, object], ...]
    session_completion_status: str
    active_language: str = "en"
    initial_language: str = "en"
    language_mode: str = "manual"
    detected_languages_seen: tuple[str, ...] = ()
    safety_guidance_attribution: dict[str, object] | None = None
    location_confirmed_at: datetime | None = None
    broad_location_text: str | None = None
    caller_landmark_text: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    utterance_records: tuple[dict[str, object], ...] = ()
    prototype_disclaimer: str = (
        "SIMULATED HANDOFF — NOT AN OFFICIAL 112 OR MEDICAL REPORT. "
        "All resources, assignments, statuses, and ETAs are simulated."
    )


class HandoffGenerator:
    """Generates auditable structured incident handoffs (Section 20)."""

    def generate_handoff(
        self,
        state: IncidentSessionState,
        *,
        audit_events: Sequence[AuditEvent],
        transcript_retention_expired: bool = False,
        generated_at: datetime,
    ) -> StructuredIncidentHandoff:
        if transcript_retention_expired:
            statements: tuple[str, ...] = ("not retained",)
        else:
            statements = tuple(u.text for u in state.utterances)

        confirmed_facts = tuple(
            {
                "field_name": f.field_name,
                "value": f.value,
                "source_utterance_id": str(f.source_utterance_id),
                "evidence_text": (
                    "not retained" if transcript_retention_expired else f.evidence_text
                ),
            }
            for f in state.validated_facts
        )

        affected_count: int | None = None
        for f in state.validated_facts:
            if f.field_name == "affected_person_count" and isinstance(f.value, int):
                affected_count = f.value

        backend_prov: list[dict[str, object]] = []
        if state.location_state is not None:
            backend_prov.append(
                {
                    "source_type": state.location_state.provenance.source_type.value,
                    "source_id": state.location_state.provenance.source_id,
                    "timestamp": state.location_state.provenance.source_timestamp.isoformat(),
                }
            )
        for a in state.assignments:
            backend_prov.append(
                {
                    "source_type": a.provenance.source_type.value,
                    "source_id": a.provenance.source_id,
                    "timestamp": a.provenance.source_timestamp.isoformat(),
                }
            )

        conflicts = tuple(
            {
                "field_name": c.field_name,
                "prior_value": c.prior_value,
                "prior_utterance_id": str(c.prior_utterance_id),
                "new_value": c.new_value,
                "new_utterance_id": str(c.new_utterance_id),
                "description": c.description,
            }
            for c in state.conflicts
        )

        sim_assignments = tuple(
            {
                "assignment_id": str(a.assignment_id),
                "category": a.category.value,
                "resource_id": a.resource_id,
                "vehicle_code": a.resource_id,
                "resource_callsign": a.resource_callsign,
                "status": a.status.value,
                "lifecycle_status": a.lifecycle_status,
                "assigned_at": a.assigned_at.isoformat(),
                "en_route_at": a.en_route_at.isoformat() if a.en_route_at else None,
                "arrived_at": a.arrived_at.isoformat() if a.arrived_at else None,
                "estimated_arrival_at": (
                    a.estimated_arrival_at.isoformat() if a.estimated_arrival_at else None
                ),
                "simulated_eta_minutes": a.simulated_eta_minutes,
                "distance_meters": a.distance_meters,
                "allocation_mode": a.allocation_mode,
                "is_simulated": True,
            }
            for a in state.assignments
        )

        utterance_records = tuple(
            {
                "utterance_id": str(u.utterance_id),
                "sequence_number": u.sequence_number,
                "text": "not retained" if transcript_retention_expired else u.text,
                "language": u.language,
                "utterance_role": u.utterance_role.value if u.utterance_role else None,
                "finalized_at": u.finalized_at.isoformat(),
            }
            for u in state.utterances
        )

        safety_attr: dict[str, object] | None = None
        if state.last_selected_instruction_id:
            inst = InstructionLibrary().get_by_id(
                state.last_selected_instruction_id,
                state.active_language,
            )
            if inst is not None:
                safety_attr = {
                    "instruction_id": inst.instruction_id,
                    "language": inst.language,
                    "approved_text": inst.approved_text,
                    "source": inst.source,
                    "review_date": inst.review_date.isoformat(),
                    "version": inst.version,
                }

        timeline = tuple(
            {
                "event_id": str(e.event_id),
                "event_type": e.event_type,
                "severity": e.severity.value,
                "timestamp": e.timestamp.isoformat(),
                "outcome": e.outcome,
                "reason_code": e.reason_code,
            }
            for e in audit_events
        )

        unanswered: list[str] = list(state.missing_information_fields)
        if state.pending_followup_question:
            unanswered.append(state.pending_followup_question)

        return StructuredIncidentHandoff(
            incident_id=state.incident_id,
            session_id=state.session_id,
            generated_at=generated_at,
            incident_category=state.category.value,
            classification_attempt_count=state.classification_attempts,
            classification_confidences=tuple(state.classification_confidences),
            provider_blocked_turn_count=state.total_provider_blocks,
            fallback_reason=state.fallback_reason,
            transcript_retained=not transcript_retention_expired,
            original_caller_statements=statements,
            confirmed_facts=confirmed_facts,
            source_utterance_ids=tuple(str(u.utterance_id) for u in state.utterances),
            backend_provenance_references=tuple(backend_prov),
            unanswered_questions=tuple(unanswered),
            conflicting_information=conflicts,
            confirmed_location_label=(
                state.location_state.resolved_label if state.location_state else None
            ),
            location_source=state.location_state.source if state.location_state else None,
            accuracy_radius_meters=(
                state.location_state.accuracy_m if state.location_state else None
            ),
            location_confidence=(
                state.location_state.match_confidence if state.location_state else None
            ),
            location_confirmation_status=(
                state.location_state.confirmation_status.value
                if state.location_state
                else "unresolved"
            ),
            affected_person_count=affected_count,
            required_resource_categories=tuple(c.value for c in state.required_resource_categories),
            simulated_assignments=sim_assignments,
            hospital_directory_suggestions=tuple(state.hospital_recommendations),
            possible_duplicate_flag=state.duplicate_match is not None,
            matched_duplicate_incident_id=(
                str(state.duplicate_match.matched_incident_id)
                if state.duplicate_match
                else None
            ),
            possible_false_or_prank_flag=(
                state.prank_flag.possible_false_or_prank_call
                if state.prank_flag
                else False
            ),
            control_room_review_required=state.control_room_review_required,
            control_room_review_reason_codes=tuple(state.review_reason_codes),
            event_timeline=timeline,
            session_completion_status=state.session_state,
            active_language=state.active_language,
            initial_language=state.initial_language,
            language_mode=getattr(state, "language_mode", "manual"),
            detected_languages_seen=tuple(
                getattr(state, "detected_languages_seen", [state.initial_language])
                or [state.initial_language]
            ),
            safety_guidance_attribution=safety_attr,
            location_confirmed_at=(
                state.location_state.confirmed_at if state.location_state else None
            ),
            broad_location_text=(
                state.location_state.broad_area_text if state.location_state else None
            ),
            caller_landmark_text=(
                (state.location_state.caller_landmark_text or state.location_state.caller_location_text)
                if state.location_state
                else None
            ),
            latitude=state.location_state.latitude if state.location_state else None,
            longitude=state.location_state.longitude if state.location_state else None,
            utterance_records=utterance_records,
        )
