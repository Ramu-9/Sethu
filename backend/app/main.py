"""
FastAPI entry point and WebSocket /ws/session route for SETU (Section 7 & Section 8).

Implements:
- WS /ws/session: Authoritative session creation, token-authenticated reconnection,
  16 kHz mono PCM audio frame ingestion through PrismaSTTAdapter (Gate 0.1 guarded),
  GPS 500m evaluation, caller location confirmation, barge-in cancellation, and
  structured incident handoff broadcasting.
- GET /api/instructions/manifest: Serves the versioned instruction audio manifest
  with SHA-256 digests and honest Gate 0.2 placeholder status.
- GET /backend-media/instructions/{lang}/{instruction_id}/{filename}: Serves the
  16 kHz mono PCM WAV files for AudioContext + AudioBufferSourceNode playback.
- Stdio WebSocket Bridge (--stdio-ws-bridge): Allows the port-3000 HTTP/WS gateway
  to forward /ws/session frames directly into this Python process when uvicorn/fastapi
  wheels are not installed in a sandboxed container.
"""

import asyncio
import base64
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any
from uuid import UUID, uuid4

from backend.app.adapters.base import (
    AdapterError,
    AudioChunk,
    EmergencyCategory,
    ProvenancedFact,
    ProviderBlockedContentError,
    ReasoningOutput,
    ReasoningUnavailableError,
    STTProviderError,
    STTSignalType,
    SupportedLanguage,
    TTSProviderError,
    UtteranceRole,
)
from backend.app.adapters.mock import MockReasoningAdapter
from backend.app.adapters.prisma import GnaniPrismaAdapter
from backend.app.adapters.reasoning import (
    GeminiReasoningAdapter,
    build_gemini_rest_transport,
)
from backend.app.adapters.timbre import GnaniTimbreAdapter
from backend.app.audit.writer import InMemoryAuditWriter
from backend.app.config import load_settings
from backend.app.models.capability_profile import DEFAULT_DEPLOYMENT_CAPABILITY_PROFILE
from backend.app.observability.metrics import MetricsRegistry
from backend.app.orchestrator.dispatch import (
    DemoLifecycleStatus,
    DispatchEngine,
    InMemoryResourceRepository,
    ResourceCategory,
    SyntheticResourceRecord,
    build_demo_fleet_resources,
)
from backend.app.orchestrator.duplicates import DuplicateDetector
from backend.app.orchestrator.handoff import HandoffGenerator
from backend.app.orchestrator.hospital import HospitalRanker, HospitalScoringWeights
from backend.app.orchestrator.instructions import InstructionLibrary
from backend.app.orchestrator.location import GPSPermissionStatus, LocationResolver
from backend.app.orchestrator.policy import PolicyValidator
from backend.app.db.pool import NeonPostgresStore
from backend.app.orchestrator.session import SessionOrchestrator, SessionTokenSigner
from backend.app.orchestrator.timeout import CallerTimeoutCoordinator


def build_default_orchestrator(
    *,
    timbre_gate_0_2_verified: bool = False,
    demo_mode: bool = True,
    hospital_weights: HospitalScoringWeights | None = None,
    session_token_secret: str | None = None,
) -> tuple[
    SessionOrchestrator,
    HandoffGenerator,
    InMemoryAuditWriter,
    MetricsRegistry,
    InstructionLibrary,
]:
    """Assemble the SETU orchestrator with synthetic resources and directory records."""
    synthetic_resources = build_demo_fleet_resources()
    repo = InMemoryResourceRepository(synthetic_resources)
    dispatch = DispatchEngine(repo, demo_mode=demo_mode)
    location_resolver = LocationResolver()
    hospital_ranker = HospitalRanker(weights=hospital_weights)
    instruction_lib = InstructionLibrary(
        default_voice_by_lang=(
            {"en": "Pranav", "te": "Pranav", "hi": "Pranav"}
            if timbre_gate_0_2_verified
            else None
        ),
        timbre_version="v2.5",
        timbre_gate_0_2_verified=timbre_gate_0_2_verified,
    )
    policy = PolicyValidator(instruction_lib)
    duplicates = DuplicateDetector()
    audit = InMemoryAuditWriter()
    metrics = MetricsRegistry()
    secret = (
        session_token_secret
        or os.environ.get("SETU_SESSION_TOKEN_SECRET")
        or os.environ.get("SETU_SESSION_SIGNING_KEY")
        or "setu-dev-signing-secret-key-2026-india"
    )
    signer = SessionTokenSigner(secret_key=secret)
    orchestrator = SessionOrchestrator(
        reasoning_adapter=MockReasoningAdapter(),
        dispatch_engine=dispatch,
        location_resolver=location_resolver,
        hospital_ranker=hospital_ranker,
        instruction_library=instruction_lib,
        policy_validator=policy,
        duplicate_detector=duplicates,
        audit_writer=audit,
        metrics=metrics,
        token_signer=signer,
        demo_mode=demo_mode,
    )
    return orchestrator, HandoffGenerator(), audit, metrics, instruction_lib


def _collect_active_demo_assignments(orchestrator: SessionOrchestrator) -> list[Any]:
    """Collect active demo assignments across all non-completed sessions."""
    sessions = getattr(orchestrator, "_sessions", {}) or {}
    active: list[Any] = []
    for sess in sessions.values():
        if str(getattr(sess, "session_state", "")).lower() == "completed":
            continue
        active.extend(getattr(sess, "assignments", []) or [])
    return active


def _serialize_demo_fleet(
    orchestrator: SessionOrchestrator,
    active_assignments: list[Any],
) -> list[dict[str, Any]]:
    """Combine static depot fleet inventory with live incident assignment coordinates/statuses."""
    repo = getattr(orchestrator._dispatch, "_repository", None)
    if repo is None or not hasattr(repo, "list_all"):
        return []
    by_vid = {a.resource_id: a for a in active_assignments if a.resource_id}
    items: list[dict[str, Any]] = []
    for rec in repo.list_all():
        # A released/AVAILABLE vehicle must report depot status and coords cleanly
        asgn = None if rec.is_available else by_vid.get(rec.resource_id)
        rem_sec = getattr(asgn, "remaining_eta_seconds", None) if asgn is not None else None
        items.append(
            {
                "vehicle_id": rec.resource_id,
                "vehicle_code": rec.resource_id,
                "category": rec.category.value,
                "callsign": rec.callsign,
                "status": (
                    asgn.lifecycle_status
                    if asgn is not None
                    else ("AVAILABLE" if rec.is_available else "ASSIGNED")
                ),
                "latitude": (
                    asgn.current_latitude
                    if asgn is not None and asgn.current_latitude is not None
                    else rec.latitude
                ),
                "longitude": (
                    asgn.current_longitude
                    if asgn is not None and asgn.current_longitude is not None
                    else rec.longitude
                ),
                "simulated_eta_minutes": asgn.simulated_eta_minutes if asgn else None,
                "remaining_eta_minutes": asgn.remaining_eta_minutes if asgn else None,
                "remaining_eta_seconds": rem_sec,
                "estimated_arrival_at": (
                    asgn.estimated_arrival_at.isoformat()
                    if asgn is not None and getattr(asgn, "estimated_arrival_at", None) is not None
                    else None
                ),
                "is_available": rec.is_available,
                "is_simulated": True,
            }
        )
    return items


def serialize_session_snapshot(
    session_id: UUID,
    orchestrator: SessionOrchestrator,
    handoff_gen: HandoffGenerator,
    audit: InMemoryAuditWriter,
) -> dict[str, Any]:
    """Serialize authoritative session state and structured handoff for API / WebSocket clients."""
    state = orchestrator.get_session(session_id)
    now = datetime.now(timezone.utc)
    events = audit.list_events(incident_id=state.incident_id)
    handoff = handoff_gen.generate_handoff(
        state,
        audit_events=events,
        transcript_retention_expired=False,
        generated_at=now,
    )
    loc = state.location_state
    return {
        "session_id": str(state.session_id),
        "incident_id": str(state.incident_id),
        "session_token": state.session_token,
        "session_state": state.session_state,
        "incident_state": state.incident_state,
        "revision": state.revision,
        "last_event_sequence": state.last_event_sequence,
        "initial_language": state.initial_language,
        "active_language": state.active_language,
        "category": state.category.value,
        "category_established_above_threshold": state.category_established_above_threshold,
        "classification_attempts": state.classification_attempts,
        "consecutive_unknown_count": state.consecutive_unknown_count,
        "consecutive_provider_blocks": state.consecutive_provider_blocks,
        "total_provider_blocks": state.total_provider_blocks,
        "unknown_fallback_applied": state.unknown_fallback_applied,
        "fallback_reason": state.fallback_reason,
        "control_room_review_required": state.control_room_review_required,
        "review_reason_codes": list(state.review_reason_codes),
        "last_spoken_response_text": state.last_spoken_response_text,
        "last_selected_instruction_id": state.last_selected_instruction_id,
        "pending_followup_question": state.pending_followup_question,
        "pending_question": (
            {
                "question_id": state.pending_question_record.question_id,
                "question_text": state.pending_question_record.question_text,
                "target_field": state.pending_question_record.target_field,
                "status": state.pending_question_record.status,
                "asked_at": state.pending_question_record.asked_at.isoformat(),
            }
            if getattr(state, "pending_question_record", None) is not None
            else None
        ),
        "identified_hazards": list(getattr(state, "identified_hazards", []) or []),
        "active_safety_guidance": getattr(state, "active_safety_guidance", None),
        "task_execution_telemetry": getattr(state, "task_execution_telemetry", None) or {},
        "required_resource_categories": [c.value for c in state.required_resource_categories],
        "utterances": [
            {
                "utterance_id": str(u.utterance_id),
                "sequence_number": u.sequence_number,
                "text": u.text,
                "language": u.language,
                "utterance_role": u.utterance_role.value if u.utterance_role else None,
                "finalized_at": u.finalized_at.isoformat(),
            }
            for u in state.utterances
        ],
        "validated_facts": [
            {
                "field_name": f.field_name,
                "value": f.value,
                "source_utterance_id": str(f.source_utterance_id),
                "evidence_text": f.evidence_text,
            }
            for f in state.validated_facts
        ],
        "conflicts": [
            {
                "field_name": c.field_name,
                "prior_value": c.prior_value,
                "prior_utterance_id": str(c.prior_utterance_id),
                "new_value": c.new_value,
                "new_utterance_id": str(c.new_utterance_id),
            }
            for c in state.conflicts
        ],
        "active_conflicts_count": len(state.conflicts),
        "location": (
            {
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "accuracy_m": loc.accuracy_m,
                "match_confidence": loc.match_confidence,
                "source_type": loc.source,
                "confirmation_status": loc.confirmation_status.value,
                "display_label": loc.resolved_label,
                "broad_area_text": getattr(loc, "broad_area_text", None),
                "caller_landmark_text": getattr(loc, "caller_landmark_text", None),
                "caller_location_text": loc.caller_location_text,
                "confirmed_at": (
                    loc.confirmed_at.isoformat() if getattr(loc, "confirmed_at", None) else None
                ),
                "is_usable_candidate": loc.is_usable_candidate,
            }
            if loc is not None
            else None
        ),
        "assignments": [
            {
                "assignment_id": str(a.assignment_id),
                "incident_id": str(a.incident_id),
                "idempotency_key": a.idempotency_key,
                "resource_id": a.resource_id,
                "vehicle_code": a.resource_id,
                "resource_callsign": a.resource_callsign,
                "resource_category": a.category.value,
                "status": a.status.value,
                "lifecycle_status": a.lifecycle_status,
                "assigned_at": a.assigned_at.isoformat(),
                "en_route_at": a.en_route_at.isoformat() if a.en_route_at else None,
                "arrived_at": a.arrived_at.isoformat() if a.arrived_at else None,
                "estimated_arrival_at": (
                    a.estimated_arrival_at.isoformat()
                    if getattr(a, "estimated_arrival_at", None)
                    else None
                ),
                "simulated_eta_minutes": a.simulated_eta_minutes,
                "remaining_eta_minutes": a.remaining_eta_minutes,
                "remaining_eta_seconds": getattr(a, "remaining_eta_seconds", None),
                "current_latitude": a.current_latitude,
                "current_longitude": a.current_longitude,
                "depot_latitude": getattr(a, "depot_latitude", None),
                "depot_longitude": getattr(a, "depot_longitude", None),
                "target_latitude": a.target_latitude,
                "target_longitude": a.target_longitude,
                "allocation_mode": a.allocation_mode,
                "distance_meters": a.distance_meters,
                "provenance_source": f"{a.provenance.source_type.value}:{a.provenance.source_id}",
            }
            for a in state.assignments
        ],
        "demo_vehicles": _serialize_demo_fleet(orchestrator, state.assignments),
        "conversational_memory": {
            "turn_count": len(state.conversation_history),
            "cleared": state.conversational_memory_cleared,
            "question_history": [
                {
                    "question_id": q.question_id,
                    "question_text": q.question_text,
                    "target_field": q.target_field,
                    "status": q.status,
                    "asked_at": q.asked_at.isoformat(),
                    "answered_by_utterance_id": (
                        str(q.answered_by_utterance_id) if q.answered_by_utterance_id else None
                    ),
                }
                for q in getattr(state, "question_history", [])
            ],
            "superseded_facts": [
                {
                    "field_name": sf.field_name,
                    "prior_value": sf.prior_value,
                    "prior_utterance_id": str(sf.prior_utterance_id),
                    "new_value": sf.new_value,
                    "new_utterance_id": str(sf.new_utterance_id),
                    "description": sf.description,
                }
                for sf in getattr(state, "superseded_facts", [])
            ],
            "location_history": list(getattr(state, "location_history", []) or []),
        },
        "hospital_recommendations": [
            {
                "hospital_id": h.hospital.hospital_id,
                "name": h.hospital.name,
                "address": h.hospital.address,
                "distance_km": h.distance_km,
                "eta_minutes": h.eta_minutes,
                "eta_source_label": h.eta_source_label,
                "matched_capabilities": list(h.matched_capabilities),
                "distance_component_score": h.distance_component_score,
                "capability_component_score": h.capability_component_score,
                "applied_distance_weight": h.applied_distance_weight,
                "applied_capability_weight": h.applied_capability_weight,
                "weighted_distance_contribution": h.weighted_distance_contribution,
                "weighted_capability_contribution": h.weighted_capability_contribution,
                "final_score": h.final_score,
                "source_identifier": h.hospital.source_identifier,
                "verification_date": h.hospital.verification_date.isoformat(),
                "is_synthetic_directory": h.hospital.is_synthetic_directory,
            }
            for h in state.hospital_recommendations
        ],
        "handoff": asdict(handoff),
    }


class LiveSessionBridge:
    """
    Executes the authoritative /ws/session protocol in Python:
      browser mic -> AudioWorklet (16 kHz mono PCM) -> /ws/session -> FastAPI ->
      PrismaSTTAdapter -> SessionOrchestrator -> GeminiReasoningAdapter ->
      TimbreTTSAdapter / cached instruction audio -> browser AudioContext.
    """

    def __init__(self, *, use_live_providers: bool = False) -> None:
        self._use_live_providers = use_live_providers
        self.settings = load_settings(env_override=None if use_live_providers else {})
        manifest_ready = False
        self._manifest_asset_count = 0
        if use_live_providers:
            manifest_path = Path("backend-media/instructions/manifest.json")
            if manifest_path.exists():
                try:
                    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest_ready = bool(
                        isinstance(rows, list)
                        and rows
                        and all(r.get("generation_status") == "ready" for r in rows)
                    )
                    if isinstance(rows, list):
                        self._manifest_asset_count = len(rows)
                except Exception:
                    manifest_ready = False

        self.gate_0_1_prisma_verified = bool(
            use_live_providers
            and self.settings.prisma_endpoint_url
            and self.settings.prisma_auth_token
        )
        self.gate_0_2_timbre_verified = bool(
            use_live_providers
            and self.settings.timbre_endpoint_url
            and self.settings.timbre_auth_token
            and manifest_ready
        )
        self.gate_0_3_gemini_verified = bool(
            use_live_providers and self.settings.gate_0_3_gemini_verified
        )

        (
            self.orchestrator,
            self.handoff_gen,
            self.audit,
            self.metrics,
            self.instruction_lib,
        ) = build_default_orchestrator(
            timbre_gate_0_2_verified=self.gate_0_2_timbre_verified,
            hospital_weights=self.settings.hospital_weights if use_live_providers else None,
            session_token_secret=self.settings.session_token_secret if use_live_providers else None,
        )
        self.capability_profile = DEFAULT_DEPLOYMENT_CAPABILITY_PROFILE
        self.timeout_coordinator = CallerTimeoutCoordinator(self.settings.timeout_policy)
        self.prisma_adapter = GnaniPrismaAdapter(
            endpoint_url=(self.settings.prisma_endpoint_url or None)
            if use_live_providers
            else None,
            api_key=(self.settings.prisma_auth_token or None) if use_live_providers else None,
        )
        self.timbre_adapter = GnaniTimbreAdapter(
            endpoint_url=(self.settings.timbre_endpoint_url or None)
            if use_live_providers
            else None,
            api_key=(self.settings.timbre_auth_token or None) if use_live_providers else None,
        )
        gemini_transport = (
            build_gemini_rest_transport(
                api_key=self.settings.gemini_api_key,
                model_id=self.settings.gemini_model_id,
                fallback_api_keys=self.settings.gemini_fallback_api_keys,
            )
            if (use_live_providers and self.settings.gemini_api_key)
            else None
        )
        self.gemini_adapter = GeminiReasoningAdapter(
            model_id=self.settings.gemini_model_id,
            transport_callable=gemini_transport,
        )
        self.pg_store = NeonPostgresStore() if use_live_providers else None
        self._pg_retry_after: float = 0.0
        self._pg_write_lock = threading.Lock()
        self._pending_persist_futures: list[asyncio.Future[Any]] = []
        self._last_pg_error: str | None = None
        self._demo_fleet_seeded: bool = False
        self._active_session_id: UUID | None = None
        self._pcm_chunks_received: dict[str, int] = {}
        self._pcm_bytes_received: dict[str, int] = {}
        self._active_stt_streams: dict[str, Any] = {}
        self._last_gemini_status_message: str | None = None

    def _refresh_gemini_runtime_status(self, incident_id: UUID) -> None:
        if not self.gemini_adapter.is_configured:
            return
        events = self.audit.list_events(incident_id=incident_id)
        last_turn_idx = max(
            (i for i, e in enumerate(events) if e.event_type == "final_transcript_received"),
            default=-1,
        )
        if last_turn_idx < 0:
            return
        turn_events = events[last_turn_idx:]
        outage_event = next(
            (e for e in reversed(turn_events) if e.event_type == "reasoning_provider_outage"),
            None,
        )
        if outage_event is not None:
            code = outage_event.error_code or "unavailable"
            self.gate_0_3_gemini_verified = False
            self._last_gemini_status_message = (
                f"Configured ({self.settings.gemini_model_id} HTTP {code} upstream unavailable/quota exceeded; DEMO deterministic fallback active)"
            )
            return
        self.gate_0_3_gemini_verified = True
        self._last_gemini_status_message = f"Live and verified ({self.settings.gemini_model_id})"

    async def _run_utterance_with_reasoning(
        self,
        *,
        session_id: UUID,
        utterance_id: UUID,
        text: str,
        detected_language: SupportedLanguage,
    ) -> Any:
        prior_adapter = self.orchestrator._reasoning
        if self.gemini_adapter.is_configured:
            self.orchestrator._reasoning = self.gemini_adapter
        try:
            state = await self.orchestrator.process_finalized_utterance(
                session_id,
                utterance_id=utterance_id,
                text=text,
                detected_language=detected_language,
                timeout_seconds=6.5,
            )
            self._refresh_gemini_runtime_status(state.incident_id)
            return state
        finally:
            self.orchestrator._reasoning = prior_adapter

    def _get_provider_status(self) -> dict[str, Any]:
        pg_live = bool(self.pg_store is not None and self.pg_store.is_configured)
        return {
            "prisma": {
                "status": "live_and_verified" if self.gate_0_1_prisma_verified else "unconfigured",
                "model": "prisma-v2.5",
                "message": (
                    "Live and verified (Gnani Prisma v2.5 WSS 16 kHz mono PCM streaming)"
                    if self.gate_0_1_prisma_verified
                    else "Live STT endpoint not configured in this environment."
                ),
            },
            "timbre": {
                "status": "live_and_verified" if self.gate_0_2_timbre_verified else "placeholder",
                "model": "timbre-v2.5",
                "voice": "Pranav" if self.gate_0_2_timbre_verified else "placeholder",
                "message": (
                    f"Live and verified (Gnani Timbre v2.5 voice=Pranav; {self._manifest_asset_count} instruction WAV assets checksum-verified)"
                    if self.gate_0_2_timbre_verified
                    else "Development placeholder audio (pending Gate 0.2 verification)."
                ),
            },
            "gemini": {
                "status": (
                    "live_and_verified"
                    if self.gate_0_3_gemini_verified
                    else (
                        "configured_upstream_unavailable"
                        if self.settings.gemini_api_key
                        else "unconfigured"
                    )
                ),
                "model": self.settings.gemini_model_id,
                "message": (
                    f"Live and verified ({self.settings.gemini_model_id})"
                    if self.gate_0_3_gemini_verified
                    else (
                        self._last_gemini_status_message
                        or (
                            f"Configured ({self.settings.gemini_model_id} awaiting/unavailable; DEMO deterministic fallback active)"
                            if self.settings.gemini_api_key
                            else "GEMINI_API_KEY not configured; MockReasoningAdapter active"
                        )
                    )
                ),
            },
            "postgresql": {
                "status": "live_and_verified" if pg_live else "in_memory_fallback",
                "message": (
                    "Live and verified (Neon PostgreSQL + PostGIS + pg_trgm + append-only audit triggers)"
                    if pg_live
                    else "In-memory session store"
                ),
            },
        }

    def _persist_sync_safe(
        self,
        snap: dict[str, Any],
        events: list[Any],
        *,
        force: bool = False,
    ) -> None:
        if self.pg_store is None:
            return
        import time as _time
        with self._pg_write_lock:
            if not force and _time.monotonic() < self._pg_retry_after:
                return
            for attempt in range(2):
                try:
                    if not self._demo_fleet_seeded:
                        self.pg_store.ensure_demo_fleet_schema_and_seed()
                        self._demo_fleet_seeded = True
                    self.pg_store.persist_session_snapshot(snap, events)
                    self._last_pg_error = None
                    self._pg_retry_after = 0.0
                    return
                except Exception as exc:
                    err_str = str(exc)
                    is_dns = "getaddrinfo failed" in err_str or "Name or service not known" in err_str
                    if is_dns and attempt == 0:
                        _time.sleep(0.35)
                        continue
                    self._last_pg_error = err_str
                    print(f"[setu-pg-persist-error] {err_str}", file=sys.stderr, flush=True)
                    if is_dns and not force:
                        self._pg_retry_after = _time.monotonic() + 15.0
                    return

    async def flush_persistence(self) -> None:
        pending = [f for f in self._pending_persist_futures if not f.done()]
        self._pending_persist_futures = pending
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            self._pending_persist_futures = [f for f in self._pending_persist_futures if not f.done()]

    async def _persist_to_postgres_if_configured(
        self,
        session_id: UUID,
        snap: dict[str, Any],
        *,
        wait: bool = False,
    ) -> None:
        if self.pg_store is None or not self.pg_store.is_configured:
            return
        import time as _time
        if not wait and _time.monotonic() < self._pg_retry_after:
            return
        try:
            state = self.orchestrator.get_session(session_id)
            events = self.audit.list_events(incident_id=state.incident_id)
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(None, lambda: self._persist_sync_safe(snap, events, force=wait))
            self._pending_persist_futures = [f for f in self._pending_persist_futures if not f.done()]
            self._pending_persist_futures.append(fut)
            if wait:
                await self.flush_persistence()
        except Exception:
            pass

    async def _synthesize_timbre_payload(
        self,
        *,
        session_id: UUID,
        text: str,
        language: SupportedLanguage,
        timeout_seconds: float = 12.0,
        emit_event: Callable[[dict[str, Any]], None] | None = None,
        req_id: Any = None,
        turn_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not text or not self.timbre_adapter._endpoint_url or not self.timbre_adapter._api_key:
            return None
        try:
            import time as _time
            t0 = _time.perf_counter()
            synthesis_id = str(uuid4())
            handle = await self.timbre_adapter.start_synthesis(
                session_id=session_id,
                synthesis_id=UUID(synthesis_id),
                text=text,
                language=language,
                voice_id="Pranav",
                timeout_seconds=timeout_seconds,
            )
            pcm_parts: list[bytes] = []
            ttfab_ms: float | None = None
            chunk_idx = 0
            async for chunk in handle.iter_audio_chunks(
                first_byte_timeout_seconds=timeout_seconds,
                chunk_timeout_seconds=8.0,
            ):
                if ttfab_ms is None:
                    ttfab_ms = round((_time.perf_counter() - t0) * 1000.0, 1)
                    self.metrics.record_latency("timbre_live_ttfab_ms", ttfab_ms)
                chunk_idx += 1
                if emit_event is not None:
                    emit_event(
                        {
                            "type": "live_timbre_audio_chunk",
                            "stream_chunk": True,
                            "req_id": req_id,
                            "turn_id": turn_id,
                            "synthesis_id": synthesis_id,
                            "sequence_number": chunk_idx,
                            "is_first_chunk": chunk.is_first_byte,
                            "pcm_base64": base64.b64encode(chunk.audio_bytes).decode("ascii"),
                            "sample_rate_hz": 16000,
                            "voice_id": "Pranav",
                            "ttfab_ms": ttfab_ms,
                            "language": language,
                        }
                    )
                pcm_parts.append(chunk.audio_bytes)
            raw_pcm = b"".join(pcm_parts)
            if not raw_pcm:
                return None
            total_tts_ms = round((_time.perf_counter() - t0) * 1000.0, 1)
            self.metrics.record_latency("timbre_total_ms", total_tts_ms)
            return {
                "synthesis_id": synthesis_id,
                "pcm_base64": base64.b64encode(raw_pcm).decode("ascii"),
                "sample_rate_hz": 16000,
                "voice_id": "Pranav",
                "ttfab_ms": ttfab_ms or total_tts_ms,
                "total_tts_ms": total_tts_ms,
                "chunk_count": chunk_idx,
                "streamed": emit_event is not None,
                "language": language,
                "text": text,
            }
        except Exception:
            return None

    def _resolve_session_id(self, payload: dict[str, Any]) -> UUID:
        raw_sid = payload.get("session_id")
        if raw_sid:
            try:
                return UUID(str(raw_sid))
            except ValueError:
                pass
        if self._active_session_id is not None:
            return self._active_session_id
        state = self.orchestrator.create_session(initial_language="en")
        self._active_session_id = state.session_id
        self._pcm_chunks_received[str(state.session_id)] = 0
        self._pcm_bytes_received[str(state.session_id)] = 0
        return state.session_id

    def _resolve_instruction_audio_payload(
        self,
        instruction_id: str | None,
        language: SupportedLanguage,
    ) -> dict[str, Any] | None:
        if not instruction_id:
            return None
        try:
            inst, entry = self.instruction_lib.resolve_cached_asset(instruction_id, language)
            if entry is None:
                return None
            return {
                "instruction_id": entry.instruction_id,
                "language": entry.language,
                "text_version": entry.text_version,
                "file_path": f"/{entry.file_path}",
                "checksum_sha256": entry.checksum_sha256,
                "generation_status": entry.generation_status,
                "approved_text": inst.approved_text,
            }
        except KeyError:
            return None

    async def _finish_utterance_turn(
        self,
        *,
        t_turn0: float,
        turn_id: str,
        req_id: Any,
        session_id: UUID,
        utterance_id: UUID,
        text: str,
        detected_language: SupportedLanguage,
        prisma_stt_ms: float | None = None,
        vad_finalize_ms: float | None = None,
        emit_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[Any, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
        import time as _time
        before_samples = self.metrics.get_metric_summary("reasoning_processing_ms").sample_count
        t_r0 = _time.perf_counter()
        state = await self._run_utterance_with_reasoning(
            session_id=session_id,
            utterance_id=utterance_id,
            text=text,
            detected_language=detected_language,
        )
        total_process_ms = round((_time.perf_counter() - t_r0) * 1000.0, 1)
        reason_summary = self.metrics.get_metric_summary("reasoning_processing_ms")
        if (
            reason_summary.sample_count > before_samples
            and reason_summary.last_value_ms is not None
        ):
            gemini_reasoning_ms = round(float(reason_summary.last_value_ms), 1)
            orchestrator_ms = round(max(0.0, total_process_ms - gemini_reasoning_ms), 1)
        else:
            gemini_reasoning_ms = total_process_ms
            orchestrator_ms = 0.0
        self.metrics.record_latency("orchestrator_ms", orchestrator_ms)

        # Emit immediate dashboard state update BEFORE blocking on Timbre synthesis + Neon persistence
        pre_tts_snap = serialize_session_snapshot(
            state.session_id,
            self.orchestrator,
            self.handoff_gen,
            self.audit,
        )
        if emit_event is not None:
            emit_event(
                {
                    "type": "orchestration_state_updated",
                    "stream_chunk": True,
                    "req_id": req_id,
                    "turn_id": turn_id,
                    "session_id": str(state.session_id),
                    "event": "orchestration_state_updated",
                    "prisma_transcript": text,
                    "prisma_detected_language": detected_language,
                    "snapshot": pre_tts_snap,
                    "orchestrator_ms": orchestrator_ms,
                    "task_execution_telemetry": pre_tts_snap.get("task_execution_telemetry"),
                }
            )

        async def _timed_persist() -> float:
            t_p0 = _time.perf_counter()
            await self._persist_to_postgres_if_configured(state.session_id, pre_tts_snap)
            return round((_time.perf_counter() - t_p0) * 1000.0, 1)

        t_overlap0 = _time.perf_counter()
        overlap_results = await asyncio.gather(
            self._synthesize_timbre_payload(
                session_id=state.session_id,
                text=state.last_spoken_response_text,
                language=state.active_language,
                timeout_seconds=12.0,
                emit_event=emit_event,
                req_id=req_id,
                turn_id=turn_id,
            ),
            _timed_persist(),
            return_exceptions=True,
        )
        tts_and_persist_wall_ms = round((_time.perf_counter() - t_overlap0) * 1000.0, 1)
        live_timbre_payload = (
            overlap_results[0]
            if not isinstance(overlap_results[0], BaseException)
            else None
        )
        postgres_persist_ms = (
            overlap_results[1]
            if not isinstance(overlap_results[1], BaseException)
            else None
        )

        audio_asset = self._resolve_instruction_audio_payload(
            state.last_selected_instruction_id,
            state.active_language,
        )
        timbre_ttfb_ms = live_timbre_payload.get("ttfab_ms") if live_timbre_payload else None
        timbre_total_ms = live_timbre_payload.get("total_tts_ms") if live_timbre_payload else None
        if isinstance(state.task_execution_telemetry, dict):
            state.task_execution_telemetry["post_orchestration_overlap"] = {
                "timbre_tts_ms": timbre_total_ms,
                "postgres_persist_ms": postgres_persist_ms,
                "parallel_wall_ms": tts_and_persist_wall_ms,
            }
        snap = dict(pre_tts_snap)
        snap["task_execution_telemetry"] = state.task_execution_telemetry
        total_backend_ms = round((_time.perf_counter() - t_turn0) * 1000.0, 1)
        self.metrics.record_latency("e2e_conversational_response_ms", total_backend_ms)
        stage_latencies = {
            "turn_id": turn_id,
            "vad_finalize_ms": round(float(vad_finalize_ms), 1) if vad_finalize_ms is not None else None,
            "prisma_stt_ms": prisma_stt_ms,
            "gemini_reasoning_ms": gemini_reasoning_ms,
            "orchestrator_ms": orchestrator_ms,
            "postgres_persist_ms": postgres_persist_ms,
            "tts_and_persist_wall_ms": tts_and_persist_wall_ms,
            "timbre_ttfb_ms": timbre_ttfb_ms,
            "timbre_total_ms": timbre_total_ms,
            "total_backend_ms": total_backend_ms,
            "task_execution_telemetry": state.task_execution_telemetry,
            "measured_at": datetime.now(timezone.utc).isoformat(),
        }
        return state, snap, live_timbre_payload, audio_asset, stage_latencies

    async def handle_ws_message(
        self,
        payload: dict[str, Any],
        *,
        emit_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        msg_type = payload.get("type")
        req_id = payload.get("req_id")

        try:
            if msg_type == "session_init":
                repo = getattr(self.orchestrator._dispatch, "_repository", None)
                if payload.get("reset_demo_fleet") and repo is not None and hasattr(repo, "reset_all_available"):
                    repo.reset_all_available()
                elif (
                    payload.get("fresh_scenario")
                    and self._active_session_id is not None
                    and not payload.get("preserve_other_incidents")
                ):
                    try:
                        prev_state = self.orchestrator.get_session(self._active_session_id)
                        if prev_state.session_state != "completed":
                            ended_prev = self.orchestrator.end_session(
                                self._active_session_id,
                                reason="replaced_by_new_session",
                            )
                            prev_snap = serialize_session_snapshot(
                                ended_prev.session_id,
                                self.orchestrator,
                                self.handoff_gen,
                                self.audit,
                            )
                            await self._persist_to_postgres_if_configured(ended_prev.session_id, prev_snap)
                    except KeyError:
                        pass
                lang_raw = payload.get("initial_language", "en")
                lang: SupportedLanguage = lang_raw if lang_raw in ("en", "te", "hi") else "en"
                state = self.orchestrator.create_session(initial_language=lang)
                self._active_session_id = state.session_id
                sid_str = str(state.session_id)
                self._pcm_chunks_received[sid_str] = 0
                self._pcm_bytes_received[sid_str] = 0

                # Default initial GPS candidate if supplied
                if payload.get("latitude") is not None and payload.get("longitude") is not None:
                    self.orchestrator.update_gps_location(
                        state.session_id,
                        latitude=float(payload["latitude"]),
                        longitude=float(payload["longitude"]),
                        accuracy_m=float(payload.get("accuracy_m", 42.0)),
                        permission_status=GPSPermissionStatus.GRANTED,
                        confirmed_by_caller=False,
                    )

                snap = serialize_session_snapshot(
                    state.session_id,
                    self.orchestrator,
                    self.handoff_gen,
                    self.audit,
                )
                await self._persist_to_postgres_if_configured(state.session_id, snap)
                audio_asset = self._resolve_instruction_audio_payload(
                    state.last_selected_instruction_id,
                    state.active_language,
                )
                return {
                    "type": "session_state",
                    "req_id": req_id,
                    "event": "session_created",
                    "snapshot": snap,
                    "cached_instruction_audio": audio_asset,
                    "gate_status": {
                        "gate_0_1_prisma_verified": self.gate_0_1_prisma_verified,
                        "gate_0_2_timbre_verified": self.gate_0_2_timbre_verified,
                        "gate_0_3_gemini_verified": self.gate_0_3_gemini_verified,
                    },
                    "provider_status": self._get_provider_status(),
                }

            if msg_type == "session_reconnect":
                raw_sid = payload.get("session_id")
                if not raw_sid:
                    return {
                        "type": "error",
                        "req_id": req_id,
                        "reason_code": "session_id_required",
                        "message": "session_id is required to reconnect.",
                    }
                try:
                    req_sid = UUID(str(raw_sid))
                except ValueError:
                    return {
                        "type": "error",
                        "req_id": req_id,
                        "reason_code": "invalid_session_id",
                        "message": "Invalid session_id format.",
                    }
                if req_sid not in getattr(self.orchestrator, "_sessions", {}):
                    return {
                        "type": "error",
                        "req_id": req_id,
                        "reason_code": "session_not_found_in_memory",
                        "message": "Previous session is no longer active in memory; start a fresh live session.",
                    }
                session_id = req_sid
                token = str(payload["session_token"])
                last_seq = int(payload.get("last_received_sequence", 0))
                state = self.orchestrator.reconnect_session(
                    session_id,
                    session_token=token,
                    last_received_sequence=last_seq,
                )
                self._active_session_id = state.session_id
                snap = serialize_session_snapshot(
                    state.session_id,
                    self.orchestrator,
                    self.handoff_gen,
                    self.audit,
                )
                await self._persist_to_postgres_if_configured(state.session_id, snap)
                return {
                    "type": "session_state",
                    "req_id": req_id,
                    "event": "session_reconnected",
                    "snapshot": snap,
                    "gate_status": {
                        "gate_0_1_prisma_verified": self.gate_0_1_prisma_verified,
                        "gate_0_2_timbre_verified": self.gate_0_2_timbre_verified,
                        "gate_0_3_gemini_verified": self.gate_0_3_gemini_verified,
                    },
                    "provider_status": self._get_provider_status(),
                }

            if msg_type == "audio_chunk_pcm16":
                session_id = self._resolve_session_id(payload)
                sid_str = str(session_id)
                pcm_b64 = str(payload.get("pcm_base64", ""))
                pcm_bytes = base64.b64decode(pcm_b64) if pcm_b64 else b""
                seq_num = int(payload.get("sequence_number", 0))
                is_finalize = bool(payload.get("finalize_utterance"))
                is_discard = bool(payload.get("discard_utterance"))

                if pcm_bytes:
                    self._pcm_chunks_received[sid_str] = self._pcm_chunks_received.get(sid_str, 0) + 1
                    self._pcm_bytes_received[sid_str] = self._pcm_bytes_received.get(sid_str, 0) + len(pcm_bytes)

                state = self.orchestrator.get_session(session_id)
                if state.session_state == "completed":
                    if self._active_session_id is not None and self._active_session_id != session_id:
                        session_id = self._active_session_id
                        sid_str = str(session_id)
                        state = self.orchestrator.get_session(session_id)
                    else:
                        return {
                            "type": "session_state",
                            "req_id": req_id,
                            "event": "prisma_non_speech_or_silence",
                            "snapshot": serialize_session_snapshot(
                                state.session_id,
                                self.orchestrator,
                                self.handoff_gen,
                                self.audit,
                            ),
                            "prisma_transcript": None,
                        }
                try:
                    stream = self._active_stt_streams.get(sid_str)
                    if is_discard:
                        self._active_stt_streams.pop(sid_str, None)
                        if stream is not None:
                            try:
                                await stream.close(timeout_seconds=0.5)
                            except Exception:
                                pass
                        return {
                            "type": "session_state",
                            "req_id": req_id,
                            "event": "prisma_non_speech_or_silence",
                            "snapshot": serialize_session_snapshot(
                                state.session_id,
                                self.orchestrator,
                                self.handoff_gen,
                                self.audit,
                            ),
                            "prisma_transcript": None,
                        }
                    if is_finalize and not pcm_bytes and (
                        stream is None
                        or not getattr(stream, "is_active", False)
                        or getattr(stream, "buffered_chunk_count", 0) == 0
                    ):
                        self._active_stt_streams.pop(sid_str, None)
                        return {
                            "type": "session_state",
                            "req_id": req_id,
                            "event": "prisma_non_speech_or_silence",
                            "snapshot": serialize_session_snapshot(
                                state.session_id,
                                self.orchestrator,
                                self.handoff_gen,
                                self.audit,
                            ),
                            "prisma_transcript": None,
                        }

                    if stream is None or not getattr(stream, "is_active", False):
                        stream = await self.prisma_adapter.open_stream(
                            session_id=session_id,
                            utterance_id=uuid4(),
                            language_hint=state.active_language,
                            timeout_seconds=2.0,
                        )
                        self._active_stt_streams[sid_str] = stream

                    if pcm_bytes:
                        chunk = AudioChunk(
                            pcm_bytes=pcm_bytes,
                            sample_rate_hz=int(payload.get("sample_rate_hz", 16000)),
                            sequence_number=seq_num,
                            captured_at=datetime.now(timezone.utc),
                        )
                        await stream.send_audio_chunk(chunk, timeout_seconds=1.0)

                    if is_finalize:
                        import time as _time
                        t_turn0 = _time.perf_counter()
                        vad_finalize_ms_raw = payload.get("vad_finalize_ms")
                        vad_finalize_ms = (
                            float(vad_finalize_ms_raw)
                            if vad_finalize_ms_raw is not None
                            else None
                        )
                        if vad_finalize_ms is not None and vad_finalize_ms >= 0.0:
                            self.metrics.record_latency("vad_finalize_ms", vad_finalize_ms)

                        self._active_stt_streams.pop(sid_str, None)
                        t_stt0 = _time.perf_counter()
                        evt = await stream.finalize_utterance(timeout_seconds=12.0)
                        prisma_stt_ms = round((_time.perf_counter() - t_stt0) * 1000.0, 1)
                        self.metrics.record_latency("prisma_final_transcript_ms", prisma_stt_ms)
                        await stream.close(timeout_seconds=1.0)
                        if evt.signal_type == STTSignalType.FINAL_TRANSCRIPT and evt.text:
                            # Guard against speakerphone acoustic self-echo of the assistant's own spoken output
                            raw_words = [
                                w.strip(".,!?;:\"'()")
                                for w in evt.text.lower().split()
                                if w.strip(".,!?;:\"'()")
                            ]
                            assistant_ref = (state.last_spoken_response_text or "").lower()
                            if len(raw_words) >= 4 and assistant_ref:
                                matched_words = sum(1 for w in raw_words if w in assistant_ref)
                                if (matched_words / len(raw_words)) >= 0.80:
                                    return {
                                        "type": "session_state",
                                        "req_id": req_id,
                                        "event": "prisma_non_speech_or_silence",
                                        "snapshot": serialize_session_snapshot(
                                            state.session_id,
                                            self.orchestrator,
                                            self.handoff_gen,
                                            self.audit,
                                        ),
                                        "prisma_transcript": None,
                                    }
                            turn_id = str(evt.utterance_id)
                            if emit_event is not None:
                                emit_event(
                                    {
                                        "type": "prisma_transcript_ready",
                                        "stream_chunk": True,
                                        "req_id": req_id,
                                        "turn_id": turn_id,
                                        "transcript": evt.text,
                                        "detected_language": evt.detected_language or state.active_language,
                                        "prisma_stt_ms": prisma_stt_ms,
                                    }
                                )
                            state, snap, live_timbre_payload, audio_asset, stage_latencies = (
                                await self._finish_utterance_turn(
                                    t_turn0=t_turn0,
                                    turn_id=turn_id,
                                    req_id=req_id,
                                    session_id=session_id,
                                    utterance_id=evt.utterance_id,
                                    text=evt.text,
                                    detected_language=evt.detected_language or state.active_language,
                                    prisma_stt_ms=prisma_stt_ms,
                                    vad_finalize_ms=vad_finalize_ms,
                                    emit_event=emit_event,
                                )
                            )
                            return {
                                "type": "session_state",
                                "req_id": req_id,
                                "turn_id": turn_id,
                                "event": "prisma_finalized_utterance_processed",
                                "snapshot": snap,
                                "prisma_transcript": evt.text,
                                "prisma_detected_language": evt.detected_language,
                                "live_timbre_audio": live_timbre_payload,
                                "cached_instruction_audio": audio_asset,
                                "stage_latencies": stage_latencies,
                                "gate_status": {
                                    "gate_0_1_prisma_verified": self.gate_0_1_prisma_verified,
                                    "gate_0_2_timbre_verified": self.gate_0_2_timbre_verified,
                                    "gate_0_3_gemini_verified": self.gate_0_3_gemini_verified,
                                },
                                "provider_status": self._get_provider_status(),
                            }
                        return {
                            "type": "session_state",
                            "req_id": req_id,
                            "event": "prisma_non_speech_or_silence",
                            "snapshot": serialize_session_snapshot(
                                state.session_id,
                                self.orchestrator,
                                self.handoff_gen,
                                self.audit,
                            ),
                            "prisma_transcript": None,
                        }
                    return {
                        "type": "stt_chunk_accepted",
                        "req_id": req_id,
                        "session_id": sid_str,
                        "pcm_chunks_ingested": self._pcm_chunks_received[sid_str],
                        "pcm_bytes_ingested": self._pcm_bytes_received[sid_str],
                        "prisma_status": "Live Prisma v2.5 WSS stream active (buffering 16 kHz mono PCM)",
                    }
                except STTProviderError as exc:
                    return {
                        "type": "stt_gate_blocked",
                        "req_id": req_id,
                        "session_id": sid_str,
                        "reason_code": exc.reason_code,
                        "message": str(exc),
                        "pcm_chunks_ingested": self._pcm_chunks_received[sid_str],
                        "pcm_bytes_ingested": self._pcm_bytes_received[sid_str],
                    }

            if msg_type == "gps_update":
                session_id = self._resolve_session_id(payload)
                acc = payload.get("accuracy_m")
                perm_str = payload.get("permission_status", "granted")
                perm = (
                    GPSPermissionStatus.GRANTED
                    if perm_str == "granted"
                    else GPSPermissionStatus.DENIED
                )
                state = self.orchestrator.update_gps_location(
                    session_id,
                    latitude=float(payload.get("latitude", 17.4435)),
                    longitude=float(payload.get("longitude", 78.3772)),
                    accuracy_m=float(acc) if acc is not None else None,
                    permission_status=perm,
                    confirmed_by_caller=bool(payload.get("confirmed_by_caller", False)),
                )
                snap = serialize_session_snapshot(
                    state.session_id,
                    self.orchestrator,
                    self.handoff_gen,
                    self.audit,
                )
                return {
                    "type": "session_state",
                    "req_id": req_id,
                    "event": "gps_updated",
                    "snapshot": snap,
                }

            if msg_type == "location_confirm":
                import time as _time
                t_conf0 = _time.perf_counter()
                session_id = self._resolve_session_id(payload)
                turn_id = f"loc-confirm-{uuid4()}"
                state = self.orchestrator.confirm_current_location(session_id)
                pre_tts_snap = serialize_session_snapshot(
                    state.session_id,
                    self.orchestrator,
                    self.handoff_gen,
                    self.audit,
                )
                if emit_event is not None:
                    emit_event(
                        {
                            "type": "orchestration_state_updated",
                            "stream_chunk": True,
                            "req_id": req_id,
                            "turn_id": turn_id,
                            "session_id": str(state.session_id),
                            "event": "orchestration_state_updated",
                            "snapshot": pre_tts_snap,
                            "task_execution_telemetry": pre_tts_snap.get("task_execution_telemetry"),
                        }
                    )
                t_ov0 = _time.perf_counter()
                overlap_res = await asyncio.gather(
                    self._synthesize_timbre_payload(
                        session_id=state.session_id,
                        text=state.last_spoken_response_text,
                        language=state.active_language,
                        timeout_seconds=12.0,
                        emit_event=emit_event,
                        req_id=req_id,
                        turn_id=turn_id,
                    ),
                    self._persist_to_postgres_if_configured(state.session_id, pre_tts_snap),
                    return_exceptions=True,
                )
                live_timbre_payload = (
                    overlap_res[0]
                    if not isinstance(overlap_res[0], BaseException)
                    else None
                )
                snap = pre_tts_snap
                audio_asset = self._resolve_instruction_audio_payload(
                    state.last_selected_instruction_id,
                    state.active_language,
                )
                _ = (_time.perf_counter() - t_conf0, _time.perf_counter() - t_ov0)
                return {
                    "type": "session_state",
                    "req_id": req_id,
                    "turn_id": turn_id,
                    "event": "location_confirmed",
                    "snapshot": snap,
                    "live_timbre_audio": live_timbre_payload,
                    "cached_instruction_audio": audio_asset,
                    "gate_status": {
                        "gate_0_1_prisma_verified": self.gate_0_1_prisma_verified,
                        "gate_0_2_timbre_verified": self.gate_0_2_timbre_verified,
                        "gate_0_3_gemini_verified": self.gate_0_3_gemini_verified,
                    },
                    "provider_status": self._get_provider_status(),
                }

            if msg_type == "location_invalidate":
                from dataclasses import replace as _dc_replace
                from backend.app.orchestrator.location import LocationConfirmationStatus
                session_id = self._resolve_session_id(payload)
                state = self.orchestrator.get_session(session_id)
                if state.location_state is not None:
                    state.location_state = _dc_replace(
                        state.location_state,
                        confirmation_status=LocationConfirmationStatus.CANDIDATE_AWAITING_CONFIRMATION,
                        confirmed_at=None,
                    )
                prompt_by_lang = {
                    "te": "సరే, దయచేసి మీ ఖచ్చితమైన ప్రదేశం లేదా సమీప ల్యాండ్‌మార్క్ చెప్పండి.",
                    "hi": "ठीक है, कृपया अपना सटीक स्थान या पास का कोई लैंडमार्क बताइए।",
                }
                state.last_spoken_response_text = prompt_by_lang.get(
                    state.active_language,
                    "Understood. Please tell me the exact location or nearest landmark so I can update it.",
                )
                state.pending_followup_question = state.last_spoken_response_text
                snap = serialize_session_snapshot(
                    state.session_id,
                    self.orchestrator,
                    self.handoff_gen,
                    self.audit,
                )
                await self._persist_to_postgres_if_configured(state.session_id, snap)
                return {
                    "type": "session_state",
                    "req_id": req_id,
                    "event": "location_invalidated",
                    "snapshot": snap,
                }

            if msg_type == "evaluate_caller_silence":
                session_id = self._resolve_session_id(payload)
                idle_sec = float(payload.get("idle_seconds", 16.0))
                check_ins_sent = int(payload.get("check_ins_sent", 0))
                eval_res = self.timeout_coordinator.evaluate_silence(
                    idle_seconds=idle_sec,
                    check_ins_sent=check_ins_sent,
                )
                state = self.orchestrator.get_session(session_id)
                if eval_res.flag_control_room:
                    state.control_room_review_required = True
                    if "caller_silence_timeout_review" not in state.review_reason_codes:
                        state.review_reason_codes.append("caller_silence_timeout_review")
                if eval_res.check_in_message_en:
                    state.last_spoken_response_text = eval_res.check_in_message_en
                snap = serialize_session_snapshot(
                    state.session_id,
                    self.orchestrator,
                    self.handoff_gen,
                    self.audit,
                )
                return {
                    "type": "session_state",
                    "req_id": req_id,
                    "event": f"silence_evaluated:{eval_res.action}",
                    "timeout_evaluation": {
                        "action": eval_res.action,
                        "flag_control_room": eval_res.flag_control_room,
                        "session_exhausted": eval_res.session_exhausted,
                        "check_in_message_en": eval_res.check_in_message_en,
                    },
                    "snapshot": snap,
                }

            if msg_type == "barge_in":
                session_id = self._resolve_session_id(payload)
                barge_mode = str(payload.get("barge_in_mode", "genuine_speech"))
                preserve_stream = bool(payload.get("preserve_active_stream", False))
                state = self.orchestrator.get_session(session_id)
                if (
                    not preserve_stream
                    and barge_mode not in {"empty_pre_gate01", "verified_false_trigger"}
                ):
                    # Drop any stale un-finalized non-speech stream when explicitly requested
                    self._active_stt_streams.pop(str(session_id), None)
                self.orchestrator.handle_barge_in(
                    session_id,
                    interrupted_Source=(
                        "cached_instruction"
                        if bool(payload.get("was_playing_cached_instruction", True))
                        else "live_timbre"
                    ),
                    interrupted_instruction_id=state.last_selected_instruction_id,
                    preserved_preroll_ms=int(payload.get("preserved_preroll_ms", 1500)),
                    transcript_text=(
                        ""
                        if barge_mode == "empty_pre_gate01"
                        else str(payload.get("transcript_text", "caller_speech_detected"))
                    ),
                    is_verified_non_speech_signal=(barge_mode == "verified_false_trigger"),
                )
                snap = serialize_session_snapshot(
                    state.session_id,
                    self.orchestrator,
                    self.handoff_gen,
                    self.audit,
                )
                audio_asset = self._resolve_instruction_audio_payload(
                    state.last_selected_instruction_id,
                    state.active_language,
                )
                return {
                    "type": "session_state",
                    "req_id": req_id,
                    "event": f"barge_in:{barge_mode}",
                    "snapshot": snap,
                    "cached_instruction_audio": audio_asset,
                }

            if msg_type == "scripted_finalized_utterance":
                # Executes the authoritative Python SessionOrchestrator pipeline for a scripted test turn
                session_id = self._resolve_session_id(payload)
                utt_id = UUID(str(payload.get("utterance_id") or uuid4()))
                text = str(payload["text"])
                lang_in = payload.get("language", "en")
                det_lang: SupportedLanguage = lang_in if lang_in in ("en", "te", "hi") else "en"

                sim_block = bool(payload.get("simulate_provider_block", False))
                sim_outage = bool(payload.get("simulate_provider_outage", False))

                if sim_block:
                    self.orchestrator._reasoning.enqueue_exception(
                        ProviderBlockedContentError(
                            "Provider blocked violence/emergency description under safety filter",
                            session_id=session_id,
                            utterance_id=utt_id,
                            provider_finish_reason="SAFETY",
                        )
                    )
                elif sim_outage:
                    self.orchestrator._reasoning.enqueue_exception(
                        ReasoningUnavailableError(
                            "Reasoning provider HTTP 503 unavailable",
                            session_id=session_id,
                            utterance_id=utt_id,
                            is_retryable=False,
                            http_status=503,
                        )
                    )
                else:
                    cat_str = str(payload.get("category", "unknown"))
                    cat = EmergencyCategory(cat_str)
                    role_str = str(payload.get("utterance_role", "new_description"))
                    role = UtteranceRole(role_str)
                    conf = float(payload.get("confidence", 0.92))
                    raw_facts = payload.get("extracted_facts", [])
                    facts = [
                        ProvenancedFact(
                            field_name=str(f.get("field") or f.get("field_name") or ""),
                            value=f["value"],
                            source_utterance_id=utt_id,
                            evidence_text=str(f.get("evidence_text") or f.get("value") or ""),
                        )
                        for f in raw_facts
                    ]
                    self.orchestrator._reasoning.enqueue_output(
                        ReasoningOutput(
                            utterance_role=role,
                            utterance_role_source_id=utt_id,
                            category=cat,
                            category_confidence=conf,
                            supporting_utterance_ids=[utt_id],
                            extracted_facts=facts,
                            conflicts=[],
                            missing_information_fields=[],
                            proposed_followup_question=payload.get("followup_question"),
                            selected_instruction_id=payload.get("selected_instruction_id"),
                            caller_response_text=str(
                                payload.get("response_text")
                                or "In this demo, your statement has been processed."
                            ),
                            prompt_tokens=120,
                            completion_tokens=48,
                            latency_ms=145.0,
                        )
                    )

                import time as _time
                t_script0 = _time.perf_counter()
                state = await self.orchestrator.process_finalized_utterance(
                    session_id,
                    utterance_id=utt_id,
                    text=text,
                    detected_language=det_lang,
                )
                snap = serialize_session_snapshot(
                    state.session_id,
                    self.orchestrator,
                    self.handoff_gen,
                    self.audit,
                )
                turn_id = str(utt_id)
                if emit_event is not None:
                    emit_event(
                        {
                            "type": "orchestration_state_updated",
                            "stream_chunk": True,
                            "req_id": req_id,
                            "turn_id": turn_id,
                            "session_id": str(state.session_id),
                            "event": "orchestration_state_updated",
                            "prisma_transcript": text,
                            "prisma_detected_language": det_lang,
                            "snapshot": snap,
                            "task_execution_telemetry": snap.get("task_execution_telemetry"),
                        }
                    )

                async def _timed_script_persist() -> float:
                    t_p0 = _time.perf_counter()
                    await self._persist_to_postgres_if_configured(state.session_id, snap)
                    return round((_time.perf_counter() - t_p0) * 1000.0, 1)

                t_ov0 = _time.perf_counter()
                overlap_results = await asyncio.gather(
                    self._synthesize_timbre_payload(
                        session_id=state.session_id,
                        text=state.last_spoken_response_text,
                        language=state.active_language,
                        timeout_seconds=12.0,
                        emit_event=emit_event,
                        req_id=req_id,
                        turn_id=turn_id,
                    ),
                    _timed_script_persist(),
                    return_exceptions=True,
                )
                tts_and_persist_wall_ms = round((_time.perf_counter() - t_ov0) * 1000.0, 1)
                live_timbre_payload = (
                    overlap_results[0]
                    if not isinstance(overlap_results[0], BaseException)
                    else None
                )
                postgres_persist_ms = (
                    overlap_results[1]
                    if not isinstance(overlap_results[1], BaseException)
                    else None
                )
                timbre_total_ms = (
                    live_timbre_payload.get("total_tts_ms") if live_timbre_payload else None
                )
                if isinstance(state.task_execution_telemetry, dict):
                    state.task_execution_telemetry["post_orchestration_overlap"] = {
                        "timbre_tts_ms": timbre_total_ms,
                        "postgres_persist_ms": postgres_persist_ms,
                        "parallel_wall_ms": tts_and_persist_wall_ms,
                    }
                snap["task_execution_telemetry"] = state.task_execution_telemetry
                audio_asset = self._resolve_instruction_audio_payload(
                    state.last_selected_instruction_id,
                    state.active_language,
                )
                total_ms = round((_time.perf_counter() - t_script0) * 1000.0, 1)
                return {
                    "type": "session_state",
                    "req_id": req_id,
                    "turn_id": turn_id,
                    "event": "utterance_processed",
                    "snapshot": snap,
                    "live_timbre_audio": live_timbre_payload,
                    "cached_instruction_audio": audio_asset,
                    "stage_latencies": {
                        "turn_id": turn_id,
                        "postgres_persist_ms": postgres_persist_ms,
                        "tts_and_persist_wall_ms": tts_and_persist_wall_ms,
                        "timbre_total_ms": timbre_total_ms,
                        "total_backend_ms": total_ms,
                        "task_execution_telemetry": state.task_execution_telemetry,
                    },
                }

            if msg_type == "live_gemini_utterance":
                import time as _time
                t_turn0 = _time.perf_counter()
                session_id = self._resolve_session_id(payload)
                utt_id = UUID(str(payload.get("utterance_id") or uuid4()))
                turn_id = str(utt_id)
                text = str(payload["text"])
                lang_in = payload.get("language", "en")
                det_lang: SupportedLanguage = lang_in if lang_in in ("en", "te", "hi") else "en"

                state, snap, live_timbre_payload, audio_asset, stage_latencies = (
                    await self._finish_utterance_turn(
                        t_turn0=t_turn0,
                        turn_id=turn_id,
                        req_id=req_id,
                        session_id=session_id,
                        utterance_id=utt_id,
                        text=text,
                        detected_language=det_lang,
                        prisma_stt_ms=None,
                        vad_finalize_ms=None,
                        emit_event=emit_event,
                    )
                )
                return {
                    "type": "session_state",
                    "req_id": req_id,
                    "turn_id": turn_id,
                    "event": "live_gemini_utterance_processed",
                    "snapshot": snap,
                    "live_timbre_audio": live_timbre_payload,
                    "cached_instruction_audio": audio_asset,
                    "stage_latencies": stage_latencies,
                    "gate_status": {
                        "gate_0_1_prisma_verified": self.gate_0_1_prisma_verified,
                        "gate_0_2_timbre_verified": self.gate_0_2_timbre_verified,
                        "gate_0_3_gemini_verified": self.gate_0_3_gemini_verified,
                    },
                    "provider_status": self._get_provider_status(),
                }

            if msg_type == "synthesize_live_timbre":
                session_id = self._resolve_session_id(payload)
                text = str(payload.get("text") or "In this simulation, SETU is staying on the line with you.")
                lang_in = payload.get("language", "en")
                lang: SupportedLanguage = lang_in if lang_in in ("en", "te", "hi") else "en"
                timbre_payload = await self._synthesize_timbre_payload(
                    session_id=session_id,
                    text=text,
                    language=lang,
                    timeout_seconds=12.0,
                )
                if timbre_payload is None:
                    raise TTSProviderError("Live Timbre synthesis returned no audio chunks or is unconfigured.")
                return {
                    "type": "live_timbre_audio",
                    "req_id": req_id,
                    "language": lang,
                    "voice_id": "Pranav",
                    "ttfab_ms": timbre_payload["ttfab_ms"],
                    "pcm_base64": timbre_payload["pcm_base64"],
                    "sample_rate_hz": timbre_payload["sample_rate_hz"],
                }

            if msg_type == "demo_lifecycle_tick":
                session_id = self._resolve_session_id(payload)
                state = self.orchestrator.get_session(session_id)
                now_utc = datetime.now(timezone.utc)
                force_stage = payload.get("force_stage")
                changed_total: list[Any] = []
                spoken_update: str | None = None

                if state.session_state == "completed":
                    snap = serialize_session_snapshot(
                        state.session_id,
                        self.orchestrator,
                        self.handoff_gen,
                        self.audit,
                    )
                    return {
                        "type": "session_state",
                        "req_id": req_id,
                        "event": "demo_lifecycle_unchanged",
                        "changed_count": 0,
                        "spoken_announcement_text": None,
                        "live_timbre_audio": None,
                        "snapshot": snap,
                    }

                status_transitioned = False
                if force_stage == "EN_ROUTE":
                    state, chg, spk = self.orchestrator.advance_demo_lifecycle(
                        session_id, target_status=DemoLifecycleStatus.EN_ROUTE, now=now_utc
                    )
                    changed_total.extend(chg)
                    spoken_update = spk or spoken_update
                    status_transitioned = bool(chg)
                elif force_stage == "ARRIVED":
                    state, chg, spk = self.orchestrator.advance_demo_lifecycle(
                        session_id, target_status=DemoLifecycleStatus.ARRIVED, now=now_utc
                    )
                    changed_total.extend(chg)
                    spoken_update = spk or spoken_update
                    status_transitioned = bool(chg)
                else:
                    # Continuous EN_ROUTE coordinate & ETA interpolation, and EN_ROUTE -> ARRIVED upon deadline
                    en_route_ids = {
                        a.assignment_id
                        for a in state.assignments
                        if a.lifecycle_status == DemoLifecycleStatus.ASSIGNED.value
                        and (now_utc - a.assigned_at).total_seconds() >= 1.5
                    }
                    arrived_ids: set[UUID] = set()
                    still_en_route_ids: set[UUID] = set()
                    for a in state.assignments:
                        if a.lifecycle_status == DemoLifecycleStatus.EN_ROUTE.value:
                            deadline = getattr(a, "estimated_arrival_at", None) or (
                                (a.en_route_at + timedelta(seconds=12.0))
                                if a.en_route_at is not None
                                else None
                            )
                            if deadline is not None and now_utc >= deadline:
                                arrived_ids.add(a.assignment_id)
                            else:
                                still_en_route_ids.add(a.assignment_id)

                    if en_route_ids or still_en_route_ids:
                        state, chg, _ = self.orchestrator.advance_demo_lifecycle(
                            session_id,
                            target_status=DemoLifecycleStatus.EN_ROUTE,
                            now=now_utc,
                            eligible_assignment_ids=(en_route_ids | still_en_route_ids),
                        )
                        changed_total.extend(chg)
                        if en_route_ids:
                            status_transitioned = True
                    if arrived_ids:
                        state, chg, spk = self.orchestrator.advance_demo_lifecycle(
                            session_id,
                            target_status=DemoLifecycleStatus.ARRIVED,
                            now=now_utc,
                            eligible_assignment_ids=arrived_ids,
                        )
                        changed_total.extend(chg)
                        spoken_update = spk or spoken_update
                        status_transitioned = True

                snap = serialize_session_snapshot(
                    state.session_id,
                    self.orchestrator,
                    self.handoff_gen,
                    self.audit,
                )
                if status_transitioned:
                    await self._persist_to_postgres_if_configured(state.session_id, snap)
                live_timbre_payload = (
                    await self._synthesize_timbre_payload(
                        session_id=state.session_id,
                        text=spoken_update,
                        language=state.active_language,
                        timeout_seconds=12.0,
                        emit_event=emit_event,
                        req_id=req_id,
                        turn_id=f"lifecycle-{uuid4()}",
                    )
                    if spoken_update
                    else None
                )
                return {
                    "type": "session_state",
                    "req_id": req_id,
                    "event": "demo_lifecycle_updated" if changed_total else "demo_lifecycle_unchanged",
                    "changed_count": len(changed_total),
                    "spoken_announcement_text": spoken_update,
                    "live_timbre_audio": live_timbre_payload,
                    "snapshot": snap,
                }

            if msg_type == "get_demo_fleet":
                active_assignments = _collect_active_demo_assignments(self.orchestrator)
                return {
                    "type": "demo_fleet_status",
                    "req_id": req_id,
                    "vehicles": _serialize_demo_fleet(self.orchestrator, active_assignments),
                }

            if msg_type == "get_handoff":
                raw_sid = payload.get("session_id")
                target_sid: UUID | None = None
                if raw_sid:
                    try:
                        target_sid = UUID(str(raw_sid))
                    except ValueError:
                        target_sid = None
                elif self._active_session_id is not None:
                    target_sid = self._active_session_id

                if target_sid is not None and target_sid in getattr(self.orchestrator, "_sessions", {}):
                    snap = serialize_session_snapshot(
                        target_sid,
                        self.orchestrator,
                        self.handoff_gen,
                        self.audit,
                    )
                    return {
                        "type": "session_handoff",
                        "req_id": req_id,
                        "session_id": str(target_sid),
                        "source": "in_memory",
                        "handoff": snap["handoff"],
                    }

                if target_sid is not None and self.pg_store is not None and self.pg_store.is_configured:
                    pg_handoff = self.pg_store.get_latest_handoff_by_session(str(target_sid))
                    if pg_handoff is not None:
                        return {
                            "type": "session_handoff",
                            "req_id": req_id,
                            "session_id": str(target_sid),
                            "source": "postgres",
                            "handoff": pg_handoff,
                        }

                return {
                    "type": "error",
                    "req_id": req_id,
                    "reason_code": "handoff_not_found",
                    "message": "No structured handoff found for the requested session_id.",
                }

            if msg_type == "session_end":
                session_id = self._resolve_session_id(payload)
                self._active_stt_streams.pop(str(session_id), None)
                state = self.orchestrator.end_session(
                    session_id,
                    reason=str(payload.get("reason") or "caller_session_ended"),
                )
                if self._active_session_id == session_id:
                    self._active_session_id = None
                snap = serialize_session_snapshot(
                    state.session_id,
                    self.orchestrator,
                    self.handoff_gen,
                    self.audit,
                )
                await self._persist_to_postgres_if_configured(state.session_id, snap, wait=True)
                audio_asset = self._resolve_instruction_audio_payload(
                    state.last_selected_instruction_id,
                    state.active_language,
                )
                return {
                    "type": "session_state",
                    "req_id": req_id,
                    "event": "session_completed",
                    "snapshot": snap,
                    "cached_instruction_audio": audio_asset,
                }

            return {
                "type": "error",
                "req_id": req_id,
                "reason_code": "unknown_ws_message_type",
                "message": f"Unsupported message type: {msg_type}",
            }
        except AdapterError as exc:
            return {
                "type": "error",
                "req_id": req_id,
                "reason_code": exc.reason_code,
                "message": str(exc),
            }
        except Exception as exc:
            return {
                "type": "error",
                "req_id": req_id,
                "reason_code": "internal_orchestrator_error",
                "message": str(exc),
            }


# Optional FastAPI wiring when running under uvicorn in Docker / production
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="SETU Voice-First Emergency Coordination Prototype")
    _bridge = LiveSessionBridge(use_live_providers=True)

    @app.get("/api/demo-fleet")
    async def get_demo_fleet() -> JSONResponse:
        active_assignments = _collect_active_demo_assignments(_bridge.orchestrator)
        return JSONResponse(
            {
                "allocation_mode": "DEMO",
                "vehicles": _serialize_demo_fleet(_bridge.orchestrator, active_assignments),
            }
        )

    @app.get("/api/session/{session_id}/handoff")
    async def get_session_handoff(session_id: str) -> JSONResponse:
        res = await _bridge.handle_ws_message({"type": "get_handoff", "session_id": session_id})
        if res.get("type") == "error":
            return JSONResponse(res, status_code=404)
        return JSONResponse(res)

    @app.get("/api/instructions/manifest")
    async def get_instruction_manifest() -> JSONResponse:
        manifest_path = Path("backend-media/instructions/manifest.json")
        if not manifest_path.exists():
            return JSONResponse({"error": "manifest_not_found"}, status_code=404)
        return JSONResponse(json.loads(manifest_path.read_text(encoding="utf-8")))

    @app.get("/backend-media/instructions/{lang}/{instruction_id}/{filename}")
    async def get_instruction_audio(
        lang: str,
        instruction_id: str,
        filename: str,
    ) -> FileResponse:
        file_path = Path("backend-media/instructions") / lang / instruction_id / filename
        return FileResponse(file_path, media_type="audio/wav")

    @app.websocket("/ws/session")
    async def ws_session_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                resp = await _bridge.handle_ws_message(msg)
                await websocket.send_text(json.dumps(resp, default=str))
        except WebSocketDisconnect:
            pass

except ImportError:
    app = None  # FastAPI wheel not installed in container; stdio bridge used by gateway


async def run_stdio_ws_bridge() -> None:
    """Line-delimited JSON bridge for /ws/session with non-blocking concurrent task dispatch."""
    bridge = LiveSessionBridge(use_live_providers=True)
    loop = asyncio.get_running_loop()
    pending_tasks: set[asyncio.Task[Any]] = set()

    def _write_stdout_line(payload_obj: dict[str, Any]) -> None:
        line_out = json.dumps(payload_obj, default=str) + "\n"
        sys.stdout.write(line_out)
        sys.stdout.flush()

    def _emit_stream_event(event: dict[str, Any]) -> None:
        _write_stdout_line(event)

    async def _handle_line(raw_line: str) -> None:
        req_id: Any = None
        try:
            payload = json.loads(raw_line)
            if isinstance(payload, dict):
                req_id = payload.get("req_id")
            response = await bridge.handle_ws_message(payload, emit_event=_emit_stream_event)
            if isinstance(response, dict) and req_id is not None and "req_id" not in response:
                response["req_id"] = req_id
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            response = {
                "type": "error",
                "req_id": req_id,
                "reason_code": "bridge_parse_error",
                "message": str(exc),
            }
        _write_stdout_line(response)

    try:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            raw_str = line.strip()
            if not raw_str:
                continue
            task = asyncio.create_task(_handle_line(raw_str))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)
    finally:
        if pending_tasks:
            await asyncio.gather(*list(pending_tasks), return_exceptions=True)


if __name__ == "__main__":
    if "--stdio-ws-bridge" in sys.argv:
        asyncio.run(run_stdio_ws_bridge())
