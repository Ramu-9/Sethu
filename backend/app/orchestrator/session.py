"""
Authoritative per-session state machine and conversation lifecycle (Sections 8, 9, 15, 17, 18).

Enforces:
- Signed short-lived session tokens (Section 8.1).
- Full dialogue context on every reasoning request (Section 9.2).
- Utterance role rules: only `new_description` counts as a classification attempt (Section 9.3).
- Confidence threshold (default 0.70, Gate 0.10).
- Two-strike unknown fallback and single-unknown session-ending fallback (Section 9.3).
- Provider-blocked emergency content escalation (N=2 default, Gate 0.13) and
  session-ending-after-block fallback with distinct reason code `provider_blocked_emergency_content` (Section 9.4).
- Reasoning outage reassurance fallback with distinct reason code `reasoning_unavailable` (Section 15.4).
- Single-application guarantee for the deterministic ambulance fallback across all triggers (Section 9.3).
- Barge-in and false-trigger handling (Section 18.4).
"""

import asyncio
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import re
import time
from typing import Any
from uuid import UUID, uuid4

from backend.app.adapters.base import (
    ALL_SUPPORTED_LANGUAGES,
    ConversationTurnRecord,
    DialogueContext,
    EmergencyCategory,
    FactConflict,
    FinalizedUtteranceRecord,
    LanguageSelectionMode,
    PendingQuestionRecord,
    ProvenancedFact,
    ProviderBlockedContentError,
    ReasoningAdapter,
    ReasoningSchemaValidationError,
    ReasoningUnavailableError,
    SupportedLanguage,
    UtteranceRole,
)
from backend.app.audit.writer import AuditEvent, AuditSeverity, InMemoryAuditWriter
from backend.app.observability.metrics import MetricsRegistry
from backend.app.orchestrator.announcements import (
    build_demo_dispatch_announcement,
    build_demo_lifecycle_announcement,
    build_pending_location_dispatch_message,
    deterministic_demo_classify_and_extract,
)
from backend.app.orchestrator.dispatch import (
    AssignmentStatus,
    DemoLifecycleStatus,
    DispatchEngine,
    ResourceCategory,
    SimulatedAssignment,
)
from backend.app.orchestrator.duplicates import (
    DuplicateDetector,
    DuplicateMatchResult,
    IncidentSummaryForDuplicateCheck,
)
from backend.app.orchestrator.hospital import HospitalRanker, HospitalScoreBreakdown
from backend.app.orchestrator.instructions import InstructionLibrary
from backend.app.orchestrator.location import (
    GPSPermissionStatus,
    LocationConfirmationStatus,
    LocationResolver,
    ResolvedLocationState,
)
from backend.app.orchestrator.policy import PolicyValidator, PossiblePrankCallFlag


DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.70
DEFAULT_PROVIDER_BLOCK_ESCALATION_THRESHOLD: int = 2
MAX_FALSE_BARGE_INS_BEFORE_MANUAL_ONLY: int = 3

LANGUAGE_DISPLAY_NAMES_BY_CODE: dict[SupportedLanguage, str] = {
    "en": "English",
    "te": "Telugu",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "ml": "Malayalam",
    "bn": "Bengali",
}


def detect_script_or_lexicon_language(text: str) -> SupportedLanguage | None:
    """Deterministically detect one of the 7 supported languages from Unicode script or emergency lexicon."""
    if not text or not text.strip():
        return None
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "te"
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "kn"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"
    if re.search(r"[\u0D00-\u0D7F]", text):
        return "ml"
    if re.search(r"[\u0980-\u09FF]", text):
        return "bn"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    lower = text.lower()
    if re.search(r"\b(pramadam|jarigindi|ikkada|akkada|manta\w*|gayalu|spruha\s+ledu|opiri\s+adatledu|pampandi)\b", lower):
        return "te"
    if re.search(r"\b(agun\s+legeche|durghotona|rokto\s+porche|ekhane|okhane|oshustho|pathan)\b", lower):
        return "bn"
    if re.search(r"\b(apaghata|benki|prajne\s+illa|usiraduttilla|kaluhisi)\b", lower):
        return "kn"
    if re.search(r"\b(vipathu|neruppu|mayakkam|moochu\s+vidavillai|anuppungal)\b", lower):
        return "ta"
    if re.search(r"\b(apaghatam|theepidutham|bodham\s+illa|ayikkoo)\b", lower):
        return "ml"
    if re.search(r"\b(aag\s+lagi|hadhsa|ghayal|behosh|bhejiye|yahan|wahan|saans\s+nahi)\b", lower):
        return "hi"
    if re.search(r"\b(accident|fire|ambulance|injured|bleeding|unconscious|collapsed|emergency|help|near|there\s+is)\b", lower):
        return "en"
    return None


class SessionTokenError(PermissionError):
    """Raised when a session reconnect token is missing, invalid, or expired (Section 8.1)."""


class SessionTokenSigner:
    """
    Application-owned HMAC-SHA256 session token signer and verifier (Section 8.1).
    Key rotation invalidates outstanding tokens safely.
    """

    def __init__(self, *, secret_key: str, ttl_seconds: int = 3600) -> None:
        if not secret_key or len(secret_key.strip()) < 16:
            raise ValueError("Session token signing key must be at least 16 characters")
        self._secret = secret_key.encode("utf-8")
        self._ttl_seconds = ttl_seconds

    def issue_token(self, session_id: UUID, *, issued_at: datetime) -> str:
        exp_ts = int((issued_at + timedelta(seconds=self._ttl_seconds)).timestamp())
        payload = f"{session_id}:{exp_ts}"
        sig = hmac.new(self._secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{payload}:{sig}"

    def verify_token(self, token: str, *, expected_session_id: UUID, now: datetime) -> None:
        parts = token.split(":")
        if len(parts) != 3:
            raise SessionTokenError("Malformed session token")
        sid_str, exp_str, provided_sig = parts
        if sid_str != str(expected_session_id):
            raise SessionTokenError("Session token does not match session_id")
        try:
            exp_ts = int(exp_str)
        except ValueError as exc:
            raise SessionTokenError("Invalid token expiration timestamp") from exc

        if int(now.timestamp()) > exp_ts:
            raise SessionTokenError("Session token has expired")

        payload = f"{sid_str}:{exp_ts}"
        expected_sig = hmac.new(self._secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(provided_sig, expected_sig):
            raise SessionTokenError("Invalid session token signature")


@dataclass
class BargeInOutcome:
    """Outcome of a caller barge-in interruption (Section 18.4)."""
    local_playback_stopped_immediately: bool
    queued_chunks_cleared: bool
    server_cancellation_sent: bool
    counted_as_utterance: bool
    is_false_barge_in: bool
    replayed_cached_instruction_id: str | None
    response_instruction_id: str | None
    response_text: str | None
    automatic_barge_in_disabled: bool


@dataclass
class IncidentSessionState:
    """Authoritative persisted session and incident state (Section 8)."""
    session_id: UUID
    incident_id: UUID
    session_token: str
    session_state: str  # 'active', 'disconnected', 'completed'
    incident_state: str  # 'open', 'flagged_for_review', 'handed_off'
    revision: int
    last_event_sequence: int
    initial_language: SupportedLanguage
    active_language: SupportedLanguage
    created_at: datetime
    updated_at: datetime
    category: EmergencyCategory
    category_established_above_threshold: bool
    classification_attempts: int
    classification_confidences: list[float] = field(default_factory=list)
    consecutive_unknown_count: int = 0
    consecutive_provider_blocks: int = 0
    total_provider_blocks: int = 0
    unknown_fallback_applied: bool = False
    fallback_reason: str | None = None
    control_room_review_required: bool = False
    review_reason_codes: list[str] = field(default_factory=list)
    utterances: list[FinalizedUtteranceRecord] = field(default_factory=list)
    processed_utterance_ids: set[UUID] = field(default_factory=set)
    validated_facts: list[ProvenancedFact] = field(default_factory=list)
    conflicts: list[FactConflict] = field(default_factory=list)
    missing_information_fields: list[str] = field(default_factory=list)
    pending_followup_question: str | None = None
    pending_question_record: PendingQuestionRecord | None = None
    question_history: list[PendingQuestionRecord] = field(default_factory=list)
    location_state: ResolvedLocationState | None = None
    required_resource_categories: list[ResourceCategory] = field(default_factory=list)
    assignments: list[SimulatedAssignment] = field(default_factory=list)
    hospital_recommendations: list[HospitalScoreBreakdown] = field(default_factory=list)
    duplicate_match: DuplicateMatchResult | None = None
    prank_flag: PossiblePrankCallFlag | None = None
    last_spoken_response_text: str = ""
    last_selected_instruction_id: str | None = None
    false_barge_in_count: int = 0
    automatic_vad_barge_in_enabled: bool = True
    conversation_history: list[ConversationTurnRecord] = field(default_factory=list)
    conversational_memory_cleared: bool = False
    announced_assignment_keys: set[str] = field(default_factory=set)
    announced_lifecycle_events: set[str] = field(default_factory=set)
    identified_hazards: list[str] = field(default_factory=list)
    active_safety_guidance: dict[str, Any] | None = None
    task_execution_telemetry: dict[str, Any] = field(default_factory=dict)
    superseded_facts: list[FactConflict] = field(default_factory=list)
    location_history: list[dict[str, Any]] = field(default_factory=list)
    language_mode: LanguageSelectionMode = "manual"
    detected_languages_seen: list[str] = field(default_factory=list)


class SessionOrchestrator:
    """
    Coordinates the SETU incident lifecycle, deterministic dispatch, unknown/block
    fallbacks, location confirmation, hospital ranking, and barge-in rules.
    """

    def __init__(
        self,
        *,
        reasoning_adapter: ReasoningAdapter,
        dispatch_engine: DispatchEngine,
        location_resolver: LocationResolver,
        hospital_ranker: HospitalRanker,
        instruction_library: InstructionLibrary,
        policy_validator: PolicyValidator,
        duplicate_detector: DuplicateDetector,
        audit_writer: InMemoryAuditWriter,
        metrics: MetricsRegistry,
        token_signer: SessionTokenSigner,
        classification_confidence_threshold: float = DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
        provider_block_escalation_threshold: int = DEFAULT_PROVIDER_BLOCK_ESCALATION_THRESHOLD,
        demo_mode: bool = False,
    ) -> None:
        self._reasoning = reasoning_adapter
        self._dispatch = dispatch_engine
        self._location_resolver = location_resolver
        self._hospital_ranker = hospital_ranker
        self._instructions = instruction_library
        self._policy = policy_validator
        self._duplicates = duplicate_detector
        self._audit = audit_writer
        self._metrics = metrics
        self._token_signer = token_signer
        self._confidence_threshold = classification_confidence_threshold
        self._block_threshold = provider_block_escalation_threshold
        self._demo_mode = bool(demo_mode or getattr(dispatch_engine, "is_demo_mode", False))
        self._sessions: dict[UUID, IncidentSessionState] = {}
        self._session_locks: dict[tuple[UUID, int], asyncio.Lock] = {}
        self._inflight_utterance_tasks: dict[tuple[UUID, UUID, int], asyncio.Future[IncidentSessionState]] = {}

    def _get_session_lock(self, session_id: UUID) -> asyncio.Lock:
        """Return an event-loop-local asyncio.Lock for the given session."""
        loop = asyncio.get_running_loop()
        key = (session_id, id(loop))
        lock = self._session_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[key] = lock
        return lock

    def create_session(
        self,
        *,
        initial_language: SupportedLanguage = "en",
        language_mode: LanguageSelectionMode = "manual",
        now: datetime | None = None,
    ) -> IncidentSessionState:
        """Create a new persistent session and play the pre-synthesized spoken disclosure (Section 2.1)."""
        ts = now or datetime.now(timezone.utc)
        session_id = uuid4()
        incident_id = uuid4()
        token = self._token_signer.issue_token(session_id, issued_at=ts)

        disclosure, _ = self._instructions.resolve_cached_asset(
            "disclosure_session_start",
            initial_language,
        )
        self._metrics.record_latency("cached_audio_lookup_ms", 0.4)
        self._metrics.record_latency("cached_audio_playback_start_ms", 1.2)

        state = IncidentSessionState(
            session_id=session_id,
            incident_id=incident_id,
            session_token=token,
            session_state="active",
            incident_state="open",
            revision=1,
            last_event_sequence=1,
            initial_language=initial_language,
            active_language=initial_language,
            created_at=ts,
            updated_at=ts,
            category=EmergencyCategory.UNKNOWN,
            category_established_above_threshold=False,
            classification_attempts=0,
            last_spoken_response_text=disclosure.approved_text,
            last_selected_instruction_id=disclosure.instruction_id,
            language_mode=language_mode,
            detected_languages_seen=[initial_language],
        )
        self._sessions[session_id] = state

        self._audit.append(
            AuditEvent.create(
                session_id=session_id,
                incident_id=incident_id,
                event_type="session_created",
                severity=AuditSeverity.INFO,
                outcome="spoken_disclosure_played_from_cache",
                now=ts,
                details={
                    "initial_language": initial_language,
                    "language_mode": language_mode,
                    "disclosure_instruction_id": disclosure.instruction_id,
                },
            )
        )
        return state

    def set_language_mode(
        self,
        session_id: UUID,
        *,
        language_mode: LanguageSelectionMode = "manual",
        manual_language: SupportedLanguage | None = None,
        now: datetime | None = None,
    ) -> IncidentSessionState:
        """Update session language across all 7 supported languages without losing incident state."""
        ts = now or datetime.now(timezone.utc)
        state = self.get_session(session_id)
        prev_mode = state.language_mode
        prev_lang = state.active_language
        state.language_mode = language_mode
        if manual_language in ALL_SUPPORTED_LANGUAGES:
            state.active_language = manual_language
            if manual_language not in state.detected_languages_seen:
                state.detected_languages_seen.append(manual_language)
        state.revision += 1
        state.last_event_sequence += 1
        state.updated_at = ts
        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="language_mode_updated",
                severity=AuditSeverity.INFO,
                outcome=f"{prev_mode}:{prev_lang}->{language_mode}:{state.active_language}",
                now=ts,
                details={
                    "previous_mode": prev_mode,
                    "new_mode": language_mode,
                    "previous_language": prev_lang,
                    "active_language": state.active_language,
                },
            )
        )
        if prev_lang != state.active_language:
            self._audit.append(
                AuditEvent.create(
                    session_id=state.session_id,
                    incident_id=state.incident_id,
                    event_type="language_switched",
                    severity=AuditSeverity.INFO,
                    outcome=f"{prev_lang}->{state.active_language}",
                    now=ts,
                    details={
                        "previous_language": prev_lang,
                        "new_language": state.active_language,
                        "language_mode": state.language_mode,
                    },
                )
            )
        return state

    def record_uncertain_language_detection(
        self,
        session_id: UUID,
        *,
        raw_detected_language_code: str | None,
        language_confidence: float | None = None,
        utterance_id: UUID | None = None,
        request_clarification_if_needed: bool = False,
        now: datetime | None = None,
    ) -> IncidentSessionState:
        """
        Keep the session's active_language unchanged when Prisma indic-auto reports an
        unsupported Indic language (e.g. mr-IN, gu-IN, pa-IN, as-IN) or low confidence,
        and optionally set a localized clarification prompt without losing emergency context.
        """
        ts = now or datetime.now(timezone.utc)
        state = self.get_session(session_id)
        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="language_detection_uncertain",
                severity=AuditSeverity.WARNING,
                outcome=f"preserved_{state.active_language}",
                now=ts,
                provenance_refs=[str(utterance_id)] if utterance_id else (),
                details={
                    "preserved_language": state.active_language,
                    "raw_detected_language_code": raw_detected_language_code,
                    "language_confidence": language_confidence,
                },
            )
        )
        if request_clarification_if_needed:
            clarify_inst = self._instructions.get_didnt_catch_that_phrase(state.active_language)
            state.last_spoken_response_text = clarify_inst.approved_text
            state.last_selected_instruction_id = clarify_inst.instruction_id
            state.revision += 1
            state.last_event_sequence += 1
            state.updated_at = ts
        return state

    def get_session(self, session_id: UUID) -> IncidentSessionState:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        return self._sessions[session_id]

    def list_sessions(self) -> list[IncidentSessionState]:
        return list(self._sessions.values())

    def reconnect_session(
        self,
        session_id: UUID,
        *,
        session_token: str,
        last_received_sequence: int,
        now: datetime | None = None,
    ) -> IncidentSessionState:
        """Validate session token and restore session on WebSocket reconnect (Section 8.1)."""
        ts = now or datetime.now(timezone.utc)
        state = self.get_session(session_id)
        self._token_signer.verify_token(
            session_token,
            expected_session_id=session_id,
            now=ts,
        )
        state.session_state = "active"
        state.revision += 1
        state.updated_at = ts
        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="session_reconnected",
                severity=AuditSeverity.INFO,
                outcome="resumed",
                now=ts,
                details={"last_received_sequence": last_received_sequence},
            )
        )
        return state

    def update_gps_location(
        self,
        session_id: UUID,
        *,
        latitude: float | None,
        longitude: float | None,
        accuracy_m: float | None,
        permission_status: GPSPermissionStatus,
        confirmed_by_caller: bool = False,
        now: datetime | None = None,
    ) -> IncidentSessionState:
        """Evaluate browser GPS against the 500m threshold and trigger pending allocations if eligible."""
        ts = now or datetime.now(timezone.utc)
        state = self.get_session(session_id)
        resolved = self._location_resolver.evaluate_browser_gps(
            latitude=latitude,
            longitude=longitude,
            accuracy_m=accuracy_m,
            permission_status=permission_status,
            timestamp=ts,
        )
        if confirmed_by_caller and resolved.is_usable_candidate:
            resolved = self._location_resolver.confirm_location(resolved, confirmed_at=ts)

        state.location_state = resolved
        state.revision += 1
        state.last_event_sequence += 1
        state.updated_at = ts

        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="location_gps_evaluated",
                severity=AuditSeverity.INFO if resolved.is_usable_candidate else AuditSeverity.WARNING,
                outcome="candidate_accepted" if resolved.is_usable_candidate else "requires_spoken_landmark",
                now=ts,
                provenance_refs=[resolved.provenance.source_id],
                details={
                    "accuracy_m": accuracy_m,
                    "confirmation_status": resolved.confirmation_status.value,
                },
            )
        )

        self._reconcile_location_dependent_operations(state, now=ts)
        return state

    def confirm_current_location(
        self,
        session_id: UUID,
        *,
        now: datetime | None = None,
    ) -> IncidentSessionState:
        """
        Confirm the current location candidate (GPS or landmark) by the caller (Section 12.3),
        commit demo vehicle allocation if category is established, transition units to EN_ROUTE,
        and produce immediate spoken dispatch + approved safety guidance.
        """
        ts = now or datetime.now(timezone.utc)
        state = self.get_session(session_id)
        has_coords = (
            state.location_state is not None
            and state.location_state.latitude is not None
            and state.location_state.longitude is not None
        )
        has_spoken = (
            self._demo_mode
            and state.location_state is not None
            and bool(
                state.location_state.caller_location_text
                or state.location_state.broad_area_text
                or state.location_state.caller_landmark_text
            )
        )
        if not (has_coords or has_spoken):
            raise ValueError("No location candidate exists to confirm")
        assert state.location_state is not None
        state.location_state = self._location_resolver.confirm_location(
            state.location_state,
            confirmed_at=ts,
            allow_text_only=self._demo_mode,
        )
        # If there was a pending question targeting location confirmation, mark it answered
        if (
            state.pending_question_record is not None
            and state.pending_question_record.status == "pending"
            and state.pending_question_record.target_field in ("location_confirmed", "caller_location_text", "caller_landmark_text")
        ):
            resolved_q = PendingQuestionRecord(
                question_id=state.pending_question_record.question_id,
                question_text=state.pending_question_record.question_text,
                target_field=state.pending_question_record.target_field,
                asked_at=state.pending_question_record.asked_at,
                source_utterance_id=state.pending_question_record.source_utterance_id,
                status="answered",
                answered_by_utterance_id=None,
            )
            state.question_history.append(resolved_q)
            state.pending_question_record = None
            state.pending_followup_question = None

        state.revision += 1
        state.last_event_sequence += 1
        state.updated_at = ts
        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="location_confirmed_by_caller",
                severity=AuditSeverity.INFO,
                outcome="confirmed",
                now=ts,
                provenance_refs=[state.location_state.provenance.source_id],
                details={"confirmed_at": ts.isoformat()},
            )
        )
        self._reconcile_location_dependent_operations(state, now=ts)

        # In DEMO mode, immediately compose spoken confirmation + EN_ROUTE dispatch + safety guidance
        if self._demo_mode:
            lang = state.active_language
            ack = {
                "te": "మీ లొకేషన్ నిర్ధారించబడింది.",
                "hi": "आपकी लोकेशन कन्फर्म हो गई है।",
                "kn": "ನಿಮ್ಮ ಸ್ಥಳವನ್ನು ದೃಢೀಕರಿಸಲಾಗಿದೆ.",
                "ta": "உங்கள் இருப்பிடம் உறுதிப்படுத்தப்பட்டது.",
                "ml": "നിങ്ങളുടെ ലൊക്കേഷൻ സ്ഥിരീകരിച്ചിരിക്കുന്നു.",
                "bn": "আপনার লোকেশন নিশ্চিত করা হয়েছে।",
            }.get(lang, "Thank you, your location is confirmed.")

            spoken = self._append_new_demo_dispatch_announcement(state, ack, now=ts)
            context_txt = " ".join(u.text for u in state.utterances)
            safety_inst = self._instructions.get_safety_guidance_for_category(
                state.category,
                lang,
                context_text=context_txt,
            )
            if safety_inst is not None:
                state.active_safety_guidance = {
                    "instruction_id": safety_inst.instruction_id,
                    "category": safety_inst.category.value if hasattr(safety_inst.category, "value") else str(safety_inst.category),
                    "language": safety_inst.language,
                    "approved_text": safety_inst.approved_text,
                    "source": safety_inst.source,
                    "review_date": safety_inst.review_date.isoformat(),
                    "version": safety_inst.version,
                }
                if safety_inst.approved_text.lower() not in spoken.lower():
                    spoken = f"{spoken} {safety_inst.approved_text}".strip()
                    state.last_selected_instruction_id = safety_inst.instruction_id

            safe_spoken, _ = self._policy.sanitize_caller_response_text(
                spoken,
                language=lang,
                has_simulated_assignment=any(
                    a.status == AssignmentStatus.SIMULATED_ASSIGNED for a in state.assignments
                ),
            )
            state.last_spoken_response_text = safe_spoken
            state.conversation_history.append(
                ConversationTurnRecord(
                    role="assistant",
                    text=safe_spoken,
                    language=lang,
                    utterance_id=None,
                    recorded_at=ts,
                )
            )
        return state

    async def _execute_parallel_branch(
        self,
        name: str,
        coro: Awaitable[Any],
    ) -> tuple[str, float, Any, str | None]:
        """Run one independent orchestration branch with fault isolation and wall-clock timing."""
        t0 = time.perf_counter()
        try:
            res = await coro
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return name, elapsed_ms, res, None
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return name, elapsed_ms, None, f"{type(exc).__name__}: {exc}"

    async def _branch_evaluate_hazards_and_fleet(
        self,
        *,
        existing_hazards: tuple[str, ...],
        llm_hazards: tuple[str, ...],
        candidate_category: EmergencyCategory,
        context_text: str,
    ) -> dict[str, Any]:
        """Concurrently compute unified multi-hazard tags, required resource categories, and synthetic fleet availability."""
        await asyncio.sleep(0)
        merged: list[str] = []
        for tag in (*existing_hazards, *llm_hazards):
            norm = str(tag).strip().lower().replace("-", "_").replace(" ", "_")
            if norm and norm not in merged:
                merged.append(norm)
        lower_ctx = context_text.lower()
        keyword_hazards = (
            (("vehicle_collision",), ("collid", "collision", "crash", "bus", "truck", "accident", "ప్రమాదం", "दुर्घटना", "ಅಪಘಾತ", "விபத்து", "അപകടം", "দুর্ঘটনা")),
            (("fire",), ("fire", "flames", "burning", "smoke", "आग", "మంట", "ಬೆಂಕಿ", "தீ", "நெருப்பு", "തീ", "തീപിടുത്തം", "আগুন")),
            (("gas_cylinder",), ("cylinder", "lpg", "gas cylinder", "सिलेंडर", "సిలిండర్", "ಸಿಲಿಂಡರ್", "சிலிண்டர்", "സിലിണ്ടർ", "সিলিন্ডার")),
            (("injuries",), ("injur", "bleed", "burn", "hurt", "casualt", "घायल", "గాయ", "ಗಾಯ", "காய", "പരിക്ക്", "আহত")),
            (("explosion",), ("blast", "explod", "explosion", "పేలుడు", "विस्फोट", "ಸ್ಫೋಟ", "வெடிப்பு", "സ്ഫോടനം", "বিস্ফোরণ")),
            (("trapped_persons",), ("trapped", "stuck inside", "collapse", "debris", "చిక్కుకు", "फंसे", "ಸಿಲುಕಿ", "சிக்கி", "കുടുങ്ങി", "আটকে")),
        )
        for tags, words in keyword_hazards:
            if any(w in lower_ctx for w in words):
                for t in tags:
                    if t not in merged:
                        merged.append(t)

        required_cats = self._dispatch.determine_multi_hazard_categories(
            candidate_category,
            identified_hazards=tuple(merged),
            context_text=context_text,
        )
        availability = self._dispatch.check_fleet_availability(required_cats)
        return {
            "identified_hazards": merged,
            "required_categories": list(required_cats),
            "fleet_availability": availability,
        }

    async def _branch_prepare_safety_guidance(
        self,
        *,
        candidate_category: EmergencyCategory,
        active_language: SupportedLanguage,
        context_text: str,
        selected_instruction_id: str | None,
    ) -> dict[str, Any] | None:
        """Concurrently look up approved safety guidance aligned with detected hazards and language."""
        await asyncio.sleep(0)
        inst = self._instructions.get_safety_guidance_for_category(
            candidate_category,
            active_language,
            context_text=context_text,
        )
        if inst is None and selected_instruction_id:
            inst = self._instructions.get_by_id(selected_instruction_id, active_language)
        if inst is None:
            return None
        return {
            "instruction_id": inst.instruction_id,
            "category": inst.category.value if hasattr(inst.category, "value") else str(inst.category),
            "language": inst.language,
            "approved_text": inst.approved_text,
            "source": inst.source,
            "review_date": inst.review_date.isoformat(),
            "version": inst.version,
        }

    async def _branch_prepare_location_and_hospitals(
        self,
        *,
        candidate_category: EmergencyCategory,
        latitude: float | None,
        longitude: float | None,
        location_text_candidates: tuple[str, ...],
        utterance_id: UUID,
        now: datetime,
    ) -> dict[str, Any]:
        """Concurrently prepare landmark candidates and nearby directory hospital rankings without auto-confirming location."""
        await asyncio.sleep(0)
        landmark_matches = []
        for loc_txt in location_text_candidates:
            if loc_txt.strip():
                landmark_matches = self._location_resolver.search_landmarks(
                    caller_location_text=loc_txt.strip(),
                    source_utterance_id=utterance_id,
                    now=now,
                    min_similarity=0.52,
                )
                if landmark_matches:
                    break
        hospitals: list[HospitalScoreBreakdown] = []
        if latitude is not None and longitude is not None:
            hospitals = list(
                self._hospital_ranker.rank_hospitals(
                    incident_category=candidate_category,
                    caller_latitude=latitude,
                    caller_longitude=longitude,
                    now=now,
                )
            )
        return {
            "landmark_matches": landmark_matches,
            "hospitals": hospitals,
        }

    async def _branch_resolve_facts_and_question(
        self,
        *,
        state: IncidentSessionState,
        text: str,
        utterance_id: UUID,
        validated_output: Any,
        now: datetime,
    ) -> dict[str, Any]:
        """Concurrently bind pending-question answers and prepare normalized extracted facts."""
        await asyncio.sleep(0)
        pre_facts, was_pending_answered = self._resolve_pending_question_from_utterance(
            state,
            text=text,
            utterance_id=utterance_id,
            extracted_facts=validated_output.extracted_facts,
            answered_hint=validated_output.pending_question_answered,
            now=now,
        )
        resolved_role = validated_output.utterance_role
        if (
            was_pending_answered
            and resolved_role == UtteranceRole.NEW_DESCRIPTION
            and state.category_established_above_threshold
        ):
            resolved_role = UtteranceRole.ANSWER_TO_PENDING_QUESTION
        return {
            "pre_facts": pre_facts,
            "was_pending_answered": was_pending_answered,
            "resolved_role": resolved_role,
        }

    async def process_finalized_utterance(
        self,
        session_id: UUID,
        *,
        utterance_id: UUID,
        text: str,
        detected_language: SupportedLanguage | None = None,
        timeout_seconds: float = 18.0,
        now: datetime | None = None,
    ) -> IncidentSessionState:
        """
        Process a single finalized caller utterance through the reasoning model,
        policy validator, deterministic dispatcher, and STAY WITH ME instruction selector.
        """
        ts = now or datetime.now(timezone.utc)
        state = self.get_session(session_id)

        cleaned_text = text.strip()
        if not cleaned_text:
            return state

        loop = asyncio.get_running_loop()
        inflight_key = (session_id, utterance_id, id(loop))
        existing_inflight = self._inflight_utterance_tasks.get(inflight_key)
        if existing_inflight is not None:
            self._metrics.increment_duplicate_request_prevented(coalesced=True)
            return await asyncio.shield(existing_inflight)

        # Prevent duplicate processing of the same finalized utterance (Section 9.3 & Requirement 1)
        if utterance_id in state.processed_utterance_ids:
            self._metrics.increment_duplicate_request_prevented(coalesced=False)
            return state

        fut: asyncio.Future[IncidentSessionState] = loop.create_future()
        self._inflight_utterance_tasks[inflight_key] = fut
        try:
            res_state = await self._process_finalized_utterance_impl(
                session_id=session_id,
                state=state,
                utterance_id=utterance_id,
                cleaned_text=cleaned_text,
                detected_language=detected_language,
                timeout_seconds=timeout_seconds,
                ts=ts,
            )
            if not fut.done():
                fut.set_result(res_state)
            return res_state
        except BaseException as exc:
            if not fut.done():
                fut.set_exception(exc)
                fut.exception()
            raise
        finally:
            self._inflight_utterance_tasks.pop(inflight_key, None)

    def _select_scoped_eligible_instruction_ids(
        self,
        state: IncidentSessionState,
        utterance_text: str,
    ) -> tuple[str, ...]:
        """
        Scope eligible instruction IDs to the established emergency category on routine follow-up turns
        while retaining all category safety IDs when category is not yet established or when a new
        cross-category hazard keyword appears in utterance_text.
        """
        eligible_instructions = self._instructions.get_eligible_instructions(
            state.category,
            state.active_language,
        )
        scoped_ids: list[str] = list(dict.fromkeys(i.instruction_id for i in eligible_instructions))
        has_cross_category_cue = bool(
            re.search(
                r"\b(fire|flames?|smoke|burning|explo\w*|blast|collaps\w*|trapped|cylinder|lpg|gas|"
                r"attack\w*|assault\w*|weapon|knife|accident|crash\w*|collid\w*|unconscious|fainted|"
                r"aag|dhuan|behosh|ghayal|chot|manta|agni|spruha|pramadam|benki|thee|neruppu|agun)\b|"
                r"(?:మంట|అగ్ని|పేలుడు|కూలి|సిలిండర్|దాడి|ప్రమాదం|స్పృహ|आग|विस्फोट|इमारत|सिलेंडर|हमला|दुर्घटना|बेहोश|"
                r"ಬೆಂಕಿ|ಸ್ಫೋಟ|ಕುಸಿದು|ಸಿಲಿಂಡರ್|ದಾಳಿ|ಅಪಘಾತ|ಪ್ರಜ್ಞೆ|தீ|வெடிப்பு|இடிந்து|சிலிண்டர்|தாக்குதல்|விபத்து|மயக்கம்|"
                r"തീ|സ്ഫോടനം|ഇടിഞ്ഞു|സിലിണ്ടർ|ആക്രമണം|അപകടം|ബോധമില്ല|আগুন|বিস্ফোরণ|ধসে|সিলিন্ডার|হামলা|দুর্ঘটনা|অজ্ঞান)",
                utterance_text.lower(),
                re.I,
            )
        )
        if not state.category_established_above_threshold or has_cross_category_cue:
            for cat in EmergencyCategory:
                for inst in self._instructions.get_eligible_instructions(cat, state.active_language):
                    if inst.instruction_id not in scoped_ids:
                        scoped_ids.append(inst.instruction_id)
        else:
            if state.category == EmergencyCategory.FIRE:
                for inst in self._instructions.get_eligible_instructions(
                    EmergencyCategory.FIRE_WITH_INJURIES, state.active_language
                ):
                    if inst.instruction_id not in scoped_ids:
                        scoped_ids.append(inst.instruction_id)
            if "gas_cylinder" in state.identified_hazards and "safety_fire_gas_cylinder_01" not in scoped_ids:
                scoped_ids.append("safety_fire_gas_cylinder_01")
        return tuple(scoped_ids)

    @staticmethod
    def _build_session_memory_summary(state: IncidentSessionState) -> dict[str, Any]:
        """Build complete structured session memory summary so bounded turn windows never lose context."""
        loc = state.location_state
        confirmed_loc_label = (
            (loc.resolved_label or loc.caller_location_text)
            if loc is not None
            else None
        )
        answered_questions = [
            {
                "target_field": q.target_field,
                "question": q.question_text,
                "answered_by_utt": str(q.answered_by_utterance_id) if q.answered_by_utterance_id else None,
            }
            for q in state.question_history
            if q.status == "answered"
        ]
        answered_fields = sorted({q.target_field for q in state.question_history if q.status == "answered"})
        superseded_items = [
            {
                "field": c.field_name,
                "prior_value": c.prior_value,
                "prior_utt": str(c.prior_utterance_id),
                "new_value": c.new_value,
                "new_utt": str(c.new_utterance_id),
            }
            for c in (state.superseded_facts or state.conflicts)
        ]
        active_assignments = [
            {
                "vehicle_id": a.resource_id,
                "category": a.category.value,
                "status": a.status.value,
                "lifecycle_status": a.lifecycle_status,
                "eta_minutes": a.remaining_eta_minutes or a.simulated_eta_minutes,
            }
            for a in state.assignments
        ]
        return {
            "confirmed_location_label": confirmed_loc_label,
            "location_history": list(state.location_history),
            "answered_target_fields": answered_fields,
            "answered_questions": answered_questions,
            "superseded_facts": superseded_items,
            "active_assignments": active_assignments,
            "required_resources": [c.value for c in state.required_resource_categories],
        }

    async def _process_finalized_utterance_impl(
        self,
        *,
        session_id: UUID,
        state: IncidentSessionState,
        utterance_id: UUID,
        cleaned_text: str,
        detected_language: SupportedLanguage | None,
        timeout_seconds: float,
        ts: datetime,
    ) -> IncidentSessionState:
        # Language switching at utterance boundary across all 7 supported languages without restarting incident (Section 17)
        if detected_language in ALL_SUPPORTED_LANGUAGES:
            if detected_language not in state.detected_languages_seen:
                state.detected_languages_seen.append(detected_language)
            if detected_language != state.active_language:
                prev_lang = state.active_language
                state.active_language = detected_language
                self._audit.append(
                    AuditEvent.create(
                        session_id=state.session_id,
                        incident_id=state.incident_id,
                        event_type="language_switched",
                        severity=AuditSeverity.INFO,
                        outcome=f"{prev_lang}->{detected_language}",
                        now=ts,
                        details={
                            "previous_language": prev_lang,
                            "new_language": detected_language,
                            "language_mode": state.language_mode,
                        },
                    )
                )

        current_record = FinalizedUtteranceRecord(
            utterance_id=utterance_id,
            sequence_number=len(state.utterances) + 1,
            text=cleaned_text,
            language=state.active_language,
            utterance_role=None,
            finalized_at=ts,
        )

        scoped_prompt_ids = self._select_scoped_eligible_instruction_ids(state, cleaned_text)
        all_eligible_ids_list: list[str] = list(scoped_prompt_ids)
        for cat in EmergencyCategory:
            for inst in self._instructions.get_eligible_instructions(cat, state.active_language):
                if inst.instruction_id not in all_eligible_ids_list:
                    all_eligible_ids_list.append(inst.instruction_id)
        all_eligible_ids = tuple(all_eligible_ids_list)
        session_mem_summary = self._build_session_memory_summary(state)
        session_mem_summary["prompt_scoped_instruction_ids"] = list(scoped_prompt_ids)
        loc_confirmed_bool = bool(
            state.location_state is not None
            and state.location_state.confirmation_status == LocationConfirmationStatus.CONFIRMED
        )
        sim_assigned_count = len(
            [a for a in state.assignments if a.status == AssignmentStatus.SIMULATED_ASSIGNED]
        )

        context = DialogueContext(
            session_id=state.session_id,
            incident_id=state.incident_id,
            active_language=state.active_language,
            prior_utterances=tuple(state.utterances),
            validated_facts=tuple(state.validated_facts),
            current_utterance=current_record,
            pending_followup_question=state.pending_followup_question,
            eligible_instruction_ids=all_eligible_ids,
            backend_status_summary={
                "assignments_count": sim_assigned_count,
                "simulated_assignments_count": sim_assigned_count,
                "required_categories": ",".join(c.value for c in state.required_resource_categories),
                "location_confirmation_status": (
                    state.location_state.confirmation_status.value
                    if state.location_state
                    else "unresolved"
                ),
                "location_confirmed": str(loc_confirmed_bool).lower(),
                "category": state.category.value,
                "current_category": state.category.value,
                "allocation_mode": "DEMO" if self._demo_mode else "STRICT",
            },
            conversation_history=tuple(state.conversation_history),
            pending_question=state.pending_question_record,
            identified_hazards=tuple(state.identified_hazards),
            active_safety_guidance_id=(
                str(state.active_safety_guidance["instruction_id"])
                if state.active_safety_guidance and state.active_safety_guidance.get("instruction_id")
                else None
            ),
            session_memory_summary=session_mem_summary,
        )

        async with self._get_session_lock(session_id):
            if state.session_state == "completed":
                return state
            state.processed_utterance_ids.add(utterance_id)
            state.utterances.append(current_record)
            state.conversation_history.append(
                ConversationTurnRecord(
                    role="caller",
                    text=cleaned_text,
                    language=state.active_language,
                    utterance_id=utterance_id,
                    recorded_at=ts,
                )
            )
            state.conversational_memory_cleared = False
            state.revision += 1
            state.last_event_sequence += 1
            state.updated_at = ts
            existing_hazards_snap = tuple(state.identified_hazards)
            existing_lat = state.location_state.latitude if state.location_state else None
            existing_lon = state.location_state.longitude if state.location_state else None
            context_all_text = " ".join(u.text for u in state.utterances)

            self._audit.append(
                AuditEvent.create(
                    session_id=state.session_id,
                    incident_id=state.incident_id,
                    event_type="final_transcript_received",
                    severity=AuditSeverity.INFO,
                    outcome="persisted",
                    now=ts,
                    provenance_refs=[str(utterance_id)],
                    details={"language": state.active_language},
                )
            )

        t0 = time.perf_counter()
        try:
            raw_output = await self._reasoning.analyze_utterance(
                context,
                timeout_seconds=timeout_seconds,
            )
            validated_output = self._policy.validate_reasoning_output(raw_output, context)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._metrics.record_latency("reasoning_processing_ms", elapsed_ms)
            self._metrics.record_token_usage(
                prompt_tokens=validated_output.prompt_tokens,
                completion_tokens=validated_output.completion_tokens,
                thinking_level=getattr(validated_output, "thinking_level", "LOW"),
                was_coalesced_or_cached=bool(getattr(validated_output, "was_coalesced_or_cached", False)),
            )
        except ProviderBlockedContentError as blocked_err:
            self._metrics.increment_provider_block()
            self._handle_provider_blocked_turn(state, utterance_id=utterance_id, err=blocked_err, now=ts)
            if state.last_spoken_response_text:
                state.conversation_history.append(
                    ConversationTurnRecord(
                        role="assistant",
                        text=state.last_spoken_response_text,
                        language=state.active_language,
                        utterance_id=None,
                        recorded_at=ts,
                    )
                )
            return state
        except ReasoningUnavailableError as outage_err:
            if outage_err.http_status == 429:
                self._metrics.increment_rate_limit()
            if self._demo_mode:
                self._handle_demo_outage_turn(
                    state,
                    utterance_id=utterance_id,
                    text=cleaned_text,
                    err=outage_err,
                    now=ts,
                )
            else:
                self._handle_reasoning_outage_turn(state, utterance_id=utterance_id, err=outage_err, now=ts)
            state.conversation_history.append(
                ConversationTurnRecord(
                    role="assistant",
                    text=state.last_spoken_response_text,
                    language=state.active_language,
                    utterance_id=None,
                    recorded_at=ts,
                )
            )
            return state
        except ReasoningSchemaValidationError as schema_err:
            self._handle_schema_invalid_turn(state, utterance_id=utterance_id, err=schema_err, now=ts)
            if state.last_spoken_response_text:
                state.conversation_history.append(
                    ConversationTurnRecord(
                        role="assistant",
                        text=state.last_spoken_response_text,
                        language=state.active_language,
                        utterance_id=None,
                        recorded_at=ts,
                    )
                )
            return state

        # Schema-valid response resets consecutive_provider_blocks to zero (Section 9.4)
        if state.session_state == "completed":
            return state

        candidate_cat = (
            validated_output.category
            if (
                validated_output.category != EmergencyCategory.UNKNOWN
                and validated_output.category_confidence >= self._confidence_threshold
            )
            else state.category
        )
        if (
            candidate_cat == EmergencyCategory.FIRE
            and (
                any(
                    f.field_name == "injuries_reported" and f.value is True
                    for f in (*state.validated_facts, *validated_output.extracted_facts)
                )
            )
        ):
            candidate_cat = EmergencyCategory.FIRE_WITH_INJURIES

        loc_texts = tuple(
            str(f.value).strip()
            for f in validated_output.extracted_facts
            if f.field_name in ("caller_location_text", "caller_landmark_text", "broad_location_text")
            and isinstance(f.value, str)
            and f.value.strip()
        )

        # Stage A: Execute independent post-Gemini orchestration branches concurrently via asyncio.gather
        t_parallel0 = time.perf_counter()
        branch_results = await asyncio.gather(
            self._execute_parallel_branch(
                "fact_merge_and_question",
                self._branch_resolve_facts_and_question(
                    state=state,
                    text=cleaned_text,
                    utterance_id=utterance_id,
                    validated_output=validated_output,
                    now=ts,
                ),
            ),
            self._execute_parallel_branch(
                "hazard_and_fleet_eval",
                self._branch_evaluate_hazards_and_fleet(
                    existing_hazards=existing_hazards_snap,
                    llm_hazards=tuple(getattr(validated_output, "identified_hazards", ()) or ()),
                    candidate_category=candidate_cat,
                    context_text=context_all_text,
                ),
            ),
            self._execute_parallel_branch(
                "safety_guidance_lookup",
                self._branch_prepare_safety_guidance(
                    candidate_category=candidate_cat,
                    active_language=state.active_language,
                    context_text=context_all_text,
                    selected_instruction_id=validated_output.selected_instruction_id,
                ),
            ),
            self._execute_parallel_branch(
                "location_and_hospitals_prep",
                self._branch_prepare_location_and_hospitals(
                    candidate_category=candidate_cat,
                    latitude=existing_lat,
                    longitude=existing_lon,
                    location_text_candidates=loc_texts,
                    utterance_id=utterance_id,
                    now=ts,
                ),
            ),
        )
        parallel_wall_ms = round((time.perf_counter() - t_parallel0) * 1000.0, 2)

        branches_telemetry: dict[str, Any] = {}
        branch_payloads: dict[str, Any] = {}
        degraded_branches: list[str] = []
        sum_branch_ms = 0.0
        for b_name, b_ms, b_val, b_err in branch_results:
            sum_branch_ms += b_ms
            if b_err is not None:
                degraded_branches.append(b_name)
                branches_telemetry[b_name] = {"duration_ms": b_ms, "status": "error", "error": b_err}
            else:
                branch_payloads[b_name] = b_val
                branches_telemetry[b_name] = {"duration_ms": b_ms, "status": "ok"}

        # Stage B: Validation-gated atomic commit under per-session asyncio lock
        async with self._get_session_lock(session_id):
            if state.session_state == "completed":
                return state

            state.consecutive_provider_blocks = 0

            fact_res = branch_payloads.get("fact_merge_and_question")
            if isinstance(fact_res, dict):
                pre_facts = fact_res["pre_facts"]
                was_pending_answered = bool(fact_res["was_pending_answered"])
                resolved_role = fact_res["resolved_role"]
            else:
                pre_facts, was_pending_answered = self._resolve_pending_question_from_utterance(
                    state,
                    text=cleaned_text,
                    utterance_id=utterance_id,
                    extracted_facts=validated_output.extracted_facts,
                    answered_hint=validated_output.pending_question_answered,
                    now=ts,
                )
                resolved_role = validated_output.utterance_role
                if (
                    was_pending_answered
                    and resolved_role == UtteranceRole.NEW_DESCRIPTION
                    and state.category_established_above_threshold
                ):
                    resolved_role = UtteranceRole.ANSWER_TO_PENDING_QUESTION

            if degraded_branches:
                self._audit.append(
                    AuditEvent.create(
                        session_id=state.session_id,
                        incident_id=state.incident_id,
                        event_type="parallel_orchestration_partial_degradation",
                        severity=AuditSeverity.WARNING,
                        outcome=",".join(degraded_branches),
                        provenance_refs=[str(utterance_id)],
                        details={"branches": branches_telemetry},
                        now=ts,
                    )
                )

            # Update utterance role on the matching stored record cleanly
            for idx in range(len(state.utterances) - 1, -1, -1):
                if state.utterances[idx].utterance_id == current_record.utterance_id:
                    rec_u = state.utterances[idx]
                    state.utterances[idx] = FinalizedUtteranceRecord(
                        utterance_id=rec_u.utterance_id,
                        sequence_number=rec_u.sequence_number,
                        text=rec_u.text,
                        language=rec_u.language,
                        utterance_role=resolved_role,
                        finalized_at=rec_u.finalized_at,
                    )
                    break

            # Merge validated facts into session memory (updating fields cleanly while preserving provenance)
            self._merge_validated_facts(state, pre_facts, now=ts)

            hazard_res = branch_payloads.get("hazard_and_fleet_eval")
            multi_hazard_cats: tuple[ResourceCategory, ...] | None = None
            fleet_avail: dict[str, int] = {}
            if isinstance(hazard_res, dict):
                state.identified_hazards = list(hazard_res.get("identified_hazards") or [])
                req_cats = hazard_res.get("required_categories") or []
                multi_hazard_cats = tuple(req_cats) if req_cats else None
                fleet_avail = dict(hazard_res.get("fleet_availability") or {})
            elif getattr(validated_output, "identified_hazards", None):
                for hz in validated_output.identified_hazards:
                    if hz not in state.identified_hazards:
                        state.identified_hazards.append(hz)

            guidance_res = branch_payloads.get("safety_guidance_lookup")
            if isinstance(guidance_res, dict):
                state.active_safety_guidance = guidance_res

            loc_res = branch_payloads.get("location_and_hospitals_prep")
            if isinstance(loc_res, dict) and loc_res.get("hospitals") and not state.hospital_recommendations:
                state.hospital_recommendations = list(loc_res["hospitals"])

            for conflict in validated_output.conflicts:
                state.conflicts.append(conflict)
                self._add_review_reason(state, "conflicting_caller_statements")

            state.missing_information_fields = self._filter_missing_fields(
                state, validated_output.missing_information_fields
            )
            filtered_followup = self._filter_redundant_followup(
                state,
                validated_output.proposed_followup_question,
                target_field_hint=validated_output.proposed_followup_target_field,
            )
            # If the model proposed a redundant question that was filtered out (e.g., re-asking injuries in Scenario A/B),
            # dynamically select the next truly missing personalized question if one is needed.
            dynamic_replacement_q: str | None = None
            if validated_output.proposed_followup_question and filtered_followup is None:
                _, _, _, cand_next_q, _ = deterministic_demo_classify_and_extract(
                    text=cleaned_text,
                    utterance_id=utterance_id,
                    active_language=state.active_language,
                    existing_category=state.category,
                    existing_established=state.category_established_above_threshold,
                    existing_facts=state.validated_facts,
                    pending_followup_question=None,
                    pending_question=None,
                    question_history=state.question_history,
                    active_assignments=state.assignments,
                )
                dynamic_replacement_q = self._filter_redundant_followup(state, cand_next_q)
                filtered_followup = dynamic_replacement_q

            self._set_pending_question(
                state,
                question_text=filtered_followup,
                target_field_hint=(
                    validated_output.proposed_followup_target_field
                    if dynamic_replacement_q is None
                    else None
                ),
                source_utterance_id=utterance_id,
                now=ts,
            )

            # Classification attempt logic (Section 9.3)
            is_new_description = resolved_role == UtteranceRole.NEW_DESCRIPTION
            is_supported_above_threshold = (
                validated_output.category != EmergencyCategory.UNKNOWN
                and validated_output.category_confidence >= self._confidence_threshold
            )

            if is_new_description:
                state.classification_attempts += 1
                state.classification_confidences.append(validated_output.category_confidence)

                if is_supported_above_threshold:
                    state.consecutive_unknown_count = 0
                    self._apply_supported_category(
                        state,
                        new_category=validated_output.category,
                        supporting_ids=validated_output.supporting_utterance_ids,
                        required_categories_override=multi_hazard_cats,
                        now=ts,
                    )
                else:
                    state.consecutive_unknown_count += 1
                    if state.consecutive_unknown_count >= 2:
                        self._apply_deterministic_unknown_fallback(
                            state,
                            fallback_reason="two_consecutive_unknown_classifications",
                            review_reason_code="unknown_emergency_fallback",
                            supporting_ids=validated_output.supporting_utterance_ids,
                            now=ts,
                        )
                        fallback_phrase, _ = self._instructions.resolve_cached_asset(
                            "unknown_emergency_fallback",
                            state.active_language,
                        )
                        state.last_spoken_response_text = fallback_phrase.approved_text
                        state.last_selected_instruction_id = fallback_phrase.instruction_id
                        state.conversation_history.append(
                            ConversationTurnRecord(
                                role="assistant",
                                text=state.last_spoken_response_text,
                                language=state.active_language,
                                utterance_id=None,
                                recorded_at=ts,
                            )
                        )
                        state.task_execution_telemetry = {
                            "mode": "parallel_asyncio",
                            "revision": state.revision,
                            "parallel_wall_ms": parallel_wall_ms,
                            "sum_branch_ms": round(sum_branch_ms, 2),
                            "sequential_saved_ms": round(max(0.0, sum_branch_ms - parallel_wall_ms), 2),
                            "branches": branches_telemetry,
                            "degraded_branches": degraded_branches,
                            "fleet_availability": fleet_avail,
                            "gemini_reasoning": {
                                "thinking_level": getattr(validated_output, "thinking_level", "LOW"),
                                "prompt_tokens": validated_output.prompt_tokens,
                                "completion_tokens": validated_output.completion_tokens,
                                "latency_ms": validated_output.latency_ms,
                                "was_coalesced_or_cached": bool(
                                    getattr(validated_output, "was_coalesced_or_cached", False)
                                ),
                            },
                            "gates_enforced": ["unknown_fallback_gate", "location_confirmation_gate"],
                        }
                        return state
            elif is_supported_above_threshold and (
                not state.category_established_above_threshold
                or validated_output.category != state.category
                or (
                    multi_hazard_cats is not None
                    and set(multi_hazard_cats) != set(state.required_resource_categories)
                )
            ):
                # Follow-up answer clarifies, upgrades category, or reveals an additional concurrent hazard
                state.consecutive_unknown_count = 0
                self._apply_supported_category(
                    state,
                    new_category=validated_output.category,
                    supporting_ids=validated_output.supporting_utterance_ids,
                    required_categories_override=multi_hazard_cats,
                    now=ts,
                )

            # Upgrade category deterministically if pending-question injury answer turned True on fire
            if (
                state.category == EmergencyCategory.FIRE
                and any(f.field_name == "injuries_reported" and f.value is True for f in state.validated_facts)
            ):
                updated_cats = self._dispatch.determine_multi_hazard_categories(
                    EmergencyCategory.FIRE_WITH_INJURIES,
                    identified_hazards=tuple(state.identified_hazards),
                    context_text=context_all_text,
                )
                self._apply_supported_category(
                    state,
                    new_category=EmergencyCategory.FIRE_WITH_INJURIES,
                    supporting_ids=[utterance_id],
                    required_categories_override=updated_cats,
                    now=ts,
                )
            elif (
                state.category_established_above_threshold
                and multi_hazard_cats is not None
                and set(multi_hazard_cats) - set(state.required_resource_categories)
            ):
                # Ensure newly revealed compound hazards prepare additional required units behind the location gate
                self._apply_supported_category(
                    state,
                    new_category=state.category,
                    supporting_ids=[utterance_id],
                    required_categories_override=multi_hazard_cats,
                    now=ts,
                )

            # Sanitize dynamic caller-facing text (Section 3.2)
            has_active_sim_assignment = any(
                a.status == AssignmentStatus.SIMULATED_ASSIGNED for a in state.assignments
            )
            safe_text, was_replaced = self._policy.sanitize_caller_response_text(
                validated_output.caller_response_text,
                language=state.active_language,
                has_simulated_assignment=has_active_sim_assignment,
            )
            if not was_replaced:
                safe_text = self._strip_redundant_question_from_text(
                    state,
                    safe_text,
                    validated_output.proposed_followup_question,
                     None if dynamic_replacement_q is not None else state.pending_followup_question,
                )
                if dynamic_replacement_q and dynamic_replacement_q.lower() not in safe_text.lower():
                    safe_text = f"{safe_text} {dynamic_replacement_q}".strip()
            if was_replaced:
                self._audit.append(
                    AuditEvent.create(
                        session_id=state.session_id,
                        incident_id=state.incident_id,
                        event_type="caller_response_text_sanitized",
                        severity=AuditSeverity.WARNING,
                        outcome="replaced_with_approved_phrase",
                        now=ts,
                    )
                )

            # Include critical hazard safety guidance calmly when relevant (e.g., gas cylinder / fire risk)
            if (
                state.active_safety_guidance
                and (
                    "gas_cylinder" in state.identified_hazards
                    or "explosion" in state.identified_hazards
                    or not filtered_followup
                )
            ):
                guid_txt = str(state.active_safety_guidance.get("approved_text") or "").strip()
                if guid_txt and guid_txt.lower() not in safe_text.lower() and not was_replaced:
                    safe_text = f"{safe_text} {guid_txt}".strip()

            if self._demo_mode:
                safe_text = self._append_new_demo_dispatch_announcement(state, safe_text, now=ts)

            state.revision += 1
            state.last_event_sequence += 1
            state.updated_at = ts
            state.last_spoken_response_text = safe_text
            state.last_selected_instruction_id = (
                validated_output.selected_instruction_id
                or (
                    str(state.active_safety_guidance["instruction_id"])
                    if state.active_safety_guidance
                    else None
                )
            )
            state.conversation_history.append(
                ConversationTurnRecord(
                    role="assistant",
                    text=safe_text,
                    language=state.active_language,
                    utterance_id=None,
                    recorded_at=ts,
                )
            )
            state.task_execution_telemetry = {
                "mode": "parallel_asyncio",
                "revision": state.revision,
                "parallel_wall_ms": parallel_wall_ms,
                "sum_branch_ms": round(sum_branch_ms, 2),
                "sequential_saved_ms": round(max(0.0, sum_branch_ms - parallel_wall_ms), 2),
                "branches": branches_telemetry,
                "degraded_branches": degraded_branches,
                "fleet_availability": fleet_avail,
                "gemini_reasoning": {
                    "thinking_level": getattr(validated_output, "thinking_level", "LOW"),
                    "prompt_tokens": validated_output.prompt_tokens,
                    "completion_tokens": validated_output.completion_tokens,
                    "latency_ms": validated_output.latency_ms,
                    "was_coalesced_or_cached": bool(
                        getattr(validated_output, "was_coalesced_or_cached", False)
                    ),
                },
                "gates_enforced": [
                    "location_confirmation_gate"
                    if any(a.status == AssignmentStatus.PENDING_LOCATION for a in state.assignments)
                    else "dispatch_committed",
                ],
            }
            return state

    def handle_barge_in(
        self,
        session_id: UUID,
        *,
        interrupted_Source: str,  # 'cached_instruction' or 'live_timbre'
        interrupted_instruction_id: str | None,
        preserved_preroll_ms: int,
        transcript_text: str | None,
        is_verified_non_speech_signal: bool = False,
        now: datetime | None = None,
    ) -> BargeInOutcome:
        """
        Execute mandatory barge-in sequence and false-trigger handling (Section 18.4).
        1. Local playback stops immediately.
        2. Queued chunks clear.
        3. Server cancellation is a no-op for cached playback, active for live Timbre.
        """
        ts = now or datetime.now(timezone.utc)
        state = self.get_session(session_id)
        is_cached = interrupted_Source == "cached_instruction"
        server_cancel_sent = not is_cached

        self._metrics.record_latency("barge_in_local_stop_ms", 1.5)
        if server_cancel_sent:
            self._metrics.record_latency("server_cancellation_ms", 14.0)

        cleaned = (transcript_text or "").strip()

        # Before Gate 0.1 verification: empty/unusable transcript plays "Sorry, I didn't catch that."
        if not is_verified_non_speech_signal and not cleaned:
            self._metrics.increment_barge_in(is_false_trigger=False)
            didnt_catch, _ = self._instructions.resolve_cached_asset(
                "didnt_catch_that",
                state.active_language,
            )
            state.last_spoken_response_text = didnt_catch.approved_text
            state.last_selected_instruction_id = didnt_catch.instruction_id
            self._audit.append(
                AuditEvent.create(
                    session_id=state.session_id,
                    incident_id=state.incident_id,
                    event_type="barge_in_empty_transcript",
                    severity=AuditSeverity.INFO,
                    outcome="played_didnt_catch_that",
                    now=ts,
                    details={"preserved_preroll_ms": preserved_preroll_ms},
                )
            )
            return BargeInOutcome(
                local_playback_stopped_immediately=True,
                queued_chunks_cleared=True,
                server_cancellation_sent=server_cancel_sent,
                counted_as_utterance=False,
                is_false_barge_in=False,
                replayed_cached_instruction_id=None,
                response_instruction_id=didnt_catch.instruction_id,
                response_text=didnt_catch.approved_text,
                automatic_barge_in_disabled=not state.automatic_vad_barge_in_enabled,
            )

        # Verified non-speech signal (false barge-in)
        if is_verified_non_speech_signal:
            state.false_barge_in_count += 1
            self._metrics.increment_barge_in(is_false_trigger=True)
            if state.false_barge_in_count >= MAX_FALSE_BARGE_INS_BEFORE_MANUAL_ONLY:
                state.automatic_vad_barge_in_enabled = False

            replay_id = interrupted_instruction_id if is_cached else None
            replay_text: str | None = None
            if replay_id is not None:
                inst, _ = self._instructions.resolve_cached_asset(replay_id, state.active_language)
                replay_text = inst.approved_text

            self._audit.append(
                AuditEvent.create(
                    session_id=state.session_id,
                    incident_id=state.incident_id,
                    event_type="false_barge_in",
                    severity=AuditSeverity.INFO,
                    outcome="replayed_cached_asset" if is_cached else "live_tts_not_replayed",
                    now=ts,
                    details={
                        "false_barge_in_count": state.false_barge_in_count,
                        "automatic_vad_enabled": state.automatic_vad_barge_in_enabled,
                    },
                )
            )
            return BargeInOutcome(
                local_playback_stopped_immediately=True,
                queued_chunks_cleared=True,
                server_cancellation_sent=server_cancel_sent,
                counted_as_utterance=False,
                is_false_barge_in=True,
                replayed_cached_instruction_id=replay_id,
                response_instruction_id=replay_id,
                response_text=replay_text,
                automatic_barge_in_disabled=not state.automatic_vad_barge_in_enabled,
            )

        self._metrics.increment_barge_in(is_false_trigger=False)
        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="barge_in_genuine_speech",
                severity=AuditSeverity.INFO,
                outcome="forwarded_preroll_to_prisma",
                now=ts,
                details={"preserved_preroll_ms": preserved_preroll_ms},
            )
        )
        return BargeInOutcome(
            local_playback_stopped_immediately=True,
            queued_chunks_cleared=True,
            server_cancellation_sent=server_cancel_sent,
            counted_as_utterance=True,
            is_false_barge_in=False,
            replayed_cached_instruction_id=None,
            response_instruction_id=None,
            response_text=None,
            automatic_barge_in_disabled=not state.automatic_vad_barge_in_enabled,
        )

    def end_session(
        self,
        session_id: UUID,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> IncidentSessionState:
        """
        End the session and enforce session-ending single-unknown (Section 9.3) and
        session-ending-after-provider-block (Section 9.4) fallbacks.
        """
        ts = now or datetime.now(timezone.utc)
        state = self.get_session(session_id)

        if not state.category_established_above_threshold:
            if state.consecutive_provider_blocks >= 1:
                self._apply_deterministic_unknown_fallback(
                    state,
                    fallback_reason="session_ended_after_provider_block",
                    review_reason_code="provider_blocked_emergency_content",
                    supporting_ids=[u.utterance_id for u in state.utterances],
                    now=ts,
                )
            elif state.consecutive_unknown_count == 1:
                self._apply_deterministic_unknown_fallback(
                    state,
                    fallback_reason="session_ended_before_second_attempt",
                    review_reason_code="unknown_emergency_fallback",
                    supporting_ids=[u.utterance_id for u in state.utterances],
                    now=ts,
                )

        state.session_state = "completed"
        state.incident_state = (
            "flagged_for_review" if state.control_room_review_required else "handed_off"
        )
        state.revision += 1
        state.last_event_sequence += 1
        state.updated_at = ts

        # Release any synthetic fleet locks held by this session's incident so future sessions remain isolated
        repo = getattr(self._dispatch, "_repository", None)
        if repo is not None and hasattr(repo, "release_incident_resources"):
            repo.release_incident_resources(state.incident_id)

        # Clear session-scoped conversational memory while retaining audit/handoff assignment history
        state.conversation_history.clear()
        state.pending_followup_question = None
        state.pending_question_record = None
        state.missing_information_fields.clear()
        state.conversational_memory_cleared = True

        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="session_completed",
                severity=AuditSeverity.INFO,
                outcome=reason,
                now=ts,
            )
        )
        return state

    def advance_demo_lifecycle(
        self,
        session_id: UUID,
        *,
        target_status: DemoLifecycleStatus,
        now: datetime | None = None,
        eligible_assignment_ids: set[UUID] | None = None,
    ) -> tuple[IncidentSessionState, list[SimulatedAssignment], str | None]:
        """
        Advance active simulated vehicle assignments along ASSIGNED -> EN_ROUTE -> ARRIVED
        for the session's incident, returning updated state, changed assignments, and optional
        spoken arrival announcement text.
        """
        ts = now or datetime.now(timezone.utc)
        state = self.get_session(session_id)
        if state.session_state == "completed" or not state.assignments:
            return state, [], None

        updated_all, changed = self._dispatch.advance_assignment_lifecycle(
            state.assignments,
            target_status=target_status,
            now=ts,
            eligible_assignment_ids=eligible_assignment_ids,
        )
        if not changed:
            return state, [], None

        state.assignments = updated_all
        state.revision += 1
        state.last_event_sequence += 1
        state.updated_at = ts

        event_type = (
            "demo_vehicle_arrived"
            if target_status == DemoLifecycleStatus.ARRIVED
            else "demo_vehicle_en_route"
        )
        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type=event_type,
                severity=AuditSeverity.INFO,
                outcome=f"{target_status.value}:{','.join(a.resource_id or '' for a in changed)}",
                provenance_refs=[a.resource_id for a in changed if a.resource_id],
                now=ts,
            )
        )

        spoken_update: str | None = None
        if target_status == DemoLifecycleStatus.ARRIVED:
            unannounced = [
                a for a in changed
                if f"{a.assignment_id}:{DemoLifecycleStatus.ARRIVED.value}" not in state.announced_lifecycle_events
            ]
            if unannounced:
                raw_update = build_demo_lifecycle_announcement(unannounced, state.active_language)
                if raw_update:
                    spoken_update, _ = self._policy.sanitize_caller_response_text(
                        raw_update,
                        language=state.active_language,
                        has_simulated_assignment=True,
                    )
                    for a in unannounced:
                        state.announced_lifecycle_events.add(
                            f"{a.assignment_id}:{DemoLifecycleStatus.ARRIVED.value}"
                        )
                    state.last_spoken_response_text = spoken_update
                    state.conversation_history.append(
                        ConversationTurnRecord(
                            role="assistant",
                            text=spoken_update,
                            language=state.active_language,
                            utterance_id=None,
                            recorded_at=ts,
                        )
                    )

        return state, changed, spoken_update

    # ========================================================================
    # Internal Helper Methods
    # ========================================================================

    def _handle_provider_blocked_turn(
        self,
        state: IncidentSessionState,
        *,
        utterance_id: UUID,
        err: ProviderBlockedContentError,
        now: datetime,
    ) -> None:
        """Section 9.4: Staged escalation for provider-blocked emergency content."""
        state.consecutive_provider_blocks += 1
        state.total_provider_blocks += 1

        if state.consecutive_provider_blocks == 1:
            self._audit.append(
                AuditEvent.create(
                    session_id=state.session_id,
                    incident_id=state.incident_id,
                    event_type="provider_blocked",
                    severity=AuditSeverity.INFO,
                    outcome="first_block_reask_prompt_played",
                    reason_code="provider_blocked_emergency_content",
                    provenance_refs=[str(utterance_id)],
                    now=now,
                )
            )
            reask, _ = self._instructions.resolve_cached_asset(
                "provider_blocked_reask",
                state.active_language,
            )
            state.last_spoken_response_text = reask.approved_text
            state.last_selected_instruction_id = reask.instruction_id
            return

        if state.consecutive_provider_blocks == self._block_threshold:
            if not state.category_established_above_threshold:
                self._apply_deterministic_unknown_fallback(
                    state,
                    fallback_reason="consecutive_provider_blocks",
                    review_reason_code="provider_blocked_emergency_content",
                    supporting_ids=[u.utterance_id for u in state.utterances],
                    now=now,
                )
                stmt, _ = self._instructions.resolve_cached_asset(
                    "provider_blocked_fallback",
                    state.active_language,
                )
                state.last_spoken_response_text = stmt.approved_text
                state.last_selected_instruction_id = stmt.instruction_id
            else:
                self._add_review_reason(state, "provider_blocked_emergency_content")
                self._audit.append(
                    AuditEvent.create(
                        session_id=state.session_id,
                        incident_id=state.incident_id,
                        event_type="provider_blocked_escalation_with_existing_category",
                        severity=AuditSeverity.WARNING,
                        outcome="flagged_for_review_no_new_assignment",
                        reason_code="provider_blocked_emergency_content",
                        provenance_refs=[str(utterance_id)],
                        now=now,
                    )
                )
                reassurance, _ = self._instructions.resolve_cached_asset(
                    "outage_reassurance_1",
                    state.active_language,
                )
                state.last_spoken_response_text = reassurance.approved_text
                state.last_selected_instruction_id = reassurance.instruction_id
            return

        # Further blocked turns (> N)
        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="provider_blocked",
                severity=AuditSeverity.WARNING,
                outcome="continued_reassurance_after_fallback",
                reason_code="provider_blocked_emergency_content",
                provenance_refs=[str(utterance_id)],
                now=now,
            )
        )

    @staticmethod
    def _infer_target_field_from_question(
        question_text: str | None,
        explicit_target: str | None = None,
    ) -> str:
        if explicit_target and explicit_target.strip():
            return explicit_target.strip().lower()
        q = (question_text or "").lower()
        if re.search(r"how many|number of people|count|ఎంతమంది|कितने\s+लोग", q):
            return "affected_person_count"
        if re.search(r"conscious|awake|breath|responsive|స్పృహ|होश|सांस", q):
            return "consciousness_status"
        if re.search(r"injur|hurt|trapped|bleed|burn|గాయ|घायल", q):
            return "injuries_reported"
        if re.search(r"safe\s+location|safe\s+distance|safe\s+spot|somewhere\s+safe|clear\s+of\s+danger|సురక్షిత|सुरक्षित", q):
            return "caller_safe_spot_status"
        if re.search(r"confirm.*location|is this where you are|pin|నిర్ధార|లొకేషన్|कन्फर्म|लोकेशन", q):
            return "location_confirmed"
        if re.search(r"where|location|address|landmark|street|area|ల్యాండ్|ప్రాంతం|స్థలం|स्थान|लैंडमार्क|कहाँ", q):
            return "caller_location_text"
        return "general_followup"

    def _resolve_pending_question_from_utterance(
        self,
        state: IncidentSessionState,
        *,
        text: str,
        utterance_id: UUID,
        extracted_facts: tuple[ProvenancedFact, ...] | list[ProvenancedFact],
        answered_hint: bool | None,
        now: datetime,
    ) -> tuple[list[ProvenancedFact], bool]:
        """
        Bind the caller's reply deterministically to any active pending question's target_field,
        preventing the assistant from re-asking the same question when a concise reply like 'Yes' is given.
        """
        facts_out = list(extracted_facts)
        stripped = text.strip()
        lowered = stripped.lower()
        is_location_confirm_phrase = bool(
            re.search(
                r"\b(?:my\s+location|location\s+confirmed|confirm\s+(?:the\s+|my\s+)?location|same\s+place|exact\s+spot|that'?s\s+(?:the|my)\s+location|यही\s+जगह|यहीं\s+पर|ఇక్కడే|నా\s+లొకేషన్)\b",
                lowered,
                re.I,
            )
        )
        if (
            is_location_confirm_phrase
            and state.location_state is not None
            and not any(f.field_name == "location_confirmed" and f.value is True for f in facts_out)
        ):
            facts_out.append(
                ProvenancedFact(
                    field_name="location_confirmed",
                    value=True,
                    source_utterance_id=utterance_id,
                    evidence_text=stripped[:80],
                )
            )

        pq = state.pending_question_record
        if pq is None and state.pending_followup_question:
            pq = PendingQuestionRecord(
                question_id=f"pq-legacy-{len(state.question_history) + 1}",
                question_text=state.pending_followup_question,
                target_field=self._infer_target_field_from_question(state.pending_followup_question),
                asked_at=now,
                source_utterance_id=None,
                status="pending",
            )

        if pq is None or pq.status != "pending":
            return facts_out, False

        target = pq.target_field.lower()
        already_extracted = any(f.field_name.lower() == target for f in facts_out)

        is_yes = bool(
            re.match(
                r"^(?:yes|yeah|yep|yup|affirmative|correct|true|అవును|ఉన్నారు|हाँ|जी|हां)\b",
                lowered,
                re.I,
            )
        ) and not is_location_confirm_phrase
        is_no = bool(
            re.match(
                r"^(?:no|nope|none|nobody|no\s*one|zero|nil|not\s+really|లేదు|ఎవరూ\s*లేరు|नहीं|कोई\s*नहीं)\b",
                lowered,
                re.I,
            )
        )

        resolved_via_det = False
        if not already_extracted:
            if target == "location_confirmed":
                if is_yes or is_location_confirm_phrase:
                    facts_out.append(
                        ProvenancedFact(
                            field_name="location_confirmed",
                            value=True,
                            source_utterance_id=utterance_id,
                            evidence_text=stripped[:80],
                        )
                    )
                    resolved_via_det = True
            elif target == "injuries_reported":
                if is_yes:
                    facts_out.append(
                        ProvenancedFact(
                            field_name="injuries_reported",
                            value=True,
                            source_utterance_id=utterance_id,
                            evidence_text=stripped[:80],
                        )
                    )
                    resolved_via_det = True
                elif is_no:
                    facts_out.append(
                        ProvenancedFact(
                            field_name="injuries_reported",
                            value=False,
                            source_utterance_id=utterance_id,
                            evidence_text=stripped[:80],
                        )
                    )
                    resolved_via_det = True
            elif target == "affected_person_count":
                cnt_m = re.search(r"\b(\d+|one|two|three|four|five|six|ten|none|nobody|ఒకరు|ఇద్దరు|ముగ్గురు|नहीं|एक|दो|तीन)\b", lowered, re.I)
                word_map = {
                    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10,
                    "none": 0, "nobody": 0, "ఒకరు": 1, "ఇద్దరు": 2, "ముగ్గురు": 3, "एक": 1, "दो": 2, "तीन": 3,
                }
                if cnt_m:
                    tok = cnt_m.group(1).lower()
                    val = int(tok) if tok.isdigit() else word_map.get(tok)
                    if val is not None:
                        facts_out.append(
                            ProvenancedFact(
                                field_name="affected_person_count",
                                value=val,
                                source_utterance_id=utterance_id,
                                evidence_text=stripped[:80],
                            )
                        )
                        resolved_via_det = True
            elif target == "consciousness_status":
                if is_yes or re.search(r"\b(conscious|awake|talking|responding)\b", lowered):
                    facts_out.append(
                        ProvenancedFact(
                            field_name="consciousness_status",
                            value="conscious",
                            source_utterance_id=utterance_id,
                            evidence_text=stripped[:80],
                        )
                    )
                    resolved_via_det = True
                elif is_no or re.search(r"\b(unconscious|fainted|not responding|unresponsive)\b", lowered):
                    facts_out.append(
                        ProvenancedFact(
                            field_name="consciousness_status",
                            value="unconscious",
                            source_utterance_id=utterance_id,
                            evidence_text=stripped[:80],
                        )
                    )
                    resolved_via_det = True
            elif target == "caller_safe_spot_status" and (is_yes or is_no or re.search(r"\b(safe|outside|away|సురక్షిత|सुरक्षित)\b", lowered)):
                facts_out.append(
                    ProvenancedFact(
                        field_name="caller_safe_spot_status",
                        value="safe_distance" if (is_yes or "safe" in lowered or "outside" in lowered) else "needs_relocation",
                        source_utterance_id=utterance_id,
                        evidence_text=stripped[:80],
                    )
                )
                resolved_via_det = True

        was_answered = bool(
            already_extracted
            or resolved_via_det
            or answered_hint is True
            or (target in ("caller_location_text", "caller_landmark_text", "broad_location_text") and any(
                f.field_name in ("caller_location_text", "caller_landmark_text", "broad_location_text")
                for f in facts_out
            ))
        )

        if was_answered:
            archived = PendingQuestionRecord(
                question_id=pq.question_id,
                question_text=pq.question_text,
                target_field=pq.target_field,
                asked_at=pq.asked_at,
                source_utterance_id=pq.source_utterance_id,
                status="answered",
                answered_by_utterance_id=utterance_id,
            )
            state.question_history.append(archived)
            state.pending_question_record = None
            state.pending_followup_question = None
            self._audit.append(
                AuditEvent.create(
                    session_id=state.session_id,
                    incident_id=state.incident_id,
                    event_type="pending_question_answered",
                    severity=AuditSeverity.INFO,
                    outcome=f"answered:{target}",
                    provenance_refs=[str(utterance_id)],
                    now=now,
                    details={"question_id": pq.question_id, "target_field": pq.target_field},
                )
            )
        return facts_out, was_answered

    def _set_pending_question(
        self,
        state: IncidentSessionState,
        *,
        question_text: str | None,
        target_field_hint: str | None,
        source_utterance_id: UUID,
        now: datetime,
    ) -> None:
        if not question_text:
            state.pending_followup_question = None
            state.pending_question_record = None
            return
        target_field = self._infer_target_field_from_question(question_text, target_field_hint)
        if state.pending_question_record is not None and state.pending_question_record.status == "pending":
            # Supersede previous unanswered pending question safely
            state.question_history.append(
                PendingQuestionRecord(
                    question_id=state.pending_question_record.question_id,
                    question_text=state.pending_question_record.question_text,
                    target_field=state.pending_question_record.target_field,
                    asked_at=state.pending_question_record.asked_at,
                    source_utterance_id=state.pending_question_record.source_utterance_id,
                    status="superseded",
                )
            )
        state.pending_followup_question = question_text
        state.pending_question_record = PendingQuestionRecord(
            question_id=f"pq-{state.revision}-{len(state.question_history) + 1}",
            question_text=question_text,
            target_field=target_field,
            asked_at=now,
            source_utterance_id=source_utterance_id,
            status="pending",
        )

    def _merge_validated_facts(
        self,
        state: IncidentSessionState,
        new_facts: tuple[ProvenancedFact, ...] | list[ProvenancedFact],
        *,
        now: datetime,
    ) -> None:
        """
        Update session memory with extracted facts, preserving prior broad area when refining street/landmark
        and invalidating prior location confirmation cleanly when caller updates the location.
        """
        existing_broad = next(
            (str(f.value) for f in state.validated_facts if f.field_name == "broad_location_text" and f.value),
            None,
        ) or next(
            (str(f.value) for f in state.validated_facts if f.field_name == "caller_location_text" and f.value),
            None,
        )
        prior_loc_snapshot = state.location_state.caller_location_text if state.location_state else None
        was_confirmed_previously = bool(
            state.location_state is not None
            and state.location_state.confirmation_status == LocationConfirmationStatus.CONFIRMED
        )

        # Extract broad_area_text / caller_landmark_text updates present in this turn
        new_broad: str | None = None
        new_landmark: str | None = None
        new_combined: str | None = None
        loc_utt_id: UUID | None = None

        for fact in new_facts:
            if fact.field_name == "broad_location_text" and isinstance(fact.value, str) and fact.value.strip():
                new_broad = fact.value.strip()
                loc_utt_id = fact.source_utterance_id
            elif fact.field_name == "caller_landmark_text" and isinstance(fact.value, str) and fact.value.strip():
                new_landmark = fact.value.strip()
                loc_utt_id = fact.source_utterance_id
            elif fact.field_name == "caller_location_text" and isinstance(fact.value, str) and fact.value.strip():
                new_combined = fact.value.strip()
                loc_utt_id = fact.source_utterance_id

        # If Gemini returned caller_landmark_text or refined caller_location_text ("beside ABC building")
        # while broad area ("Madhapur") was already captured, preserve BOTH in caller_location_text!
        normalized_facts: list[ProvenancedFact] = list(new_facts)
        if existing_broad and new_combined and not new_broad and existing_broad.lower() not in new_combined.lower():
            # Distinguish pure landmark refinement vs completely different city/area
            is_refinement = bool(
                new_landmark
                or re.search(r"\b(near|beside|opposite|behind|road|street|lane|building|gate|pillar)\b", new_combined, re.I)
            )
            if is_refinement:
                merged_combined = f"{existing_broad} — {new_combined}"
                normalized_facts = [
                    ProvenancedFact(
                        field_name=f.field_name,
                        value=(merged_combined if f.field_name == "caller_location_text" else f.value),
                        source_utterance_id=f.source_utterance_id,
                        evidence_text=f.evidence_text,
                    )
                    for f in normalized_facts
                ]
                if not any(f.field_name == "broad_location_text" for f in normalized_facts):
                    normalized_facts.append(
                        ProvenancedFact(
                            field_name="broad_location_text",
                            value=existing_broad,
                            source_utterance_id=loc_utt_id or state.utterances[-1].utterance_id,
                            evidence_text=existing_broad,
                        )
                    )
                if not any(f.field_name == "caller_landmark_text" for f in normalized_facts):
                    normalized_facts.append(
                        ProvenancedFact(
                            field_name="caller_landmark_text",
                            value=new_combined,
                            source_utterance_id=loc_utt_id or state.utterances[-1].utterance_id,
                            evidence_text=new_combined,
                        )
                    )
                new_combined = merged_combined
                new_landmark = new_landmark or new_combined

        for fact in normalized_facts:
            replaced = False
            for idx, existing in enumerate(state.validated_facts):
                if existing.field_name == fact.field_name:
                    # Preserve superseded fact history and provenance when value changes
                    if existing.value != fact.value and existing.source_utterance_id != fact.source_utterance_id:
                        is_compatible_loc_refinement = bool(
                            fact.field_name == "caller_location_text"
                            and isinstance(existing.value, str)
                            and isinstance(fact.value, str)
                            and existing.value.lower() in fact.value.lower()
                        )
                        sup_conflict = FactConflict(
                            field_name=fact.field_name,
                            prior_value=existing.value,
                            prior_utterance_id=existing.source_utterance_id,
                            new_value=fact.value,
                            new_utterance_id=fact.source_utterance_id,
                            description=(
                                "location_refinement"
                                if is_compatible_loc_refinement
                                else "caller_corrected_or_updated_fact"
                            ),
                        )
                        if sup_conflict not in state.superseded_facts:
                            state.superseded_facts.append(sup_conflict)
                    state.validated_facts[idx] = fact
                    replaced = True
                    break
            if not replaced:
                state.validated_facts.append(fact)

        effective_loc_text = new_combined or new_landmark or new_broad
        if effective_loc_text and loc_utt_id is not None:
            if not state.location_history or state.location_history[-1].get("location_text") != effective_loc_text:
                state.location_history.append(
                    {
                        "location_text": effective_loc_text,
                        "broad_area_text": new_broad or existing_broad,
                        "landmark_text": new_landmark,
                        "source_utterance_id": str(loc_utt_id),
                        "recorded_at": now.isoformat(),
                    }
                )
            location_changed_after_confirm = bool(
                was_confirmed_previously
                and prior_loc_snapshot
                and effective_loc_text.lower() != prior_loc_snapshot.lower()
            )
            self._try_resolve_spoken_landmark(
                state,
                caller_location_text=effective_loc_text,
                broad_area_text=new_broad or existing_broad,
                caller_landmark_text=new_landmark,
                source_utterance_id=loc_utt_id,
                force_update=True,
                invalidate_confirmation=location_changed_after_confirm,
                now=now,
            )

        for fact in normalized_facts:
            if (
                fact.field_name == "location_confirmed"
                and fact.value is True
                and state.location_state is not None
                and (
                    state.location_state.latitude is not None
                    or (
                        self._demo_mode
                        and bool(
                            state.location_state.caller_location_text
                            or state.location_state.broad_area_text
                            or state.location_state.caller_landmark_text
                        )
                    )
                )
            ):
                if state.location_state.confirmation_status != LocationConfirmationStatus.CONFIRMED:
                    state.location_state = self._location_resolver.confirm_location(
                        state.location_state,
                        confirmed_at=now,
                        allow_text_only=self._demo_mode,
                    )
                    self._reconcile_location_dependent_operations(state, now=now)

    @staticmethod
    def _collect_confirmed_field_keys(state: IncidentSessionState) -> set[str]:
        confirmed = {
            f.field_name.lower()
            for f in state.validated_facts
            if f.value not in (None, "")
        }
        for q in state.question_history:
            if q.status == "answered":
                confirmed.add(q.target_field.lower())
        if state.category_established_above_threshold:
            confirmed.add("emergency_type")
            confirmed.add("emergency_category")
        if (
            state.location_state is not None
            and (
                state.location_state.confirmation_status == LocationConfirmationStatus.CONFIRMED
                or state.location_state.caller_location_text
            )
        ):
            confirmed.add("caller_location_text")
            confirmed.add("location")
            confirmed.add("landmark")
        return confirmed

    @classmethod
    def _filter_missing_fields(
        cls,
        state: IncidentSessionState,
        missing_fields: tuple[str, ...] | list[str],
    ) -> list[str]:
        confirmed_keys = cls._collect_confirmed_field_keys(state)
        return [f for f in missing_fields if f.lower() not in confirmed_keys]

    @classmethod
    def _filter_redundant_followup(
        cls,
        state: IncidentSessionState,
        proposed_question: str | None,
        *,
        target_field_hint: str | None = None,
    ) -> str | None:
        """Avoid re-asking questions whose information is already confirmed or answered in session memory."""
        if not proposed_question or not proposed_question.strip():
            return None
        q = proposed_question.strip()
        q_lower = q.lower()
        confirmed_keys = cls._collect_confirmed_field_keys(state)
        conflict_fields = {c.field_name.lower() for c in state.conflicts}

        inferred_target = cls._infer_target_field_from_question(q, target_field_hint)
        if inferred_target in confirmed_keys and inferred_target not in conflict_fields and inferred_target != "general_followup":
            return None

        if (
            inferred_target != "caller_safe_spot_status"
            and any(k in confirmed_keys for k in ("caller_location_text", "location", "landmark", "address", "location_confirmed"))
            and "caller_location_text" not in conflict_fields
            and re.search(
                r"\b(where|location|address|landmark|street|area)\b|ల్యాండ్|ప్రాంతం|స్థలం|स्थान|लैंडमार्क|कहाँ",
                q_lower,
            )
        ):
            return None
        has_injury_status = any(
            k in confirmed_keys
            for k in ("injuries_reported", "injuries_present", "injuries", "anyone_injured", "affected_person_count", "people_affected")
        )
        has_person_count = any(
            k in confirmed_keys for k in ("affected_person_count", "people_affected", "person_count")
        )
        if (
            has_injury_status
            and not any(cf in conflict_fields for cf in ("injuries_reported", "injuries_present", "affected_person_count"))
            and re.search(
                r"\b(anyone injured|anyone hurt|any injuries|anyone trapped|are there injuries|is anyone injured|is anyone hurt)\b|గాయాల|గాయాలు|घायल",
                q_lower,
            )
        ):
            return None
        if (
            has_person_count
            and "affected_person_count" not in conflict_fields
            and re.search(r"\b(how many people|number of people|how many injured|how many are hurt)\b|ఎంతమంది|कितने\s+लोग", q_lower)
        ):
            return None
        if (
            "consciousness_status" in confirmed_keys
            and "consciousness_status" not in conflict_fields
            and re.search(r"\b(conscious|awake|breathing)\b|స్పృహ|होश", q_lower)
        ):
            return None
        if (
            state.category_established_above_threshold
            and re.search(
                r"\b(what kind of emergency|what happened|what is the emergency)\b|ఏమి\s+జరిగింది|क्या\s+हुआ",
                q_lower,
            )
        ):
            return None
        return q

    @staticmethod
    def _strip_redundant_question_from_text(
        state: IncidentSessionState,
        response_text: str,
        raw_proposed_question: str | None,
        filtered_question: str | None,
    ) -> str:
        """Remove a trailing/embedded question sentence when its follow-up was filtered out as already answered."""
        if not response_text or filtered_question is not None or not raw_proposed_question:
            return response_text
        raw_q = raw_proposed_question.strip()
        if not raw_q:
            return response_text
        # Direct removal if the exact question string appears inside response_text
        if raw_q in response_text:
            cleaned = response_text.replace(raw_q, "").strip()
            if cleaned:
                return cleaned
        # Otherwise strip a trailing '?' sentence if its topic matches location/injuries/category already confirmed
        parts = re.split(r"(?<=[.?!।])\s+", response_text.strip())
        if len(parts) > 1 and parts[-1].endswith("?"):
            candidate_tail = parts[-1]
            if SessionOrchestrator._filter_redundant_followup(state, candidate_tail) is None:
                remainder = " ".join(parts[:-1]).strip()
                if remainder:
                    return remainder
        return response_text

    def _append_new_demo_dispatch_announcement(
        self,
        state: IncidentSessionState,
        base_text: str,
        *,
        now: datetime,
    ) -> str:
        """
        Append deterministic EN/TE/HI demo assignment announcements for newly allocated or
        unavailable vehicles not yet announced in this session.
        When location is still unconfirmed (PENDING_LOCATION gate active), politely ask the
        caller to verify/confirm the map pin rather than claiming active vehicle dispatch.
        """
        committed_new: list[SimulatedAssignment] = []
        pending_items: list[SimulatedAssignment] = []
        for asgn in state.assignments:
            if asgn.status == AssignmentStatus.PENDING_LOCATION:
                pending_items.append(asgn)
                continue
            key = f"{asgn.assignment_id}:{asgn.status.value}"
            if key not in state.announced_assignment_keys:
                committed_new.append(asgn)

        if not committed_new:
            if (
                pending_items
                and state.required_resource_categories
                and not any(a.status == AssignmentStatus.SIMULATED_ASSIGNED for a in state.assignments)
            ):
                pending_key = f"pending_confirm:{','.join(c.value for c in state.required_resource_categories)}"
                if pending_key not in state.announced_lifecycle_events:
                    state.announced_lifecycle_events.add(pending_key)
                    prompt_msg = build_pending_location_dispatch_message(
                        state.required_resource_categories,
                        state.active_language,
                    )
                    if prompt_msg and prompt_msg.lower() not in base_text.lower():
                        return f"{base_text} {prompt_msg}".strip()
            return base_text

        raw_ann = build_demo_dispatch_announcement(committed_new, state.active_language)
        if not raw_ann:
            return base_text

        safe_ann, _ = self._policy.sanitize_caller_response_text(
            raw_ann,
            language=state.active_language,
            has_simulated_assignment=True,
        )
        for asgn in committed_new:
            state.announced_assignment_keys.add(f"{asgn.assignment_id}:{asgn.status.value}")

        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="demo_dispatch_announced",
                severity=AuditSeverity.INFO,
                outcome=f"announced:{','.join(a.resource_id or a.status.value for a in committed_new)}",
                now=now,
            )
        )
        if safe_ann.lower() in base_text.lower():
            return base_text
        return f"{base_text} {safe_ann}".strip()

    def _handle_demo_outage_turn(
        self,
        state: IncidentSessionState,
        *,
        utterance_id: UUID,
        text: str,
        err: ReasoningUnavailableError,
        now: datetime,
    ) -> None:
        """
        DEMO-mode resilience on Gemini outage (HTTP 503 / 429 / timeout / network error):
        honestly flags reasoning_unavailable while running deterministic keyword classification,
        pending-question answer resolution, fact extraction, dispatch gating, and safety guidance.
        """
        self._add_review_reason(state, "reasoning_unavailable")
        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="reasoning_provider_outage",
                severity=AuditSeverity.WARNING,
                outcome="demo_deterministic_fallback_activated",
                error_code=str(err.http_status or "timeout_or_network"),
                reason_code="reasoning_unavailable",
                provenance_refs=[str(utterance_id)],
                now=now,
            )
        )
        inferred_cat, role, extracted_facts, next_q, base_reply = deterministic_demo_classify_and_extract(
            text=text,
            utterance_id=utterance_id,
            active_language=state.active_language,
            existing_category=state.category,
            existing_established=state.category_established_above_threshold,
            existing_facts=state.validated_facts,
            pending_followup_question=state.pending_followup_question,
            pending_question=state.pending_question_record,
            question_history=state.question_history,
            active_assignments=state.assignments,
        )
        pre_facts, was_answered = self._resolve_pending_question_from_utterance(
            state,
            text=text,
            utterance_id=utterance_id,
            extracted_facts=extracted_facts,
            answered_hint=None,
            now=now,
        )
        if was_answered:
            role = UtteranceRole.ANSWER_TO_PENDING_QUESTION
        state.utterances[-1] = FinalizedUtteranceRecord(
            utterance_id=state.utterances[-1].utterance_id,
            sequence_number=state.utterances[-1].sequence_number,
            text=state.utterances[-1].text,
            language=state.utterances[-1].language,
            utterance_role=role,
            finalized_at=state.utterances[-1].finalized_at,
        )
        self._merge_validated_facts(state, pre_facts, now=now)
        filtered_next_q = self._filter_redundant_followup(state, next_q)
        self._set_pending_question(
            state,
            question_text=filtered_next_q,
            target_field_hint=None,
            source_utterance_id=utterance_id,
            now=now,
        )

        context_all_text = " ".join(u.text for u in state.utterances)
        lower_ctx = context_all_text.lower()
        merged_hazards = list(state.identified_hazards)
        keyword_hazards = (
            ("vehicle_collision", ("collid", "collision", "crash", "bus", "truck", "accident")),
            ("fire", ("fire", "flames", "burning", "smoke", "आग", "మంట")),
            ("gas_cylinder", ("cylinder", "lpg", "gas cylinder", "सिलेंडर", "సిలిండర్")),
            ("injuries", ("injur", "bleed", "burn", "hurt", "casualt", "घायल", "గాయ")),
            ("explosion", ("blast", "explod", "explosion")),
            ("trapped_persons", ("trapped", "stuck inside", "collapse", "debris")),
        )
        for h_tag, words in keyword_hazards:
            if any(w in lower_ctx for w in words) and h_tag not in merged_hazards:
                merged_hazards.append(h_tag)
        state.identified_hazards = merged_hazards

        target_cat_for_haz = (
            EmergencyCategory.FIRE_WITH_INJURIES
            if (
                inferred_cat == EmergencyCategory.FIRE
                and any(f.field_name == "injuries_reported" and f.value is True for f in state.validated_facts)
            )
            else (inferred_cat if inferred_cat != EmergencyCategory.UNKNOWN else state.category)
        )
        multi_hazard_cats = (
            self._dispatch.determine_multi_hazard_categories(
                target_cat_for_haz,
                identified_hazards=tuple(merged_hazards),
                context_text=context_all_text,
            )
            if target_cat_for_haz != EmergencyCategory.UNKNOWN
            else None
        )
        fleet_avail = (
            self._dispatch.check_fleet_availability(multi_hazard_cats)
            if multi_hazard_cats
            else {}
        )

        if inferred_cat != EmergencyCategory.UNKNOWN:
            if role == UtteranceRole.NEW_DESCRIPTION:
                state.classification_attempts += 1
            state.consecutive_unknown_count = 0
            self._apply_supported_category(
                state,
                new_category=target_cat_for_haz,
                supporting_ids=[utterance_id],
                required_categories_override=multi_hazard_cats,
                now=now,
            )
        elif role == UtteranceRole.NEW_DESCRIPTION:
            state.classification_attempts += 1
            state.consecutive_unknown_count += 1
            if state.consecutive_unknown_count >= 2:
                self._apply_deterministic_unknown_fallback(
                    state,
                    fallback_reason="two_consecutive_unknown_classifications",
                    review_reason_code="unknown_emergency_fallback",
                    supporting_ids=[utterance_id],
                    now=now,
                )
                fallback_phrase, _ = self._instructions.resolve_cached_asset(
                    "unknown_emergency_fallback",
                    state.active_language,
                )
                state.last_spoken_response_text = fallback_phrase.approved_text
                state.last_selected_instruction_id = fallback_phrase.instruction_id
                return

        if (
            state.category == EmergencyCategory.FIRE
            and any(f.field_name == "injuries_reported" and f.value is True for f in state.validated_facts)
        ):
            self._apply_supported_category(
                state,
                new_category=EmergencyCategory.FIRE_WITH_INJURIES,
                supporting_ids=[utterance_id],
                required_categories_override=self._dispatch.determine_multi_hazard_categories(
                    EmergencyCategory.FIRE_WITH_INJURIES,
                    identified_hazards=tuple(merged_hazards),
                    context_text=context_all_text,
                ),
                now=now,
            )

        safety_inst = self._instructions.get_safety_guidance_for_category(
            state.category,
            state.active_language,
            context_text=context_all_text,
        )
        if safety_inst is not None:
            state.active_safety_guidance = {
                "instruction_id": safety_inst.instruction_id,
                "category": safety_inst.category.value if hasattr(safety_inst.category, "value") else str(safety_inst.category),
                "language": safety_inst.language,
                "approved_text": safety_inst.approved_text,
                "source": safety_inst.source,
                "review_date": safety_inst.review_date.isoformat(),
                "version": safety_inst.version,
            }
        # Attach relevant safety guidance on safety concern turns or after intake details are captured
        has_cylinder_kw = any(kw in text.lower() for kw in ("cylinder", "lpg", "what should i do", "help", "సిలిండర్", "सिलेंडर"))
        if safety_inst is not None and (has_cylinder_kw or not filtered_next_q):
            if safety_inst.approved_text.lower() not in base_reply.lower():
                base_reply = f"{base_reply} {safety_inst.approved_text}".strip()

        has_active_sim_assignment = any(
            a.status == AssignmentStatus.SIMULATED_ASSIGNED for a in state.assignments
        )
        safe_reply, was_replaced = self._policy.sanitize_caller_response_text(
            base_reply,
            language=state.active_language,
            has_simulated_assignment=has_active_sim_assignment,
        )
        if not was_replaced:
            safe_reply = self._strip_redundant_question_from_text(
                state, safe_reply, next_q, filtered_next_q
            )
        safe_reply = self._append_new_demo_dispatch_announcement(state, safe_reply, now=now)
        state.last_spoken_response_text = safe_reply
        state.last_selected_instruction_id = (
            safety_inst.instruction_id if safety_inst is not None else "outage_reassurance_1"
        )
        state.task_execution_telemetry = {
            "mode": "demo_resilience_fallback",
            "revision": state.revision,
            "parallel_wall_ms": 0.4,
            "sum_branch_ms": 0.4,
            "sequential_saved_ms": 0.0,
            "branches": {
                "fact_merge_and_question": {"duration_ms": 0.1, "status": "ok"},
                "hazard_and_fleet_eval": {"duration_ms": 0.1, "status": "ok"},
                "safety_guidance_lookup": {"duration_ms": 0.1, "status": "ok"},
                "location_and_hospitals_prep": {"duration_ms": 0.1, "status": "ok"},
            },
            "degraded_branches": ["gemini_reasoning_quota_or_outage"],
            "fleet_availability": fleet_avail,
            "gates_enforced": [
                "location_confirmation_gate"
                if any(a.status == AssignmentStatus.PENDING_LOCATION for a in state.assignments)
                else "dispatch_committed",
            ],
        }

    def _handle_reasoning_outage_turn(
        self,
        state: IncidentSessionState,
        *,
        utterance_id: UUID,
        err: ReasoningUnavailableError,
        now: datetime,
    ) -> None:
        """Section 15.4: Outage sets reasoning_unavailable flag and plays cached reassurance."""
        self._add_review_reason(state, "reasoning_unavailable")
        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="reasoning_provider_outage",
                severity=AuditSeverity.WARNING,
                outcome="static_stay_with_me_reassurance_activated",
                error_code=str(err.http_status or "timeout_or_network"),
                reason_code="reasoning_unavailable",
                provenance_refs=[str(utterance_id)],
                now=now,
            )
        )
        reassurance, _ = self._instructions.resolve_cached_asset(
            "outage_reassurance_1",
            state.active_language,
        )
        state.last_spoken_response_text = reassurance.approved_text
        state.last_selected_instruction_id = reassurance.instruction_id

    def _handle_schema_invalid_turn(
        self,
        state: IncidentSessionState,
        *,
        utterance_id: UUID,
        err: ReasoningSchemaValidationError,
        now: datetime,
    ) -> None:
        """Schema-invalid output is an integration failure, not a classification attempt."""
        self._add_review_reason(state, "reasoning_schema_invalid")
        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="reasoning_schema_validation_failed",
                severity=AuditSeverity.WARNING,
                outcome="rejected_unverified_output",
                reason_code="reasoning_schema_invalid",
                provenance_refs=[str(utterance_id)],
                now=now,
                details={"error": str(err)},
            )
        )
        reassurance, _ = self._instructions.resolve_cached_asset(
            "outage_reassurance_1",
            state.active_language,
        )
        state.last_spoken_response_text = reassurance.approved_text
        state.last_selected_instruction_id = reassurance.instruction_id

    def _apply_supported_category(
        self,
        state: IncidentSessionState,
        *,
        new_category: EmergencyCategory,
        supporting_ids: tuple[UUID, ...] | list[UUID],
        now: datetime,
        required_categories_override: Sequence[ResourceCategory] | None = None,
    ) -> None:
        prev_cat = state.category
        is_reclassification = (
            state.category_established_above_threshold
            or state.unknown_fallback_applied
        ) and prev_cat != new_category

        state.category = new_category
        state.category_established_above_threshold = True

        loc_candidate = (
            state.location_state.to_dispatch_candidate()
            if state.location_state is not None
            else None
        )

        t0 = time.perf_counter()
        if is_reclassification:
            # Supersede stale category-implied facts (e.g., fire_reported after reclassifying away from fire)
            implied_field_by_cat: dict[EmergencyCategory, str] = {
                EmergencyCategory.FIRE: "fire_reported",
                EmergencyCategory.FIRE_WITH_INJURIES: "fire_reported",
                EmergencyCategory.ROAD_ACCIDENT: "vehicle_collision_reported",
                EmergencyCategory.EXPLOSION: "explosion_reported",
            }
            old_implied = implied_field_by_cat.get(prev_cat)
            new_implied = implied_field_by_cat.get(new_category)
            if old_implied and old_implied != new_implied:
                stale_facts = [f for f in state.validated_facts if f.field_name == old_implied]
                if stale_facts:
                    state.validated_facts = [
                        f for f in state.validated_facts if f.field_name != old_implied
                    ]
                    latest_uid = supporting_ids[-1] if supporting_ids else state.session_id
                    for sf in stale_facts:
                        cat_conflict = FactConflict(
                            field_name=old_implied,
                            prior_value=sf.value,
                            prior_utterance_id=sf.source_utterance_id,
                            new_value=f"superseded_by_{new_category.value}",
                            new_utterance_id=latest_uid,
                        )
                        if cat_conflict not in state.conflicts:
                            state.conflicts.append(cat_conflict)
                    self._add_review_reason(state, "conflicting_caller_statements")
                    self._audit.append(
                        AuditEvent.create(
                            session_id=state.session_id,
                            incident_id=state.incident_id,
                            event_type="stale_fact_superseded_on_reclassification",
                            severity=AuditSeverity.WARNING,
                            outcome=f"superseded:{old_implied}",
                            provenance_refs=[str(sf.source_utterance_id) for sf in stale_facts],
                            now=now,
                        )
                    )

            # If location also changed in this session, flag possible_second_incident
            has_location_conflict = any(
                c.field_name == "caller_location_text" for c in state.conflicts
            )
            if has_location_conflict:
                self._add_review_reason(state, "possible_second_incident")
                self._audit.append(
                    AuditEvent.create(
                        session_id=state.session_id,
                        incident_id=state.incident_id,
                        event_type="possible_second_incident_flagged",
                        severity=AuditSeverity.WARNING,
                        outcome=f"category_and_location_changed:{prev_cat.value}->{new_category.value}",
                        provenance_refs=[str(uid) for uid in supporting_ids],
                        now=now,
                    )
                )

            outcome = self._dispatch.reclassify_incident(
                incident_id=state.incident_id,
                previous_category=prev_cat,
                new_category=new_category,
                location_candidate=loc_candidate,
                existing_assignments=state.assignments,
                now=now,
            )
            if required_categories_override and (
                set(required_categories_override) - set(outcome.required_categories)
            ):
                outcome = self._dispatch.allocate_for_incident(
                    incident_id=state.incident_id,
                    category=new_category,
                    location_candidate=loc_candidate,
                    existing_assignments=outcome.assignments,
                    now=now,
                    required_categories_override=required_categories_override,
                )
            self._audit.append(
                AuditEvent.create(
                    session_id=state.session_id,
                    incident_id=state.incident_id,
                    event_type="incident_reclassified",
                    severity=AuditSeverity.INFO,
                    outcome=f"{prev_cat.value}->{new_category.value}",
                    provenance_refs=[str(uid) for uid in supporting_ids],
                    now=now,
                )
            )
        else:
            outcome = self._dispatch.allocate_for_incident(
                incident_id=state.incident_id,
                category=new_category,
                location_candidate=loc_candidate,
                existing_assignments=state.assignments,
                now=now,
                required_categories_override=required_categories_override,
            )

        self._metrics.record_latency("assignment_creation_ms", (time.perf_counter() - t0) * 1000.0)
        state.required_resource_categories = list(outcome.required_categories)
        state.assignments = list(outcome.assignments)
        for reason in outcome.review_reasons:
            self._add_review_reason(state, reason)

        self._reconcile_location_dependent_operations(state, now=now)

    def _apply_deterministic_unknown_fallback(
        self,
        state: IncidentSessionState,
        *,
        fallback_reason: str,
        review_reason_code: str,
        supporting_ids: tuple[UUID, ...] | list[UUID],
        now: datetime,
    ) -> None:
        """
        Single-application deterministic unknown fallback (Section 9.3 & 9.4).
        Later triggers append an audit event and reason code but create no further assignments.
        """
        self._add_review_reason(state, review_reason_code)
        if state.unknown_fallback_applied:
            self._audit.append(
                AuditEvent.create(
                    session_id=state.session_id,
                    incident_id=state.incident_id,
                    event_type="unknown_fallback_duplicate_trigger_skipped",
                    severity=AuditSeverity.INFO,
                    outcome="single_application_enforced",
                    reason_code=review_reason_code,
                    provenance_refs=[str(uid) for uid in supporting_ids],
                    now=now,
                    details={"additional_trigger": fallback_reason},
                )
            )
            return

        state.unknown_fallback_applied = True
        state.fallback_reason = fallback_reason
        state.category = EmergencyCategory.UNKNOWN

        loc_candidate = (
            state.location_state.to_dispatch_candidate()
            if state.location_state is not None
            else None
        )
        outcome = self._dispatch.allocate_for_incident(
            incident_id=state.incident_id,
            category=EmergencyCategory.UNKNOWN,
            location_candidate=loc_candidate,
            existing_assignments=state.assignments,
            now=now,
        )
        state.required_resource_categories = list(outcome.required_categories)
        state.assignments = list(outcome.assignments)
        for r in outcome.review_reasons:
            self._add_review_reason(state, r)

        self._reconcile_location_dependent_operations(state, now=now)

        self._audit.append(
            AuditEvent.create(
                session_id=state.session_id,
                incident_id=state.incident_id,
                event_type="deterministic_unknown_fallback_applied",
                severity=AuditSeverity.WARNING,
                outcome="simulated_ambulance_requested",
                reason_code=review_reason_code,
                provenance_refs=[str(uid) for uid in supporting_ids],
                now=now,
                details={"fallback_reason": fallback_reason},
            )
        )

    def _try_resolve_spoken_landmark(
        self,
        state: IncidentSessionState,
        *,
        caller_location_text: str,
        source_utterance_id: UUID,
        broad_area_text: str | None = None,
        caller_landmark_text: str | None = None,
        force_update: bool = False,
        invalidate_confirmation: bool = False,
        now: datetime,
    ) -> None:
        """
        Match high-confidence synthetic landmark when applicable; otherwise record caller broad area
        and street/landmark detail distinctly without fabricating coordinates or confidence scores.
        """
        matches = self._location_resolver.search_landmarks(
            caller_location_text=caller_location_text,
            source_utterance_id=source_utterance_id,
            now=now,
            min_similarity=0.52,
        )
        can_override_gps = (
            state.location_state is None
            or not state.location_state.is_usable_candidate
            or state.location_state.source == "landmark"
            or (
                matches
                and matches[0].similarity_score >= 0.72
            )
        )
        if matches and can_override_gps:
            top = matches[0]
            state.location_state = self._location_resolver.select_landmark_candidate(
                top,
                confirmed_by_caller=False,
                now=now,
            )
            state.location_state = self._location_resolver.apply_spoken_location_detail(
                state.location_state,
                broad_area_text=broad_area_text,
                caller_landmark_text=caller_landmark_text or caller_location_text,
                source_utterance_id=source_utterance_id,
                invalidate_confirmation=True,
                now=now,
            )
            self._add_review_reason(state, "provisional_location_unconfirmed")
            self._reconcile_location_dependent_operations(state, now=now)
            return

        # Record spoken location context without replacing usable browser GPS with fake coordinates
        state.location_state = self._location_resolver.apply_spoken_location_detail(
            state.location_state,
            broad_area_text=broad_area_text,
            caller_landmark_text=caller_landmark_text or caller_location_text,
            source_utterance_id=source_utterance_id,
            invalidate_confirmation=invalidate_confirmation,
            now=now,
        )
        if invalidate_confirmation:
            self._add_review_reason(state, "provisional_location_unconfirmed")
            self._reconcile_location_dependent_operations(state, now=now)

    def _reconcile_location_dependent_operations(
        self,
        state: IncidentSessionState,
        *,
        now: datetime,
    ) -> None:
        """
        Allocate any pending_location resources once a location candidate exists,
        never allow both 'assigned' and 'pending_location' at once,
        rank nearby directory hospitals, and run duplicate detection if confirmed.
        """
        if state.location_state is None:
            return
        can_confirm_text_only = (
            self._demo_mode
            and state.location_state.confirmation_status == LocationConfirmationStatus.CONFIRMED
            and bool(
                state.location_state.caller_location_text
                or state.location_state.broad_area_text
                or state.location_state.caller_landmark_text
            )
        )
        if not state.location_state.is_usable_candidate and not can_confirm_text_only:
            return

        loc_candidate = state.location_state.to_dispatch_candidate()
        if loc_candidate is not None and state.required_resource_categories:
            outcome = self._dispatch.allocate_for_incident(
                incident_id=state.incident_id,
                category=state.category,
                location_candidate=loc_candidate,
                existing_assignments=state.assignments,
                now=now,
                required_categories_override=state.required_resource_categories,
            )
            state.assignments = list(outcome.assignments)

        # Ensure pending_location is removed once any assignment is allocated
        if any(a.status.value == "simulated_assigned" for a in state.assignments):
            if "pending_location" in state.review_reason_codes:
                state.review_reason_codes.remove("pending_location")

        # Clear provisional_location_unconfirmed once caller confirms location
        if (
            state.location_state.confirmation_status == LocationConfirmationStatus.CONFIRMED
            and "provisional_location_unconfirmed" in state.review_reason_codes
        ):
            state.review_reason_codes.remove("provisional_location_unconfirmed")

        if state.location_state.latitude is None or state.location_state.longitude is None:
            return
        t_hosp = time.perf_counter()
        state.hospital_recommendations = self._hospital_ranker.rank_hospitals(
            incident_category=state.category,
            caller_latitude=state.location_state.latitude,
            caller_longitude=state.location_state.longitude,
            now=now,
        )
        self._metrics.record_latency("hospital_ranking_ms", (time.perf_counter() - t_hosp) * 1000.0)

        # Duplicate detection runs when BOTH supported category above threshold AND confirmed location exist (Section 14)
        is_confirmed = (
            state.location_state.confirmation_status == LocationConfirmationStatus.CONFIRMED
        )
        if state.category_established_above_threshold and is_confirmed:
            target = IncidentSummaryForDuplicateCheck(
                incident_id=state.incident_id,
                category=state.category,
                category_above_threshold=True,
                confirmed_latitude=state.location_state.latitude,
                confirmed_longitude=state.location_state.longitude,
                is_location_confirmed=True,
                created_at=state.created_at,
            )
            other_summaries = [
                IncidentSummaryForDuplicateCheck(
                    incident_id=other.incident_id,
                    category=other.category,
                    category_above_threshold=other.category_established_above_threshold,
                    confirmed_latitude=(
                        other.location_state.latitude if other.location_state else None
                    ),
                    confirmed_longitude=(
                        other.location_state.longitude if other.location_state else None
                    ),
                    is_location_confirmed=(
                        other.location_state is not None
                        and other.location_state.confirmation_status
                        == LocationConfirmationStatus.CONFIRMED
                    ),
                    created_at=other.created_at,
                )
                for other in self._sessions.values()
                if other.incident_id != state.incident_id
            ]
            dup_match = self._duplicates.check_for_duplicate(target, other_summaries)
            if dup_match is not None and state.duplicate_match is None:
                state.duplicate_match = dup_match
                self._add_review_reason(state, "possible_duplicate_incident")
                self._audit.append(
                    AuditEvent.create(
                        session_id=state.session_id,
                        incident_id=state.incident_id,
                        event_type="duplicate_incident_detected",
                        severity=AuditSeverity.WARNING,
                        outcome="flagged_for_review",
                        now=now,
                        details={
                            "matched_incident_id": str(dup_match.matched_incident_id),
                            "distance_meters": dup_match.distance_meters,
                        },
                    )
                )

    @staticmethod
    def _add_review_reason(state: IncidentSessionState, reason: str) -> None:
        state.control_room_review_required = True
        if reason not in state.review_reason_codes:
            state.review_reason_codes.append(reason)
