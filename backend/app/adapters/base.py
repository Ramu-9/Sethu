"""
Application-owned adapter interfaces, domain structures, provenance models,
and typed exceptions for SETU.

Satisfies Sections 3, 5 (Rule 3), 9.1–9.4, 15.4, 18.4, and 31.3.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol
from uuid import UUID


SupportedLanguage = Literal["en", "te", "hi"]


class EmergencyCategory(str, Enum):
    """Fixed emergency classification taxonomy per Section 9."""
    MEDICAL_EMERGENCY = "medical_emergency"
    FIRE = "fire"
    FIRE_WITH_INJURIES = "fire_with_injuries"
    ROAD_ACCIDENT = "road_accident"
    BUILDING_COLLAPSE = "building_collapse"
    PHYSICAL_ATTACK = "physical_attack"
    GAS_LEAK = "gas_leak"
    EXPLOSION = "explosion"
    UNKNOWN = "unknown"


class UtteranceRole(str, Enum):
    """Required utterance role classification per Section 9.2."""
    NEW_DESCRIPTION = "new_description"
    ANSWER_TO_PENDING_QUESTION = "answer_to_pending_question"
    ACKNOWLEDGEMENT = "acknowledgement"
    OTHER = "other"


class STTSignalType(str, Enum):
    """Typed signal emitted by Prisma STT, shaped per Gate 0.1 verification."""
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    NON_SPEECH_OR_EMPTY = "non_speech_or_empty"


class BackendSourceType(str, Enum):
    """Conceptual provenance source types per Section 5 (Rule 3)."""
    CALLER_UTTERANCE = "caller_utterance"
    BROWSER_GPS = "browser_gps"
    DATABASE_RECORD = "database_record"
    PROVIDER_RESPONSE = "provider_response"
    BACKEND_RULE = "backend_rule"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class BackendProvenance:
    """
    Application-owned provenance for backend-generated facts (Rule 3).
    Never attaches fabricated caller utterance IDs to backend facts.
    """
    source_type: BackendSourceType
    source_id: str
    source_timestamp: datetime
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")


@dataclass(frozen=True)
class AudioChunk:
    """16 kHz, 16-bit mono PCM audio frame."""
    pcm_bytes: bytes
    sample_rate_hz: int
    sequence_number: int
    captured_at: datetime

    def __post_init__(self) -> None:
        if not self.pcm_bytes:
            raise ValueError("pcm_bytes must not be empty")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")


@dataclass(frozen=True)
class TranscriptEvent:
    """Normalized event yielded from STTAdapter streaming."""
    session_id: UUID
    utterance_id: UUID
    signal_type: STTSignalType
    text: str | None
    detected_language: SupportedLanguage | None
    language_confidence: float | None
    transcript_confidence: float | None
    provider_timestamp: datetime


@dataclass(frozen=True)
class ProvenancedFact:
    """
    Single LLM-extracted incident fact with mandatory caller utterance provenance (Rule 3).
    """
    field_name: str
    value: bool | int | str | None
    source_utterance_id: UUID
    evidence_text: str

    def __post_init__(self) -> None:
        if not self.field_name or not self.field_name.strip():
            raise ValueError("field_name must be non-empty")
        if not self.evidence_text or not self.evidence_text.strip():
            raise ValueError("evidence_text must be non-empty for LLM-extracted facts")


@dataclass(frozen=True)
class FactConflict:
    """Conflicting statements preserved with both source utterance references (Rule 3)."""
    field_name: str
    prior_value: bool | int | str
    prior_utterance_id: UUID
    new_value: bool | int | str
    new_utterance_id: UUID
    description: str = ""


@dataclass(frozen=True)
class ConversationTurnRecord:
    """Single conversational turn (caller utterance or assistant response) in session memory."""
    role: Literal["caller", "assistant"]
    text: str
    language: SupportedLanguage | None
    utterance_id: UUID | None
    recorded_at: datetime


@dataclass(frozen=True)
class FinalizedUtteranceRecord:
    """Immutable record of a finalized caller utterance for dialogue context."""
    utterance_id: UUID
    sequence_number: int
    text: str
    language: SupportedLanguage | None
    utterance_role: UtteranceRole | None
    finalized_at: datetime


@dataclass(frozen=True)
class PendingQuestionRecord:
    """Structured state for an active or resolved follow-up question in session memory."""
    question_id: str
    question_text: str
    target_field: str
    asked_at: datetime
    source_utterance_id: UUID | None = None
    status: Literal["pending", "answered", "clarified", "superseded"] = "pending"
    answered_by_utterance_id: UUID | None = None


@dataclass(frozen=True)
class DialogueContext:
    """Full dialogue context required for every classification and extraction request (Section 9.2)."""
    session_id: UUID
    incident_id: UUID
    active_language: SupportedLanguage
    prior_utterances: Sequence[FinalizedUtteranceRecord]
    validated_facts: Sequence[ProvenancedFact]
    current_utterance: FinalizedUtteranceRecord
    pending_followup_question: str | None
    eligible_instruction_ids: Sequence[str]
    backend_status_summary: Mapping[str, str | int | None]
    conversation_history: Sequence[ConversationTurnRecord] = ()
    pending_question: PendingQuestionRecord | None = None
    identified_hazards: Sequence[str] = ()
    active_safety_guidance_id: str | None = None

    def known_utterance_ids(self) -> set[UUID]:
        """Return the set of all finalized utterance IDs in this dialogue context."""
        ids = {u.utterance_id for u in self.prior_utterances}
        ids.add(self.current_utterance.utterance_id)
        return ids


# ============================================================================
# Typed Adapter Exceptions (Distinct Failure Modes & Reason Codes)
# ============================================================================

class AdapterError(Exception):
    """Base class for all external provider adapter failures."""
    reason_code: str = "adapter_error"


class ProviderBlockedContentError(AdapterError):
    """
    Raised when the reasoning provider blocks, filters, truncates, or empties
    a response via safety systems (Section 9.4).
    Maps strictly to reason_code = 'provider_blocked_emergency_content'.
    Must NEVER be treated as an 'unknown' classification or an outage.
    """
    reason_code: str = "provider_blocked_emergency_content"

    def __init__(
        self,
        message: str,
        *,
        session_id: UUID,
        utterance_id: UUID,
        provider_finish_reason: str | None = None,
        safety_ratings: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.utterance_id = utterance_id
        self.provider_finish_reason = provider_finish_reason
        self.safety_ratings = dict(safety_ratings) if safety_ratings else None


class ReasoningUnavailableError(AdapterError):
    """
    Raised on reasoning provider outage, timeout, disconnect, authentication
    failure, rate limit (HTTP 429), or quota exhaustion (Section 15.4).
    Maps strictly to reason_code = 'reasoning_unavailable'.
    """
    reason_code: str = "reasoning_unavailable"

    def __init__(
        self,
        message: str,
        *,
        session_id: UUID,
        utterance_id: UUID,
        is_retryable: bool,
        http_status: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.utterance_id = utterance_id
        self.is_retryable = is_retryable
        self.http_status = http_status
        self.retry_after_seconds = retry_after_seconds


class ReasoningSchemaValidationError(AdapterError):
    """
    Raised when the reasoning provider returns malformed JSON, missing
    utterance_role, or invalid provenance references.
    Treated as an integration failure, not a classification attempt.
    """
    reason_code: str = "reasoning_schema_invalid"

    def __init__(
        self,
        message: str,
        *,
        session_id: UUID | None = None,
        utterance_id: UUID | None = None,
    ) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.utterance_id = utterance_id


class STTProviderError(AdapterError):
    """Raised on Prisma unavailability, stream failure, auth error, or timeout."""
    reason_code: str = "stt_provider_unavailable"


class TTSProviderError(AdapterError):
    """Raised on Timbre unavailability, stream failure, auth error, or timeout."""
    reason_code: str = "tts_provider_unavailable"


@dataclass(frozen=True)
class ReasoningOutput:
    """Strict schema-validated output from the Gemini reasoning model."""
    utterance_role: UtteranceRole
    utterance_role_source_id: UUID
    category: EmergencyCategory
    category_confidence: float
    supporting_utterance_ids: Sequence[UUID]
    extracted_facts: Sequence[ProvenancedFact]
    conflicts: Sequence[FactConflict]
    missing_information_fields: Sequence[str]
    proposed_followup_question: str | None
    selected_instruction_id: str | None
    caller_response_text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    pending_question_answered: bool | None = None
    proposed_followup_target_field: str | None = None
    identified_hazards: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not (0.0 <= self.category_confidence <= 1.0):
            raise ValueError("category_confidence must be between 0.0 and 1.0")

    def validate_against_context(self, context: DialogueContext) -> None:
        """
        Verify that all provenance references exist in the DialogueContext and
        that selected_instruction_id (if present) is in eligible_instruction_ids.

        Inputs:
            context: The DialogueContext submitted for this reasoning turn.
        Outputs:
            None.
        Failure modes:
            Raises ReasoningSchemaValidationError if any utterance ID or instruction ID is invalid.
        """
        known_ids = context.known_utterance_ids()
        if self.utterance_role_source_id != context.current_utterance.utterance_id:
            raise ReasoningSchemaValidationError(
                "utterance_role_source_id must match current_utterance.utterance_id",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
            )
        for sup_id in self.supporting_utterance_ids:
            if sup_id not in known_ids:
                raise ReasoningSchemaValidationError(
                    f"Supporting utterance ID {sup_id} does not exist in dialogue context",
                    session_id=context.session_id,
                    utterance_id=context.current_utterance.utterance_id,
                )
        for fact in self.extracted_facts:
            if fact.source_utterance_id not in known_ids:
                raise ReasoningSchemaValidationError(
                    f"Fact '{fact.field_name}' references unknown utterance ID {fact.source_utterance_id}",
                    session_id=context.session_id,
                    utterance_id=context.current_utterance.utterance_id,
                )
        for conflict in self.conflicts:
            if conflict.prior_utterance_id not in known_ids or conflict.new_utterance_id not in known_ids:
                raise ReasoningSchemaValidationError(
                    f"Conflict on '{conflict.field_name}' references unknown utterance ID",
                    session_id=context.session_id,
                    utterance_id=context.current_utterance.utterance_id,
                )
        if (
            self.selected_instruction_id is not None
            and self.selected_instruction_id not in context.eligible_instruction_ids
        ):
            raise ReasoningSchemaValidationError(
                f"Selected instruction ID '{self.selected_instruction_id}' is not in eligible set",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
            )


@dataclass(frozen=True)
class SynthesisAudioChunk:
    """Audio chunk emitted by Timbre TTS synthesis."""
    audio_bytes: bytes
    audio_format: str
    sample_rate_hz: int
    sequence_number: int
    is_first_byte: bool
    is_final_chunk: bool


# ============================================================================
# Application-Owned Adapter Interfaces
# ============================================================================

class STTStreamSession(Protocol):
    """Active streaming recognition handle coordinated per caller session."""

    @property
    def session_id(self) -> UUID: ...

    @property
    def is_active(self) -> bool: ...

    async def send_audio_chunk(
        self,
        chunk: AudioChunk,
        *,
        timeout_seconds: float,
    ) -> None:
        """
        Send a single 16 kHz mono PCM frame to the active Prisma stream.

        Inputs:
            chunk: Validated 16 kHz 16-bit mono PCM audio frame.
            timeout_seconds: Explicit write deadline in seconds.
        Outputs:
            None.
        Failure modes:
            Raises STTProviderError on stream disconnect, write timeout, or unverified format.
            Raises asyncio.CancelledError if cancelled during barge-in or session teardown.
        """
        ...

    def receive_events(
        self,
        *,
        idle_timeout_seconds: float,
    ) -> AsyncIterator[TranscriptEvent]:
        """
        Yield partial transcripts, finalized transcripts, and verified non-speech/empty signals.

        Inputs:
            idle_timeout_seconds: Maximum wait between stream frames before timeout.
        Outputs:
            AsyncIterator of TranscriptEvent objects.
        Failure modes:
            Raises STTProviderError on provider disconnect, invalid payload, or timeout.
            Raises asyncio.CancelledError on stream cancellation.
        """
        ...

    async def finalize_utterance(
        self,
        *,
        timeout_seconds: float,
    ) -> TranscriptEvent:
        """
        Signal end-of-speech to Prisma and await the finalized transcript or non-speech signal.

        Inputs:
            timeout_seconds: Explicit deadline to receive the final event.
        Outputs:
            Final TranscriptEvent (FINAL_TRANSCRIPT or NON_SPEECH_OR_EMPTY).
        Failure modes:
            Raises STTProviderError on timeout or broken stream without fabricating transcript text.
        """
        ...

    async def close(self, *, timeout_seconds: float) -> None:
        """
        Cleanly close and tear down the Prisma stream session (idempotent and cancellation-safe).

        Inputs:
            timeout_seconds: Explicit deadline for provider stream teardown confirmation.
        Outputs:
            None.
        Failure modes:
            Logs and raises STTProviderError if remote teardown fails beyond deadline.
        """
        ...


class STTAdapter(ABC):
    """Application-owned interface for Gnani Prisma speech-to-text."""

    @abstractmethod
    async def open_stream(
        self,
        *,
        session_id: UUID,
        utterance_id: UUID,
        language_hint: SupportedLanguage,
        timeout_seconds: float,
    ) -> STTStreamSession:
        """
        Open a new streaming STT session for the given caller session and utterance.

        Inputs:
            session_id: Authoritative PostgreSQL session UUID.
            utterance_id: Backend-generated UUID for the incoming utterance.
            language_hint: Active session language ('en', 'te', or 'hi').
            timeout_seconds: Explicit connection deadline in seconds.
        Outputs:
            An active STTStreamSession handle.
        Failure modes:
            Raises STTProviderError on auth failure, quota/concurrency rejection, or timeout.
            Raises asyncio.CancelledError if cancelled before stream establishment.
        """
        ...


class ReasoningAdapter(ABC):
    """Application-owned interface for the Google Gemini classification and extraction model."""

    @abstractmethod
    async def analyze_utterance(
        self,
        context: DialogueContext,
        *,
        timeout_seconds: float,
    ) -> ReasoningOutput:
        """
        Classify the emergency and extract provenanced facts using full dialogue context.

        Executes at most one bounded retry (250 ms default delay or Retry-After within deadline)
        for transient HTTP 408/429/5xx or connection errors only. Never retries safety blocks,
        auth errors, or schema validation failures.

        Inputs:
            context: Full DialogueContext containing prior finalized utterances, validated facts,
                     the current finalized utterance, pending follow-up question, and eligible
                     instruction IDs. Partial transcripts must never be passed.
            timeout_seconds: Explicit total deadline (including any single transient retry).
        Outputs:
            Validated ReasoningOutput with mandatory utterance_role and source_utterance_id provenance.
        Failure modes:
            Raises ProviderBlockedContentError (reason_code='provider_blocked_emergency_content')
                if Gemini blocks, filters, truncates, or returns an empty response due to safety filters.
            Raises ReasoningUnavailableError (reason_code='reasoning_unavailable')
                on timeout, network outage, auth failure, rate limit, or quota exhaustion.
            Raises ReasoningSchemaValidationError (reason_code='reasoning_schema_invalid')
                if the structured output violates the schema or omits required provenance.
            Raises asyncio.CancelledError on caller cancellation/teardown.
        """
        ...


class TTSStreamHandle(Protocol):
    """Active streaming TTS synthesis handle supporting immediate barge-in cancellation."""

    @property
    def session_id(self) -> UUID: ...

    @property
    def synthesis_id(self) -> UUID: ...

    @property
    def is_active(self) -> bool: ...

    def iter_audio_chunks(
        self,
        *,
        first_byte_timeout_seconds: float,
        chunk_timeout_seconds: float,
    ) -> AsyncIterator[SynthesisAudioChunk]:
        """
        Yield synthesized audio chunks and record exact TTFAB on the first byte.

        Inputs:
            first_byte_timeout_seconds: Explicit deadline for the first audio byte (TTFAB).
            chunk_timeout_seconds: Explicit deadline between subsequent streaming chunks.
        Outputs:
            AsyncIterator of SynthesisAudioChunk objects.
        Failure modes:
            Raises TTSProviderError on TTFAB timeout, mid-stream disconnect, or invalid audio payload.
            Raises asyncio.CancelledError when cancelled via cancel() during caller barge-in.
        """
        ...

    async def cancel(self, *, timeout_seconds: float) -> None:
        """
        Abort active server-side Timbre generation and confirm stream teardown (idempotent).

        Called during barge-in strictly AFTER local client playback has already stopped.

        Inputs:
            timeout_seconds: Explicit deadline to confirm provider stream teardown.
        Outputs:
            None.
        Failure modes:
            Raises TTSProviderError if teardown confirmation times out or fails.
        """
        ...


class TTSAdapter(ABC):
    """Application-owned interface for Gnani Timbre text-to-speech."""

    @abstractmethod
    async def start_synthesis(
        self,
        *,
        session_id: UUID,
        synthesis_id: UUID,
        text: str,
        language: SupportedLanguage,
        voice_id: str,
        timeout_seconds: float,
    ) -> TTSStreamHandle:
        """
        Initiate plain-text Timbre v2.5 speech synthesis for validated caller-facing text.

        Never called on a cache hit for pre-synthesized safety or holding instructions.

        Inputs:
            session_id: Authoritative PostgreSQL session UUID.
            synthesis_id: Backend-generated UUID for this synthesis operation.
            text: Policy-validated plain text (no unsupported v3.0 expressive tags).
            language: Target synthesis language ('en', 'te', or 'hi').
            voice_id: Verified Timbre voice identifier for the language.
            timeout_seconds: Explicit connection initiation deadline.
        Outputs:
            An active TTSStreamHandle for streaming consumption and barge-in cancellation.
        Failure modes:
            Raises TTSProviderError on auth error, concurrency violation, unsupported voice, or timeout.
            Raises asyncio.CancelledError if cancelled during initiation.
        """
        ...
