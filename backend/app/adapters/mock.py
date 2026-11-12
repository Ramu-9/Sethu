"""
Test-only provider fakes for automated unit and integration testing (Section 7 & 27).

Never used as a silent substitute in production.
"""

from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from uuid import UUID

from backend.app.adapters.base import (
    AudioChunk,
    DialogueContext,
    ReasoningAdapter,
    ReasoningOutput,
    STTAdapter,
    STTSignalType,
    STTStreamSession,
    SupportedLanguage,
    SynthesisAudioChunk,
    TranscriptEvent,
    TTSAdapter,
    TTSStreamHandle,
)


class MockReasoningAdapter(ReasoningAdapter):
    """Deterministic scriptable ReasoningAdapter for automated tests (Section 28.1 & 28.2)."""

    def __init__(self) -> None:
        self._queue: deque[ReasoningOutput | Exception] = deque()
        self.received_contexts: list[DialogueContext] = []

    def enqueue_output(self, output: ReasoningOutput) -> None:
        self._queue.append(output)

    def enqueue_exception(self, exc: Exception) -> None:
        self._queue.append(exc)

    async def analyze_utterance(
        self,
        context: DialogueContext,
        *,
        timeout_seconds: float,
    ) -> ReasoningOutput:
        self.received_contexts.append(context)
        if not self._queue:
            raise RuntimeError("MockReasoningAdapter queue is empty; enqueue an output or exception in test.")
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item


class MockSTTStreamSession(STTStreamSession):
    """Test-only STT stream handle."""

    def __init__(self, session_id: UUID, utterance_id: UUID) -> None:
        self._session_id = session_id
        self._utterance_id = utterance_id
        self._is_active = True
        self.received_chunks: list[AudioChunk] = []

    @property
    def session_id(self) -> UUID:
        return self._session_id

    @property
    def is_active(self) -> bool:
        return self._is_active

    async def send_audio_chunk(self, chunk: AudioChunk, *, timeout_seconds: float) -> None:
        self.received_chunks.append(chunk)

    async def receive_events(self, *, idle_timeout_seconds: float) -> AsyncIterator[TranscriptEvent]:
        if False:
            yield  # pragma: no cover

    async def finalize_utterance(self, *, timeout_seconds: float) -> TranscriptEvent:
        return TranscriptEvent(
            session_id=self._session_id,
            utterance_id=self._utterance_id,
            signal_type=STTSignalType.FINAL_TRANSCRIPT,
            text="test finalized utterance",
            detected_language="en",
            language_confidence=0.99,
            transcript_confidence=0.95,
            provider_timestamp=datetime.now(timezone.utc),
        )

    async def close(self, *, timeout_seconds: float) -> None:
        self._is_active = False


class MockSTTAdapter(STTAdapter):
    """Test-only STTAdapter."""

    async def open_stream(
        self,
        *,
        session_id: UUID,
        utterance_id: UUID,
        language_hint: SupportedLanguage,
        timeout_seconds: float,
    ) -> STTStreamSession:
        return MockSTTStreamSession(session_id=session_id, utterance_id=utterance_id)


class MockTTSStreamHandle(TTSStreamHandle):
    """Test-only TTS stream handle."""

    def __init__(self, session_id: UUID, synthesis_id: UUID) -> None:
        self._session_id = session_id
        self._synthesis_id = synthesis_id
        self._is_active = True
        self.cancelled = False

    @property
    def session_id(self) -> UUID:
        return self._session_id

    @property
    def synthesis_id(self) -> UUID:
        return self._synthesis_id

    @property
    def is_active(self) -> bool:
        return self._is_active

    async def iter_audio_chunks(
        self,
        *,
        first_byte_timeout_seconds: float,
        chunk_timeout_seconds: float,
    ) -> AsyncIterator[SynthesisAudioChunk]:
        yield SynthesisAudioChunk(
            audio_bytes=b"\x00\x00" * 160,
            audio_format="pcm_s16le",
            sample_rate_hz=16000,
            sequence_number=0,
            is_first_byte=True,
            is_final_chunk=True,
        )

    async def cancel(self, *, timeout_seconds: float) -> None:
        self.cancelled = True
        self._is_active = False


class MockTTSAdapter(TTSAdapter):
    """Test-only TTSAdapter."""

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
        return MockTTSStreamHandle(session_id=session_id, synthesis_id=synthesis_id)
