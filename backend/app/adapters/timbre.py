"""
Gnani Timbre v2.5 Text-to-Speech adapter and streaming handle (Sections 3.3, 4.2, 18.4, and Gate 0.2).

Enforces:
- Section 4.2: Default compatibility target is plain-text Timbre v2.5. Never sends unverified
  v3.0 expressive tags (<pauses>, <slow>, <whispers>, <excited>, <sigh>).
- Verified Vachana SSE synthesis contract (POST https://api.vachana.ai/api/v1/tts/sse):
  - Header authentication via X-API-Key-ID
  - Request payload: {"text": ..., "model": "timbre-v2.5", "voice": "Pranav", "language": "en-IN"|"hi-IN"|"te-IN", "audio_config": {"container": "wav", "sample_rate": 16000}}
  - Parses SSE 'event: chunk' / 'data: {"chunk_index": ..., "audio": "<base64>"}'
- Section 18.4: Supports immediate barge-in cancellation via cancel(timeout_seconds=...) after
  local AudioBufferSourceNode playback has already stopped.
- Gate 0.2: Raises TTSProviderError when participant endpoint_url or api_key are not configured.
"""

import asyncio
import base64
from collections.abc import AsyncIterator, Callable, Coroutine
import json
import re
from typing import Any
import urllib.error
import urllib.request
from uuid import UUID

from backend.app.adapters.base import (
    SupportedLanguage,
    SynthesisAudioChunk,
    TTSAdapter,
    TTSProviderError,
    TTSStreamHandle,
)


_UNSUPPORTED_V3_TAGS_RE = re.compile(r"<(?:pauses|slow|whispers|excited|sigh)>", re.I)

TIMBRE_LANGUAGE_CODE_MAP: dict[SupportedLanguage, str] = {
    "en": "en-IN",
    "te": "te-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "ta": "ta-IN",
    "ml": "ml-IN",
    "bn": "bn-IN",
}

SUPPORTED_TIMBRE_V25_VOICES = frozenset(
    {
        "Nalini",
        "Bhavna",
        "Yashvi",
        "Urmila",
        "Jwala",
        "Chitra",
        "Ambuja",
        "Deepak",
        "Roopesh",
        "Vikrant",
        "Hemraj",
        "Jalaj",
        "Omkar",
        "Aarohi",
        "Bhavini",
        "Charvi",
        "Eishani",
        "Falguni",
        "Gauri",
        "Iravati",
        "Janaki",
        "Kamakshi",
        "Madhuri",
        "Radhika",
        "Shweta",
        "Tanvi",
        "Vidya",
        "Wamika",
        "Yamini",
        "Abhimanyu",
        "Chirag",
        "Deven",
        "Farhan",
        "Jatin",
        "Kartik",
        "Kaveri",
        "Trupti",
        "Devika",
        "Pranav",
        "Shlok",
        "Girish",
        "Asmita",
        "Trisha",
        "Brinda",
        "Vedika",
    }
)

DEFAULT_TIMBRE_V25_VOICE_BY_LANGUAGE: dict[SupportedLanguage, str] = {
    "en": "Pranav",
    "hi": "Pranav",
    "te": "Pranav",
    "kn": "Pranav",
    "ta": "Pranav",
    "ml": "Pranav",
    "bn": "Pranav",
}


def resolve_timbre_v25_voice(voice_id: str, language: SupportedLanguage) -> str:
    """Resolve a requested voice identifier to a verified Timbre v2.5 catalog voice."""
    if voice_id in SUPPORTED_TIMBRE_V25_VOICES:
        return voice_id
    return DEFAULT_TIMBRE_V25_VOICE_BY_LANGUAGE.get(language, "Pranav")


def strip_unverified_v3_expressive_tags(text: str) -> str:
    """Ensure plain-text v2.5 synthesis never speaks v3.0 expressive tags literally (Section 4.2)."""
    return _UNSUPPORTED_V3_TAGS_RE.sub("", text).strip()


class TimbreTTSStreamHandle(TTSStreamHandle):
    """Active streaming TTS synthesis handle for Gnani Timbre v2.5."""

    def __init__(
        self,
        *,
        session_id: UUID,
        synthesis_id: UUID,
        clean_text: str,
        language: SupportedLanguage,
        voice_id: str,
        endpoint_url: str,
        api_key: str,
        chunk_transport: (
            Callable[[str, SupportedLanguage, str, float], Coroutine[Any, Any, list[bytes]]]
            | None
        ) = None,
    ) -> None:
        self._session_id = session_id
        self._synthesis_id = synthesis_id
        self._clean_text = clean_text
        self._language = language
        self._voice_id = resolve_timbre_v25_voice(voice_id, language)
        self._endpoint_url = endpoint_url
        self._api_key = api_key
        self._chunk_transport = chunk_transport
        self._is_active = True
        self._cancelled = False

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
        if not self._is_active or self._cancelled:
            raise asyncio.CancelledError("Timbre synthesis stream was cancelled prior to iteration.")
        if first_byte_timeout_seconds <= 0 or chunk_timeout_seconds <= 0:
            raise TTSProviderError("Timbre timeout_seconds must be positive.")

        if self._chunk_transport is not None:
            raw_chunks = await asyncio.wait_for(
                self._chunk_transport(
                    self._clean_text,
                    self._language,
                    self._voice_id,
                    first_byte_timeout_seconds,
                ),
                timeout=first_byte_timeout_seconds,
            )
            if not raw_chunks:
                self._is_active = False
                raise TTSProviderError("Timbre v2.5 returned zero audio bytes.")
            total = len(raw_chunks)
            for idx, pcm_bytes in enumerate(raw_chunks):
                if self._cancelled:
                    self._is_active = False
                    raise asyncio.CancelledError("Timbre stream cancelled during caller barge-in.")
                if not pcm_bytes:
                    raise TTSProviderError("Timbre v2.5 emitted an empty audio chunk.")
                yield SynthesisAudioChunk(
                    audio_bytes=pcm_bytes,
                    audio_format="pcm_s16le",
                    sample_rate_hz=16000,
                    sequence_number=idx + 1,
                    is_first_byte=(idx == 0),
                    is_final_chunk=(idx == total - 1),
                )
            self._is_active = False
            return

        loop = asyncio.get_running_loop()
        chunk_queue: asyncio.Queue[object] = asyncio.Queue()
        worker_task = asyncio.create_task(
            asyncio.to_thread(
                self._sync_stream_sse_chunks,
                max(first_byte_timeout_seconds, chunk_timeout_seconds),
                chunk_queue,
                loop,
                strip_wav_header=True,
            )
        )
        emitted_count = 0
        try:
            while True:
                item_timeout = (
                    first_byte_timeout_seconds if emitted_count == 0 else chunk_timeout_seconds
                )
                try:
                    item = await asyncio.wait_for(chunk_queue.get(), timeout=item_timeout)
                except TimeoutError as exc:
                    self._cancelled = True
                    raise TTSProviderError(
                        "Timbre v2.5 synthesis timed out waiting for audio chunk."
                    ) from exc

                if self._cancelled:
                    self._is_active = False
                    raise asyncio.CancelledError("Timbre stream cancelled during caller barge-in.")
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                assert isinstance(item, bytes)
                if not item:
                    raise TTSProviderError("Timbre v2.5 emitted an empty audio chunk.")
                emitted_count += 1
                yield SynthesisAudioChunk(
                    audio_bytes=item,
                    audio_format="pcm_s16le",
                    sample_rate_hz=16000,
                    sequence_number=emitted_count,
                    is_first_byte=(emitted_count == 1),
                    is_final_chunk=False,
                )
            if emitted_count == 0:
                raise TTSProviderError("Timbre v2.5 returned zero audio bytes.")
        finally:
            self._is_active = False
            if not worker_task.done():
                self._cancelled = True
                try:
                    await asyncio.wait_for(worker_task, timeout=0.25)
                except Exception:
                    pass
            else:
                try:
                    await worker_task
                except Exception:
                    pass

    def _build_timbre_request(self) -> urllib.request.Request:
        body = {
            "text": self._clean_text,
            "model": "timbre-v2.5",
            "voice": self._voice_id,
            "language": TIMBRE_LANGUAGE_CODE_MAP.get(self._language, "en-IN"),
            "audio_config": {
                "container": "wav",
                "sample_rate": 16000,
            },
        }
        return urllib.request.Request(
            self._endpoint_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key-ID": self._api_key,
                "User-Agent": "setu-timbre-client/2.5",
            },
            method="POST",
        )

    @staticmethod
    def _decode_sse_audio_line(
        raw_line: str,
        *,
        strip_wav_header: bool,
        is_first_chunk: bool,
    ) -> bytes | None:
        stripped = raw_line.strip()
        if not stripped.startswith("data:"):
            return None
        data_json = stripped[5:].strip()
        if not data_json:
            return None
        try:
            event_obj = json.loads(data_json)
        except Exception:
            return None
        b64_audio = event_obj.get("audio") or event_obj.get("audio_chunk")
        if not b64_audio:
            return None
        chunk_bytes = base64.b64decode(b64_audio)
        if (
            strip_wav_header
            and is_first_chunk
            and len(chunk_bytes) > 44
            and chunk_bytes[:4] == b"RIFF"
        ):
            chunk_bytes = chunk_bytes[44:]
        return chunk_bytes or None

    def _sync_stream_sse_chunks(
        self,
        timeout_sec: float,
        chunk_queue: asyncio.Queue[object],
        loop: asyncio.AbstractEventLoop,
        *,
        strip_wav_header: bool = True,
    ) -> None:
        """Stream SSE chunks onto chunk_queue line-by-line immediately as they arrive."""
        req = self._build_timbre_request()
        first_chunk_pending = True
        try:
            with urllib.request.urlopen(req, timeout=max(1.0, timeout_sec)) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "text/event-stream" in ctype:
                    for raw_bytes in resp:
                        if self._cancelled:
                            break
                        line = raw_bytes.decode("utf-8", errors="ignore")
                        chunk_bytes = self._decode_sse_audio_line(
                            line,
                            strip_wav_header=strip_wav_header,
                            is_first_chunk=first_chunk_pending,
                        )
                        if chunk_bytes:
                            first_chunk_pending = False
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk_bytes)
                else:
                    raw_text = resp.read().decode("utf-8", errors="ignore")
                    if "event:" in raw_text or "data:" in raw_text:
                        for line in raw_text.splitlines():
                            chunk_bytes = self._decode_sse_audio_line(
                                line,
                                strip_wav_header=strip_wav_header,
                                is_first_chunk=first_chunk_pending,
                            )
                            if chunk_bytes:
                                first_chunk_pending = False
                                loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk_bytes)
                    else:
                        payload = json.loads(raw_text)
                        for c in (payload.get("audio_chunks_base64") or []):
                            if c:
                                chunk_bytes = base64.b64decode(c)
                                if (
                                    strip_wav_header
                                    and first_chunk_pending
                                    and len(chunk_bytes) > 44
                                    and chunk_bytes[:4] == b"RIFF"
                                ):
                                    chunk_bytes = chunk_bytes[44:]
                                if chunk_bytes:
                                    first_chunk_pending = False
                                    loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk_bytes)
        except Exception as exc:
            loop.call_soon_threadsafe(
                chunk_queue.put_nowait,
                TTSProviderError(f"Timbre v2.5 synthesis request failed: {exc}"),
            )
        finally:
            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

    def _sync_http_synthesize(
        self,
        timeout_sec: float,
        *,
        strip_wav_header: bool = True,
    ) -> list[bytes]:
        req = self._build_timbre_request()
        try:
            with urllib.request.urlopen(req, timeout=max(1.0, timeout_sec)) as resp:
                raw_text = resp.read().decode("utf-8", errors="ignore")
                ctype = (resp.headers.get("Content-Type") or "").lower()

                if "text/event-stream" in ctype or "event:" in raw_text or "data:" in raw_text:
                    decoded_chunks: list[bytes] = []
                    for line in raw_text.splitlines():
                        chunk_bytes = self._decode_sse_audio_line(
                            line,
                            strip_wav_header=strip_wav_header,
                            is_first_chunk=(not decoded_chunks),
                        )
                        if chunk_bytes:
                            decoded_chunks.append(chunk_bytes)
                    return decoded_chunks

                payload = json.loads(raw_text)
                b64_chunks = payload.get("audio_chunks_base64") or []
                return [base64.b64decode(c) for c in b64_chunks if c]
        except Exception as exc:
            raise TTSProviderError(f"Timbre v2.5 synthesis request failed: {exc}") from exc

    async def cancel(self, *, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise TTSProviderError("Cancel timeout_seconds must be positive.")
        self._cancelled = True
        self._is_active = False


class GnaniTimbreAdapter(TTSAdapter):
    """Gnani Timbre v2.5 plain-text Text-to-Speech adapter."""

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        chunk_transport: (
            Callable[[str, SupportedLanguage, str, float], Coroutine[Any, Any, list[bytes]]]
            | None
        ) = None,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._api_key = api_key
        self._chunk_transport = chunk_transport

    @property
    def is_configured(self) -> bool:
        return bool(self._endpoint_url and self._api_key)

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
        clean_text = strip_unverified_v3_expressive_tags(text)
        if not clean_text:
            raise TTSProviderError("Cannot synthesize empty text after v2.5 tag sanitization.")
        if not self._endpoint_url or not self._api_key:
            raise TTSProviderError(
                "Gate 0.2 unresolved: Gnani Timbre v2.5 endpoint and participant credentials "
                "are not configured."
            )
        if timeout_seconds <= 0:
            raise TTSProviderError("Timbre start_synthesis timeout_seconds must be positive.")
        return TimbreTTSStreamHandle(
            session_id=session_id,
            synthesis_id=synthesis_id,
            clean_text=clean_text,
            language=language,
            voice_id=voice_id,
            endpoint_url=self._endpoint_url,
            api_key=self._api_key,
            chunk_transport=self._chunk_transport,
        )

    def synthesize_pcm_sync(
        self,
        *,
        text: str,
        language: SupportedLanguage,
        voice_id: str = "Pranav",
        timeout_seconds: float = 20.0,
    ) -> bytes:
        """
        Synchronously synthesize plain text via Gnani Timbre v2.5 and return raw
        16 kHz, 16-bit signed little-endian mono PCM frames.
        """
        clean_text = strip_unverified_v3_expressive_tags(text)
        if not clean_text:
            raise TTSProviderError("Cannot synthesize empty text.")
        if not self._endpoint_url or not self._api_key:
            raise TTSProviderError("Gnani Timbre v2.5 endpoint and API key are required.")
        from uuid import uuid4

        handle = TimbreTTSStreamHandle(
            session_id=uuid4(),
            synthesis_id=uuid4(),
            clean_text=clean_text,
            language=language,
            voice_id=voice_id,
            endpoint_url=self._endpoint_url,
            api_key=self._api_key,
        )
        chunks = handle._sync_http_synthesize(timeout_seconds, strip_wav_header=True)
        if not chunks:
            raise TTSProviderError("Timbre v2.5 returned zero audio chunks.")
        return b"".join(chunks)
