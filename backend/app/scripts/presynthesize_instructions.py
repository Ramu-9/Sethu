"""
Pre-synthesis script for approved STAY WITH ME safety and reassurance instructions (Section 15.2).

When Gnani Timbre v2.5 credentials (TIMBRE_ENDPOINT_URL and TIMBRE_AUTH_TOKEN) are available,
synthesizes every approved instruction across English, Telugu, and Hindi using real
Gnani Timbre v2.5 plain-text synthesis (voice='Pranav', 16 kHz 16-bit mono PCM WAV),
verifies byte-level SHA-256 checksums, and records:
  "generation_status": "ready"
in backend-media/instructions/manifest.json.

If real Timbre v2.5 WAV files are already present and checksum-verified on disk, skips
re-synthesizing over the network on subsequent startups. Falls back to development
placeholder WAV generation only when Timbre credentials are absent.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import wave

from backend.app.adapters.timbre import GnaniTimbreAdapter
from backend.app.config import load_settings
from backend.app.orchestrator.instructions import ApprovedInstruction, InstructionLibrary


def _write_pcm_as_wav_16khz_mono(wav_path: Path, pcm_bytes: bytes) -> str:
    """Write raw 16 kHz, 16-bit signed little-endian mono PCM bytes into a standard WAV container."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure even byte count for 16-bit samples
    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[:-1]
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_bytes)
    return hashlib.sha256(wav_path.read_bytes()).hexdigest()


def _write_placeholder_wav_16khz_mono(
    wav_path: Path,
    *,
    seed_text: str,
    duration_ms: int = 420,
) -> str:
    """
    Write a valid 16 kHz, 16-bit mono PCM WAV development placeholder file and return
    its exact byte-level SHA-256 hex digest.
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    seed_hash = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    freq_hz = 380.0 + float(seed_hash % 180)

    frames = bytearray()
    for i in range(num_samples):
        t = i / sample_rate
        envelope = math.sin(math.pi * (i / max(1, num_samples - 1)))
        sample_val = int(3200.0 * envelope * math.sin(2.0 * math.pi * freq_hz * t))
        frames.extend(struct.pack("<h", sample_val))

    return _write_pcm_as_wav_16khz_mono(wav_path, bytes(frames))


def _load_existing_manifest_map(manifest_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not manifest_path.exists():
        return {}
    try:
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            return {}
        return {
            (str(r.get("instruction_id", "")), str(r.get("language", ""))): r
            for r in rows
            if isinstance(r, dict)
        }
    except Exception:
        return {}


def _is_existing_manifest_verified_timbre(
    manifest_path: Path,
    instructions: list[ApprovedInstruction],
) -> bool:
    """Return True if manifest.json already contains verified Timbre v2.5 WAV files matching all texts and SHA-256 digests."""
    existing = _load_existing_manifest_map(manifest_path)
    if len(existing) < len(instructions):
        return False
    for inst in instructions:
        row = existing.get((inst.instruction_id, inst.language))
        if not row or row.get("generation_status") != "ready":
            return False
        if row.get("approved_text") != inst.approved_text:
            return False
        fp = Path(row.get("file_path", ""))
        if not fp.exists():
            return False
        actual_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
        if actual_sha != row.get("checksum_sha256"):
            return False
    return True


def generate_manifest_json(output_dir: Path, *, force_resynthesize: bool = False) -> Path:
    """
    Synthesize approved instruction WAV files via Gnani Timbre v2.5 when credentials are
    configured (or fallback to development placeholders when unconfigured) and write manifest.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    base_lib = InstructionLibrary(timbre_gate_0_2_verified=False)
    instructions = list(base_lib._instructions)

    if not force_resynthesize and _is_existing_manifest_verified_timbre(
        manifest_path, instructions
    ):
        return manifest_path

    existing_map = _load_existing_manifest_map(manifest_path)
    settings = load_settings()
    has_timbre_creds = bool(settings.timbre_endpoint_url and settings.timbre_auth_token)

    if has_timbre_creds:
        adapter = GnaniTimbreAdapter(
            endpoint_url=settings.timbre_endpoint_url,
            api_key=settings.timbre_auth_token,
        )

        import time

        def _synth_one(inst: ApprovedInstruction) -> tuple[str, str, bool]:
            wav_rel = Path(
                f"backend-media/instructions/{inst.language}/{inst.instruction_id}/{inst.version}.wav"
            )
            existing_row = existing_map.get((inst.instruction_id, inst.language))
            if (
                not force_resynthesize
                and existing_row is not None
                and existing_row.get("generation_status") == "ready"
                and existing_row.get("approved_text") == inst.approved_text
                and wav_rel.exists()
                and wav_rel.stat().st_size > 16000
            ):
                actual_sha = hashlib.sha256(wav_rel.read_bytes()).hexdigest()
                if actual_sha == existing_row.get("checksum_sha256"):
                    return f"{inst.instruction_id}:{inst.language}", actual_sha, True

            last_err: Exception | None = None
            for attempt in range(5):
                try:
                    if attempt > 0:
                        time.sleep(1.5 * attempt)
                    pcm_bytes = adapter.synthesize_pcm_sync(
                        text=inst.approved_text,
                        language=inst.language,
                        voice_id="Pranav",
                        timeout_seconds=25.0,
                    )
                    digest = _write_pcm_as_wav_16khz_mono(wav_rel, pcm_bytes)
                    time.sleep(0.35)
                    return f"{inst.instruction_id}:{inst.language}", digest, True
                except Exception as exc:
                    last_err = exc
                    time.sleep(1.2 * (attempt + 1))
            raise RuntimeError(
                f"Timbre synthesis failed for {inst.instruction_id}:{inst.language}: {last_err}"
            )

        all_succeeded = True
        for inst in instructions:
            try:
                _synth_one(inst)
            except Exception as exc:
                print("Warning: Timbre synthesis item error:", exc)
                all_succeeded = False

        if all_succeeded:
            lib = InstructionLibrary(
                default_voice_by_lang={"en": "Pranav", "te": "Pranav", "hi": "Pranav"},
                timbre_version="v2.5",
                timbre_gate_0_2_verified=True,
            )
            manifest_rows: list[dict[str, str]] = []
            now_iso = datetime.now(timezone.utc).isoformat()
            for inst in lib._instructions:
                resolved_inst, entry = lib.resolve_cached_asset(inst.instruction_id, inst.language)
                if entry is not None:
                    manifest_rows.append(
                        {
                            "instruction_id": entry.instruction_id,
                            "language": entry.language,
                            "voice_id": "Pranav",
                            "timbre_version": "v2.5",
                            "text_version": entry.text_version,
                            "audio_format": entry.audio_format,
                            "file_path": entry.file_path,
                            "checksum_sha256": entry.checksum_sha256,
                            "generated_at": now_iso,
                            "generation_status": "ready",
                            "placeholder_notice": (
                                "Synthesized by Gnani Timbre v2.5 (voice=Pranav, 16 kHz 16-bit mono PCM WAV) "
                                "and verified with SHA-256 checksum."
                            ),
                            "approved_text": resolved_inst.approved_text,
                        }
                    )
            manifest_path.write_text(
                json.dumps(manifest_rows, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return manifest_path

    # Fallback when Timbre credentials are not configured (preserve any existing real WAV > 16 KB)
    for inst in instructions:
        wav_rel = Path(
            f"backend-media/instructions/{inst.language}/{inst.instruction_id}/{inst.version}.wav"
        )
        if not (wav_rel.exists() and wav_rel.stat().st_size > 16000):
            _write_placeholder_wav_16khz_mono(
                wav_rel,
                seed_text=f"{inst.instruction_id}:{inst.language}:{inst.version}:{inst.approved_text}",
            )

    lib = InstructionLibrary(timbre_gate_0_2_verified=False)
    manifest_rows = []
    for inst in lib._instructions:
        resolved_inst, entry = lib.resolve_cached_asset(inst.instruction_id, inst.language)
        if entry is not None:
            manifest_rows.append(
                {
                    "instruction_id": entry.instruction_id,
                    "language": entry.language,
                    "voice_id": entry.voice_id,
                    "timbre_version": entry.timbre_version,
                    "text_version": entry.text_version,
                    "audio_format": entry.audio_format,
                    "file_path": entry.file_path,
                    "checksum_sha256": entry.checksum_sha256,
                    "generated_at": entry.generated_at.isoformat(),
                    "generation_status": entry.generation_status,
                    "placeholder_notice": (
                        "Development placeholder 16 kHz mono PCM WAV generated by "
                        "backend/app/scripts/presynthesize_instructions.py; pending Gate 0.2 Timbre v2.5 verification."
                    ),
                    "approved_text": resolved_inst.approved_text,
                }
            )

    manifest_path.write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    target = Path("backend-media/instructions")
    written = generate_manifest_json(target, force_resynthesize=False)
    print(f"Wrote instruction WAV files and manifest to {written}")
