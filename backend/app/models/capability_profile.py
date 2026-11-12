"""
Verified deployment capability profile (Section 4.3) and SQLAlchemy 2.0 / Pydantic v2
domain models (Section 6 & Section 7).

Records the actual verified configuration at deployment and disables all unverified
v3.0 capabilities so the application runs strictly on the v2.5 baseline until
participant account access is confirmed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentCapabilityProfile:
    """
    Verified provider capability profile per Section 4.3.
    Never invents an automatic model-discovery endpoint or probes undocumented parameters.
    """
    prisma_model_version: str
    prisma_endpoint_transport: str
    prisma_v3_codemix_mode_enabled: bool
    prisma_v3_keyterm_prompting_enabled: bool
    prisma_non_speech_signal_shape: str
    timbre_model_version: str
    timbre_endpoint_transport: str
    timbre_v3_expressive_tags_enabled: bool
    reasoning_provider: str
    reasoning_model_identifier: str
    reasoning_safety_block_signalling: str
    concurrent_stt_tts_permitted: bool
    audio_encoding: str
    barge_in_cancellation_mode: str


DEFAULT_DEPLOYMENT_CAPABILITY_PROFILE = DeploymentCapabilityProfile(
    prisma_model_version="v2.5",
    prisma_endpoint_transport="pending_gate_0_1_verification",
    prisma_v3_codemix_mode_enabled=False,
    prisma_v3_keyterm_prompting_enabled=False,
    prisma_non_speech_signal_shape="unverified_gate_0_1_treat_all_barge_in_as_genuine",
    timbre_model_version="v2.5",
    timbre_endpoint_transport="pending_gate_0_2_verification",
    timbre_v3_expressive_tags_enabled=False,
    reasoning_provider="google_gemini_api",
    reasoning_model_identifier="configured_by_lead_engineer_gate_0_3",
    reasoning_safety_block_signalling="finish_reason_safety_or_empty_candidates_gate_0_3",
    concurrent_stt_tts_permitted=False,
    audio_encoding="pcm_s16le_16000_mono",
    barge_in_cancellation_mode="local_stop_first_then_sequential_teardown_gate_0_4",
)


def validate_capability_profile(profile: DeploymentCapabilityProfile) -> None:
    """
    Enforce Section 4.1 & 4.2 rules:
    - Never enable v3.0 codemix parameter or keyterm prompting when Prisma is v2.5.
    - Never enable v3.0 expressive tags when Timbre is v2.5.
    """
    if profile.prisma_model_version == "v2.5":
        if profile.prisma_v3_codemix_mode_enabled:
            raise ValueError("Cannot send v3.0 codemix parameter to Prisma v2.5 (Section 4.1)")
        if profile.prisma_v3_keyterm_prompting_enabled:
            raise ValueError("Cannot send v3.0 keyterm prompting to Prisma v2.5 (Section 4.1)")

    if profile.timbre_model_version == "v2.5" and profile.timbre_v3_expressive_tags_enabled:
        raise ValueError("Cannot send v3.0 expressive tags to Timbre v2.5 (Section 4.2)")
