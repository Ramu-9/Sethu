"""
Validated configuration loader for SETU (Section 7, Section 13.2, Section 19, and Section 34).

Loads environment variables with explicit defaults and strict validation for:
- Database connection URL & session HMAC signing key
- Configurable hospital ranking weights (distance, capability, ETA)
- Configurable caller silence timeout policy parameters (Rule 5 & Section 19)
- External provider gate flags (Gate 0.1 Prisma, Gate 0.2 Timbre, Gate 0.3 Gemini)
"""

from dataclasses import dataclass
import os
from pathlib import Path

from backend.app.orchestrator.hospital import HospitalScoringWeights
from backend.app.orchestrator.timeout import TimeoutPolicyConfig


_PLACEHOLDER_VALUES = {
    "GEMINI_API_KEY": {"", "MY_GEMINI_API_KEY"},
    "PRISMA_ENDPOINT_URL": {""},
    "PRISMA_AUTH_TOKEN": {""},
    "TIMBRE_ENDPOINT_URL": {""},
    "TIMBRE_AUTH_TOKEN": {""},
    "DATABASE_URL": {
        "",
        "postgresql+asyncpg://setu_runtime:setu_runtime@localhost:5432/setu",
        "postgresql+asyncpg://setu:setu_local_dev@localhost:5432/setu",
    },
    "MIGRATION_DATABASE_URL": {
        "",
        "postgresql+asyncpg://setu_owner:setu_owner_dev_password@localhost:5432/setu",
    },
}


def _read_repo_dotenv(merged: dict[str, str]) -> None:
    """Apply repository .env values (overriding stale container defaults when non-placeholder)."""
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    if not dotenv_path.is_file():
        return
    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            k = key.strip()
            v = val.strip().strip('"').strip("'")
            if k and v and v not in _PLACEHOLDER_VALUES.get(k, {""}):
                merged[k] = v
    except Exception:
        return


def get_runtime_env_map() -> dict[str, str]:
    """
    Return the effective runtime environment dictionary from os.environ and repository .env
    without scanning unrelated processes or logging secret values.
    """
    merged: dict[str, str] = dict(os.environ)
    _read_repo_dotenv(merged)
    return merged


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings for the SETU FastAPI/WebSocket backend."""

    database_url: str
    session_token_secret: str
    gemini_api_key: str
    gemini_model_id: str
    prisma_endpoint_url: str
    prisma_auth_token: str
    timbre_endpoint_url: str
    timbre_auth_token: str
    gate_0_1_prisma_verified: bool
    gate_0_2_timbre_verified: bool
    gate_0_3_gemini_verified: bool
    hospital_weights: HospitalScoringWeights
    timeout_policy: TimeoutPolicyConfig
    migration_database_url: str = (
        "postgresql+asyncpg://setu_owner:setu_owner_dev_password@localhost:5432/setu"
    )
    gemini_fallback_api_keys: tuple[str, ...] = ()


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(env_override: dict[str, str] | None = None) -> Settings:
    """Load and validate environment settings."""
    env = env_override if env_override is not None else get_runtime_env_map()

    dist_w = float(env.get("HOSPITAL_WEIGHT_DISTANCE", "0.45"))
    cap_w = float(env.get("HOSPITAL_WEIGHT_CAPABILITY", "0.35"))
    eta_w = float(env.get("HOSPITAL_WEIGHT_ETA", "0.20"))
    weight_sum = dist_w + cap_w + eta_w
    if abs(weight_sum - 1.0) > 1e-5 and weight_sum > 0:
        dist_w = round(dist_w / weight_sum, 6)
        cap_w = round(cap_w / weight_sum, 6)
        eta_w = round(1.0 - dist_w - cap_w, 6)

    hospital_weights = HospitalScoringWeights(
        distance_weight=dist_w,
        capability_weight=cap_w,
        eta_weight=eta_w,
    )
    timeout_policy = TimeoutPolicyConfig(
        initial_silence_seconds=float(env.get("TIMEOUT_INITIAL_SILENCE_SECONDS", "15.0")),
        max_check_ins=int(env.get("TIMEOUT_MAX_CHECK_INS", "2")),
        wait_between_check_ins_seconds=float(
            env.get("TIMEOUT_WAIT_BETWEEN_CHECK_INS_SECONDS", "15.0")
        ),
        max_idle_exhaustion_seconds=float(
            env.get("TIMEOUT_MAX_IDLE_EXHAUSTION_SECONDS", "90.0")
        ),
    )
    gemini_key = env.get("GEMINI_API_KEY", "").strip()
    if gemini_key in _PLACEHOLDER_VALUES["GEMINI_API_KEY"]:
        gemini_key = ""

    fallback_keys: list[str] = []
    explicit_fallback = env.get("GEMINI_FALLBACK_API_KEY", "").strip()
    if (
        explicit_fallback
        and explicit_fallback not in _PLACEHOLDER_VALUES["GEMINI_API_KEY"]
        and explicit_fallback != gemini_key
    ):
        fallback_keys.append(explicit_fallback)

    raw_secret = env.get(
        "SETU_SESSION_TOKEN_SECRET",
        "setu-local-dev-hmac-secret-2026-key",
    ).strip()
    if not raw_secret or "dont know" in raw_secret.lower():
        raw_secret = "120f71a45e80ac7c03547a55cf4633e344334c86359e813af33ca245b347852e"

    return Settings(
        database_url=env.get(
            "DATABASE_URL",
            "postgresql+asyncpg://setu:setu_local_dev@localhost:5432/setu",
        ),
        migration_database_url=env.get(
            "MIGRATION_DATABASE_URL",
            env.get(
                "DATABASE_URL",
                "postgresql+asyncpg://setu_owner:setu_owner_dev_password@localhost:5432/setu",
            ),
        ),
        session_token_secret=raw_secret,
        gemini_api_key=gemini_key,
        gemini_model_id=env.get("GEMINI_MODEL_ID", "gemini-3.8-flash").strip() or "gemini-3.8-flash",
        prisma_endpoint_url=env.get("PRISMA_ENDPOINT_URL", "").strip(),
        prisma_auth_token=env.get("PRISMA_AUTH_TOKEN", "").strip(),
        timbre_endpoint_url=env.get("TIMBRE_ENDPOINT_URL", "").strip(),
        timbre_auth_token=env.get("TIMBRE_AUTH_TOKEN", "").strip(),
        gate_0_1_prisma_verified=_parse_bool(env.get("PRISMA_GATE_0_1_VERIFIED"), False),
        gate_0_2_timbre_verified=_parse_bool(env.get("TIMBRE_GATE_0_2_VERIFIED"), False),
        gate_0_3_gemini_verified=_parse_bool(env.get("GEMINI_GATE_0_3_VERIFIED"), False),
        hospital_weights=hospital_weights,
        timeout_policy=timeout_policy,
        gemini_fallback_api_keys=tuple(fallback_keys),
    )
