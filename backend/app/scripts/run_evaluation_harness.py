"""
Section 26 Evaluation Harness CLI Runner (`backend/app/scripts/run_evaluation_harness.py`).

Enforces Section 26.5 ('Never claim measured performance that has not actually been measured'):
- Pre-flight check `verify_live_gates_before_measurement(settings)` verifies that Gate 0.1
  (Gnani Prisma v2.5), Gate 0.2 (Gnani Timbre v2.5), and Gate 0.3 (Google Gemini API) are
  all closed and configured with live participant credentials before any live measurement run.
- Never runs against mock adapters to claim measured WER, CER, Macro-F1, or TTFAB numbers.
"""

import json
import sys
from typing import Any

from backend.app.config import Settings, load_settings
from backend.app.observability.evaluation import build_minimum_viable_evaluation_dataset


class EvaluationGateBlockedError(RuntimeError):
    """Raised when attempting to run live evaluation measurement while Gates 0.1–0.3 are open."""


def verify_live_gates_before_measurement(settings: Settings) -> dict[str, Any]:
    """
    Verify that all three external provider gates are closed before running live evaluation.
    Raises EvaluationGateBlockedError if any gate is unverified or missing credentials.
    """
    dataset = build_minimum_viable_evaluation_dataset()
    missing_gates: list[str] = []

    if not settings.gate_0_1_prisma_verified or not settings.prisma_endpoint_url or not settings.prisma_auth_token:
        missing_gates.append("Gate 0.1 (Gnani Prisma v2.5 live STT endpoint & credentials)")
    if not settings.gate_0_2_timbre_verified or not settings.timbre_endpoint_url or not settings.timbre_auth_token:
        missing_gates.append("Gate 0.2 (Gnani Timbre v2.5 live TTS endpoint & credentials)")
    if not settings.gate_0_3_gemini_verified or not settings.gemini_api_key:
        missing_gates.append("Gate 0.3 (Google Gemini API live model & key verification)")

    if missing_gates:
        raise EvaluationGateBlockedError(
            "Section 26.5 Guard: Live evaluation measurement is blocked until all provider gates "
            f"are closed ({', '.join(missing_gates)}). Refusing to produce fabricated WER/CER/F1 "
            f"numbers from mock adapters across the {len(dataset.part_a_utterances)} Part A utterances "
            f"and {len(dataset.part_b_conversations)} Part B conversations."
        )

    return {
        "status": "gates_verified_ready_for_live_measurement",
        "part_a_utterance_count": len(dataset.part_a_utterances),
        "part_b_conversation_count": len(dataset.part_b_conversations),
    }


def main() -> int:
    settings = load_settings()
    dataset = build_minimum_viable_evaluation_dataset()
    try:
        status = verify_live_gates_before_measurement(settings)
        print(json.dumps(status, indent=2))
        return 0
    except EvaluationGateBlockedError as exc:
        report = {
            "status": "blocked_pending_provider_gates",
            "reason": str(exc),
            "dataset_ready": {
                "part_a_unique_utterances": len(dataset.part_a_utterances),
                "planned_latency_observations_across_2_runs": len(dataset.part_a_utterances) * 2,
                "part_b_multi_turn_conversations": len(dataset.part_b_conversations),
            },
            "gate_status": {
                "gate_0_1_prisma_verified": settings.gate_0_1_prisma_verified,
                "gate_0_2_timbre_verified": settings.gate_0_2_timbre_verified,
                "gate_0_3_gemini_verified": settings.gate_0_3_gemini_verified,
            },
        }
        print(json.dumps(report, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
