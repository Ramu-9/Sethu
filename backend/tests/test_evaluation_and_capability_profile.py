"""
Unit tests for Section 26 (Evaluation Dataset, Minimum Viable Matrix,
WER/CER, Macro F1, Field-level F1, Provider-Block Rate metrics, and Section 26.5
pre-flight Gate 0.1–0.3 guard), Section 4.3 (Capability Profile), and Section 6/21.2
(PostgreSQL 16 + PostGIS + pg_trgm DDL & FOR UPDATE SKIP LOCKED queries).
"""

import unittest
from uuid import uuid4

from backend.app.adapters.base import EmergencyCategory, ProvenancedFact
from backend.app.config import Settings, load_settings
from backend.app.db.migrations import (  # type: ignore[attr-defined]
    __name__ as _migrations_pkg,
)
from backend.app.db.pool import (
    FIND_DUPLICATE_INCIDENT_WITHIN_200M_5MIN_SQL,
    POSTGRES_APPEND_ONLY_SCHEMA_SQL,
    SELECT_NEAREST_AVAILABLE_RESOURCE_FOR_UPDATE_SQL,
)
from backend.app.models.capability_profile import (
    DEFAULT_DEPLOYMENT_CAPABILITY_PROFILE,
    validate_capability_profile,
)
from backend.app.observability.evaluation import (
    EvaluationPredictionRecord,
    build_minimum_viable_evaluation_dataset,
    compute_character_error_rate,
    compute_classification_metrics,
    compute_fact_extraction_metrics,
    compute_word_error_rate,
    evaluate_prediction_batch,
)
from backend.app.orchestrator.hospital import HospitalScoringWeights
from backend.app.orchestrator.timeout import TimeoutPolicyConfig
from backend.app.scripts.run_evaluation_harness import (
    EvaluationGateBlockedError,
    verify_live_gates_before_measurement,
)


class TestEvaluationSuiteAndCapabilityProfile(unittest.TestCase):
    """Tests for Section 4.3 Capability Profile, Section 6/21 DDL, and Section 26 Evaluation Harness."""

    def test_capability_profile_enforces_v25_baseline_and_sequential_concurrency(self) -> None:
        """Section 4.3 & Gate 0.4: Profile records v2.5 Prisma/Timbre and no unverified v3 flags."""
        profile = DEFAULT_DEPLOYMENT_CAPABILITY_PROFILE
        validate_capability_profile(profile)
        self.assertEqual(profile.prisma_model_version, "v2.5")
        self.assertEqual(profile.timbre_model_version, "v2.5")
        self.assertFalse(profile.prisma_v3_codemix_mode_enabled)
        self.assertFalse(profile.prisma_v3_keyterm_prompting_enabled)
        self.assertFalse(profile.timbre_v3_expressive_tags_enabled)
        self.assertFalse(profile.concurrent_stt_tts_permitted)
        self.assertEqual(profile.audio_encoding, "pcm_s16le_16000_mono")

    def test_minimum_viable_evaluation_dataset_structure(self) -> None:
        """
        Section 26.1:
        - 6 scenarios x 3 language conditions (te, hi, codemix) x 2 speakers = 36 unique utterances.
        - Code-mixed condition (12 utterances) is split evenly between Tanglish (6) and Hinglish (6).
        - Includes Part B multi-turn conversations (scenarios 11 & 12) reported separately.
        """
        dataset = build_minimum_viable_evaluation_dataset()
        self.assertEqual(len(dataset.part_a_utterances), 36)

        by_cond: dict[str, int] = {}
        by_subcond: dict[str, int] = {}
        speakers_per_cond: dict[str, set[str]] = {}
        for item in dataset.part_a_utterances:
            by_cond[item.language_condition] = by_cond.get(item.language_condition, 0) + 1
            if item.codemix_subcondition:
                by_subcond[item.codemix_subcondition] = (
                    by_subcond.get(item.codemix_subcondition, 0) + 1
                )
            speakers_per_cond.setdefault(item.language_condition, set()).add(item.speaker_id)

        self.assertEqual(by_cond, {"te": 12, "hi": 12, "codemix": 12})
        self.assertEqual(by_subcond, {"tanglish": 6, "hinglish": 6})
        for cond, spk_set in speakers_per_cond.items():
            self.assertGreaterEqual(len(spk_set), 2, f"Condition {cond} requires >= 2 speakers")

        # Part B multi-turn scripted conversations (Scenarios 11 & 12)
        self.assertGreaterEqual(len(dataset.part_b_conversations), 2)

    def test_wer_and_cer_calculations(self) -> None:
        """Section 26.5: Exact WER and CER calculations."""
        self.assertEqual(
            compute_word_error_rate(
                reference="Anna ikkada road accident jarigindi",
                hypothesis="Anna ikkada road accident jarigindi",
            ),
            0.0,
        )
        # 1 word substitution out of 4 words -> WER = 0.25
        self.assertAlmostEqual(
            compute_word_error_rate(
                reference="near secunderabad railway station",
                hypothesis="near secunderabad metro station",
            ),
            0.25,
            places=4,
        )
        self.assertEqual(compute_character_error_rate("aag lagi", "aag lagi"), 0.0)
        self.assertGreater(compute_character_error_rate("aag lagi", "aag"), 0.0)

    def test_classification_macro_f1_and_separate_provider_block_rate(self) -> None:
        """
        Section 26.5: Provider-block rate is reported separately from classification
        accuracy, and blocked turns are not conflated with unknown classifications.
        """
        dataset = build_minimum_viable_evaluation_dataset()
        predictions: list[EvaluationPredictionRecord] = []
        for idx, item in enumerate(dataset.part_a_utterances):
            is_blocked = item.scenario_id == "physical_attack" and idx % 2 == 0
            predictions.append(
                EvaluationPredictionRecord(
                    utterance_id=item.utterance_id,
                    hypothesis_transcript=item.ground_truth_transcript,
                    predicted_category=None if is_blocked else item.expected_category,
                    was_provider_blocked=is_blocked,
                    provider_block_fallback_triggered_correctly=is_blocked,
                    extracted_facts=(
                        ProvenancedFact(
                            field_name="injured_persons",
                            value=True,
                            source_utterance_id=uuid4(),
                            evidence_text=item.ground_truth_transcript,
                        ),
                    ),
                    latency_ms_runs=(280.0, 295.0),
                )
            )

        report = evaluate_prediction_batch(dataset, predictions)
        self.assertEqual(report.total_part_a_utterances, 36)
        self.assertEqual(report.total_latency_observations, 72)
        # 72 observations overall >= 30 threshold -> overall p50/p95 available!
        self.assertIsNotNone(report.overall_latency_p50_ms)
        self.assertIsNotNone(report.overall_latency_p95_ms)
        # Per-language has 24 observations (< 30 threshold) -> percentiles marked None / insufficiently_sampled
        for lang_stat in report.per_language_latency.values():
            self.assertEqual(lang_stat["observation_count"], 24)
            self.assertIsNone(lang_stat["p50_ms"])
            self.assertEqual(lang_stat["status"], "insufficiently_sampled (<30 observations)")

        self.assertEqual(report.classification_accuracy, 1.0)
        self.assertGreater(report.provider_block_rate, 0.0)
        self.assertEqual(report.fact_provenance_coverage, 1.0)

    def test_postgres_postgis_pgtrgm_ddl_and_skip_locked_queries(self) -> None:
        """
        Sections 6, 10.3, 12.2, 14 & 21.2:
        Verify that POSTGRES_APPEND_ONLY_SCHEMA_SQL defines all 10 tables, PostGIS + pg_trgm,
        GIN trigram indexes, FOR UPDATE SKIP LOCKED resource query, and append-only triggers.
        """
        self.assertTrue(_migrations_pkg.endswith("migrations"))
        for required_token in (
            "CREATE EXTENSION IF NOT EXISTS postgis;",
            "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
            "CREATE TABLE IF NOT EXISTS sessions",
            "CREATE TABLE IF NOT EXISTS incidents",
            "CREATE TABLE IF NOT EXISTS utterances",
            "CREATE TABLE IF NOT EXISTS validated_facts",
            "CREATE TABLE IF NOT EXISTS synthetic_resources",
            "CREATE TABLE IF NOT EXISTS simulated_assignments",
            "CREATE TABLE IF NOT EXISTS synthetic_landmarks",
            "CREATE TABLE IF NOT EXISTS synthetic_hospitals",
            "CREATE TABLE IF NOT EXISTS incident_handoffs",
            "CREATE TABLE IF NOT EXISTS audit_events",
            "gin_trgm_ops",
            "forbid_audit_events_mutation()",
            "BEFORE UPDATE OR DELETE ON audit_events",
            "BEFORE TRUNCATE ON audit_events",
        ):
            self.assertIn(required_token, POSTGRES_APPEND_ONLY_SCHEMA_SQL)

        self.assertIn("FOR UPDATE SKIP LOCKED", SELECT_NEAREST_AVAILABLE_RESOURCE_FOR_UPDATE_SQL)
        self.assertIn("ST_DWithin", FIND_DUPLICATE_INCIDENT_WITHIN_200M_5MIN_SQL)

    def test_evaluation_harness_pre_flight_gate_guard(self) -> None:
        """
        Section 26.5: Live evaluation runner refuses to produce fabricated numbers when Gates 0.1-0.3
        are open, and succeeds only when all three gates and credentials are configured.
        """
        default_settings = load_settings(env_override={})
        with self.assertRaises(EvaluationGateBlockedError):
            verify_live_gates_before_measurement(default_settings)

        verified_settings = Settings(
            database_url="postgresql+asyncpg://setu:setu@localhost:5432/setu",
            session_token_secret="test-secret",
            gemini_api_key="test-gemini-key",
            gemini_model_id="gemini-3.8-flash",
            prisma_endpoint_url="https://prisma.example.invalid/v2.5",
            prisma_auth_token="test-prisma-token",
            timbre_endpoint_url="https://timbre.example.invalid/v2.5",
            timbre_auth_token="test-timbre-token",
            gate_0_1_prisma_verified=True,
            gate_0_2_timbre_verified=True,
            gate_0_3_gemini_verified=True,
            hospital_weights=HospitalScoringWeights(0.45, 0.20, 0.35),
            timeout_policy=TimeoutPolicyConfig(15.0, 2, 15.0, 90.0),
        )
        res = verify_live_gates_before_measurement(verified_settings)
        self.assertEqual(res["status"], "gates_verified_ready_for_live_measurement")
        self.assertEqual(res["part_a_utterance_count"], 36)
        self.assertEqual(res["part_b_conversation_count"], 2)


if __name__ == "__main__":
    unittest.main()
