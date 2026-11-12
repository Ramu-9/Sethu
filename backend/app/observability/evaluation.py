"""
Systematic evaluation dataset and metric harness for SETU (Section 26).

Implements:
- Section 26.0 & 26.1: Minimum Viable Evaluation matrix:
  6 scenarios x 3 language conditions (Telugu, Hindi, Code-Mixed) x 2 speakers = 36 unique utterances,
  with Code-Mixed split evenly between Tanglish (6) and Hinglish (6) and reported separately,
  plus 2 latency runs per utterance = 72 total observations, and Part B multi-turn conversations.
- Section 26.4: Reports p50 and p95 only when >= 30 valid observations exist; marks per-language
  percentiles (24 observations in minimum viable mode) as insufficiently sampled.
- Section 26.5: Computes WER, CER, classification accuracy, Macro F1, provider-block rate
  (reported separately from classification accuracy), and fact provenance coverage.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

from backend.app.adapters.base import EmergencyCategory, ProvenancedFact
from backend.app.observability.metrics import MIN_OBSERVATIONS_FOR_PERCENTILES
from backend.app.orchestrator.dispatch import BASE_DISPATCH_MAPPING, ResourceCategory


@dataclass(frozen=True)
class EvaluationGroundTruthUtterance:
    """Human-reviewed ground truth for a single Part A evaluation utterance (Section 26.3)."""
    utterance_id: str
    scenario_id: str
    language_condition: str  # 'te', 'hi', 'codemix'
    codemix_subcondition: str | None  # 'tanglish' or 'hinglish' when language_condition == 'codemix'
    speaker_id: str
    ground_truth_transcript: str
    expected_category: EmergencyCategory
    expected_resources: tuple[ResourceCategory, ...]
    expected_fact_fields: tuple[str, ...]
    tests_safety_filter_sensitivity: bool


@dataclass(frozen=True)
class MultiTurnScriptedConversation:
    """Part B multi-turn scripted conversation ground truth (Section 26.0)."""
    conversation_id: str
    scenario_number: int
    title: str
    turns: tuple[dict[str, str], ...]
    expected_final_category: EmergencyCategory
    expected_review_flag: str | None


@dataclass(frozen=True)
class EvaluationDataset:
    """Complete evaluation dataset combining Part A matrix and Part B multi-turn conversations."""
    part_a_utterances: tuple[EvaluationGroundTruthUtterance, ...]
    part_b_conversations: tuple[MultiTurnScriptedConversation, ...]


@dataclass(frozen=True)
class EvaluationPredictionRecord:
    """Measured model & orchestrator prediction for a single Part A utterance across repeated runs."""
    utterance_id: str
    hypothesis_transcript: str
    predicted_category: EmergencyCategory | None
    was_provider_blocked: bool
    provider_block_fallback_triggered_correctly: bool
    extracted_facts: tuple[ProvenancedFact, ...]
    latency_ms_runs: tuple[float, ...]


@dataclass(frozen=True)
class EvaluationBatchReport:
    """Complete evaluation summary adhering strictly to Section 26.1, 26.4, and 26.5."""
    total_part_a_utterances: int
    tanglish_sample_count: int
    hinglish_sample_count: int
    combined_codemix_sample_count: int
    total_latency_observations: int
    mean_wer: float
    mean_cer: float
    classification_accuracy: float
    macro_f1: float
    provider_block_rate: float
    provider_block_fallback_correctness: float
    fact_provenance_coverage: float
    overall_latency_p50_ms: float | None
    overall_latency_p95_ms: float | None
    per_language_latency: dict[str, dict[str, object]]


def _edit_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    n, m = len(seq_a), len(seq_b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m]


def compute_word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) per Section 26.5."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return round(_edit_distance(ref_words, hyp_words) / len(ref_words), 4)


def compute_character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) per Section 26.5."""
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return round(_edit_distance(ref_chars, hyp_chars) / len(ref_chars), 4)


def compute_classification_metrics(
    pairs: Sequence[tuple[EmergencyCategory, EmergencyCategory]],
) -> tuple[float, float]:
    """Return (accuracy, macro_f1) across non-blocked classification evaluations."""
    if not pairs:
        return 0.0, 0.0
    correct = sum(1 for exp, pred in pairs if exp == pred)
    accuracy = round(correct / len(pairs), 4)

    categories = sorted({exp for exp, _ in pairs} | {pred for _, pred in pairs}, key=lambda c: c.value)
    f1_scores: list[float] = []
    for cat in categories:
        tp = sum(1 for exp, pred in pairs if exp == cat and pred == cat)
        fp = sum(1 for exp, pred in pairs if exp != cat and pred == cat)
        fn = sum(1 for exp, pred in pairs if exp == cat and pred != cat)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
    return accuracy, macro_f1


def compute_fact_extraction_metrics(
    facts: Sequence[ProvenancedFact],
) -> float:
    """Return provenance coverage ratio (Rule 3 & Section 26.5)."""
    if not facts:
        return 1.0
    valid = sum(
        1
        for f in facts
        if f.source_utterance_id is not None and bool(f.evidence_text.strip())
    )
    return round(valid / len(facts), 4)


def _percentile(sorted_vals: list[float], q: float) -> float:
    idx = (len(sorted_vals) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return round(sorted_vals[lo], 2)
    w = idx - lo
    return round(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w, 2)


def build_minimum_viable_evaluation_dataset() -> EvaluationDataset:
    """
    Build the exact Section 26.1 Minimum Viable Evaluation dataset:
    - 6 scenarios (including violence/injury scenarios 4, 6, 8 for safety-filter measurement)
    - 3 language conditions: Telugu ('te'), Hindi ('hi'), and Code-Mixed ('codemix')
    - 2 distinct consented synthetic speakers per condition = 36 unique utterances
    - Code-mixed split evenly: 6 Tanglish + 6 Hinglish.
    """
    scenarios = (
        (
            "medical_emergency",
            EmergencyCategory.MEDICAL_EMERGENCY,
            ("injured_persons",),
            False,
            {
                "te": "ఇక్కడ ఒక వ్యక్తికి గుండె నొప్పిగా ఉంది వెంటనే సహాయం కావాలి",
                "hi": "यहाँ एक बुजुर्ग व्यक्ति बेहोश हो गए हैं और सांस लेने में दिक्कत है",
                "tanglish": "Anna ikkada oka person unconscious ayyaru chest pain antunnaru",
                "hinglish": "Bhaiya ek person suddenly collapse ho gaye hain breathing problem hai",
            },
        ),
        (
            "fire_with_injuries",
            EmergencyCategory.FIRE_WITH_INJURIES,
            ("fire_reported", "injured_persons"),
            True,
            {
                "te": "భవనంలో మంటలు చెలరేగాయి ఇద్దరు వ్యక్తులకు గాయాలయ్యాయి",
                "hi": "दुकान में भीषण आग लग गई है और दो लोग झुलस गए हैं",
                "tanglish": "Shop lo pedda fire start ayindi two people ki burn injuries ayyayi",
                "hinglish": "Building ke ground floor par fire lagi hai aur do log injured hain",
            },
        ),
        (
            "road_accident",
            EmergencyCategory.ROAD_ACCIDENT,
            ("vehicle_collision_reported", "injured_persons"),
            True,
            {
                "te": "సికింద్రాబాద్ స్టేషన్ దగ్గర రోడ్డు ప్రమాదం జరిగింది ఒకరికి రక్తస్రావం అవుతోంది",
                "hi": "रेलवे स्टेशन के पास सड़क दुर्घटना हुई है एक व्यक्ति घायल है",
                "tanglish": "Anna ikkada road accident jarigindi oka person unconscious ga unnaru near railway station",
                "hinglish": "Main road par car aur bike ka accident hua hai ek aadmi injured hai",
            },
        ),
        (
            "physical_attack",
            EmergencyCategory.PHYSICAL_ATTACK,
            ("threat_present",),
            True,
            {
                "te": "మార్కెట్ దగ్గర కత్తితో దాడి చేస్తున్నారు వెంటనే పోలీసులను పంపండి",
                "hi": "बाज़ार के पास कुछ लोग हथियार से हमला कर रहे हैं",
                "tanglish": "Market entrance daggara knife tho physical attack chestunnaru please help",
                "hinglish": "Market ke paas ek aadmi knife se attack kar raha hai",
            },
        ),
        (
            "explosion",
            EmergencyCategory.EXPLOSION,
            ("explosion_reported", "injured_persons"),
            True,
            {
                "te": "ఫ్యాక్టరీలో పెద్ద పేలుడు జరిగింది మంటలు మరియు గాయపడిన వారు ఉన్నారు",
                "hi": "गोदाम में सिलेंडर फटने से बड़ा धमाका हुआ है और लोग घायल हैं",
                "tanglish": "Godown lo pedda cylinder explosion ayindi chala mandi injured ga unnaru",
                "hinglish": "Factory mein cylinder blast explosion hua hai aur fire lag gayi hai",
            },
        ),
        (
            "unknown",
            EmergencyCategory.UNKNOWN,
            ("caller_observations",),
            False,
            {
                "te": "ఇక్కడ వీధిలో జనాలు గుమిగూడి అరుస్తున్నారు ఏమి జరిగిందో తెలియడం లేదు",
                "hi": "गली के बाहर बहुत भीड़ जमा है और लोग चिल्ला रहे हैं पता नहीं क्या हुआ",
                "tanglish": "Bayata road meeda chala crowd gather ayyi shouting chestunnaru em jarigindo telidu",
                "hinglish": "Bahar street par log gather hokar shout kar rahe hain kuch samajh nahi aa raha",
            },
        ),
    )

    utterances: list[EvaluationGroundTruthUtterance] = []
    idx = 0
    for sc_index, (sc_id, cat, fact_fields, tests_filter, texts) in enumerate(scenarios):
        # 1. Telugu (2 speakers)
        for spk in ("spk_te_01", "spk_te_02"):
            idx += 1
            utterances.append(
                EvaluationGroundTruthUtterance(
                    utterance_id=f"eval_a_{idx:02d}",
                    scenario_id=sc_id,
                    language_condition="te",
                    codemix_subcondition=None,
                    speaker_id=spk,
                    ground_truth_transcript=texts["te"],
                    expected_category=cat,
                    expected_resources=BASE_DISPATCH_MAPPING[cat],
                    expected_fact_fields=fact_fields,
                    tests_safety_filter_sensitivity=tests_filter,
                )
            )
        # 2. Hindi (2 speakers)
        for spk in ("spk_hi_01", "spk_hi_02"):
            idx += 1
            utterances.append(
                EvaluationGroundTruthUtterance(
                    utterance_id=f"eval_a_{idx:02d}",
                    scenario_id=sc_id,
                    language_condition="hi",
                    codemix_subcondition=None,
                    speaker_id=spk,
                    ground_truth_transcript=texts["hi"],
                    expected_category=cat,
                    expected_resources=BASE_DISPATCH_MAPPING[cat],
                    expected_fact_fields=fact_fields,
                    tests_safety_filter_sensitivity=tests_filter,
                )
            )
        # 3. Code-mixed: split evenly (first 3 scenarios x 2 speakers = 6 Tanglish;
        #    last 3 scenarios x 2 speakers = 6 Hinglish)
        subcond = "tanglish" if sc_index < 3 else "hinglish"
        for spk in ("spk_cm_01", "spk_cm_02"):
            idx += 1
            utterances.append(
                EvaluationGroundTruthUtterance(
                    utterance_id=f"eval_a_{idx:02d}",
                    scenario_id=sc_id,
                    language_condition="codemix",
                    codemix_subcondition=subcond,
                    speaker_id=spk,
                    ground_truth_transcript=texts[subcond],
                    expected_category=cat,
                    expected_resources=BASE_DISPATCH_MAPPING[cat],
                    expected_fact_fields=fact_fields,
                    tests_safety_filter_sensitivity=tests_filter,
                )
            )

    part_b = (
        MultiTurnScriptedConversation(
            conversation_id="eval_b_11_language_switch",
            scenario_number=11,
            title="Mid-session language switching (Telugu -> English -> Hindi) without restarting incident",
            turns=(
                {"lang": "te", "text": "అన్నా ఇక్కడ రోడ్డు ప్రమాదం జరిగింది"},
                {"lang": "en", "text": "Yes, two people are injured near Secunderabad Railway Station"},
                {"lang": "hi", "text": "हाँ मैं सुरक्षित दूरी पर फुटपाथ पर खड़ा हूँ"},
            ),
            expected_final_category=EmergencyCategory.ROAD_ACCIDENT,
            expected_review_flag=None,
        ),
        MultiTurnScriptedConversation(
            conversation_id="eval_b_12_duplicate_and_review",
            scenario_number=12,
            title="Possible duplicate incident within 200m and 5 minutes flagged for control-room review",
            turns=(
                {"lang": "en", "text": "There is a road accident at Secunderabad Railway Station."},
                {"lang": "en", "text": "Yes, I confirm Secunderabad Railway Station is my exact location."},
            ),
            expected_final_category=EmergencyCategory.ROAD_ACCIDENT,
            expected_review_flag="possible_duplicate_incident",
        ),
    )

    return EvaluationDataset(
        part_a_utterances=tuple(utterances),
        part_b_conversations=part_b,
    )


def evaluate_prediction_batch(
    dataset: EvaluationDataset,
    predictions: Sequence[EvaluationPredictionRecord],
) -> EvaluationBatchReport:
    """
    Evaluate a batch of Part A predictions and enforce Section 26.4 sample-count rules.
    """
    pred_by_id = {p.utterance_id: p for p in predictions}
    wers: list[float] = []
    cers: list[float] = []
    cls_pairs: list[tuple[EmergencyCategory, EmergencyCategory]] = []
    all_facts: list[ProvenancedFact] = []
    blocked_count = 0
    blocked_correct = 0
    all_latencies: list[float] = []
    latencies_by_lang: dict[str, list[float]] = {"te": [], "hi": [], "codemix": []}

    tanglish_n = 0
    hinglish_n = 0

    for gt in dataset.part_a_utterances:
        if gt.codemix_subcondition == "tanglish":
            tanglish_n += 1
        elif gt.codemix_subcondition == "hinglish":
            hinglish_n += 1

        pred = pred_by_id[gt.utterance_id]
        wers.append(compute_word_error_rate(gt.ground_truth_transcript, pred.hypothesis_transcript))
        cers.append(compute_character_error_rate(gt.ground_truth_transcript, pred.hypothesis_transcript))
        all_facts.extend(pred.extracted_facts)

        for ms in pred.latency_ms_runs:
            all_latencies.append(ms)
            latencies_by_lang.setdefault(gt.language_condition, []).append(ms)

        if pred.was_provider_blocked:
            blocked_count += 1
            if pred.provider_block_fallback_triggered_correctly:
                blocked_correct += 1
        elif pred.predicted_category is not None:
            cls_pairs.append((gt.expected_category, pred.predicted_category))

    acc, macro_f1 = compute_classification_metrics(cls_pairs)
    prov_cov = compute_fact_extraction_metrics(all_facts)

    sorted_all = sorted(all_latencies)
    if len(sorted_all) >= MIN_OBSERVATIONS_FOR_PERCENTILES:
        overall_p50: float | None = _percentile(sorted_all, 0.50)
        overall_p95: float | None = _percentile(sorted_all, 0.95)
    else:
        overall_p50 = None
        overall_p95 = None

    per_lang_summary: dict[str, dict[str, object]] = {}
    for lang, vals in latencies_by_lang.items():
        s_vals = sorted(vals)
        if len(s_vals) >= MIN_OBSERVATIONS_FOR_PERCENTILES:
            per_lang_summary[lang] = {
                "observation_count": len(s_vals),
                "p50_ms": _percentile(s_vals, 0.50),
                "p95_ms": _percentile(s_vals, 0.95),
                "status": "sufficiently_sampled",
            }
        else:
            per_lang_summary[lang] = {
                "observation_count": len(s_vals),
                "p50_ms": None,
                "p95_ms": None,
                "status": "insufficiently_sampled (<30 observations)",
            }

    total_u = len(dataset.part_a_utterances)
    return EvaluationBatchReport(
        total_part_a_utterances=total_u,
        tanglish_sample_count=tanglish_n,
        hinglish_sample_count=hinglish_n,
        combined_codemix_sample_count=tanglish_n + hinglish_n,
        total_latency_observations=len(all_latencies),
        mean_wer=round(sum(wers) / max(1, len(wers)), 4),
        mean_cer=round(sum(cers) / max(1, len(cers)), 4),
        classification_accuracy=acc,
        macro_f1=macro_f1,
        provider_block_rate=round(blocked_count / max(1, total_u), 4),
        provider_block_fallback_correctness=(
            round(blocked_correct / blocked_count, 4) if blocked_count > 0 else 1.0
        ),
        fact_provenance_coverage=prov_cov,
        overall_latency_p50_ms=overall_p50,
        overall_latency_p95_ms=overall_p95,
        per_language_latency=per_lang_summary,
    )
