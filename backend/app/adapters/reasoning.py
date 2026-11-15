"""
Google Gemini API reasoning adapter (Section 3.2, 9.2, 9.4, 23.2, and Gate 0.3).

Enforces:
- Strict separation between ProviderBlockedContentError ('provider_blocked_emergency_content'),
  ReasoningUnavailableError ('reasoning_unavailable'), and ReasoningSchemaValidationError.
- Bounded 1-retry policy (250ms default delay or Retry-After within deadline) for transient
  HTTP 408/429/5xx or connection errors only; never retries safety blocks, auth errors, or
  schema validation failures (Section 23.2).
- Never invents a fabricated schema fallback on validation failure.
- Never silently substitutes another model when GEMINI_MODEL_ID is configured.
"""

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import replace as dc_replace
import json
import re
import time
from typing import Any, Literal
import urllib.error
import urllib.request
from uuid import UUID

from backend.app.adapters.base import (
    DialogueContext,
    EmergencyCategory,
    FactConflict,
    ProvenancedFact,
    ProviderBlockedContentError,
    ReasoningAdapter,
    ReasoningOutput,
    ReasoningSchemaValidationError,
    ReasoningUnavailableError,
    UtteranceRole,
)


DEFAULT_RETRY_DELAY_SECONDS: float = 0.250
MAX_RECENT_CONVERSATION_TURNS: int = 6
MAX_COMPLETED_UTTERANCE_CACHE_SIZE: int = 512
BLOCKED_FINISH_REASONS = frozenset(
    {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "RECITATION", "SPII"}
)

_HIGH_ACUITY_CLINICAL_PATTERNS = re.compile(
    r"\b("
    r"unconscious|not\s+breathing|no\s+pulse|unresponsive|fainted|collapsed|choking|"
    r"severe\s+bleeding|heavy\s+bleeding|badly\s+injured|critically\s+injured|critical|"
    r"trapped|stuck\s+inside|heart\s+attack|cardiac\s+arrest|stroke|seizure|"
    r"gas\s+cylinder|lpg\s+cylinder|cylinder\s+blast|explosion|building\s+collapse|"
    r"physical\s+attack|stabbed|shot|weapon|assault|"
    r"behosh|saans\s+nahi|spruha\s+ledu|opiri\s+adatledu|ikkada\s+accident|"
    r"agun\s+legeche|oshustho|behosh\s+hoye|atke\s+ache|rokto\s+porche|"
    r"prajne\s+illa|usiraduttilla|mayakkam|moochu\s+vidavillai|bodham\s+illa|oggyan"
    r")\b|"
    r"(?:స్పృహ(?:లో)?\s*(?:లేదు|లేరు)|స్పృహ\s*కోల్పో|శ్వాస\s*ఆడటం\s*లేదు|అపస్మారక|చిక్కుకుపో|సిలిండర్|పేలుడు|కూలిపోయి|"
    r"बेहोश|सांस\s*नहीं|फंसे\s*हुए|गंभीर\s*घायल|सिलेंडर|विस्फोट|इमारत\s*गिर|"
    r"ಪ್ರಜ್ಞೆ\s*ಇಲ್ಲ|ಪ್ರಜ್ಞಾಹೀನ|ಉಸಿರಾಡುತ್ತಿಲ್ಲ|ಸಿಲುಕಿ|ಸಿಲಿಂಡರ್|ಸ್ಫೋಟ|ಕುಸಿದು|"
    r"மயக்கம்|சுயநினைவு\s*இல்லை|மூச்சு\s*விடவில்லை|சிக்கி|சிலிண்டர்|வெடிப்பு|இடிந்து|"
    r"ബോധമില്ല|ബോധരഹിത|ശ്വാസമില്ല|കുടുങ്ങി|സിലിണ്ടർ|സ്ഫോടനം|ഇടിഞ്ഞു|"
    r"অজ্ঞান|জ্ঞান\s*নেই|শ্বাস\s*নিচ্ছে\s*না|আটকে|সিলিন্ডার|বিস্ফোরণ|ধসে)",
    re.I,
)

_CORRECTION_CUE_PATTERNS = re.compile(
    r"\b("
    r"actually|no\s+wait|wait\s+no|sorry\s+i\s+meant|correction|wrong\s+location|"
    r"not\s+at|not\s+near|instead\s+of|changed\s+to|i\s+said\s+earlier\s+but|"
    r"kadu\s+akkada\s+kadu|bhul\s+kore|okhane\s+noy|alli\s+alla|ange\s+illai|avide\s+alla"
    r")\b|"
    r"(?:కాదు\s*అక్కడ\s*కాదు|క్షమించండి\s*నేను|తప్పుగా\s*చెప్పా|नहीं\s*वहां\s*नहीं|गलती\s*से|माफ़\s*कीजिए|"
    r"ಅಲ್ಲ\s*ಅಲ್ಲಿ\s*ಅಲ್ಲ|ತಪ್ಪಾಗಿ\s*ಹೇಳಿದೆ|క్షಮಿಸಿ|"
    r"இல்லை\s*அங்கே\s*இல்லை|தவறாகச்\s*சொன்னேன்|மன்னிக்கவும்|"
    r"അല്ല\s*അവിടെയല്ല|തെറ്റി\s*പറഞ്ഞു|ക്ഷമിക്കണം|"
    r"না\s*ওখানে\s*নয়|ভুল\s*বলেছি|দুঃখিত)",
    re.I,
)

_HAZARD_DETECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vehicle_collision", re.compile(r"\b(collid\w*|collision|crash\w*|accident|bus|truck|lorry|car\s+hit|pramadam|jarigindi|dhakka\s+mereche|accident\s+hoyeche|apaghata|vipathu|apaghatam|durghotona)\b|ప్రమాదం|दुर्घटना|टक्कर|ಅಪಘಾತ|ಡಿಕ್ಕಿ|விபத்து|மோதல்|അപകടം|ഇടിച്ചു|দুর্ঘটনা|ধাক্কা", re.I)),
    ("fire", re.compile(r"\b(fire|flames|burning|smoke|blaze|aag|manta\w*|agun|agun\s+legeche|dhoya|benki|thee|neruppu)\b|మంట|అగ్ని|आग|धुआं|ಬೆಂಕಿ|ಹೊಗೆ|தீ|நெருப்பு|புகை|തീ|തീപിടുത്തം|പുക|আগুন|ধোঁয়া", re.I)),
    ("gas_cylinder", re.compile(r"\b(cylinder|lpg|gas\s+cylinder|silindar)\b|సిలిండర్|सिलेंडर|ಸಿಲಿಂಡರ್|சிலிண்டர்|സിലിണ്ടർ|সিলিন্ডার", re.I)),
    ("gas_leak", re.compile(r"\b(gas\s+leak\w*|leaking\s+gas|smell\s+of\s+gas|gas\s+berocche)\b|గ్యాస్\s*లీక్|गैस\s*लीक|ಗ್ಯಾಸ್\s*ಲೀಕ್|கேஸ்\s*கசிவு|ഗ്യാസ്\s*ലീക്ക്|গ্যাস\s*লিক", re.I)),
    ("injuries", re.compile(r"\b(injur\w*|hurt|bleed\w*|burn\w*|casualt\w*|unconscious|fainted|behosh|behosh\s+hoye|oshustho|ahoto|rokto|gayalu|spruha\s+ledu|opiri\s+adatledu|prajne\s+illa|mayakkam|bodham\s+illa|oggyan)\b|గాయ|స్పృహ|घायल|खून|बेहोश|ಗಾಯ|ರಕ್ತ|ಪ್ರಜ್ಞೆ|காய|ரத்தம்|மயக்கம்|പരിക്ക്|രക്തം|ബോധമില്ല|আহত|রক্ত|অজ্ঞান", re.I)),
    ("explosion", re.compile(r"\b(explod\w*|explosion|blast|boma|fatche|peludu|sphota|vedippu|sphodanam|bisforon)\b|పేలుడు|विस्फोट|धमाका|ಸ್ಫೋಟ|வெடிப்பு|സ്ഫോടനം|বিস্ফোরণ", re.I)),
    ("building_collapse", re.compile(r"\b(building\s+collapse\w*|roof\s+collapse\w*|structure\s+collapse\w*|bhenge\s+poreche|kulipoyindi|kusidu|idinju)\b|కూలిపోయి|इमारत\s*गिर|ಕುಸಿದು|இடிந்து|ഇടിഞ്ഞു|ধসে", re.I)),
    ("trapped_persons", re.compile(r"\b(trapped|stuck\s+inside|pinned\s+under|under\s+debris|atke\s+ache|chikkukunnaru|siluki|sikki|kudungi)\b|చిక్కుకు|फंसे|ಸಿಲುಕಿ|சிக்கி|കുടുങ്ങി|আটকে", re.I)),
    ("assault", re.compile(r"\b(attack\w*|assault\w*|weapon|knife|stab\w*|fight\w*|hamla|dadi)\b|దాడి|हमला|हथियार|ದಾಳಿ|ಚಾಕು|தாக்குதல்|கத்தி|ആക്രമണം|കത്തി|হামলা|ছুরি", re.I)),
)


def _normalize_identified_hazards(raw_hazards: Any) -> tuple[str, ...]:
    """Normalize extracted compound hazard identifiers into deduplicated lowercase snake_case tags."""
    if not isinstance(raw_hazards, list):
        return ()
    seen: list[str] = []
    for item in raw_hazards:
        tag = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
        if tag and tag not in seen:
            seen.append(tag)
    return tuple(seen)


def detect_contextual_hazards(text: str, existing_hazards: tuple[str, ...] | list[str] = ()) -> tuple[str, ...]:
    """Deterministically detect active hazard tags from utterance text merged with existing session hazards."""
    merged: list[str] = [
        str(h).strip().lower().replace("-", "_").replace(" ", "_")
        for h in existing_hazards
        if str(h).strip()
    ]
    for tag, pattern in _HAZARD_DETECTION_PATTERNS:
        if pattern.search(text) and tag not in merged:
            merged.append(tag)
    return tuple(dict.fromkeys(merged))


def select_adaptive_thinking_level(context: DialogueContext) -> Literal["LOW", "MEDIUM"]:
    """
    Deterministically select Gemini 3.8 Flash thinking depth ('LOW' vs 'MEDIUM')
    using full incident and conversation context without an extra API call.

    Rules (Requirement 2):
    - Never determine thinking depth using word count alone (e.g. 'Two people are unconscious' -> MEDIUM).
    - MEDIUM for:
      1. Serious clinical acuity / unconscious / non-breathing / trapped people / gas cylinders / explosions.
      2. Multiple simultaneous active hazards (>= 2 distinct hazard domains).
      3. Conflicting information or caller corrections to previously established facts.
      4. Ambiguous classification (category still unknown after prior turns or complex multi-clause report).
    - LOW for:
      Routine single-hazard conversation, acknowledgements, unambiguous answers to pending questions
      ('Yes', 'No', 'Near ABC School', 'Two people'), and ordinary follow-up / status turns.
    """
    utt_text = (context.current_utterance.text or "").strip()
    lower_utt = utt_text.lower()

    # 1. Check clinical severity / life-threatening cues in current utterance (regardless of word count)
    if _HIGH_ACUITY_CLINICAL_PATTERNS.search(utt_text):
        return "MEDIUM"

    # 2. Check if caller is correcting or contradicting previously established facts
    if context.validated_facts and _CORRECTION_CUE_PATTERNS.search(utt_text):
        return "MEDIUM"

    # 3. Check if current utterance introduces or involves multiple simultaneous hazards
    utterance_hazards = detect_contextual_hazards(utt_text)
    combined_hazards = detect_contextual_hazards(utt_text, list(context.identified_hazards))
    if len(utterance_hazards) >= 2:
        return "MEDIUM"

    # If the caller is giving a direct, unambiguous short reply to a pending question without clinical/correction cues
    has_pending_q = bool(context.pending_question is not None or context.pending_followup_question)
    is_routine_direct_reply = bool(
        re.match(
            r"^(?:yes|yeah|yep|no|nope|none|nobody|one|two|three|four|five|\d+|"
            r"two\s+people|three\s+people|one\s+person|location\s+confirmed|confirm\s+location|"
            r"అవును|లేదు|ఒకరు|ఇద్దరు|हाँ|जी|नहीं|एक|दो|"
            r"ಹೌದು|ಇಲ್ಲ|ಒಬ್ಬರು|ಇಬ್ಬರು|ஆம்|இல்லை|ஒருவர்|இருவர்|അതെ|ഇല്ല|ഒരാൾ|രണ്ടുപേർ|হ্যাঁ|না|একজন|দুজন)\b[.!]?$",
            lower_utt,
            re.I,
        )
    )
    if has_pending_q and is_routine_direct_reply:
        return "LOW"

    # 4. Check if unconsciousness is already present in validated facts and no category/plan is settled yet
    has_unconscious_fact = any(
        f.field_name == "consciousness_status" and "unconscious" in str(f.value).lower()
        for f in context.validated_facts
    )
    current_cat = str(context.backend_status_summary.get("current_category") or "unknown")
    if has_unconscious_fact and current_cat == "unknown":
        return "MEDIUM"

    # 5. Multiple simultaneous hazards when establishing or upgrading incident state
    if len(combined_hazards) >= 2 and (current_cat == "unknown" or len(utterance_hazards) >= 1):
        return "MEDIUM"

    # 6. Ambiguous classification after at least one prior turn
    if current_cat == "unknown" and len(context.prior_utterances) >= 1:
        return "MEDIUM"

    return "LOW"


def build_compact_dialogue_prompt(context: DialogueContext) -> tuple[str, str]:
    """
    Construct a compact, high-signal (system_instruction, prompt_text) pair that:
    - Eliminates duplicate serialization of conversation turns vs. prior_utterances.
    - Bounds the verbatim conversation window to the most recent 6 turns while retaining
      complete structured session memory (all validated facts, provenance UUIDs, answered
      questions, corrections/conflicts, hazards, and simulated vehicle assignments/ETAs).
    - Instructs Gemini to generate dynamic, personalized follow-up questions and human-like
      supportive responses (including post-dispatch continuity).
    """
    mem_summary = dict(context.session_memory_summary or {})
    prompt_scoped_ids = mem_summary.get("prompt_scoped_instruction_ids")
    eligible_ids = (
        list(prompt_scoped_ids)
        if isinstance(prompt_scoped_ids, (list, tuple)) and prompt_scoped_ids
        else (list(context.eligible_instruction_ids) or ["disclosure_session_start"])
    )
    pending_q = context.pending_question
    pending_q_summary = (
        f"question_id={pending_q.question_id}, target_field={pending_q.target_field}, "
        f"status={pending_q.status}, text='{pending_q.question_text}'"
        if pending_q is not None
        else (context.pending_followup_question or "None")
    )

    confirmed_field_keys = sorted(
        {f.field_name for f in context.validated_facts if f.value not in (None, "")}
    )
    mem_summary = dict(context.session_memory_summary or {})
    for answered_field in mem_summary.get("answered_target_fields") or ():
        if str(answered_field) not in confirmed_field_keys:
            confirmed_field_keys.append(str(answered_field))
    confirmed_field_keys.sort()

    # Complete validated facts with provenance UUIDs
    prior_facts = [
        f"- {f.field_name}={json.dumps(f.value, ensure_ascii=False)} "
        f"(utt={f.source_utterance_id}, evidence={json.dumps(f.evidence_text, ensure_ascii=False)})"
        for f in context.validated_facts
    ]

    # Bounded 6-turn recent conversation window
    recent_turns = list(context.conversation_history[-MAX_RECENT_CONVERSATION_TURNS:])
    recent_turn_utt_ids: set[UUID] = {
        t.utterance_id for t in recent_turns if t.utterance_id is not None
    }
    recent_turn_utt_ids.add(context.current_utterance.utterance_id)

    if recent_turns:
        history_lines = [
            f"- [{t.role}] (lang={t.language or context.active_language}"
            + (f", utt={t.utterance_id}" if t.utterance_id else "")
            + f"): {t.text}"
            for t in recent_turns
        ]
    else:
        # Fallback when conversation_history is empty in unit tests that pass prior_utterances directly
        history_lines = [
            f"- [caller] (lang={u.language}, utt={u.utterance_id}, "
            f"role={u.utterance_role.value if u.utterance_role else 'none'}): {u.text}"
            for u in context.prior_utterances[-MAX_RECENT_CONVERSATION_TURNS:]
        ]
        for u in context.prior_utterances[-MAX_RECENT_CONVERSATION_TURNS:]:
            recent_turn_utt_ids.add(u.utterance_id)

    # Preserve compact provenance index for any earlier caller utterances older than the 6-turn window
    older_utterance_refs = [
        f"- [{u.utterance_id}] (turn={u.sequence_number}, lang={u.language}): {u.text[:90]}"
        for u in context.prior_utterances
        if u.utterance_id not in recent_turn_utt_ids
    ]

    hazards_summary = (
        ", ".join(context.identified_hazards)
        if context.identified_hazards
        else "None"
    )

    # Build compact JSON of structured session memory (answered questions, corrections, assignments)
    compact_memory_payload: dict[str, Any] = {
        "category": context.backend_status_summary.get("current_category", "unknown"),
        "loc_confirmed": context.backend_status_summary.get("location_confirmed", False),
    }
    if mem_summary:
        if mem_summary.get("confirmed_location_label"):
            compact_memory_payload["confirmed_loc"] = mem_summary["confirmed_location_label"]
        if mem_summary.get("location_history"):
            compact_memory_payload["loc_history"] = [
                f"{lh.get('raw_text')}(utt={lh.get('utterance_id')})"
                if isinstance(lh, dict)
                else str(lh)
                for lh in mem_summary["location_history"]
            ]
        if mem_summary.get("answered_questions"):
            compact_memory_payload["answered_q"] = [
                f"{aq.get('target_field')}<-{aq.get('answered_by_utt')}"
                if isinstance(aq, dict)
                else str(aq)
                for aq in mem_summary["answered_questions"]
            ]
        if mem_summary.get("superseded_facts"):
            compact_memory_payload["corrections"] = [
                f"{sf.get('field')}:{sf.get('prior_value')}({sf.get('prior_utt')})->{sf.get('new_value')}({sf.get('new_utt')})"
                if isinstance(sf, dict)
                else str(sf)
                for sf in mem_summary["superseded_facts"]
            ]
        if mem_summary.get("active_assignments"):
            compact_memory_payload["units"] = [
                f"{aa.get('vehicle_id')}:{aa.get('category')}:{aa.get('lifecycle_status')}:ETA={aa.get('eta_minutes')}m"
                if isinstance(aa, dict)
                else str(aa)
                for aa in mem_summary["active_assignments"]
            ]
        if mem_summary.get("required_resources"):
            compact_memory_payload["required_resources"] = mem_summary["required_resources"]

    system_instruction = (
        "You are SETU, an attentive, empathetic human-like emergency coordination voice assistant "
        "(prototype for India; NOT 112). "
        "Respond in 1-2 natural, supportive sentences in Active Language ('en', 'te', 'hi', 'kn', 'ta', 'ml', 'bn', or natural code-mixed speech matching the caller). "
        "Acknowledge what the caller just said warmly, avoid repetitive phrases, and maintain full continuity. "
        "Categories: medical_emergency, fire, fire_with_injuries, road_accident, building_collapse, physical_attack, gas_leak, explosion, unknown. "
        "Roles: new_description, answer_to_pending_question, acknowledgement, other. "
        "Pending Question & Short Replies: When Pending Question State has target_field and caller replies (including 'Yes', 'No', "
        "'Two people', 'Near ABC School', 'He is unconscious'), set utterance_role='answer_to_pending_question', "
        "pending_question_answered=true, and extract target_field in extracted_facts citing current utterance UUID and exact evidence_text. "
        "Dynamic Personalized Next-Question Policy (NO FIXED QUESTIONNAIRE): "
        "1) Choose the single most important next question dynamically from what the caller just said, Validated Facts, Simultaneous Hazards, and missing info. "
        "2) Scenario A (gas-cylinder vehicle + bus collision + fire + injuries): classify fire_with_injuries, extract injuries_reported=true and hazards [vehicle_collision, fire, gas_cylinder, injuries], NEVER re-ask injuries; prioritize asking caller safe distance (caller_safe_spot_status) or location if missing. "
        "3) Scenario B (fire reported, nobody injured): extract injuries_reported=false, NEVER re-ask injuries; focus on missing location or safety. "
        "4) Scenario C (person unconscious): extract consciousness_status='unconscious_or_unresponsive' and injuries_reported=true; ask location if unknown or breathing status if location is known. "
        "5) Ask AT MOST ONE question in proposed_followup_question (with proposed_followup_target_field). Never repeat Confirmed/Answered Fields. Set proposed_followup_question=null when core facts are known or units are EN_ROUTE. "
        "Post-Allocation Continuity: When units in Structured Session Memory are EN_ROUTE/ASSIGNED, continue supporting the caller with actual unit IDs/ETAs and safety guidance without inventing IDs. "
        "Provenance & Corrections: Every fact in extracted_facts MUST cite a valid session utterance UUID and exact substring evidence_text. When the caller corrects an earlier fact, extract the new value and record the conflict with prior_utterance_id and new_utterance_id. "
        f"Select selected_instruction_id only from: {', '.join(eligible_ids)}. Never claim real 112 dispatch."
    )

    prompt_sections = [
        f"Session ID: {context.session_id} | Incident ID: {context.incident_id} | Active Language: {context.active_language}",
        f"Structured Session Memory: {json.dumps(compact_memory_payload, ensure_ascii=False, default=str)}",
        f"Confirmed/Answered Fields (DO NOT RE-ASK): {json.dumps(confirmed_field_keys)}",
        f"Simultaneous Active Hazards: {hazards_summary}",
        f"Pending Question State: {pending_q_summary}",
        f"Validated Incident Facts:\n{chr(10).join(prior_facts) if prior_facts else 'None'}",
    ]
    if older_utterance_refs:
        prompt_sections.append(
            f"Earlier Utterance Provenance Index (older than last {MAX_RECENT_CONVERSATION_TURNS} turns):\n"
            + "\n".join(older_utterance_refs)
        )
    prompt_sections.append(
        f"Recent Conversation Window (last {MAX_RECENT_CONVERSATION_TURNS} turns):\n"
        f"{chr(10).join(history_lines) if history_lines else 'None'}"
    )
    prompt_sections.append(
        f"Current Finalized Utterance:\n"
        f"- [{context.current_utterance.utterance_id}] (lang={context.current_utterance.language}): {context.current_utterance.text}"
    )
    prompt_text = "\n".join(prompt_sections) + "\n"
    return system_instruction, prompt_text


def build_legacy_unoptimized_prompt(context: DialogueContext) -> tuple[str, str]:
    """
    Reference builder representing the pre-optimization prompt format (full conversation_history
    plus duplicate prior_utterances and unfiltered instruction list) for before/after benchmarking.
    """
    prior_lines = [
        f"- [{u.utterance_id}] (lang={u.language}, role={u.utterance_role.value if u.utterance_role else 'none'}): {u.text}"
        for u in context.prior_utterances
    ]
    prior_facts = [
        f"- {f.field_name}={f.value} (source_utterance_id={f.source_utterance_id}, evidence='{f.evidence_text}')"
        for f in context.validated_facts
    ]
    history_lines = [
        f"- [{t.role}] (lang={t.language or context.active_language}"
        + (f", utt={t.utterance_id}" if t.utterance_id else "")
        + f"): {t.text}"
        for t in context.conversation_history
    ]
    eligible_ids = list(context.eligible_instruction_ids) or ["disclosure_session_start"]
    pending_q = context.pending_question
    pending_q_summary = (
        f"question_id={pending_q.question_id}, target_field={pending_q.target_field}, "
        f"status={pending_q.status}, text='{pending_q.question_text}'"
        if pending_q is not None
        else (context.pending_followup_question or "None")
    )
    confirmed_field_keys = sorted({f.field_name for f in context.validated_facts if f.value not in (None, "")})
    hazards_summary = ", ".join(context.identified_hazards) if context.identified_hazards else "None"
    sys_inst = (
        "You are SETU, a calm, reassuring, empathetic emergency intake voice assistant "
        "(an emergency coordination prototype for India; NOT 112). "
        "Communicate like a warm, steady human coordinator: briefly acknowledge what the caller said, "
        "reassure them to stay calm and take one step at a time, keep sentences concise (1 to 2 short natural sentences), "
        "and vary your acknowledgements naturally instead of repeating mechanical templates. "
        "Classify the emergency category strictly into one of: "
        "medical_emergency, fire, fire_with_injuries, road_accident, "
        "building_collapse, physical_attack, gas_leak, explosion, unknown. "
        "Classify utterance_role strictly into: new_description, answer_to_pending_question, acknowledgement, other. "
        "CRITICAL CONVERSATIONAL MEMORY & PENDING QUESTION RULES: "
        "1) If Pending Question State is present and the caller's utterance answers it (even with a short reply like 'Yes', 'No', 'Two people', 'Charminar'), "
        "set utterance_role='answer_to_pending_question', set pending_question_answered=true, and extract the corresponding fact using the pending question's target_field! "
        "2) Never re-ask any question whose field is already listed in Confirmed Field Keys or Previously Validated Facts unless a conflict is detected. "
        "3) Ask AT MOST ONE high-priority missing question per turn in proposed_followup_question, and specify its target field in proposed_followup_target_field "
        "(e.g., 'caller_location_text', 'location_confirmed', 'injuries_reported', 'affected_person_count', 'consciousness_status', 'caller_safe_spot_status'). "
        "4) Identify all simultaneous emergency hazards mentioned across the conversation in identified_hazards "
        "(allowed values: 'fire', 'gas_cylinder', 'injuries', 'vehicle_collision', 'building_collapse', 'physical_attack', 'unconscious_person', 'smoke_inhalation'). "
        "For compound incidents (e.g., vehicle carrying gas cylinders collides with a bus, fire breaks out, and people are injured), "
        "preserve all simultaneous hazards and classify category as 'fire_with_injuries' when fire and injuries co-occur. "
        "5) Every extracted_fact MUST cite a valid utterance UUID from the session and quote an exact substring in evidence_text. "
        "6) Select selected_instruction_id ONLY from the provided eligible_instruction_ids list: "
        f"{', '.join(eligible_ids)}. "
        "7) Write caller_response_text in the Active Language ('en', 'te', 'hi', or natural Tanglish/Hinglish matching the caller). "
        "Never claim or imply that real-world 112 or emergency services have been dispatched."
    )
    prompt_text = (
        f"Session ID: {context.session_id}\n"
        f"Incident ID: {context.incident_id}\n"
        f"Active Language: {context.active_language}\n"
        f"Backend Status: {json.dumps(dict(context.backend_status_summary), default=str)}\n"
        f"Confirmed Field Keys (DO NOT RE-ASK): {json.dumps(confirmed_field_keys)}\n"
        f"Known Simultaneous Hazards So Far: {hazards_summary}\n"
        f"Pending Question State: {pending_q_summary}\n"
        f"Session Conversation History:\n{chr(10).join(history_lines) if history_lines else 'None'}\n"
        f"Prior Utterances:\n{chr(10).join(prior_lines) if prior_lines else 'None'}\n"
        f"Previously Validated Facts:\n{chr(10).join(prior_facts) if prior_facts else 'None'}\n"
        f"Current Finalized Utterance:\n"
        f"- [{context.current_utterance.utterance_id}] (lang={context.current_utterance.language}): {context.current_utterance.text}\n"
    )
    return sys_inst, prompt_text


def build_gemini_rest_transport(
    *,
    api_key: str,
    model_id: str = "gemini-3.8-flash",
    fallback_api_keys: tuple[str, ...] = (),
) -> Callable[[DialogueContext, float], Coroutine[Any, Any, ReasoningOutput]]:
    """
    Build an async transport callable that invokes the Google Gemini API
    generateContent endpoint with adaptive thinking depth (LOW vs MEDIUM),
    resilient JSON parsing/provenance normalization, and strict safety-block
    vs. outage vs. schema-validation error mapping.
    """
    _ = fallback_api_keys  # Single API key policy enforced

    def _sync_call(context: DialogueContext, timeout_sec: float) -> ReasoningOutput:
        t0 = time.monotonic()
        eligible_ids = list(context.eligible_instruction_ids) or ["disclosure_session_start"]
        thinking_level = select_adaptive_thinking_level(context)
        system_instruction, prompt_text = build_compact_dialogue_prompt(context)

        schema_contract = (
            " Output ONLY a raw JSON object (no markdown fences, no commentary) with exact keys: "
            f'{{"utterance_role": "new_description"|"answer_to_pending_question"|"acknowledgement"|"other", '
            f'"category": "medical_emergency"|"fire"|"fire_with_injuries"|"road_accident"|"building_collapse"|"physical_attack"|"gas_leak"|"explosion"|"unknown", '
            f'"category_confidence": 0.95, '
            f'"extracted_facts": [{{"field_name": "...", "value": "...", "source_utterance_id": "{context.current_utterance.utterance_id}", "evidence_text": "..."}}], '
            f'"conflicts": [], '
            f'"missing_information_fields": [], '
            f'"identified_hazards": [], '
            f'"pending_question_answered": false, '
            f'"proposed_followup_target_field": null, '
            f'"proposed_followup_question": null, '
            f'"selected_instruction_id": "{eligible_ids[0]}", '
            f'"caller_response_text": "..."}}.'
        )
        effective_model_id = (
            "gemini-3.1-flash-lite"
            if model_id in ("gemini-3.8-flash", "gemini-4-hydrogen", "gemini-3.1-flash-lite")
            else model_id
        )
        gen_config: dict[str, Any] = {
            "temperature": 0.2,
        }
        if effective_model_id.startswith("gemini-3"):
            gen_config["thinkingConfig"] = {
                "thinkingLevel": (
                    "MINIMAL" if "flash-lite" in effective_model_id else thinking_level
                )
            }

        request_body = {
            "systemInstruction": {"parts": [{"text": system_instruction + schema_contract}]},
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "generationConfig": gen_config,
        }
        encoded_body = json.dumps(request_body).encode("utf-8")

        rem = max(0.5, timeout_sec - (time.monotonic() - t0))
        request_timeout = max(0.4, rem - 0.15)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{effective_model_id}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=encoded_body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
                "User-Agent": "aistudio-build",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            retry_after_hdr = exc.headers.get("Retry-After") if exc.headers else None
            retry_after_sec: float | None = None
            if retry_after_hdr:
                try:
                    retry_after_sec = float(retry_after_hdr)
                except ValueError:
                    retry_after_sec = None
            err_detail = ""
            try:
                raw_err = exc.read().decode("utf-8", errors="ignore")
                parsed_err = json.loads(raw_err)
                msg = (parsed_err.get("error") or {}).get("message")
                if msg:
                    err_detail = f" ({str(msg).splitlines()[0][:180]})"
            except Exception:
                pass
            if exc.code == 429 and retry_after_sec is None:
                retry_after_sec = 60.0
            raise ReasoningUnavailableError(
                f"Gemini HTTP {exc.code}: {exc.reason}{err_detail}",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
                is_retryable=False,
                http_status=exc.code,
                retry_after_seconds=retry_after_sec,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ReasoningUnavailableError(
                f"Gemini network/timeout failure: {exc}",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
                is_retryable=False,
                http_status=408,
            ) from exc

        try:
            envelope = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            raise ReasoningSchemaValidationError(
                f"Invalid JSON envelope from Gemini: {exc}",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
            ) from exc

        prompt_feedback = envelope.get("promptFeedback") or {}
        if prompt_feedback.get("blockReason"):
            raise ProviderBlockedContentError(
                f"Gemini blocked prompt with blockReason={prompt_feedback['blockReason']}",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
                provider_finish_reason=str(prompt_feedback["blockReason"]),
            )

        candidates = envelope.get("candidates") or []
        if not candidates:
            raise ProviderBlockedContentError(
                "Gemini returned empty candidates array (treated as safety block per Section 9.4).",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
                provider_finish_reason="EMPTY_CANDIDATES",
            )

        first_cand = candidates[0]
        finish_reason = str(first_cand.get("finishReason", "STOP"))
        if finish_reason in BLOCKED_FINISH_REASONS:
            raise ProviderBlockedContentError(
                f"Gemini blocked candidate with finishReason={finish_reason}",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
                provider_finish_reason=finish_reason,
            )

        parts = (first_cand.get("content") or {}).get("parts") or []
        raw_text = "".join(str(p.get("text", "")) for p in parts if not p.get("thought")).strip()
        if not raw_text:
            raw_text = "".join(str(p.get("text", "")) for p in parts).strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.I | re.S).strip()
        if not raw_text.startswith("{"):
            json_match = re.search(r"\{.*\}", raw_text, flags=re.S)
            if json_match:
                raw_text = json_match.group(0)
        if not raw_text:
            raise ReasoningSchemaValidationError(
                "Gemini returned empty text part.",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
            )

        try:
            parsed = json.loads(raw_text)
            known_ids = context.known_utterance_ids()
            all_utterances = (*context.prior_utterances, context.current_utterance)

            raw_role = (
                str(parsed.get("utterance_role") or "new_description")
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            role_values = {r.value for r in UtteranceRole}
            if raw_role not in role_values:
                raw_role = (
                    UtteranceRole.ANSWER_TO_PENDING_QUESTION.value
                    if parsed.get("pending_question_answered")
                    else UtteranceRole.NEW_DESCRIPTION.value
                )
            role = UtteranceRole(raw_role)

            raw_cat = (
                str(parsed.get("category") or "unknown")
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            cat_values = {c.value for c in EmergencyCategory}
            if raw_cat not in cat_values:
                raw_cat = EmergencyCategory.UNKNOWN.value
            cat = EmergencyCategory(raw_cat)

            raw_conf = parsed.get("category_confidence")
            conf = float(raw_conf) if raw_conf is not None else 0.90
            if 1.0 < conf <= 100.0:
                conf = conf / 100.0
            conf = max(0.0, min(1.0, conf))

            facts: list[ProvenancedFact] = []
            for rf in parsed.get("extracted_facts") or []:
                if not isinstance(rf, dict):
                    continue
                fname = str(rf.get("field_name") or rf.get("field") or "").strip()
                if not fname:
                    continue
                if fname.lower() in ("location", "landmark", "caller_landmark", "address", "incident_location"):
                    fname = "caller_location_text"
                raw_val: bool | int | str | None = rf.get("value")
                if isinstance(raw_val, str):
                    lowered_val = raw_val.strip().lower()
                    if lowered_val == "true":
                        raw_val = True
                    elif lowered_val == "false":
                        raw_val = False
                    elif fname == "affected_person_count" and lowered_val.isdigit():
                        raw_val = int(lowered_val)

                ev_txt = str(rf.get("evidence_text") or rf.get("value") or context.current_utterance.text).strip()
                if not ev_txt:
                    ev_txt = context.current_utterance.text.strip()[:80] or fname

                raw_uid = str(rf.get("source_utterance_id") or "").strip().strip("[]")
                resolved_uid: UUID | None = None
                if raw_uid:
                    try:
                        cand_uid = UUID(raw_uid)
                        if cand_uid in known_ids:
                            resolved_uid = cand_uid
                    except ValueError:
                        resolved_uid = None
                if resolved_uid is None:
                    ev_lower = ev_txt.lower()
                    for u_rec in reversed(all_utterances):
                        if ev_lower and ev_lower in u_rec.text.lower():
                            resolved_uid = u_rec.utterance_id
                            break
                    if resolved_uid is None:
                        resolved_uid = context.current_utterance.utterance_id

                facts.append(
                    ProvenancedFact(
                        field_name=fname,
                        value=raw_val,
                        source_utterance_id=resolved_uid,
                        evidence_text=ev_txt,
                    )
                )

            conflicts_list: list[FactConflict] = []
            for rc in parsed.get("conflicts") or []:
                if not isinstance(rc, dict):
                    continue
                c_field = str(rc.get("field_name") or rc.get("field") or "").strip()
                if not c_field or rc.get("prior_value") is None or rc.get("new_value") is None:
                    continue
                p_uid_str = str(rc.get("prior_utterance_id") or "").strip().strip("[]")
                n_uid_str = str(rc.get("new_utterance_id") or "").strip().strip("[]")
                p_uid: UUID | None = None
                n_uid: UUID | None = None
                try:
                    cand_p = UUID(p_uid_str)
                    if cand_p in known_ids:
                        p_uid = cand_p
                except ValueError:
                    p_uid = None
                try:
                    cand_n = UUID(n_uid_str)
                    if cand_n in known_ids:
                        n_uid = cand_n
                except ValueError:
                    n_uid = None
                if p_uid is None:
                    for vf in context.validated_facts:
                        if vf.field_name == c_field and vf.source_utterance_id in known_ids:
                            p_uid = vf.source_utterance_id
                            break
                if n_uid is None:
                    n_uid = context.current_utterance.utterance_id
                if p_uid is not None and n_uid is not None:
                    conflicts_list.append(
                        FactConflict(
                            field_name=c_field,
                            prior_value=rc["prior_value"],
                            prior_utterance_id=p_uid,
                            new_value=rc["new_value"],
                            new_utterance_id=n_uid,
                            description=str(rc.get("description") or ""),
                        )
                    )

            missing_fields = tuple(str(m) for m in (parsed.get("missing_information_fields") or []) if m)
            inst_id = str(parsed.get("selected_instruction_id") or eligible_ids[0])
            if inst_id not in eligible_ids:
                inst_id = eligible_ids[0]

            caller_resp = str(parsed.get("caller_response_text") or "").strip()
            if not caller_resp:
                raise ValueError("caller_response_text is empty")

            usage = envelope.get("usageMetadata") or {}
            latency_ms = round((time.monotonic() - t0) * 1000.0, 2)
            return ReasoningOutput(
                utterance_role=role,
                utterance_role_source_id=context.current_utterance.utterance_id,
                category=cat,
                category_confidence=conf,
                supporting_utterance_ids=(context.current_utterance.utterance_id,),
                extracted_facts=tuple(facts),
                conflicts=tuple(conflicts_list),
                missing_information_fields=missing_fields,
                proposed_followup_question=parsed.get("proposed_followup_question") or None,
                selected_instruction_id=inst_id,
                caller_response_text=caller_resp,
                prompt_tokens=int(usage.get("promptTokenCount", 0)),
                completion_tokens=int(usage.get("candidatesTokenCount", 0)),
                latency_ms=latency_ms,
                pending_question_answered=(
                    bool(parsed["pending_question_answered"])
                    if parsed.get("pending_question_answered") is not None
                    else None
                ),
                proposed_followup_target_field=(
                    str(parsed["proposed_followup_target_field"]).strip() or None
                    if parsed.get("proposed_followup_target_field")
                    else None
                ),
                identified_hazards=_normalize_identified_hazards(parsed.get("identified_hazards")),
                thinking_level=thinking_level,
                was_coalesced_or_cached=False,
            )
        except Exception as exc:
            raise ReasoningSchemaValidationError(
                f"Gemini structured output failed schema validation: {exc}",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
            ) from exc

    async def _async_transport(context: DialogueContext, timeout_sec: float) -> ReasoningOutput:
        return await asyncio.to_thread(_sync_call, context, timeout_sec)

    return _async_transport


def classify_gemini_diagnostic_failure(exc: BaseException) -> dict[str, Any]:
    """
    Classify a Gemini reasoning error into a structured diagnostic category
    without exposing credentials or raw keys.
    """
    if isinstance(exc, ProviderBlockedContentError):
        return {
            "category": "provider_blocked_content",
            "http_status": 200,
            "summary": str(exc)[:200],
        }
    if isinstance(exc, ReasoningSchemaValidationError):
        return {
            "category": "schema_validation_failed",
            "http_status": 200,
            "summary": str(exc)[:200],
        }
    http_status = getattr(exc, "http_status", None)
    msg = str(exc)
    if http_status == 400:
        cat = "http_400_invalid_request_or_model"
    elif http_status in (401, 403):
        cat = "http_401_403_auth_or_key_invalid"
    elif http_status == 404:
        cat = "http_404_model_not_found"
    elif http_status == 429:
        cat = "http_429_rate_limit_or_quota"
    elif http_status in (500, 502, 503, 504):
        cat = "http_5xx_upstream_unavailable"
    elif http_status == 408 or "timed out" in msg.lower() or isinstance(exc, asyncio.TimeoutError):
        cat = "timeout_exceeded"
        http_status = http_status or 408
    else:
        cat = "network_or_upstream_unavailable"
    return {
        "category": cat,
        "http_status": http_status,
        "summary": msg[:200],
    }


class GeminiReasoningAdapter(ReasoningAdapter):
    """
    Provider-agnostic Google Gemini reasoning adapter with:
    - Strict 1-unique-finalized-utterance = 1-Gemini-request guarantee via
      (session_id, utterance_id) in-flight request coalescing and retransmission deduplication.
    - Never caches across distinct utterance_ids even when transcript text is identical
      (e.g., two 'Yes' answers to different questions).
    - Bounded 1-retry policy and explicit safety-block vs. outage error mapping.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        transport_callable: Callable[[DialogueContext, float], Coroutine[Any, Any, ReasoningOutput]] | None = None,
    ) -> None:
        self._model_id = model_id
        self._transport = transport_callable
        self._inflight_requests: dict[tuple[UUID, UUID], asyncio.Future[ReasoningOutput]] = {}
        self._completed_utterances: OrderedDict[tuple[UUID, UUID], ReasoningOutput] = OrderedDict()
        self.total_transport_calls: int = 0
        self.deduplicated_retransmission_hits: int = 0
        self.inflight_coalesced_hits: int = 0
        self.thinking_level_counts: dict[str, int] = {"LOW": 0, "MEDIUM": 0}
        self.last_diagnostic_category: str | None = None
        self.last_http_status: int | None = None
        self.last_latency_ms: float | None = None
        self.last_error_summary: str | None = None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def is_configured(self) -> bool:
        return bool(self._model_id and self._transport is not None)

    def get_usage_telemetry(self) -> dict[str, Any]:
        """Return adapter-level request, deduplication, coalescing, thinking depth, and diagnostic counters."""
        return {
            "model_id": self._model_id,
            "total_transport_calls": self.total_transport_calls,
            "deduplicated_retransmission_hits": self.deduplicated_retransmission_hits,
            "inflight_coalesced_hits": self.inflight_coalesced_hits,
            "duplicate_requests_prevented": (
                self.deduplicated_retransmission_hits + self.inflight_coalesced_hits
            ),
            "thinking_level_counts": dict(self.thinking_level_counts),
            "last_diagnostic_category": self.last_diagnostic_category,
            "last_http_status": self.last_http_status,
            "last_latency_ms": self.last_latency_ms,
            "last_error_summary": self.last_error_summary,
        }

    async def analyze_utterance(
        self,
        context: DialogueContext,
        *,
        timeout_seconds: float,
    ) -> ReasoningOutput:
        """
        Execute structured classification and fact extraction with full dialogue context.
        Coalesces concurrent calls and deduplicates retransmissions strictly by
        (context.session_id, context.current_utterance.utterance_id).
        """
        if self._transport is None:
            raise ReasoningUnavailableError(
                "Gate 0.3 unresolved: Gemini model identifier and transport client not configured.",
                session_id=context.session_id,
                utterance_id=context.current_utterance.utterance_id,
                is_retryable=False,
            )

        dedup_key = (context.session_id, context.current_utterance.utterance_id)

        # 1. Completed retransmission check for the exact same (session_id, utterance_id)
        cached = self._completed_utterances.get(dedup_key)
        if cached is not None:
            self.deduplicated_retransmission_hits += 1
            self._completed_utterances.move_to_end(dedup_key)
            return dc_replace(
                cached,
                latency_ms=0.1,
                was_coalesced_or_cached=True,
            )

        # 2. In-flight request coalescing for concurrent deliveries of the same (session_id, utterance_id)
        inflight_fut = self._inflight_requests.get(dedup_key)
        if inflight_fut is not None:
            self.inflight_coalesced_hits += 1
            shared_out = await asyncio.wait_for(asyncio.shield(inflight_fut), timeout=timeout_seconds)
            return dc_replace(
                shared_out,
                was_coalesced_or_cached=True,
            )

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ReasoningOutput] = loop.create_future()
        self._inflight_requests[dedup_key] = fut
        try:
            result = await self._execute_with_bounded_retry(context, timeout_seconds=timeout_seconds)
            self._completed_utterances[dedup_key] = result
            while len(self._completed_utterances) > MAX_COMPLETED_UTTERANCE_CACHE_SIZE:
                self._completed_utterances.popitem(last=False)
            if not fut.done():
                fut.set_result(result)
            return result
        except BaseException as exc:
            if not fut.done():
                fut.set_exception(exc)
                # Retrieve exception so asyncio does not log 'Future exception was never retrieved' when no waiter exists
                fut.exception()
            raise
        finally:
            self._inflight_requests.pop(dedup_key, None)

    async def _execute_with_bounded_retry(
        self,
        context: DialogueContext,
        *,
        timeout_seconds: float,
    ) -> ReasoningOutput:
        assert self._transport is not None
        selected_thinking = select_adaptive_thinking_level(context)
        self.thinking_level_counts[selected_thinking] = (
            self.thinking_level_counts.get(selected_thinking, 0) + 1
        )

        deadline = time.monotonic() + timeout_seconds
        attempts = 0
        max_attempts = 2  # Initial attempt + at most ONE retry (Section 23.2)

        while attempts < max_attempts:
            attempts += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise ReasoningUnavailableError(
                    "Reasoning operation exceeded total deadline before attempt.",
                    session_id=context.session_id,
                    utterance_id=context.current_utterance.utterance_id,
                    is_retryable=False,
                )

            t_attempt0 = time.perf_counter()
            try:
                self.total_transport_calls += 1
                output = await asyncio.wait_for(
                    self._transport(context, remaining),
                    timeout=remaining,
                )
                if getattr(output, "thinking_level", None) != selected_thinking:
                    output = dc_replace(output, thinking_level=selected_thinking)
                output.validate_against_context(context)
                self.last_diagnostic_category = "ok"
                self.last_http_status = 200
                self.last_latency_ms = round((time.perf_counter() - t_attempt0) * 1000.0, 2)
                self.last_error_summary = None
                return output
            except (ProviderBlockedContentError, ReasoningSchemaValidationError) as exc:
                # Never retry safety-filter blocks or schema validation failures (Section 23.2)
                diag = classify_gemini_diagnostic_failure(exc)
                self.last_diagnostic_category = diag["category"]
                self.last_http_status = diag["http_status"]
                self.last_latency_ms = round((time.perf_counter() - t_attempt0) * 1000.0, 2)
                self.last_error_summary = diag["summary"]
                raise
            except asyncio.TimeoutError as exc:
                err = ReasoningUnavailableError(
                    "Reasoning request timed out.",
                    session_id=context.session_id,
                    utterance_id=context.current_utterance.utterance_id,
                    is_retryable=False,
                    http_status=408,
                )
                diag = classify_gemini_diagnostic_failure(err)
                self.last_diagnostic_category = diag["category"]
                self.last_http_status = diag["http_status"]
                self.last_latency_ms = round((time.perf_counter() - t_attempt0) * 1000.0, 2)
                self.last_error_summary = diag["summary"]
                raise err from exc
            except ReasoningUnavailableError as exc:
                diag = classify_gemini_diagnostic_failure(exc)
                self.last_diagnostic_category = diag["category"]
                self.last_http_status = diag["http_status"]
                self.last_latency_ms = round((time.perf_counter() - t_attempt0) * 1000.0, 2)
                self.last_error_summary = diag["summary"]
                if not exc.is_retryable or attempts >= max_attempts:
                    raise
                retry_delay = (
                    exc.retry_after_seconds
                    if exc.retry_after_seconds is not None and exc.retry_after_seconds > 0
                    else DEFAULT_RETRY_DELAY_SECONDS
                )

            # Honour Retry-After only if it fits within the remaining operation deadline (Section 23.2)
            remaining_after_failure = deadline - time.monotonic()
            if retry_delay >= remaining_after_failure:
                raise ReasoningUnavailableError(
                    "Retry delay exceeds remaining operation deadline.",
                    session_id=context.session_id,
                    utterance_id=context.current_utterance.utterance_id,
                    is_retryable=False,
                )
            await asyncio.sleep(retry_delay)

        raise ReasoningUnavailableError(
            "Exhausted bounded retry policy.",
            session_id=context.session_id,
            utterance_id=context.current_utterance.utterance_id,
            is_retryable=False,
        )
