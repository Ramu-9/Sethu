"""
Multilingual deterministic demo dispatch announcements, lifecycle spoken updates,
and outage-safe keyword fallback classifier for SETU DEMO mode.

All generated spoken text explicitly preserves SETU safety rules:
- Explicitly labels resources, ETAs, and assignments as DEMO / SIMULATED.
- Never claims real-world dispatch, real vehicles on the way, or India 112 connection.
"""

from collections.abc import Sequence
import re
from uuid import UUID

from backend.app.adapters.base import (
    EmergencyCategory,
    PendingQuestionRecord,
    ProvenancedFact,
    SupportedLanguage,
    UtteranceRole,
)
from backend.app.orchestrator.dispatch import (
    AssignmentStatus,
    DemoLifecycleStatus,
    ResourceCategory,
    SimulatedAssignment,
)


_CATEGORY_LABELS: dict[SupportedLanguage, dict[ResourceCategory, str]] = {
    "en": {
        ResourceCategory.AMBULANCE: "ambulance",
        ResourceCategory.FIRE_ENGINE: "fire engine",
        ResourceCategory.FIRE_AND_RESCUE: "fire and rescue unit",
        ResourceCategory.POLICE_PATROL: "police patrol vehicle",
    },
    "te": {
        ResourceCategory.AMBULANCE: "అంబులెన్స్",
        ResourceCategory.FIRE_ENGINE: "ఫైర్ ఇంజిన్",
        ResourceCategory.FIRE_AND_RESCUE: "ఫైర్ మరియు రెస్క్యూ వాహనం",
        ResourceCategory.POLICE_PATROL: "పోలీస్ పెట్రోల్ వాహనం",
    },
    "hi": {
        ResourceCategory.AMBULANCE: "एम्बुलेंस",
        ResourceCategory.FIRE_ENGINE: "फायर इंजन",
        ResourceCategory.FIRE_AND_RESCUE: "फायर और रेस्क्यू यूनिट",
        ResourceCategory.POLICE_PATROL: "पुलिस पेट्रोल वाहन",
    },
}


def build_demo_dispatch_announcement(
    assignments: Sequence[SimulatedAssignment],
    language: SupportedLanguage,
) -> str:
    """
    Generate a deterministic spoken summary of backend-assigned demo vehicles and demo ETAs
    in English, Telugu, or Hindi without ever invoking LLMs for resource facts.
    """
    if not assignments:
        return ""

    labels = _CATEGORY_LABELS.get(language, _CATEGORY_LABELS["en"])
    assigned_parts: list[str] = []
    unavailable_parts: list[str] = []

    for asgn in assignments:
        cat_label = labels.get(asgn.category, asgn.category.value.replace("_", " "))
        if asgn.status == AssignmentStatus.SIMULATED_ASSIGNED and asgn.resource_id:
            eta_val = (
                asgn.remaining_eta_minutes
                if asgn.remaining_eta_minutes is not None
                else asgn.simulated_eta_minutes
            )
            lifecycle = asgn.lifecycle_status or DemoLifecycleStatus.ASSIGNED.value
            if language == "te":
                eta_str = f"డెమో ETA {eta_val} నిమిషాలు" if eta_val is not None else "డెమో ETA లభ్యం కాదు"
                assigned_parts.append(f"{cat_label} {asgn.resource_id} (స్థితి: {lifecycle}, {eta_str})")
            elif language == "hi":
                eta_str = f"डेमो ETA {eta_val} मिनट" if eta_val is not None else "डेमो ETA उपलब्ध नहीं"
                assigned_parts.append(f"{cat_label} {asgn.resource_id} (स्थिति: {lifecycle}, {eta_str})")
            else:
                eta_str = f"demo ETA {eta_val} minutes" if eta_val is not None else "demo ETA pending"
                assigned_parts.append(f"{cat_label} {asgn.resource_id} (status: {lifecycle}, {eta_str})")
        elif asgn.status == AssignmentStatus.SIMULATED_UNAVAILABLE:
            if language == "te":
                unavailable_parts.append(f"{cat_label} ప్రస్తుతం డెమో ఫ్లీట్‌లో అందుబాటులో లేదు")
            elif language == "hi":
                unavailable_parts.append(f"{cat_label} अभी डेमो बेड़े में उपलब्ध नहीं है")
            else:
                unavailable_parts.append(f"no synthetic {cat_label} is currently available in the demo fleet")

    clauses: list[str] = []
    if assigned_parts:
        joined = ", ".join(assigned_parts)
        if language == "te":
            clauses.append(
                f"ఈ SETU డెమో సిమ్యులేషన్‌లో {joined} కేటాయించబడింది. ఇది నమూనా మాత్రమే, నిజమైన 112 పంపకం కాదు."
            )
        elif language == "hi":
            clauses.append(
                f"इस SETU डेमो सिमुलेशन में {joined} आवंटित किया गया है। यह केवल एक प्रोटोटाइप सिमुलेशन है, वास्तविक 112 सेवा नहीं है।"
            )
        else:
            clauses.append(
                f"In this SETU demo simulation, simulated resource assigned: {joined}. This is a prototype simulation and not real 112 dispatch."
            )
    if unavailable_parts:
        unavail_joined = "; ".join(unavailable_parts)
        if language == "te":
            clauses.append(f"డెమో వనరుల స్థితి: {unavail_joined}.")
        elif language == "hi":
            clauses.append(f"डेमो संसाधन स्थिति: {unavail_joined}।")
        else:
            clauses.append(f"Demo fleet status notice: {unavail_joined}.")

    return " ".join(clauses)


def build_demo_lifecycle_announcement(
    changed_assignments: Sequence[SimulatedAssignment],
    language: SupportedLanguage,
) -> str:
    """
    Generate a spoken demo lifecycle update when simulated vehicles reach ARRIVED (or EN_ROUTE).
    """
    arrived = [
        a for a in changed_assignments
        if a.lifecycle_status == DemoLifecycleStatus.ARRIVED.value and a.resource_id
    ]
    if not arrived:
        return ""

    labels = _CATEGORY_LABELS.get(language, _CATEGORY_LABELS["en"])
    units = ", ".join(
        f"{labels.get(a.category, a.category.value)} {a.resource_id}"
        for a in arrived
    )
    if language == "te":
        return (
            f"డెమో స్థితి నవీకరణ: సిమ్యులేటెడ్ {units} ఘటనా స్థలానికి ARRIVED స్థితికి చేరుకుంది (ఇది డెమో సిమ్యులేషన్ మాత్రమే)."
        )
    if language == "hi":
        return (
            f"डेमो स्थिति अपडेट: सिमुलेटेड {units} घटना स्थल पर ARRIVED स्थिति में पहुँच गई है (यह केवल डेमो सिमुलेशन है)।"
        )
    return (
        f"Demo update: simulated {units} has now arrived at the scene in this SETU simulation."
    )


def build_pending_location_dispatch_message(
    required_categories: Sequence[ResourceCategory],
    language: SupportedLanguage,
) -> str:
    """Ask the caller calmly to verify and confirm the map pin before simulated EN_ROUTE dispatch."""
    if not required_categories:
        return ""
    labels = _CATEGORY_LABELS.get(language, _CATEGORY_LABELS["en"])
    cat_list = ", ".join(labels.get(c, c.value.replace("_", " ")) for c in required_categories)
    if language == "te":
        return (
            f"ఈ సిమ్యులేషన్‌లో {cat_list} సిద్ధంగా ఉంచాను. దయచేసి మ్యాప్‌లో మీ లొకేషన్ పిన్ చూసి "
            "'Confirm Location' నొక్కండి, అప్పుడు సిమ్యులేటెడ్ వాహనాలు బయలుదేరుతాయి."
        )
    if language == "hi":
        return (
            f"मैंने इस सिमुलेशन में {cat_list} तैयार कर लिया है। कृपया मैप पर अपनी लोकेशन पिन जांचें और "
            "'Confirm Location' दबाएं ताकि सिमुलेटेड डिस्पैच शुरू हो सके।"
        )
    return (
        f"I've prepared a {cat_list} in this simulation. Please check the pin on the map and tap "
        "Confirm Location so I can start the simulated dispatch."
    )


def _extract_regex_fact(
    text: str,
    utterance_id: UUID,
    *,
    field_name: str,
    value: bool | int | str,
    pattern: re.Pattern[str],
) -> ProvenancedFact | None:
    m = pattern.search(text)
    if not m:
        return None
    evidence = m.group(0).strip() or text[:80].strip()
    return ProvenancedFact(
        field_name=field_name,
        value=value,
        source_utterance_id=utterance_id,
        evidence_text=evidence,
    )


def deterministic_demo_classify_and_extract(
    *,
    text: str,
    utterance_id: UUID,
    active_language: SupportedLanguage,
    existing_category: EmergencyCategory,
    existing_established: bool,
    existing_facts: Sequence[ProvenancedFact],
    pending_followup_question: str | None,
    pending_question: PendingQuestionRecord | None = None,
    question_history: Sequence[PendingQuestionRecord] = (),
    active_assignments: Sequence[SimulatedAssignment] = (),
) -> tuple[EmergencyCategory, UtteranceRole, list[ProvenancedFact], str | None, str]:
    """
    Outage-resilient deterministic classifier & fact extractor used strictly in DEMO mode
    when Gemini returns ReasoningUnavailableError (e.g. HTTP 429/503 or timeout), ensuring
    conversational memory, explicit pending-question capture, personalized next-question
    selection (Scenarios A/B/C), and post-allocation caller support continue reliably.
    """
    lowered = text.lower()
    stripped = text.strip()
    extracted: list[ProvenancedFact] = []
    pending_target = pending_question.target_field if pending_question is not None else None
    asking_injuries = bool(
        pending_target == "injuries_reported"
        or (
            pending_followup_question
            and re.search(r"injur|hurt|trapped|గాయ|घायल", pending_followup_question, re.I)
        )
    )
    asking_count = bool(
        pending_target == "affected_person_count"
        or (
            pending_followup_question
            and re.search(r"how many|number of people|count|ఎంతమంది|कितने", pending_followup_question, re.I)
        )
    )
    asking_location = bool(
        pending_target in ("caller_location_text", "caller_landmark_text", "broad_location_text")
        or (
            pending_followup_question
            and re.search(r"landmark|location|street|where|ల్యాండ్|ప్రాంతం|స్థలం|स्थान|लैंडमार्क|कहाँ", pending_followup_question, re.I)
        )
    )
    asking_consciousness = bool(
        pending_target == "consciousness_status"
        or (
            pending_followup_question
            and re.search(r"conscious|awake|breath|స్పృహ|होश|सांस", pending_followup_question, re.I)
        )
    )
    asking_safe_spot = bool(
        pending_target == "caller_safe_spot_status"
        or (
            pending_followup_question
            and re.search(r"safe\s+location|safe\s+distance|safe\s+spot|somewhere\s+safe|సురక్షిత|सुरक्षित", pending_followup_question, re.I)
        )
    )

    has_injury_negated = bool(
        re.search(
            r"\b(?:no\s+(?:one|body)\s+(?:is\s+)?(?:injur\w*|hurt|trapped)|no\s+injur\w*|nobody\s+(?:is\s+)?(?:injur\w*|hurt)|not\s+injured)\b|"
            r"(?:గాయాలు\s*లేవు|ఎవరికీ\s*గాయాలు\s*(?:లేవు|కాలేదు)|कोई\s+घायल\s+नहीं)",
            lowered,
            re.I,
        )
    ) or (
        asking_injuries
        and bool(
            re.match(
                r"^(?:no|nope|none|nobody|no\s*one|zero|nil|లేదు|ఎవరూ\s*లేరు|नहीं|कोई\s*नहीं)\b[.!]?$",
                lowered.strip(),
                re.I,
            )
        )
    )
    has_unconscious_kw = bool(
        re.search(
            r"\b(unconscious|unresponsive|not\s+breathing|fainted|no\s+pulse|behosh|spruha\s+ledu)\b|"
            r"(?:స్పృహ\s*లేదు|స్పృహ\s*కోల్పో|అపస్మారక|శ్వాస\s*ఆడటం\s*లేదు|बेहोश|सांस\s*नहीं)",
            lowered,
            re.I,
        )
    )
    has_injury_kw = (not has_injury_negated) and (
        has_unconscious_kw
        or bool(
            re.search(
                r"\b(injur|hurt|bleed|burn|unconscious|critical|pain|patient|chest pain|heart attack|fracture|trapped)\w*|"
                r"(గాయ|దెబ్బ|రక్త|కాలిన|అపస్మారక|घायल|चोट|खून|जल|बेहोश|दर्द)",
                lowered,
                re.I,
            )
        )
        or (
            asking_injuries
            and bool(
                re.search(
                    r"^(?:yes|yeah|yep|yup|someone|somebody|few|many|multiple|అవును|हाँ|जी)\b",
                    lowered.strip(),
                    re.I,
                )
            )
        )
    )
    has_fire_kw = bool(
        re.search(
            r"\b(fire|flames?|smoke|burning)\w*|(ఫైర్|మంట|అగ్ని|పొగ|फायर|आग|धुआं|लपट)",
            lowered,
            re.I,
        )
    )
    has_explosion_kw = bool(
        re.search(r"\b(explo\w*|blast)\b|(పేలుడు|బ్లాస్ట్|धमाका|विस्फोट)", lowered, re.I)
    )
    has_collapse_kw = bool(
        re.search(
            r"\b(collaps\w*|trapped under|building fell)\b|(కూలి|బిల్డింగ్|ढह|मलबे)",
            lowered,
            re.I,
        )
    )
    has_gas_kw = bool(
        re.search(r"\b(gas\s*leak|lpg|cylinder\s*leak|smell\s*of\s*gas)\b|(గ్యాస్|सिलेण्डर|गैस)", lowered, re.I)
    )
    has_accident_kw = bool(
        re.search(
            r"\b(accident|crash|collision|collided|hit by|knocked down|road)\b|(ప్రమాదం|యాక్సిడెంట్|రోడ్|दुर्घटना|एक्सीडेंट|टक्कर)",
            lowered,
            re.I,
        )
    )
    has_attack_kw = bool(
        re.search(
            r"\b(attack|assault|knife|weapon|robbery|fight|threat)\w*|(దాడి|కత్తి|हमला|चाकू|मारपीट)",
            lowered,
            re.I,
        )
    )
    has_medical_kw = bool(
        has_unconscious_kw
        or re.search(
            r"\b(ambulance|medical|doctor|cardiac|stroke|seizure|breathing|fainted)\w*|"
            r"(అంబులెన్స్|ఆసుపత్రి|వైద్య|एम्बुलेंस|अस्पताल|सांस)",
            lowered,
            re.I,
        )
    )

    # Extract person count if stated (English, Telugu, Hindi) — preferring corrected number when "not X" is used
    corr_num_match = re.search(
        r"\b(?:actually|wait|no[,\s]+)?(\d+|one|two|three|four|five|six|ten|ఒక[రవ]\w*|ఇద్ద[రురి]\w*|ముగ్గు[రురి]\w*|నలుగు[రురి]\w*|ఐదుగు[రురి]\w*|एक|दो|तीन|चार|पांच|पाँच)"
        r"(?:\s*(?:people|persons?|injured|victims?|are\s+injured|are\s+hurt|మంది|लोग))?\s*,?\s*(?:not|instead\s+of)\s+\w+",
        lowered,
        re.I,
    )
    num_match = corr_num_match or re.search(
        r"(\d+|one|two|three|four|five|six|ten|ఒక[రవ]\w*|ఇద్ద[రురి]\w*|ముగ్గు[రురి]\w*|నలుగు[రురి]\w*|ఐదుగు[రురి]\w*|एक|दो|तीन|चार|पांच|पाँच)(?:\s*(?:people|persons?|injured|victims?|log|లో|మంది|लोग))?",
        lowered,
        re.I,
    )
    word_to_int = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10,
        "ఒకరు": 1, "ఒక": 1, "ఒకరి": 1, "ఒకరికి": 1,
        "ఇద్దరు": 2, "ఇద్దరి": 2, "ఇద్దరికి": 2,
        "ముగ్గురు": 3, "ముగ్గురి": 3, "ముగ్గురికి": 3,
        "నలుగురు": 4, "నలుగురి": 4, "నలుగురికి": 4,
        "ఐదుగురు": 5, "ఐదుగురి": 5, "ఐదుగురికి": 5,
        "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5,
    }
    count_val: int | None = None
    if num_match:
        tok = num_match.group(1).lower()
        count_val = int(tok) if tok.isdigit() else word_to_int.get(tok)
        if count_val is None:
            for prefix, mapped in (("ఒక", 1), ("ఇద్ద", 2), ("ముగ్గు", 3), ("నలుగు", 4), ("ఐదుగు", 5)):
                if tok.startswith(prefix):
                    count_val = mapped
                    break
        if count_val is not None:
            extracted.append(
                ProvenancedFact(
                    field_name="affected_person_count",
                    value=count_val,
                    source_utterance_id=utterance_id,
                    evidence_text=num_match.group(0).strip(),
                )
            )

    if asking_injuries and count_val is not None and count_val > 0 and not has_injury_negated:
        has_injury_kw = True

    inferred = existing_category
    if has_explosion_kw:
        inferred = EmergencyCategory.EXPLOSION
    elif has_collapse_kw:
        inferred = EmergencyCategory.BUILDING_COLLAPSE
    elif (has_fire_kw or existing_category in (EmergencyCategory.FIRE, EmergencyCategory.FIRE_WITH_INJURIES)) and (
        has_injury_kw or existing_category == EmergencyCategory.FIRE_WITH_INJURIES
    ):
        inferred = EmergencyCategory.FIRE_WITH_INJURIES
    elif has_fire_kw:
        inferred = EmergencyCategory.FIRE
    elif has_gas_kw:
        inferred = EmergencyCategory.GAS_LEAK
    elif has_accident_kw:
        inferred = EmergencyCategory.ROAD_ACCIDENT
    elif has_attack_kw:
        inferred = EmergencyCategory.PHYSICAL_ATTACK
    elif has_medical_kw or has_injury_kw:
        if not existing_established or existing_category == EmergencyCategory.UNKNOWN:
            inferred = EmergencyCategory.MEDICAL_EMERGENCY

    if has_injury_negated:
        extracted.append(
            ProvenancedFact(
                field_name="injuries_reported",
                value=False,
                source_utterance_id=utterance_id,
                evidence_text=stripped[:80],
            )
        )
    elif has_injury_kw:
        inj_m = re.search(
            r"[^\s,;.]*(?:injur|hurt|bleed|burn|trapped|unconscious|fainted|yes|గాయ|దెబ్బ|స్పృహ|అవును|घायल|चोट|बेहोश|हाँ)[^\s,;.]*",
            text,
            re.I,
        )
        extracted.append(
            ProvenancedFact(
                field_name="injuries_reported",
                value=True,
                source_utterance_id=utterance_id,
                evidence_text=(inj_m.group(0).strip() if inj_m else stripped[:80]),
            )
        )

    # Extract caller spoken GPS/location confirmation across EN / TE / HI
    if re.search(
        r"\b(?:location\s+confirm\w*|confirm(?:ed)?(?:\s+(?:the\s+)?location)?|yes\s+confirm\w*)\b|"
        r"(?:లొకేషన్\s*కన్ఫర్మ్|కన్ఫర్మ్\s*చేశా|लोकेशन\s*कन्फर्म|कन्फर्म\s*किया)",
        lowered,
        re.I,
    ):
        extracted.append(
            ProvenancedFact(
                field_name="location_confirmed",
                value=True,
                source_utterance_id=utterance_id,
                evidence_text=stripped[:80],
            )
        )

    has_yes_reply = bool(
        re.match(
            r"^(?:yes|yeah|yep|yup|correct|someone|somebody|few|many|multiple|అవును|ఉన్నారు|ఆ|हाँ|जी|हां)\b",
            lowered.strip(),
            re.I,
        )
    )
    has_no_reply = bool(
        re.match(
            r"^(?:no|nope|none|nobody|no\s*one|zero|nil|లేదు|ఎవరూ\s*లేరు|లేరు|नहीं|कोई\s*नहीं)\b[.!]?$",
            lowered.strip(),
            re.I,
        )
    )

    # Extract consciousness status whether spontaneous (Scenario C: "One person is unconscious") or answering a question
    if has_unconscious_kw:
        extracted.append(
            ProvenancedFact(
                field_name="consciousness_status",
                value="unconscious_or_unresponsive",
                source_utterance_id=utterance_id,
                evidence_text=stripped[:80],
            )
        )
    elif asking_consciousness and (has_yes_reply or has_no_reply or re.search(r"\b(conscious|awake|unconscious|fainted|breathing)\b", lowered)):
        c_val = "conscious" if (has_yes_reply or "awake" in lowered or ("conscious" in lowered and "unconscious" not in lowered)) else "unconscious_or_unresponsive"
        extracted.append(
            ProvenancedFact(
                field_name="consciousness_status",
                value=c_val,
                source_utterance_id=utterance_id,
                evidence_text=stripped[:80],
            )
        )

    # Extract caller safe spot status when stated or answered
    if (
        asking_safe_spot and (has_yes_reply or has_no_reply)
    ) or re.search(
        r"\b(safe\s+distance|safe\s+location|safe\s+spot|stepped\s+outside|we\s+are\s+safe|standing\s+far)\b|"
        r"(?:సురక్షితంగా\s*ఉన్నా|దూరంగా\s*ఉన్నా|सुरक्षित\s*दूरी|सुरक्षित\s*जगह)",
        lowered,
        re.I,
    ):
        is_safe = not has_no_reply
        extracted.append(
            ProvenancedFact(
                field_name="caller_safe_spot_status",
                value="safe_distance" if is_safe else "needs_relocation",
                source_utterance_id=utterance_id,
                evidence_text=stripped[:80],
            )
        )

    if any(kw in lowered for kw in ("cylinder", "lpg", "gas cylinder", "సిలిండర్", "सिलेंडर")):
        extracted.append(
            ProvenancedFact(
                field_name="hazard_details",
                value="gas_cylinder_involved",
                source_utterance_id=utterance_id,
                evidence_text=stripped[:80],
            )
        )

    # Extract location / landmark across English, Telugu, Hindi, Tanglish & Hinglish
    loc_val: str | None = None
    loc_ev: str | None = None
    loc_patterns = [
        # Street / building / landmark phrasing ("near the road beside ABC building", "beside xyz")
        re.compile(
            r"((?:near|beside|opposite|behind|in\s+front\s+of|next\s+to)\s+"
            r"[A-Za-z0-9\u0C00-\u0C7F\u0900-\u097F][A-Za-z0-9\u0C00-\u0C7F\u0900-\u097F\s.,-]{2,65}?)"
            r"(?=\s*(?:[,;]|\bnot\s+|\band\s+(?:\d|one|two|three|four|five|someone|nobody|no\s+one|people|injur|hurt|trapped)|\bwith\s|\bthere\s+is\b|[.?!]|$))",
            re.I,
        ),
        # English / Tanglish / Hinglish prepositional phrase (stop before injury/count/correction conjunctions)
        re.compile(
            r"\b(?:at|in)\s+"
            r"([A-Za-z0-9\u0C00-\u0C7F\u0900-\u097F][A-Za-z0-9\u0C00-\u0C7F\u0900-\u097F\s.-]{2,45}?)"
            r"(?=\s*(?:[,;]|\bnot\s+|\band\s+(?:\d|one|two|three|four|five|someone|nobody|no\s+one|people|injur|hurt|trapped)|\bwith\s|\bthere\s+is\b|[.?!]|$))",
            re.I,
        ),
        # Telugu postpositional landmark: "... దగ్గర", "... వద్ద", "... సమీపంలో"
        re.compile(r"([A-Za-z0-9\u0C00-\u0C7F][A-Za-z0-9\u0C00-\u0C7F .-]{1,45}?\s+(?:దగ్గర|వద్ద|సమీపంలో))(?=\s|$|[,.?!])", re.I),
        # Hindi postpositional landmark: "... के पास", "... के सामने", "... नजदीक"
        re.compile(r"([A-Za-z0-9\u0900-\u097F][A-Za-z0-9\u0900-\u097F .-]{1,45}?\s+(?:के\s+पास|के\s+सामने|नजदीक))(?=\s|$|[,.?!])", re.I),
        # Telugu place with locative suffix (-లో) or Telugu landmark noun
        re.compile(r"((?:[^\s,.?!]+\s+){0,2}[^\s,.?!]*(?:సిటీ|నగర్|రోడ్|రోడ్డు|స్టేషన్|స్కూల్|కాలేజ్|ఆసుపత్రి|కంపెనీ|బస్టాండ్|లో))(?=\s|$|[,.?!])", re.I),
        # Hindi place with locative noun
        re.compile(r"((?:[^\s,.?!]+\s+){0,2}[^\s,.?!]*(?:सिटी|नगर|रोड|स्टेशन|स्कूल|अस्पताल|कॉलोनी|चौक))(?=\s|$|[,.?!])", re.I),
    ]
    for pat in loc_patterns:
        m = pat.search(text)
        if m:
            candidate_loc = re.sub(
                r"^(?:ఇక్కడ|అక్కడ|ల్యాండ్\s*మార్క్|ల్యాండ్మార్క్|ఒక|yahan|wahan|landmark)\s+",
                "",
                m.group(1).strip(),
                flags=re.I,
            ).strip(" ,.-")
            # Strip bare leading "near " when no sub-landmark qualifier follows (e.g., "near Charminar" -> "Charminar")
            if candidate_loc.lower().startswith("near ") and not re.search(
                r"\b(beside|opposite|behind|in\s+front|street|road|building|gate|pillar|lane|junction|station|hospital|school|tower|stand)\b",
                candidate_loc[5:],
                re.I,
            ):
                candidate_loc = candidate_loc[5:].strip(" ,.-")
            # Ignore meta-complaints or non-place pronouns
            if (
                len(candidate_loc) >= 3
                and not re.search(r"చెప్పాను\s*కదా|గుర్తుపెట్టుకో", candidate_loc)
                and candidate_loc.lower() not in {
                    "a fire", "fire", "danger", "trouble", "pain", "emergency",
                    "critical", "hospital", "my area", "safe location", "safe spot", "a safe location",
                    "the shop", "the shop outside", "a shop", "the house", "the building",
                    "our street", "the street", "the stall", "the room", "the kitchen",
                }
            ):
                loc_val = candidate_loc
                loc_ev = m.group(0).strip()
                break

    # If the system asked for landmark/location and caller gave a direct place answer (e.g. "Madhapur" or "Near ABC building")
    if loc_val is None and asking_location and len(stripped) >= 3:
        first_clause = re.split(r"[,;]", stripped, maxsplit=1)[0].strip(" .")
        fc_lower = first_clause.lower()
        has_kw_in_clause = bool(
            re.search(
                r"\b(fire|accident|explosion|collapse|attack|gas|injur|hurt|yes|no|nope|none|nobody|confirm\w*|people)\b|"
                r"(చెప్పాను\s*కదా|గుర్తుపెట్టుకో|కన్ఫర్మ్|గాయ|అగ్ని|ఫైర్|ప్రమాదం|घायल|आग|कन्फर्म)",
                fc_lower,
                re.I,
            )
        )
        if len(first_clause) >= 3 and not has_kw_in_clause and not first_clause.isdigit():
            loc_val = first_clause[:80]
            loc_ev = loc_val

    if loc_val:
        is_explicit_correction = bool(
            re.search(r"\b(actually|not\s+at|not\s+in|instead\s+of|wrong\s+location|changed)\b|కాదు|नहीं\s+वहां", lowered, re.I)
        )
        is_street_or_landmark = bool(
            re.search(r"\b(near|beside|opposite|behind|in\s+front|street|road|building|gate|pillar|lane|junction|station|hospital|school|tower|stand)\b", loc_val, re.I)
        )
        existing_broad = next((str(f.value) for f in existing_facts if f.field_name == "broad_location_text" and f.value), None)
        existing_loc = next((str(f.value) for f in existing_facts if f.field_name == "caller_location_text" and f.value), None)
        if is_street_or_landmark and not is_explicit_correction:
            extracted.append(
                ProvenancedFact(
                    field_name="caller_landmark_text",
                    value=loc_val,
                    source_utterance_id=utterance_id,
                    evidence_text=loc_ev or loc_val,
                )
            )
            combined = (
                f"{existing_broad or existing_loc} — {loc_val}"
                if (existing_broad or existing_loc) and (existing_broad or existing_loc or "").lower() not in loc_val.lower()
                else loc_val
            )
            extracted.append(
                ProvenancedFact(
                    field_name="caller_location_text",
                    value=combined,
                    source_utterance_id=utterance_id,
                    evidence_text=loc_ev or loc_val,
                )
            )
        else:
            extracted.append(
                ProvenancedFact(
                    field_name="broad_location_text",
                    value=loc_val,
                    source_utterance_id=utterance_id,
                    evidence_text=loc_ev or loc_val,
                )
            )
            extracted.append(
                ProvenancedFact(
                    field_name="caller_location_text",
                    value=loc_val,
                    source_utterance_id=utterance_id,
                    evidence_text=loc_ev or loc_val,
                )
            )

    if inferred != EmergencyCategory.UNKNOWN:
        extracted.append(
            ProvenancedFact(
                field_name="emergency_type",
                value=inferred.value,
                source_utterance_id=utterance_id,
                evidence_text=stripped[:100],
            )
        )

    answered_pending = bool(
        (pending_followup_question or pending_question)
        and (
            any(
                f.field_name in (
                    "caller_location_text",
                    "broad_location_text",
                    "caller_landmark_text",
                    "injuries_reported",
                    "affected_person_count",
                    "consciousness_status",
                    "caller_safe_spot_status",
                    "location_confirmed",
                )
                for f in extracted
            )
            or (asking_count and count_val is not None)
            or (existing_established and not (has_explosion_kw or has_collapse_kw))
        )
    )
    role = (
        UtteranceRole.ANSWER_TO_PENDING_QUESTION
        if answered_pending
        else UtteranceRole.NEW_DESCRIPTION
    )

    answered_history_fields = {
        q.target_field.lower()
        for q in (question_history or ())
        if q.status == "answered"
    }
    known_fields = (
        {f.field_name.lower() for f in existing_facts if f.value not in (None, "")}
        | {f.field_name.lower() for f in extracted if f.value not in (None, "")}
        | answered_history_fields
    )
    all_facts_merged = list(existing_facts) + list(extracted)
    injuries_yes = any(
        f.field_name == "injuries_reported" and f.value is True
        for f in all_facts_merged
    )
    has_cylinder_hazard = any(
        f.field_name == "hazard_details" and f.value == "gas_cylinder_involved"
        for f in all_facts_merged
    )
    has_unconscious_person = any(
        f.field_name == "consciousness_status" and "unconscious" in str(f.value).lower()
        for f in all_facts_merged
    )
    location_known = any(
        k in known_fields for k in ("caller_location_text", "broad_location_text", "caller_landmark_text", "location_confirmed")
    )
    known_loc_text = next(
        (str(f.value) for f in reversed(all_facts_merged) if f.field_name in ("caller_location_text", "broad_location_text") and f.value),
        None,
    )
    committed_assignments = [
        a for a in (active_assignments or ())
        if a.status == AssignmentStatus.SIMULATED_ASSIGNED and a.resource_id
    ]

    # Check if caller is asking a question about earlier information or unit status
    asking_earlier_memory = bool(
        re.search(
            r"\b(remember\s+where|what\s+location|where\s+did\s+i\s+say|how\s+many\s+did\s+i\s+say|"
            r"which\s+units|what\s+units|how\s+long|eta|when\s+will\s+they\s+arrive|status)\b|"
            r"(?:ఎక్కడ\s*చెప్పాను|గుర్తుందా|ఎంత\s*సమయం|ఏ\s*వాహనాలు|कहाँ\s*बताया|याद\s*है|कितना\s*समय|कौन\s*सी\s*गाड़ी)",
            lowered,
            re.I,
        )
    )

    # Dynamic, personalized next-question generation (Requirement 3 — NO FIXED QUESTIONNAIRE)
    next_q: str | None = None
    if asking_earlier_memory and location_known:
        next_q = None
    elif has_cylinder_hazard and "caller_safe_spot_status" not in known_fields:
        # Scenario A: Gas-cylinder collision/fire -> prioritize caller safety (and location if still missing)
        if not location_known:
            next_q = (
                "మీరు గ్యాస్ సిలిండర్లకు దూరంగా సురక్షిత ప్రదేశంలో ఉన్నారా, మరియు మీ సమీప ల్యాండ్‌మార్క్ ఏమిటి?"
                if active_language == "te"
                else "क्या आप गैस सिलेंडरों से दूर सुरक्षित स्थान पर हैं, और आपके पास का लैंडमार्क क्या है?"
                if active_language == "hi"
                else "Are you at a safe distance away from the gas cylinders, and what area or landmark are you near?"
            )
        else:
            next_q = (
                "మీరు ప్రస్తుతం మంటలు మరియు గ్యాస్ సిలిండర్లకు దూరంగా సురక్షిత ప్రదేశంలో ఉన్నారా?"
                if active_language == "te"
                else "क्या आप इस समय आग और गैस सिलेंडरों से दूर सुरक्षित स्थान पर हैं?"
                if active_language == "hi"
                else "Are you in a safe location well away from the fire and gas cylinders right now?"
            )
    elif not location_known:
        next_q = (
            "మీరు ఉన్న ప్రాంతం లేదా సమీప ల్యాండ్‌మార్క్ ఏమిటో చెప్పగలరా?"
            if active_language == "te"
            else "क्या आप अपने पास का क्षेत्र या लैंडमार्क बता सकते हैं?"
            if active_language == "hi"
            else "Could you tell me the area and nearest landmark so we can verify your pin?"
        )
    elif has_unconscious_person and has_unconscious_kw:
        # Scenario C: Caller reports unconscious person and location is known -> check breathing status
        next_q = (
            "స్పృహ కోల్పోయిన వ్యక్తి ప్రస్తుతం సాధారణంగా శ్వాస తీసుకుంటున్నారా?"
            if active_language == "te"
            else "क्या बेहोश व्यक्ति इस समय सामान्य रूप से सांस ले रहा है?"
            if active_language == "hi"
            else "Is the unconscious person breathing normally right now?"
        )
    elif "injuries_reported" not in known_fields and inferred in (
        EmergencyCategory.FIRE,
        EmergencyCategory.ROAD_ACCIDENT,
        EmergencyCategory.GAS_LEAK,
        EmergencyCategory.BUILDING_COLLAPSE,
        EmergencyCategory.EXPLOSION,
    ):
        next_q = (
            "ఎవరికైనా గాయాలు అయ్యాయా లేదా దయచేసి చెప్పగలరా?"
            if active_language == "te"
            else "क्या वहां कोई घायल हुआ है?"
            if active_language == "hi"
            else "Is anyone hurt or trapped there?"
        )
    elif injuries_yes and "affected_person_count" not in known_fields:
        next_q = (
            "సుమారుగా ఎంతమంది గాయపడ్డారు?"
            if active_language == "te"
            else "लगभग कितने लोग घायल हैं?"
            if active_language == "hi"
            else "Roughly how many people are hurt?"
        )

    # Human-like, empathetic, non-repetitive response generation (Requirement 4)
    unit_summary_en = ", ".join(
        f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} min)"
        for a in committed_assignments
    )
    known_count_val = next(
        (f.value for f in reversed(all_facts_merged) if f.field_name == "affected_person_count" and f.value is not None),
        None,
    )
    if active_language == "te":
        if asking_earlier_memory:
            parts_te = []
            if known_loc_text:
                parts_te.append(f"మీరు చెప్పిన ప్రదేశం {known_loc_text} నాకు గుర్తుంది")
            if known_count_val is not None:
                parts_te.append(f"మరియు {known_count_val} మంది ప్రభావితమయ్యారని నమోదు చేశాను")
            if committed_assignments:
                units_te = ", ".join(f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} నిమిషాలు)" for a in committed_assignments)
                parts_te.append(f"ఈ డెమోలో {units_te} మీ వైపు వస్తున్నాయి")
            base_reply = (". ".join(parts_te) + ".") if parts_te else "నేను మీ వివరాలన్నీ గుర్తుపెట్టుకున్నాను, మీతోనే లైన్‌లో ఉన్నాను."
        elif has_cylinder_hazard and any(f.field_name == "hazard_details" for f in extracted):
            base_reply = "అర్థమైంది, శాంతంగా ఉండండి. గ్యాస్ సిలిండర్ దగ్గరికి అస్సలు వెళ్లవద్దు మరియు వెంటనే దూరంగా సురక్షిత స్థలంలో ఉండండి."
        elif has_unconscious_kw:
            base_reply = "అర్థమైంది, పరిస్థితి తీవ్రత నాకు తెలుస్తోంది — దయచేసి ఆ వ్యక్తి పక్కనే ఉండి గాలి ఆడేలా చూడండి."
        elif committed_assignments:
            units_te = ", ".join(f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} నిమిషాలు)" for a in committed_assignments)
            base_reply = f"ఆ వివరాలు నమోదు చేశాను. ఈ డెమోలో {units_te} మార్గంలో ఉన్నాయి — దయచేసి సురక్షితంగా ఉండండి, నేను మీతోనే లైన్‌లో ఉన్నాను."
        elif role == UtteranceRole.ANSWER_TO_PENDING_QUESTION:
            if loc_val:
                base_reply = f"ధన్యవాదాలు, {loc_val} ప్రదేశాన్ని నమోదు చేసుకున్నాను."
            elif count_val is not None:
                base_reply = f"అర్థమైంది, {count_val} మంది అని నమోదు చేశాను."
            else:
                base_reply = "ధన్యవాదాలు, ఆ ముఖ్యమైన వివరాలు నమోదు చేశాను."
        else:
            base_reply = "నేను మీ మాటలు వింటున్నాను, ప్రశాంతంగా ఉండండి. ముందుగా మీరు సురక్షిత ప్రదేశంలో ఉన్నారని నిర్ధారించుకోండి."
    elif active_language == "hi":
        if asking_earlier_memory:
            parts_hi = []
            if known_loc_text:
                parts_hi.append(f"मुझे याद है आपने स्थान {known_loc_text} बताया है")
            if known_count_val is not None:
                parts_hi.append(f"और {known_count_val} लोगों की जानकारी दर्ज है")
            if committed_assignments:
                units_hi = ", ".join(f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} मिनट)" for a in committed_assignments)
                parts_hi.append(f"इस डेमो में {units_hi} आपकी ओर रास्ते में हैं")
            base_reply = ("। ".join(parts_hi) + "।") if parts_hi else "मैंने आपकी सारी जानकारी याद रखी है और मैं आपके साथ लाइन पर हूँ।"
        elif has_cylinder_hazard and any(f.field_name == "hazard_details" for f in extracted):
            base_reply = "मैं समझ रहा हूँ, शांत रहें। जलते या लीक होते गैस सिलेंडर के बिल्कुल पास न जाएँ और खुले सुरक्षित स्थान पर हट जाएँ।"
        elif has_unconscious_kw:
            base_reply = "मैं स्थिति की गंभीरता समझ रहा हूँ — कृपया उनके पास रहें और ध्यान रखें कि सांस का रास्ता खुला रहे।"
        elif committed_assignments:
            units_hi = ", ".join(f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} मिनट)" for a in committed_assignments)
            base_reply = f"मैंने यह अपडेट दर्ज कर लिया है। इस डेमो में {units_hi} रास्ते में हैं — कृपया सुरक्षित रहें, मैं आपके साथ लाइन पर हूँ।"
        elif role == UtteranceRole.ANSWER_TO_PENDING_QUESTION:
            if loc_val:
                base_reply = f"धन्यवाद, मैंने {loc_val} लोकेशन दर्ज कर ली है।"
            elif count_val is not None:
                base_reply = f"समझ गया, मैंने {count_val} लोगों की संख्या नोट कर ली है।"
            else:
                base_reply = "ठीक है, धन्यवाद — मैंने यह जानकारी दर्ज कर ली है।"
        else:
            base_reply = "मैं आपकी बात ध्यान से सुन रहा हूँ, कृपया शांत रहें। सबसे पहले किसी सुरक्षित जगह पर हट जाएँ।"
    else:
        if asking_earlier_memory:
            parts_en = []
            if known_loc_text:
                parts_en.append(f"I have your location recorded as {known_loc_text}")
            if known_count_val is not None:
                parts_en.append(f"with {known_count_val} affected people noted")
            if committed_assignments:
                parts_en.append(f"and in this demo {unit_summary_en} are currently en route to you")
            base_reply = (", ".join(parts_en) + ".") if parts_en else "I have all your earlier details recorded and I am right here with you."
        elif has_cylinder_hazard and any(f.field_name == "hazard_details" for f in extracted):
            base_reply = (
                "I understand. Stay calm and step well outside right now — do not go near, touch, "
                "or try to move a burning or leaking gas cylinder."
            )
        elif has_unconscious_kw:
            base_reply = (
                "I understand how serious this is — stay right beside them, keep their airway clear, "
                "and do not move them unless there is immediate danger."
            )
        elif committed_assignments:
            base_reply = (
                f"I've noted that update. In this demo, {unit_summary_en} are on the way to "
                f"{known_loc_text or 'your location'} — stay in a safe spot and I will stay on the line with you."
            )
        elif role == UtteranceRole.ANSWER_TO_PENDING_QUESTION:
            if loc_val:
                base_reply = f"Thank you, I've updated the location to {loc_val}."
            elif count_val is not None:
                base_reply = f"Understood, I've recorded {count_val} people affected."
            elif has_injury_negated:
                base_reply = "I'm relieved to hear nobody is injured — I've noted that down."
            else:
                base_reply = "Got it, thank you — I've noted that detail down."
        else:
            base_reply = (
                "I understand. Stay calm, and let's take this one step at a time. "
                "First, please make sure you are somewhere safe."
            )
    if next_q:
        base_reply = f"{base_reply} {next_q}"

    return inferred, role, extracted, next_q, base_reply
