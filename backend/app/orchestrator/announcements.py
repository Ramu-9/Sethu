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
) -> tuple[EmergencyCategory, UtteranceRole, list[ProvenancedFact], str | None, str]:
    """
    Outage-resilient deterministic classifier & fact extractor used strictly in DEMO mode
    when Gemini returns ReasoningUnavailableError (e.g. HTTP 429/503 or timeout), ensuring
    conversational memory, explicit pending-question capture, and safety guidance continue reliably.
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
    has_injury_kw = (not has_injury_negated) and (
        bool(
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
            r"\b(accident|crash|collision|hit by|knocked down|road)\b|(ప్రమాదం|యాక్సిడెంట్|రోడ్|दुर्घटना|एक्सीडेंट|टक्कर)",
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
        re.search(
            r"\b(ambulance|medical|doctor|cardiac|stroke|seizure|breathing|fainted)\w*|"
            r"(అంబులెన్స్|ఆసుపత్రి|వైద్య|एम्बुलेंस|अस्पताल|सांस)",
            lowered,
            re.I,
        )
    )

    # Extract person count if stated (English, Telugu, Hindi)
    num_match = re.search(
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
        inj_m = re.search(r"[^\s,;.]*(?:injur|hurt|bleed|burn|trapped|yes|గాయ|దెబ్బ|అవును|घायल|चोट|हाँ)[^\s,;.]*", text, re.I)
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
    if asking_consciousness and (has_yes_reply or has_no_reply or re.search(r"\b(conscious|awake|unconscious|fainted|breathing)\b", lowered)):
        c_val = "conscious" if (has_yes_reply or "awake" in lowered or ("conscious" in lowered and "unconscious" not in lowered)) else "unconscious_or_unresponsive"
        extracted.append(
            ProvenancedFact(
                field_name="consciousness_status",
                value=c_val,
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
            r"(?=\s*(?:[,;]|\band\s+(?:\d|one|two|three|four|five|someone|nobody|no\s+one|people|injur|hurt|trapped)|\bwith\s|\bthere\s+is\b|[.?!]|$))",
            re.I,
        ),
        # English / Tanglish / Hinglish prepositional phrase (stop before injury/count conjunctions)
        re.compile(
            r"\b(?:at|in)\s+"
            r"([A-Za-z0-9\u0C00-\u0C7F\u0900-\u097F][A-Za-z0-9\u0C00-\u0C7F\u0900-\u097F\s.-]{2,45}?)"
            r"(?=\s*(?:[,;]|\band\s+(?:\d|one|two|three|four|five|someone|nobody|no\s+one|people|injur|hurt|trapped)|\bwith\s|\bthere\s+is\b|[.?!]|$))",
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
                and candidate_loc.lower() not in {"a fire", "fire", "danger", "trouble", "pain", "emergency", "critical", "hospital", "my area"}
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
        is_street_or_landmark = bool(
            re.search(r"\b(near|beside|opposite|behind|in\s+front|street|road|building|gate|pillar|lane|junction|station|hospital|school|tower|stand)\b", loc_val, re.I)
        )
        existing_broad = next((str(f.value) for f in existing_facts if f.field_name == "broad_location_text" and f.value), None)
        existing_loc = next((str(f.value) for f in existing_facts if f.field_name == "caller_location_text" and f.value), None)
        if is_street_or_landmark:
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

    known_fields = {f.field_name for f in existing_facts} | {f.field_name for f in extracted}
    injuries_yes = any(
        f.field_name == "injuries_reported" and f.value is True
        for f in (*existing_facts, *extracted)
    )

    next_q: str | None = None
    if "caller_location_text" not in known_fields and "broad_location_text" not in known_fields and "location_confirmed" not in known_fields:
        next_q = (
            "మీరు ఉన్న ప్రాంతం లేదా సమీప ల్యాండ్‌మార్క్ ఏమిటో చెప్పగలరా?"
            if active_language == "te"
            else "क्या आप अपने पास का क्षेत्र या लैंडमार्क बता सकते हैं?"
            if active_language == "hi"
            else "Could you tell me the area and nearest landmark so we can verify your pin?"
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

    # Natural, calm human-to-human response (varied by whether injuries/hazard/guidance apply)
    has_cylinder_hazard = any(
        f.field_name == "hazard_details" and f.value == "gas_cylinder_involved"
        for f in (*existing_facts, *extracted)
    )
    if active_language == "te":
        if has_cylinder_hazard:
            base_reply = "అర్థమైంది, శాంతంగా ఉండండి. గ్యాస్ సిలిండర్ దగ్గరికి అస్సలు వెళ్లవద్దు మరియు దూరంగా సురక్షిత స్థలంలో ఉండండి."
        elif role == UtteranceRole.ANSWER_TO_PENDING_QUESTION:
            base_reply = "ధన్యవాదాలు, ఆ వివరాలు నమోదు చేశాను."
        else:
            base_reply = "నేను మీ మాటలు వింటున్నాను, ప్రశాంతంగా ఉండండి. ముందుగా మీరు సురక్షిత ప్రదేశంలో ఉన్నారని నిర్ధారించుకోండి."
    elif active_language == "hi":
        if has_cylinder_hazard:
            base_reply = "मैं समझ रहा हूँ, शांत रहें। जलते या लीक होते गैस सिलेंडर के बिल्कुल पास न जाएँ और खुले सुरक्षित स्थान पर हट जाएँ।"
        elif role == UtteranceRole.ANSWER_TO_PENDING_QUESTION:
            base_reply = "ठीक है, धन्यवाद — मैंने यह जानकारी दर्ज कर ली है।"
        else:
            base_reply = "मैं आपकी बात ध्यान से सुन रहा हूँ, कृपया शांत रहें। सबसे पहले किसी सुरक्षित जगह पर हट जाएँ।"
    else:
        if has_cylinder_hazard:
            base_reply = (
                "I understand. Stay calm and step well outside right now — do not go near, touch, "
                "or try to move a burning or leaking gas cylinder."
            )
        elif role == UtteranceRole.ANSWER_TO_PENDING_QUESTION:
            base_reply = "Got it, thank you — I've noted that detail down."
        else:
            base_reply = (
                "I understand. Stay calm, and let's take this one step at a time. "
                "First, please make sure you are somewhere safe."
            )
    if next_q:
        base_reply = f"{base_reply} {next_q}"

    return inferred, role, extracted, next_q, base_reply
