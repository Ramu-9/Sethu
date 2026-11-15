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
    "kn": {
        ResourceCategory.AMBULANCE: "ಆಂಬ್ಯುಲೆನ್ಸ್",
        ResourceCategory.FIRE_ENGINE: "ಅಗ್ನಿಶಾಮಕ ವಾಹನ",
        ResourceCategory.FIRE_AND_RESCUE: "ಅಗ್ನಿಶಾಮಕ ಮತ್ತು ರಕ್ಷಣಾ ಘಟಕ",
        ResourceCategory.POLICE_PATROL: "ಪೊಲೀಸ್ ಗಸ್ತು ವಾಹನ",
    },
    "ta": {
        ResourceCategory.AMBULANCE: "ஆம்புலன்ஸ்",
        ResourceCategory.FIRE_ENGINE: "தீயணைப்பு வாகனம்",
        ResourceCategory.FIRE_AND_RESCUE: "தீயணைப்பு மற்றும் மீட்பு வாகனம்",
        ResourceCategory.POLICE_PATROL: "காவல்துறை ரோந்து வாகனம்",
    },
    "ml": {
        ResourceCategory.AMBULANCE: "ആംബുലൻസ്",
        ResourceCategory.FIRE_ENGINE: "ഫയർ എഞ്ചിൻ",
        ResourceCategory.FIRE_AND_RESCUE: "ഫയർ ആൻഡ് റെസ്ക്യൂ യൂണിറ്റ്",
        ResourceCategory.POLICE_PATROL: "പോലീസ് പട്രോൾ വാഹനം",
    },
    "bn": {
        ResourceCategory.AMBULANCE: "অ্যাম্বুলেন্স",
        ResourceCategory.FIRE_ENGINE: "ফায়ার ইঞ্জিন",
        ResourceCategory.FIRE_AND_RESCUE: "ফায়ার ও রেসকিউ ইউনিট",
        ResourceCategory.POLICE_PATROL: "পুলিশ পেট্রোল গাড়ি",
    },
}


def build_demo_dispatch_announcement(
    assignments: Sequence[SimulatedAssignment],
    language: SupportedLanguage,
) -> str:
    """
    Generate a deterministic spoken summary of backend-assigned demo vehicles and demo ETAs
    across all 7 supported languages without ever invoking LLMs for resource facts.
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
            elif language == "kn":
                eta_str = f"ಡೆಮೊ ETA {eta_val} ನಿಮಿಷಗಳು" if eta_val is not None else "ಡೆಮೊ ETA ಲಭ್ಯವಿಲ್ಲ"
                assigned_parts.append(f"{cat_label} {asgn.resource_id} (ಸ್ಥಿತಿ: {lifecycle}, {eta_str})")
            elif language == "ta":
                eta_str = f"டெமோ ETA {eta_val} நிமிடங்கள்" if eta_val is not None else "டெமோ ETA நிலுவையில்"
                assigned_parts.append(f"{cat_label} {asgn.resource_id} (நிலை: {lifecycle}, {eta_str})")
            elif language == "ml":
                eta_str = f"ഡെമോ ETA {eta_val} മിനിറ്റ്" if eta_val is not None else "ഡെമോ ETA ലഭ്യമല്ല"
                assigned_parts.append(f"{cat_label} {asgn.resource_id} (നില: {lifecycle}, {eta_str})")
            elif language == "bn":
                eta_str = f"ডেমো ETA {eta_val} মিনিট" if eta_val is not None else "ডেমো ETA অনুপলব্ধ"
                assigned_parts.append(f"{cat_label} {asgn.resource_id} (অবস্থা: {lifecycle}, {eta_str})")
            else:
                eta_str = f"demo ETA {eta_val} minutes" if eta_val is not None else "demo ETA pending"
                assigned_parts.append(f"{cat_label} {asgn.resource_id} (status: {lifecycle}, {eta_str})")
        elif asgn.status == AssignmentStatus.SIMULATED_UNAVAILABLE:
            if language == "te":
                unavailable_parts.append(f"{cat_label} ప్రస్తుతం డెమో ఫ్లీట్‌లో అందుబాటులో లేదు")
            elif language == "hi":
                unavailable_parts.append(f"{cat_label} अभी डेमो बेड़े में उपलब्ध नहीं है")
            elif language == "kn":
                unavailable_parts.append(f"{cat_label} ಪ್ರಸ್ತುತ ಡೆಮೊ ಫ್ಲೀಟ್‌ನಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ")
            elif language == "ta":
                unavailable_parts.append(f"{cat_label} தற்போது டெமோ தொகுப்பில் கிடைக்கவில்லை")
            elif language == "ml":
                unavailable_parts.append(f"{cat_label} നിലവിൽ ഡെമോ ഫ്ലീറ്റിൽ ലഭ്യമല്ല")
            elif language == "bn":
                unavailable_parts.append(f"{cat_label} বর্তমানে ডেমো বহরে উপলব্ধ নেই")
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
        elif language == "kn":
            clauses.append(
                f"ಈ SETU ಡೆಮೊ ಸಿಮ್ಯುಲೇಶನ್‌ನಲ್ಲಿ {joined} ನಿಯೋಜಿಸಲಾಗಿದೆ. ಇದು ಪ್ರোটೋಟೈಪ್ ಸಿಮ್ಯುಲೇಶನ್ ಮಾತ್ರ, ನಿಜವಾದ 112 ಸೇವೆ ಅಲ್ಲ."
            )
        elif language == "ta":
            clauses.append(
                f"இந்த SETU டெமோ சிமுலேஷனில் {joined} ஒதுக்கப்பட்டுள்ளது. இது ஒரு மாதிரி சிமுலேஷன் மட்டுமே, உண்மையான 112 சேவை அல்ல."
            )
        elif language == "ml":
            clauses.append(
                f"ഈ SETU ഡെമോ സിമുലേഷനിൽ {joined} അനുവദിച്ചിട്ടുണ്ട്. ഇത് ഒരു പ്രോട്ടോടൈപ്പ് സിമുലേഷൻ മാത്രമാണ്, യഥാർത്ഥ 112 സേവനമല്ല."
            )
        elif language == "bn":
            clauses.append(
                f"এই SETU ডেমো সিমুলেশনে {joined} বরাদ্দ করা হয়েছে। এটি কেবল একটি প্রোটোটাইপ সিমুলেশন, প্রকৃত ১১২ পরিষেবা নয়।"
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
        elif language == "kn":
            clauses.append(f"ಡೆಮೊ ಸಂಪನ್ಮೂಲ ಸ್ಥಿತಿ: {unavail_joined}.")
        elif language == "ta":
            clauses.append(f"டெமோ வள நிலை: {unavail_joined}.")
        elif language == "ml":
            clauses.append(f"ഡെമോ റിസോഴ്സ് നില: {unavail_joined}.")
        elif language == "bn":
            clauses.append(f"ডেমো রিসোর্স স্থিতি: {unavail_joined}।")
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
    if language == "kn":
        return (
            f"ಡೆಮೊ ಸ್ಥಿತಿ ನವೀಕರಣ: ಸಿಮ್ಯುಲೇಟೆಡ್ {units} ಘಟನಾ ಸ್ಥಳಕ್ಕೆ ARRIVED ಸ್ಥಿತಿಯಲ್ಲಿ ತಲುಪಿದೆ (ಇದು ಡೆಮೊ ಸಿಮ್ಯುಲೇಶನ್ ಮಾತ್ರ)."
        )
    if language == "ta":
        return (
            f"டெமோ நிலை புதுப்பிப்பு: சிமுலேட்டட் {units} சம்பவ இடத்திற்கு ARRIVED நிலையில் வந்துவிட்டது (இது டெமோ சிமுலேஷன் மட்டுமே)."
        )
    if language == "ml":
        return (
            f"ഡെമോ നില അപ്ഡേറ്റ്: സിമുലേറ്റഡ് {units} സംഭവസ്ഥലത്ത് ARRIVED നിലയിൽ എത്തിയിരിക്കുന്നു (ഇത് ഡെമോ സിമുലേഷൻ മാത്രമാണ്)."
        )
    if language == "bn":
        return (
            f"ডেমো স্থিতি আপডেট: সিমুলেটেড {units} ঘটনাস্থলে ARRIVED অবস্থায় পৌঁছেছে (এটি কেবল ডেমো সিমুলেশন)।"
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
    if language == "kn":
        return (
            f"ಈ ಸಿಮ್ಯುಲೇಶನ್‌ನಲ್ಲಿ {cat_list} ಸಿದ್ಧಪಡಿಸಿದ್ದೇನೆ. ದಯವಿಟ್ಟು ನಕ್ಷೆಯಲ್ಲಿ ನಿಮ್ಮ ಸ್ಥಳದ ಪಿನ್ ಪರಿಶೀಲಿಸಿ "
            "'Confirm Location' ಒತ್ತಿರಿ, ಆಗ ಸಿಮ್ಯುಲೇಟೆಡ್ ವಾಹನಗಳು ಹೊರಡುತ್ತವೆ."
        )
    if language == "ta":
        return (
            f"இந்த சிமுலேஷனில் {cat_list} தயாராக வைத்துள்ளேன். தயவுசெய்து வரைபடத்தில் உங்கள் இருப்பிட பின்னை சரிபார்த்து "
            "'Confirm Location' அழுத்தவும்."
        )
    if language == "ml":
        return (
            f"ഈ സിമുലേഷനിൽ {cat_list} സജ്ജമാക്കിയിട്ടുണ്ട്. ദയവായി മാപ്പിൽ നിങ്ങളുടെ ലൊക്കേഷൻ പിൻ പരിശോധിച്ച് "
            "'Confirm Location' അമർത്തുക."
        )
    if language == "bn":
        return (
            f"আমি এই সিমুলেশনে {cat_list} প্রস্তুত রেখেছি। অনুগ্রহ করে মানচিত্রে আপনার লোকেশন পিন যাচাই করে "
            "'Confirm Location' চাপুন যাতে সিমুলেটেড ডিসপ্যাচ শুরু হতে পারে।"
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
    selection (Scenarios A/B/C), and post-allocation caller support continue reliably
    across all 7 supported languages (en, te, hi, kn, ta, ml, bn).
    """
    lowered = text.lower()
    stripped = text.strip()
    extracted: list[ProvenancedFact] = []
    pending_target = pending_question.target_field if pending_question is not None else None
    asking_injuries = bool(
        pending_target == "injuries_reported"
        or (
            pending_followup_question
            and re.search(
                r"injur|hurt|trapped|గాయ|घायल|ಗಾಯ|காய|പരിക്ക്|പരിക്കേ|আহত|চোট",
                pending_followup_question,
                re.I,
            )
        )
    )
    asking_count = bool(
        pending_target == "affected_person_count"
        or (
            pending_followup_question
            and re.search(
                r"how many|number of people|count|ఎంతమంది|कितने|ಎಷ್ಟು|எத்தனை|എത്ര|কতজন|কত\s*মানুষ",
                pending_followup_question,
                re.I,
            )
        )
    )
    asking_location = bool(
        pending_target in ("caller_location_text", "caller_landmark_text", "broad_location_text")
        or (
            pending_followup_question
            and re.search(
                r"landmark|location|street|where|ల్యాండ్|ప్రాంతం|స్థలం|स्थान|लैंडमार्क|कहाँ|"
                r"ಲ್ಯಾಂಡ್|ಸ್ಥಳ|ಪ್ರದೇಶ|ಎಲ್ಲಿ|இடம்|அடையாளம்|எங்கே|സ്ഥലം|ലാൻഡ്‌മാർക്ക്|എവിടെ|স্থান|ল্যান্ডমার্ক|কোথায়",
                pending_followup_question,
                re.I,
            )
        )
    )
    asking_consciousness = bool(
        pending_target == "consciousness_status"
        or (
            pending_followup_question
            and re.search(
                r"conscious|awake|breath|స్పృహ|होश|सांस|ಪ್ರಜ್ಞೆ|ಉಸಿರಾ|சுயநினைவு|மூச்சு|ബോധം|ശ്വാസം|জ্ঞান|শ্বাস",
                pending_followup_question,
                re.I,
            )
        )
    )
    asking_safe_spot = bool(
        pending_target == "caller_safe_spot_status"
        or (
            pending_followup_question
            and re.search(
                r"safe\s+location|safe\s+distance|safe\s+spot|somewhere\s+safe|సురక్షిత|सुरक्षित|"
                r"ಸುರಕ್ಷಿತ|பாதுகாப்பான|സുരക്ഷിത|নিরাপদ",
                pending_followup_question,
                re.I,
            )
        )
    )

    has_injury_negated = bool(
        re.search(
            r"\b(?:no\s+(?:one|body)\s+(?:is\s+)?(?:injur\w*|hurt|trapped)|no\s+injur\w*|nobody\s+(?:is\s+)?(?:injur\w*|hurt)|not\s+injured)\b|"
            r"(?:గాయాలు\s*లేవు|ఎవరికీ\s*గాయాలు\s*(?:లేవు|కాలేదు)|कोई\s+घायल\s+नहीं|"
            r"ಯಾರಿಗೂ\s*ಗಾಯವಾಗಿಲ್ಲ|ಗಾಯಗಳಿಲ್ಲ|யாருக்கும்\s*காயமில்லை|காயம்\s*இல்லை|"
            r"ആർക്കും\s*പരിക്കില്ല|പരിക്കുകളില്ല|কেউ\s*আহত\s*নয়|কোনো\s*আহত\s*নেই)",
            lowered,
            re.I,
        )
    ) or (
        asking_injuries
        and bool(
            re.match(
                r"^(?:no|nope|none|nobody|no\s*one|zero|nil|లేదు|ఎవరూ\s*లేరు|नहीं|कोई\s*नहीं|"
                r"ಇಲ್ಲ|ಯಾರೂ\s*ಇಲ್ಲ|இல்லை|யாரும்\s*இல்லை|ഇല്ല|ആരുമില്ല|না|কেউ\s*না)\b[.!]?$",
                lowered.strip(),
                re.I,
            )
        )
    )
    has_unconscious_kw = bool(
        re.search(
            r"\b(unconscious|unresponsive|not\s+breathing|fainted|no\s+pulse|behosh|spruha\s+ledu|"
            r"prajne\s+illa|mayakkam|bodham\s+illa|oggyan)\b|"
            r"(?:స్పృహ\s*లేదు|స్పృహ\s*కోల్పో|అపస్మారక|శ్వాస\s*ఆడటం\s*లేదు|बेहोश|सांस\s*नहीं|"
            r"ಪ್ರಜ್ಞೆ\s*ಇಲ್ಲ|ಪ್ರಜ್ಞಾಹೀನ|ಉಸಿರಾಡುತ್ತಿಲ್ಲ|மயக்கம்|சுயநினைவு\s*இல்லை|மூச்சு\s*விடவில்லை|"
            r"ബോധമില്ല|ബോധരഹിത|ശ്വാസമില്ല|অজ্ঞান|জ্ঞান\s*নেই|শ্বাস\s*নিচ্ছে\s*না)",
            lowered,
            re.I,
        )
    )
    has_injury_kw = (not has_injury_negated) and (
        has_unconscious_kw
        or bool(
            re.search(
                r"\b(injur|hurt|bleed|burn|unconscious|critical|pain|patient|chest pain|heart attack|fracture|trapped|"
                r"gaya|rakta|kettadagi|kayam|ratham|parikku|ahoto|rokto)\w*|"
                r"(గాయ|దెబ్బ|రక్త|కాలిన|అపస్మారక|घायल|चोट|खून|जल|बेहोश|दर्द|"
                r"ಗಾಯ|ರಕ್ತ|ಸುಟ್ಟ|ನೋವು|ಸಿಲುಕಿ|காய|ரத்தம்|தீக்காயம்|வலி|சிக்கி|"
                r"പരിക്ക്|രക്തം|പൊള്ളൽ|വേദന|കുടുങ്ങി|আহত|রক্ত|দগ্ধ|ব্যথা|আটকে)",
                lowered,
                re.I,
            )
        )
        or (
            asking_injuries
            and bool(
                re.search(
                    r"^(?:yes|yeah|yep|yup|someone|somebody|few|many|multiple|అవును|हाँ|जी|ಹೌದು|ஆம்|ஆமா|അതെ|হ্যাঁ)\b",
                    lowered.strip(),
                    re.I,
                )
            )
        )
    )
    has_fire_kw = bool(
        re.search(
            r"\b(fire|flames?|smoke|burning|benki|thee|neruppu|agun)\w*|"
            r"(ఫైర్|మంట|అగ్ని|పొగ|फायर|आग|धुआं|लपट|"
            r"ಬೆಂಕಿ|ಹೊಗೆ|ಅಗ್ನಿ|தீ|நெருப்பு|புகை|തീ|തീപിടുത്തം|പുക|আগুন|ধোঁয়া|অগ্নি)",
            lowered,
            re.I,
        )
    )
    has_explosion_kw = bool(
        re.search(
            r"\b(explo\w*|blast|स्फोट|sphota|vedippu|sphodanam|bisforon)\b|"
            r"(పేలుడు|బ్లాస్ట్|धमाका|विस्फोट|ಸ್ಫೋಟ|வெடிப்பு|സ്ഫോടനം|বিস্ফোরণ)",
            lowered,
            re.I,
        )
    )
    has_collapse_kw = bool(
        re.search(
            r"\b(collaps\w*|trapped under|building fell|kusidu|idinju|bhenge)\b|"
            r"(కూలి|బిల్డింగ్|ढह|मलबे|ಕುಸಿದು|ಕಟ್ಟಡ\s*ಬಿದ್ದ|இடிந்து|கட்டிடம்\s*விழு|ഇടിഞ്ഞു|കെട്ടിടം\s*തകർ|ধসে|ভবন\s*ধসে)",
            lowered,
            re.I,
        )
    )
    has_gas_kw = bool(
        re.search(
            r"\b(gas\s*leak|lpg|cylinder\s*leak|smell\s*of\s*gas)\b|"
            r"(గ్యాస్|सिलेण्डर|गैस|ಗ್ಯಾಸ್|ಸಿಲಿಂಡರ್|கேஸ்|சிலிண்டர்|ഗ്യാസ്|സിലിണ്ടർ|গ্যাস|সিলিন্ডার)",
            lowered,
            re.I,
        )
    )
    has_accident_kw = bool(
        re.search(
            r"\b(accident|crash|collision|collided|hit by|knocked down|road|apaghata|vipathu|apaghatam|durghotona)\b|"
            r"(ప్రమాదం|యాక్సిడెంట్|రోడ్|दुर्घटना|हादसा|सड़क|एक्सीडेंट|टक्कर|"
            r"ಅಪಘಾತ|ಆಕ್ಸಿಡೆಂಟ್|ಡಿಕ್ಕಿ|விபத்து|ஆக்சிடென்ட்|மோதல்|അപകടം|ആക്സിഡന്റ്|ഇടിച്ചു|দুর্ঘটনা|অ্যাক্সিডেন্ট|ধাক্কা)",
            lowered,
            re.I,
        )
    )
    has_attack_kw = bool(
        re.search(
            r"\b(attack|assault|knife|weapon|robbery|fight|threat)\w*|"
            r"(దాడి|కత్తి|हमला|चाकू|मारपीट|ದಾಳಿ|ಚಾಕು|தாக்குதல்|கத்தி|ആക്രമണം|കത്തി|হামলা|ছুরি)",
            lowered,
            re.I,
        )
    )
    has_medical_kw = bool(
        has_unconscious_kw
        or re.search(
            r"\b(ambulance|medical|doctor|cardiac|stroke|seizure|breathing|fainted)\w*|"
            r"(అంబులెన్స్|ఆసుపత్రి|వైద్య|एम्बुलेंस|अस्पताल|सांस|"
            r"ಆಂಬ್ಯುಲೆನ್ಸ್|ಆಸ್ಪತ್ರೆ|ವೈದ್ಯ|ஆம்புலன்ஸ்|மருத்துவமனை|மூச்சு|"
            r"ആംബുലൻസ്|ആശുപത്രി|വൈദ്യ|অ্যাম্বুলেন্স|হাসপাতাল|শ্বাসকষ্ট|হৃদরোগ)",
            lowered,
            re.I,
        )
    )

    # Extract person count if stated across all 7 languages — preferring corrected number when "not X" is used
    num_tokens_pattern = (
        r"\d+|one|two|three|four|five|six|ten|"
        r"ఒక[రవ]\w*|ఇద్ద[రురి]\w*|ముగ్గు[రురి]\w*|నలుగు[రురి]\w*|ఐదుగు[రురి]\w*|"
        r"एक|दो|तीन|चार|पांच|पाँच|"
        r"ಒಬ್ಬ\w*|ಒಂದು|ಇಬ್ಬ\w*|ಎರಡು|ಮೂವ\w*|ಮೂರು|ನಾಲ್ವ\w*|ನಾಲ್ಕು|ಐವ\w*|ಐದು|"
        r"ஒரு\w*|ஒன்று|இரு\w*|இரண்டு|மூன்று|நான்கு|ஐந்து|"
        r"ഒരാൾ|ഒന്ന്|രണ്ടു\w*|രണ്ട്|മൂന്നു\w*|മൂന്ന്|നാലു\w*|നാല്|അഞ്ചു\w*|അഞ്ച്|"
        r"একজন|এক|দুইজন|দুজন|দুই|তিনজন|তিন|চারজন|চার|পাঁচজন|পাঁচ"
    )
    corr_num_match = re.search(
        rf"\b(?:actually|wait|no[,\s]+)?({num_tokens_pattern})"
        r"(?:\s*(?:people|persons?|injured|victims?|are\s+injured|are\s+hurt|మంది|लोग|ಜನ|பேர்|പേർ|জন|মানুষ))?\s*,?\s*(?:not|instead\s+of)\s+\w+",
        lowered,
        re.I,
    )
    num_match = corr_num_match or re.search(
        rf"({num_tokens_pattern})(?:\s*(?:people|persons?|injured|victims?|log|లో|మంది|लोग|ಜನ|பேர்|പേർ|জন|মানুষ))?",
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
        "ಒಬ್ಬರು": 1, "ಒಬ್ಬ": 1, "ಒಂದು": 1,
        "ಇಬ್ಬರು": 2, "ಎರಡು": 2,
        "ಮೂವರು": 3, "ಮೂರು": 3,
        "ನಾಲ್ವರು": 4, "ನಾಲ್ಕು": 4,
        "ಐವರು": 5, "ಐದು": 5,
        "ஒருவர்": 1, "ஒரு": 1, "ஒன்று": 1,
        "இருவர்": 2, "இரண்டு": 2,
        "மூன்று": 3, "நான்கு": 4, "ஐந்து": 5,
        "ഒരാൾ": 1, "ഒന്ന്": 1,
        "രണ്ടുപേർ": 2, "രണ്ട്": 2,
        "മൂന്നുപേർ": 3, "മൂന്ന്": 3,
        "നാലുപേർ": 4, "നാല്": 4,
        "അഞ്ചുപേർ": 5, "അഞ്ച്": 5,
        "একজন": 1, "এক": 1,
        "দুইজন": 2, "দুজন": 2, "দুই": 2,
        "তিনজন": 3, "তিন": 3,
        "চারজন": 4, "চার": 4,
        "পাঁচজন": 5, "পাঁচ": 5,
    }
    count_val: int | None = None
    if num_match:
        tok = num_match.group(1).lower()
        count_val = int(tok) if tok.isdigit() else word_to_int.get(tok)
        if count_val is None:
            for prefix, mapped in (
                ("ఒక", 1), ("ఇద్ద", 2), ("ముగ్గు", 3), ("నలుగు", 4), ("ఐదుగు", 5),
                ("ಒಬ್ಬ", 1), ("ಇಬ್ಬ", 2), ("ಮೂವ", 3), ("ನಾಲ್ವ", 4), ("ಐವ", 5),
                ("ஒரு", 1), ("இரு", 2),
                ("രണ്ടു", 2), ("മൂന്നു", 3), ("നാലു", 4), ("അഞ്ചു", 5),
            ):
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
            r"[^\s,;.]*(?:injur|hurt|bleed|burn|trapped|unconscious|fainted|yes|గాయ|దెబ్బ|స్పృహ|అవును|घायल|चोट|बेहोश|हाँ|"
            r"ಗಾಯ|ರಕ್ತ|ಪ್ರಜ್ಞೆ|ಹೌದು|காய|ரத்தம்|மயக்கம்|ஆம்|പരിക്ക്|രക്തം|ബോധമില്ല|അതെ|আহত|রক্ত|অজ্ঞান|হ্যাঁ)[^\s,;.]*",
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

    # Extract caller spoken GPS/location confirmation across all 7 languages
    if re.search(
        r"\b(?:location\s+confirm\w*|confirm(?:ed)?(?:\s+(?:the\s+)?location)?|yes\s+confirm\w*)\b|"
        r"(?:లొకేషన్\s*కన్ఫర్మ్|కన్ఫర్మ్\s*చేశా|लोकेशन\s*कन्फर्म|कन्फर्म\s*किया|"
        r"ಲೊಕೇಶನ್\s*ಕನ್ಫರ್ಮ್|ಸ್ಥಳ\s*ದೃಢೀಕರಿಸಿ|லொகேஷன்\s*கன்ஃபார்ம்|இருப்பிடம்\s*உறுதி|"
        r"ലൊക്കേഷൻ\s*കൺഫേം|സ്ഥലം\s*സ്ഥിരീകരിച്ചു|লোকেশন\s*কনফার্ম|স্থান\s*নিশ্চিত)",
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
            r"^(?:yes|yeah|yep|yup|correct|someone|somebody|few|many|multiple|అవును|ఉన్నారు|ఆ|हाँ|जी|हां|"
            r"ಹೌದು|ஆம்|ஆமா|അതെ|হ্যাঁ)\b",
            lowered.strip(),
            re.I,
        )
    )
    has_no_reply = bool(
        re.match(
            r"^(?:no|nope|none|nobody|no\s*one|zero|nil|లేదు|ఎవరూ\s*లేరు|లేరు|नहीं|कोई\s*नहीं|"
            r"ಇಲ್ಲ|ಯಾರೂ\s*ಇಲ್ಲ|இல்லை|யாரும்\s*இல்லை|ഇല്ല|ആരുമില്ല|না|কেউ\s*না)\b[.!]?$",
            lowered.strip(),
            re.I,
        )
    )

    # Extract consciousness status whether spontaneous or answering a question
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
        r"(?:సురక్షితంగా\s*ఉన్నా|దూరంగా\s*ఉన్నా|सुरक्षित\s*दूरी|सुरक्षित\s*जगह|"
        r"ಸುರಕ್ಷಿತ\s*ಸ್ಥಳ|ದೂರದಲ್ಲಿದ್ದೇನೆ|பாதுகாப்பான\s*இடம்|தள்ளி\s*இருக்கிறேன்|"
        r"സുരക്ഷിത\s*സ്ഥലത്ത്|മാറി\s*നിൽക്കുന്നു|নিরাপদ\s*দূরত্বে|নিরাপদ\s*স্থানে)",
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

    if any(
        kw in lowered
        for kw in (
            "cylinder", "lpg", "gas cylinder", "సిలిండర్", "सिलेंडर",
            "ಸಿಲಿಂಡರ್", "சிலிண்டர்", "സിലിണ്ടർ", "সিলিন্ডার",
        )
    ):
        extracted.append(
            ProvenancedFact(
                field_name="hazard_details",
                value="gas_cylinder_involved",
                source_utterance_id=utterance_id,
                evidence_text=stripped[:80],
            )
        )

    # Extract location / landmark across all 7 languages (including Indic Unicode blocks)
    loc_val: str | None = None
    loc_ev: str | None = None
    indic_char_class = r"A-Za-z0-9\u0900-\u0D7F"
    loc_patterns = [
        # Street / building / landmark phrasing ("near the road beside ABC building", "beside xyz")
        re.compile(
            rf"((?:near|beside|opposite|behind|in\s+front\s+of|next\s+to)\s+"
            rf"[{indic_char_class}][{indic_char_class}\s.,-]{{2,65}}?)"
            r"(?=\s*(?:[,;]|\bnot\s+|\band\s+(?:\d|one|two|three|four|five|someone|nobody|no\s+one|people|injur|hurt|trapped)|\bwith\s|\bthere\s+is\b|[.?!]|$))",
            re.I,
        ),
        # English / Code-mixed prepositional phrase
        re.compile(
            rf"\b(?:at|in)\s+"
            rf"([{indic_char_class}][{indic_char_class}\s.-]{{2,45}}?)"
            r"(?=\s*(?:[,;]|\bnot\s+|\band\s+(?:\d|one|two|three|four|five|someone|nobody|no\s+one|people|injur|hurt|trapped)|\bwith\s|\bthere\s+is\b|[.?!]|$))",
            re.I,
        ),
        # Telugu postpositional landmark: "... దగ్గర", "... వద్ద", "... సమీపంలో"
        re.compile(r"([A-Za-z0-9\u0C00-\u0C7F][A-Za-z0-9\u0C00-\u0C7F .-]{1,45}?\s+(?:దగ్గర|వద్ద|సమీపంలో))(?=\s|$|[,.?!])", re.I),
        # Hindi postpositional landmark: "... के पास", "... के सामने", "... नजदीक"
        re.compile(r"([A-Za-z0-9\u0900-\u097F][A-Za-z0-9\u0900-\u097F .-]{1,45}?\s+(?:के\s+पास|के\s+सामने|नजदीक))(?=\s|$|[,.?!])", re.I),
        # Kannada postpositional landmark: "... ಹತ್ತಿರ", "... ಬಳಿ", "... ಮುಂದೆ"
        re.compile(r"([A-Za-z0-9\u0C80-\u0CFF][A-Za-z0-9\u0C80-\u0CFF .-]{1,45}?\s+(?:ಹತ್ತಿರ|ಬಳಿ|ಮುಂದೆ))(?=\s|$|[,.?!])", re.I),
        # Tamil postpositional landmark: "... அருகில்", "... பக்கத்தில்", "... எதிரில்"
        re.compile(r"([A-Za-z0-9\u0B80-\u0BFF][A-Za-z0-9\u0B80-\u0BFF .-]{1,45}?\s+(?:அருகில்|பக்கத்தில்|எதிரில்))(?=\s|$|[,.?!])", re.I),
        # Malayalam postpositional landmark: "... അടുത്ത്", "... സമീപം", "... മുന്നിൽ"
        re.compile(r"([A-Za-z0-9\u0D00-\u0D7F][A-Za-z0-9\u0D00-\u0D7F .-]{1,45}?\s+(?:അടുത്ത്|സമീപം|മുന്നിൽ))(?=\s|$|[,.?!])", re.I),
        # Bengali postpositional landmark: "... কাছে", "... সামনে", "... পাশে"
        re.compile(r"([A-Za-z0-9\u0980-\u09FF][A-Za-z0-9\u0980-\u09FF .-]{1,45}?\s+(?:কাছে|সামনে|পাশে))(?=\s|$|[,.?!।])", re.I),
        # Telugu place with locative suffix (-లో) or Telugu landmark noun
        re.compile(r"((?:[^\s,.?!]+\s+){0,2}[^\s,.?!]*(?:సిటీ|నగర్|రోడ్|రోడ్డు|స్టేషన్|స్కూల్|కాలేజ్|ఆసుపత్రి|కంపెనీ|బస్టాండ్|లో))(?=\s|$|[,.?!])", re.I),
        # Hindi place with locative noun
        re.compile(r"((?:[^\s,.?!]+\s+){0,2}[^\s,.?!]*(?:सिटी|नगर|रोड|स्टेशन|स्कूल|अस्पताल|कॉलोनी|चौक))(?=\s|$|[,.?!])", re.I),
        # Kannada / Tamil / Malayalam / Bengali landmark nouns
        re.compile(
            r"((?:[^\s,.?!।]+\s+){0,2}[^\s,.?!।]*(?:ನಗರ|ರಸ್ತೆ|ನಿಲ್ದಾಣ|ಆಸ್ಪತ್ರೆ|ಶಾಲೆ|ವೃತ್ತ|"
            r"நகர்|சாலை|நிலையம்|மருத்துவமனை|பள்ளி|"
            r"നഗർ|റോഡ്|സ്റ്റേഷൻ|ആശുപത്രി|സ്കൂൾ|ജംഗ്ഷൻ|"
            r"নগর|রোড|স্টেশন|হাসপাতাল|স্কুল|মোড়|বাজার))(?=\s|$|[,.?!।])",
            re.I,
        ),
    ]
    for pat in loc_patterns:
        m = pat.search(text)
        if m:
            candidate_loc = re.sub(
                r"^(?:ఇక్కడ|అక్కడ|ల్యాండ్\s*మార్క్|ల్యాండ్మార్క్|ఒక|yahan|wahan|landmark|"
                r"ಇಲ್ಲಿ|ಅಲ್ಲಿ|இங்கே|அங்கே|ഇവിടെ|അവിടെ|এখানে|ওখানে)\s+",
                "",
                m.group(1).strip(),
                flags=re.I,
            ).strip(" ,.-।")
            if candidate_loc.lower().startswith("near ") and not re.search(
                r"\b(beside|opposite|behind|in\s+front|street|road|building|gate|pillar|lane|junction|station|hospital|school|tower|stand)\b",
                candidate_loc[5:],
                re.I,
            ):
                candidate_loc = candidate_loc[5:].strip(" ,.-")
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

    if loc_val is None and asking_location and len(stripped) >= 3:
        first_clause = re.split(r"[,;।]", stripped, maxsplit=1)[0].strip(" .")
        fc_lower = first_clause.lower()
        has_kw_in_clause = bool(
            re.search(
                r"\b(fire|accident|explosion|collapse|attack|gas|injur|hurt|yes|no|nope|none|nobody|confirm\w*|people)\b|"
                r"(చెప్పాను\s*కదా|గుర్తుపెట్టుకో|కన్ఫర్మ్|గాయ|అగ్ని|ఫైర్|ప్రమాదం|घायल|आग|कन्फर्म|"
                r"ಗಾಯ|ಬೆಂಕಿ|ಅಪಘಾತ|காய|தீ|விபத்து|പരിക്ക്|തീ|അപകടം|আহত|আগুন|দুর্ঘটনা)",
                fc_lower,
                re.I,
            )
        )
        if len(first_clause) >= 3 and not has_kw_in_clause and not first_clause.isdigit():
            loc_val = first_clause[:80]
            loc_ev = loc_val

    if loc_val:
        is_explicit_correction = bool(
            re.search(
                r"\b(actually|not\s+at|not\s+in|instead\s+of|wrong\s+location|changed)\b|"
                r"కాదు|नहीं\s+वहां|ಅಲ್ಲ|அல்ல|അല്ല|নয়",
                lowered,
                re.I,
            )
        )
        is_street_or_landmark = bool(
            re.search(
                r"\b(near|beside|opposite|behind|in\s+front|street|road|building|gate|pillar|lane|junction|station|hospital|school|tower|stand)\b|"
                r"(?:ಹತ್ತಿರ|ಬಳಿ|அருகில்|பக்கத்தில்|അടുത്ത്|സമീപം|কাছে|পাশে)",
                loc_val,
                re.I,
            )
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

    asking_earlier_memory = bool(
        re.search(
            r"\b(remember\s+where|what\s+location|where\s+did\s+i\s+say|how\s+many\s+did\s+i\s+say|"
            r"which\s+units|what\s+units|how\s+long|eta|when\s+will\s+they\s+arrive|status)\b|"
            r"(?:ఎక్కడ\s*చెప్పాను|గుర్తుందా|ఎంత\s*సమయం|ఏ\s*వాహనాలు|कहाँ\s*बताया|याद\s*है|कितना\s*समय|कौन\s*सी\s*गाड़ी|"
            r"ಎಲ್ಲಿ\s*ಹೇಳಿದೆ|ನೆನಪಿದೆಯೇ|ಎಷ್ಟು\s*ಸಮಯ|எங்கே\s*சொன்னேன்|நினைவிருக்கிறதா|എവിടെ\s*പറഞ്ഞു|ഓർമ്മയുണ്ടോ|কোথায়\s*বলেছি|মনে\s*আছে)",
            lowered,
            re.I,
        )
    )

    next_q: str | None = None
    if asking_earlier_memory and location_known:
        next_q = None
    elif has_cylinder_hazard and "caller_safe_spot_status" not in known_fields:
        if not location_known:
            next_q = {
                "te": "మీరు గ్యాస్ సిలిండర్లకు దూరంగా సురక్షిత ప్రదేశంలో ఉన్నారా, మరియు మీ సమీప ల్యాండ్‌మార్క్ ఏమిటి?",
                "hi": "क्या आप गैस सिलेंडरों से दूर सुरक्षित स्थान पर हैं, और आपके पास का लैंडमार्क क्या है?",
                "kn": "ನೀವು ಗ್ಯಾಸ್ ಸಿಲಿಂಡರ್‌ಗಳಿಂದ ದೂರ ಸುರಕ್ಷಿತ ಸ್ಥಳದಲ್ಲಿದ್ದೀರಾ, ಮತ್ತು ನಿಮ್ಮ ಹತ್ತಿರದ ಲ್ಯಾಂಡ್‌ಮಾರ್ಕ್ ಯಾವುದು?",
                "ta": "நீங்கள் கேஸ் சிலிண்டர்களிலிருந்து தள்ளி பாதுகாப்பான இடத்தில் இருக்கிறீர்களா, உங்கள் அருகிலுள்ள அடையாளம் என்ன?",
                "ml": "നിങ്ങൾ ഗ്യാസ് സിലിണ്ടറുകളിൽ നിന്ന് മാറി സുരക്ഷിത സ്ഥലത്താണോ, അടുത്തുള്ള ലാൻഡ്‌മാർക്ക് എന്താണ്?",
                "bn": "আপনি কি গ্যাস সিলিন্ডার থেকে নিরাপদ দূরত্বে আছেন, এবং আপনার কাছের ল্যান্ডমার্ক কী?",
            }.get(
                active_language,
                "Are you at a safe distance away from the gas cylinders, and what area or landmark are you near?",
            )
        else:
            next_q = {
                "te": "మీరు ప్రస్తుతం మంటలు మరియు గ్యాస్ సిలిండర్లకు దూరంగా సురక్షిత ప్రదేశంలో ఉన్నారా?",
                "hi": "क्या आप इस समय आग और गैस सिलेंडरों से दूर सुरक्षित स्थान पर हैं?",
                "kn": "ನೀವು ಪ್ರಸ್ತುತ ಬೆಂಕಿ ಮತ್ತು ಗ್ಯಾಸ್ ಸಿಲಿಂಡರ್‌ಗಳಿಂದ ದೂರ ಸುರಕ್ಷಿತ ಸ್ಥಳದಲ್ಲಿದ್ದೀರಾ?",
                "ta": "நீங்கள் இப்போது தீ மற்றும் கேஸ் சிலிண்டர்களிலிருந்து தள்ளி பாதுகாப்பான இடத்தில் இருக்கிறீர்களா?",
                "ml": "നിങ്ങൾ ഇപ്പോൾ തീയിൽ നിന്നും ഗ്യാസ് സിലിണ്ടറുകളിൽ നിന്നും അകലെ സുരക്ഷിത സ്ഥലത്താണോ?",
                "bn": "আপনি কি এখন আগুন এবং গ্যাস সিলিন্ডার থেকে নিরাপদ দূরত্বে আছেন?",
            }.get(
                active_language,
                "Are you in a safe location well away from the fire and gas cylinders right now?",
            )
    elif not location_known:
        next_q = {
            "te": "మీరు ఉన్న ప్రాంతం లేదా సమీప ల్యాండ్‌మార్క్ ఏమిటో చెప్పగలరా?",
            "hi": "क्या आप अपने पास का क्षेत्र या लैंडमार्क बता सकते हैं?",
            "kn": "ನೀವು ಇರುವ ಪ್ರದೇಶ ಅಥವಾ ಹತ್ತಿರದ ಲ್ಯಾಂಡ್‌ಮಾರ್ಕ್ ಏನೆಂದು ಹೇಳಬಹುದೇ?",
            "ta": "நீங்கள் இருக்கும் பகுதி அல்லது அருகிலுள்ள அடையாளத்தைச் சொல்ல முடியுமா?",
            "ml": "നിങ്ങൾ നിൽക്കുന്ന സ്ഥലം അല്ലെങ്കിൽ അടുത്തുള്ള ലാൻഡ്‌മാർക്ക് പറയാമോ?",
            "bn": "আপনি কোন এলাকায় বা কোন ল্যান্ডমার্কের কাছে আছেন তা বলতে পারবেন?",
        }.get(
            active_language,
            "Could you tell me the area and nearest landmark so we can verify your pin?",
        )
    elif has_unconscious_person and has_unconscious_kw:
        next_q = {
            "te": "స్పృహ కోల్పోయిన వ్యక్తి ప్రస్తుతం సాధారణంగా శ్వాస తీసుకుంటున్నారా?",
            "hi": "क्या बेहोश व्यक्ति इस समय सामान्य रूप से सांस ले रहा है?",
            "kn": "ಪ್ರಜ್ಞೆ ತಪ್ಪಿದ ವ್ಯಕ್ತಿ ಪ್ರಸ್ತುತ ಸಹಜವಾಗಿ ಉಸಿರಾಡುತ್ತಿದ್ದಾರೆಯೇ?",
            "ta": "மயக்கமடைந்த நபர் இப்போது சாதாரணமாக மூச்சு விடுகிறாரா?",
            "ml": "ബോധരഹിതനായ വ്യക്തി ഇപ്പോൾ സാധാരണ നിലയിൽ ശ്വസിക്കുന്നുണ്ടോ?",
            "bn": "অজ্ঞান ব্যক্তিটি কি এখন স্বাভাবিকভাবে শ্বাস নিচ্ছেন?",
        }.get(
            active_language,
            "Is the unconscious person breathing normally right now?",
        )
    elif "injuries_reported" not in known_fields and inferred in (
        EmergencyCategory.FIRE,
        EmergencyCategory.ROAD_ACCIDENT,
        EmergencyCategory.GAS_LEAK,
        EmergencyCategory.BUILDING_COLLAPSE,
        EmergencyCategory.EXPLOSION,
    ):
        next_q = {
            "te": "ఎవరికైనా గాయాలు అయ్యాయా లేదా దయచేసి చెప్పగలరా?",
            "hi": "क्या वहां कोई घायल हुआ है?",
            "kn": "ಅಲ್ಲಿ ಯಾರಿಗಾದರೂ ಗಾಯವಾಗಿದೆಯೇ ಅಥವಾ ಯಾರಾದರೂ ಸಿಲುಕಿಕೊಂಡಿದ್ದಾರೆಯೇ?",
            "ta": "அங்கே யாருக்காவது காயம் ஏற்பட்டுள்ளதா அல்லது சிக்கியுள்ளார்களா?",
            "ml": "അവിടെ ആർക്കെങ്കിലും പരിക്കേറ്റിട്ടുണ്ടോ അല്ലെങ്കിൽ കുടുങ്ങിക്കിടക്കുന്നുണ്ടോ?",
            "bn": "সেখানে কি কেউ আহত হয়েছেন বা আটকে আছেন?",
        }.get(
            active_language,
            "Is anyone hurt or trapped there?",
        )
    elif injuries_yes and "affected_person_count" not in known_fields:
        next_q = {
            "te": "సుమారుగా ఎంతమంది గాయపడ్డారు?",
            "hi": "लगभग कितने लोग घायल हैं?",
            "kn": "ಸುಮಾರು ಎಷ್ಟು ಜನರು ಗಾಯಗೊಂಡಿದ್ದಾರೆ?",
            "ta": "தோராயமாக எத்தனை பேர் காயமடைந்துள்ளனர்?",
            "ml": "ഏകദേശം എത്ര പേർക്ക് പരിക്കേറ്റിട്ടുണ്ട്?",
            "bn": "আনুমানিক কতজন মানুষ আহত হয়েছেন?",
        }.get(
            active_language,
            "Roughly how many people are hurt?",
        )

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
    elif active_language == "kn":
        if asking_earlier_memory:
            parts_kn = []
            if known_loc_text:
                parts_kn.append(f"ನೀವು ತಿಳಿಸಿದ ಸ್ಥಳ {known_loc_text} ನನ್ನ ಗಮನದಲ್ಲಿದೆ")
            if known_count_val is not None:
                parts_kn.append(f"ಮತ್ತು {known_count_val} ಜನರ ವಿವರ ದಾಖಲಾಗಿದೆ")
            if committed_assignments:
                units_kn = ", ".join(f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} ನಿಮಿಷ)" for a in committed_assignments)
                parts_kn.append(f"ಈ ಡೆಮೊದಲ್ಲಿ {units_kn} ನಿಮ್ಮ ಕಡೆಗೆ ಬರುತ್ತಿವೆ")
            base_reply = (". ".join(parts_kn) + ".") if parts_kn else "ನಿಮ್ಮ ಎಲ್ಲಾ ವಿವರಗಳನ್ನು ನಾನು ದಾಖಲಿಸಿಕೊಂಡಿದ್ದೇನೆ ಮತ್ತು ನಿಮ್ಮೊಂದಿಗೆ ಲೈನ್‌ನಲ್ಲಿದ್ದೇನೆ."
        elif has_cylinder_hazard and any(f.field_name == "hazard_details" for f in extracted):
            base_reply = "ಅರ್ಥವಾಯಿತು, ಶಾಂತವಾಗಿರಿ. ಗ್ಯಾಸ್ ಸಿಲಿಂಡರ್ ಹತ್ತಿರ ಹೋಗಬೇಡಿ ಮತ್ತು ತಕ್ಷಣ ಸುರಕ್ಷಿತ ಸ್ಥಳಕ್ಕೆ ಸರಿಯಿರಿ."
        elif has_unconscious_kw:
            base_reply = "ಪರಿಸ್ಥಿತಿಯ ಗಂಭೀರತೆ ನನಗೆ ಅರ್ಥವಾಗುತ್ತಿದೆ — ದಯವಿಟ್ಟು ಅವರ ಪಕ್ಕದಲ್ಲೇ ಇರಿ ಮತ್ತು ಉಸಿರಾಟದ ದಾರಿ ಮುಕ್ತವಾಗಿರಲಿ."
        elif committed_assignments:
            units_kn = ", ".join(f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} ನಿಮಿಷ)" for a in committed_assignments)
            base_reply = f"ಆ ಮಾಹಿತಿಯನ್ನು ದಾಖಲಿಸಿದ್ದೇನೆ. ಈ ಡೆಮೊದಲ್ಲಿ {units_kn} ಮಾರ್ಗದಲ್ಲಿವೆ — ದಯವಿಟ್ಟು ಸುರಕ್ಷಿತವಾಗಿರಿ, ನಾನು ಲೈನ್‌ನಲ್ಲಿರುತ್ತೇನೆ."
        elif role == UtteranceRole.ANSWER_TO_PENDING_QUESTION:
            if loc_val:
                base_reply = f"ಧನ್ಯವಾದಗಳು, {loc_val} ಸ್ಥಳವನ್ನು ದಾಖಲಿಸಿಕೊಂಡಿದ್ದೇನೆ."
            elif count_val is not None:
                base_reply = f"ಅರ್ಥವಾಯಿತು, {count_val} ಜನರು ಎಂದು ದಾಖಲಿಸಿದ್ದೇನೆ."
            else:
                base_reply = "ಧನ್ಯವಾದಗಳು, ಆ ಪ್ರಮುಖ ವಿವರವನ್ನು ದಾಖಲಿಸಿದ್ದೇನೆ."
        else:
            base_reply = "ನಾನು ನಿಮ್ಮ ಮಾತು ಕೇಳುತ್ತಿದ್ದೇನೆ, ದಯವಿಟ್ಟು ಶಾಂತವಾಗಿರಿ. ಮೊದಲು ನೀವು ಸುರಕ್ಷಿತ ಸ್ಥಳದಲ್ಲಿರುವುದನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ."
    elif active_language == "ta":
        if asking_earlier_memory:
            parts_ta = []
            if known_loc_text:
                parts_ta.append(f"நீங்கள் கூறிய இடம் {known_loc_text} என் நினைவில் உள்ளது")
            if known_count_val is not None:
                parts_ta.append(f"மேலும் {known_count_val} பேர் பாதிக்கப்பட்டுள்ளனர் என பதிவு செய்துள்ளேன்")
            if committed_assignments:
                units_ta = ", ".join(f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} நிமிடம்)" for a in committed_assignments)
                parts_ta.append(f"இந்த டெமோவில் {units_ta} உங்கள் இடத்திற்கு வருகின்றன")
            base_reply = (". ".join(parts_ta) + ".") if parts_ta else "உங்கள் அனைத்து விவரங்களையும் பதிவு செய்துள்ளேன், நான் இணைப்பில் இருக்கிறேன்."
        elif has_cylinder_hazard and any(f.field_name == "hazard_details" for f in extracted):
            base_reply = "புரிகிறது, அமைதியாக இருங்கள். கேஸ் சிலிண்டர் அருகில் செல்ல வேண்டாம், உடனடியாக பாதுகாப்பான இடத்திற்குச் செல்லுங்கள்."
        elif has_unconscious_kw:
            base_reply = "நிலைமையின் தீவிரம் புரிகிறது — தயவுசெய்து அவர் அருகிலேயே இருங்கள், சுவாசப் பாதை தடையின்றி இருக்கட்டும்."
        elif committed_assignments:
            units_ta = ", ".join(f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} நிமிடம்)" for a in committed_assignments)
            base_reply = f"அந்த விவரத்தைப் பதிவு செய்தேன். இந்த டெமோவில் {units_ta} வழியில் உள்ளன — பாதுகாப்பாக இருங்கள், நான் இணைப்பில் இருக்கிறேன்."
        elif role == UtteranceRole.ANSWER_TO_PENDING_QUESTION:
            if loc_val:
                base_reply = f"நன்றி, {loc_val} என்ற இடத்தைப் பதிவு செய்துள்ளேன்."
            elif count_val is not None:
                base_reply = f"புரிந்தது, {count_val} பேர் எனப் பதிவு செய்துள்ளேன்."
            else:
                base_reply = "நன்றி, அந்த முக்கிய விவரத்தைப் பதிவு செய்துள்ளேன்."
        else:
            base_reply = "நான் உங்கள் பேச்சைக் கேட்டுக்கொண்டிருக்கிறேன், அமைதியாக இருங்கள். முதலில் நீங்கள் பாதுகாப்பான இடத்தில் இருப்பதை உறுதிப்படுத்திக் கொள்ளுங்கள்."
    elif active_language == "ml":
        if asking_earlier_memory:
            parts_ml = []
            if known_loc_text:
                parts_ml.append(f"നിങ്ങൾ പറഞ്ഞ സ്ഥലം {known_loc_text} ഞാൻ രേഖപ്പെടുത്തിയിട്ടുണ്ട്")
            if known_count_val is not None:
                parts_ml.append(f"കൂടാതെ {known_count_val} പേർക്ക് ബുദ്ധിമുട്ടുണ്ടെന്ന് കുറിച്ചിട്ടുണ്ട്")
            if committed_assignments:
                units_ml = ", ".join(f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} മിനിറ്റ്)" for a in committed_assignments)
                parts_ml.append(f"ഈ ഡെമോയിൽ {units_ml} നിങ്ങളുടെ അടുത്തേക്ക് വരുന്നുണ്ട്")
            base_reply = (". ".join(parts_ml) + ".") if parts_ml else "നിങ്ങളുടെ എല്ലാ വിവരങ്ങളും ഞാൻ രേഖപ്പെടുത്തിയിട്ടുണ്ട്, ഞാൻ ലൈനിൽ തുടരുന്നു."
        elif has_cylinder_hazard and any(f.field_name == "hazard_details" for f in extracted):
            base_reply = "മനസ്സിലായി, സമാധാനമായിരിക്കുക. ഗ്യാസ് സിലിണ്ടറിന് സമീപം പോകരുത്, ഉടൻ സുരക്ഷിതമായ ദൂരത്തേക്ക് മാറി നിൽക്കുക."
        elif has_unconscious_kw:
            base_reply = "സാഹചര്യത്തിന്റെ ഗൗരവം ഞാൻ മനസ്സിലാക്കുന്നു — ദയവായി അവരുടെ അരികിൽ തന്നെ തുടരുക, ശ്വാസതടസ്സം ഇല്ലെന്ന് ഉറപ്പാക്കുക."
        elif committed_assignments:
            units_ml = ", ".join(f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} മിനിറ്റ്)" for a in committed_assignments)
            base_reply = f"ആ വിവരം രേഖപ്പെടുത്തിയിട്ടുണ്ട്. ഈ ഡെമോയിൽ {units_ml} വഴിയിലാണ് — സുരക്ഷിതമായി ഇരിക്കുക, ഞാൻ ലൈനിലുണ്ട്."
        elif role == UtteranceRole.ANSWER_TO_PENDING_QUESTION:
            if loc_val:
                base_reply = f"നന്ദി, {loc_val} എന്ന സ്ഥലം ഞാൻ രേഖപ്പെടുത്തിയിട്ടുണ്ട്."
            elif count_val is not None:
                base_reply = f"മനസ്സിലായി, {count_val} പേർ എന്ന് രേഖപ്പെടുത്തിയിട്ടുണ്ട്."
            else:
                base_reply = "നന്ദി, ആ പ്രധാന വിവരം ഞാൻ രേഖപ്പെടുത്തിയിട്ടുണ്ട്."
        else:
            base_reply = "ഞാൻ നിങ്ങൾ പറയുന്നത് കേൾക്കുന്നുണ്ട്, സമാധാനമായിരിക്കുക. ആദ്യം നിങ്ങൾ സുരക്ഷിതമായ ഒരിടത്താണെന്ന് ഉറപ്പാക്കുക."
    elif active_language == "bn":
        if asking_earlier_memory:
            parts_bn = []
            if known_loc_text:
                parts_bn.append(f"আপনার বলা স্থান {known_loc_text} আমার মনে আছে")
            if known_count_val is not None:
                parts_bn.append(f"এবং {known_count_val} জন আক্রান্তের তথ্য নথিভুক্ত আছে")
            if committed_assignments:
                units_bn = ", ".join(f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} মিনিট)" for a in committed_assignments)
                parts_bn.append(f"এই ডেমোতে {units_bn} আপনার দিকে আসছে")
            base_reply = ("। ".join(parts_bn) + "।") if parts_bn else "আমি আপনার সব তথ্য নথিভুক্ত করেছি এবং আমি আপনার সাথে লাইনে আছি।"
        elif has_cylinder_hazard and any(f.field_name == "hazard_details" for f in extracted):
            base_reply = "আমি বুঝতে পারছি, শান্ত থাকুন। জ্বলন্ত বা লিক হওয়া গ্যাস সিলিন্ডারের কাছে যাবেন না এবং নিরাপদ দূরত্বে সরে যান।"
        elif has_unconscious_kw:
            base_reply = "আমি পরিস্থিতির গুরুত্ব বুঝতে পারছি — অনুগ্রহ করে তাঁর পাশে থাকুন এবং শ্বাসপ্রশ্বাসের পথ খোলা রাখুন।"
        elif committed_assignments:
            units_bn = ", ".join(f"{a.resource_id} (ETA {a.remaining_eta_minutes or a.simulated_eta_minutes or 6} মিনিট)" for a in committed_assignments)
            base_reply = f"আমি এই তথ্য নথিভুক্ত করেছি। এই ডেমোতে {units_bn} পথে রয়েছে — নিরাপদে থাকুন, আমি লাইনে আছি।"
        elif role == UtteranceRole.ANSWER_TO_PENDING_QUESTION:
            if loc_val:
                base_reply = f"ধন্যবাদ, আমি {loc_val} লোকেশন নথিভুক্ত করেছি।"
            elif count_val is not None:
                base_reply = f"বুঝেছি, আমি {count_val} জন মানুষের সংখ্যা নোট করেছি।"
            else:
                base_reply = "ধন্যবাদ, আমি এই গুরুত্বপূর্ণ তথ্যটি নথিভুক্ত করেছি।"
        else:
            base_reply = "আমি আপনার কথা মনোযোগ দিয়ে শুনছি, অনুগ্রহ করে শান্ত থাকুন। প্রথমে নিশ্চিত করুন যে আপনি নিরাপদ স্থানে আছেন।"
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
