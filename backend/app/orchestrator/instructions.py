"""
Approved safety instruction library, audio asset manifest, and STAY WITH ME
cached reassurance phrases (Section 15 & Section 16).

Enforces:
- Rule 4: No free-form medical or first-aid advice; only pre-reviewed safety instructions.
- Section 2.1: Pre-synthesized spoken disclosure in English, Telugu, and Hindi.
- Section 9.3 & 9.4: Pre-synthesized required fallback statements and re-ask prompts.
- Section 15.2 & 15.3: Checksummed audio asset manifest with cache hit/miss tracking.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib

from backend.app.adapters.base import EmergencyCategory, SupportedLanguage


REQUIRED_PROTOTYPE_DISCLOSURE_EN = (
    "SETU is a prototype for emergency coordination. It does not replace India's 112 "
    "emergency service and is not connected to 112. This demo uses simulated emergency "
    "resources, availability, assignments and ETAs. No real emergency response is "
    "requested or dispatched. For a real emergency, call 112."
)

REQUIRED_UNKNOWN_FALLBACK_STATEMENT_EN = (
    "SETU couldn't confidently identify the emergency. In this demo, I've created a "
    "simulated ambulance assignment and flagged the incident for review. This is not a real dispatch."
)

REQUIRED_PROVIDER_BLOCKED_STATEMENT_EN = (
    "SETU is having trouble processing what you described. In this demo, I've created a "
    "simulated ambulance assignment and flagged the incident for review. This is not a real dispatch."
)


@dataclass(frozen=True)
class ApprovedInstruction:
    """Single versioned, pre-reviewed safety or reassurance phrase (Rule 4 & Section 15.1)."""
    instruction_id: str
    category: EmergencyCategory | str
    language: SupportedLanguage
    approved_text: str
    source: str
    review_date: date
    version: str


@dataclass(frozen=True)
class AudioAssetManifestEntry:
    """Manifest entry for pre-synthesized instruction audio (Section 15.2)."""
    instruction_id: str
    language: SupportedLanguage
    voice_id: str
    timbre_version: str
    text_version: str
    audio_format: str
    file_path: str
    checksum_sha256: str
    generated_at: datetime
    generation_status: str

    @property
    def cache_key(self) -> str:
        """Cache identity includes instruction_id, text_version, language, voice, and model version."""
        return f"{self.instruction_id}:{self.text_version}:{self.language}:{self.voice_id}:{self.timbre_version}"


def _build_default_instructions() -> tuple[ApprovedInstruction, ...]:
    rev_date = date(2026, 11, 1)
    src = "placeholder, pending review"
    items: list[ApprovedInstruction] = [
        # Spoken prototype disclosures (Section 2.1)
        ApprovedInstruction(
            instruction_id="disclosure_session_start",
            category="system_disclosure",
            language="en",
            approved_text=REQUIRED_PROTOTYPE_DISCLOSURE_EN,
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="disclosure_session_start",
            category="system_disclosure",
            language="te",
            approved_text=(
                "సేతు (SETU) అత్యవసర సమన్వయం కోసం రూపొందించిన నమూనా (prototype). ఇది భారతదేశ అధికారిక 112 "
                "సేవకు ప్రత్యామ్నాయం కాదు మరియు 112కి అనుసంధానించబడి లేదు. ఈ డెమోలో వనరులు మరియు కేటాయింపులు "
                "అన్నీ నమూనా (simulated) మాత్రమే. నిజమైన అత్యవసర పరిస్థితికి 112కి కాల్ చేయండి."
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="disclosure_session_start",
            category="system_disclosure",
            language="hi",
            approved_text=(
                "सेतु (SETU) आपातकालीन समन्वय का एक प्रोटोटाइप है। यह भारत की आधिकारिक 112 सेवा का स्थान "
                "नहीं लेता और 112 से जुड़ा नहीं है। इस डेमो में सभी संसाधन, आवंटन और समय (ETA) सिम्युलेटेड हैं। "
                "वास्तविक आपातकाल के लिए कृपया 112 पर कॉल करें।"
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # Unknown fallback statements (Section 9.3)
        ApprovedInstruction(
            instruction_id="unknown_emergency_fallback",
            category="system_fallback",
            language="en",
            approved_text=REQUIRED_UNKNOWN_FALLBACK_STATEMENT_EN,
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="unknown_emergency_fallback",
            category="system_fallback",
            language="te",
            approved_text=(
                "సేతు అత్యవసర రకాన్ని ఖచ్చితంగా గుర్తించలేకపోయింది. ఈ డెమోలో, నమూనా అంబులెన్స్ కేటాయింపు "
                "సృష్టించబడింది మరియు సమీక్ష కోసం ఫ్లాగ్ చేయబడింది. ఇది నిజమైన డిస్పాచ్ కాదు."
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="unknown_emergency_fallback",
            category="system_fallback",
            language="hi",
            approved_text=(
                "सेतु आपातकाल की स्पष्ट पहचान नहीं कर सका। इस डेमो में एक सिम्युलेटेड एम्बुलेंस आवंटन "
                "बनाया गया है और समीक्षा के लिए फ़्लैग किया गया है। यह वास्तविक डिस्पैच नहीं है।"
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # Provider-blocked prompts and fallback statements (Section 9.4)
        ApprovedInstruction(
            instruction_id="provider_blocked_reask",
            category="system_fallback",
            language="en",
            approved_text=(
                "I'm here with you in this simulation. Could you please describe the situation again "
                "in your own words so I can record the details?"
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="provider_blocked_reask",
            category="system_fallback",
            language="te",
            approved_text=(
                "నేను మీతోనే ఉన్నాను. దయచేసి అక్కడ ఏమి జరుగుతుందో మరొకసారి మీ మాటల్లో చెప్పగలరా?"
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="provider_blocked_reask",
            category="system_fallback",
            language="hi",
            approved_text=(
                "मैं इस सिम्युलेशन में आपके साथ हूँ। कृपया स्थिति को अपने शब्दों में एक बार फिर बताएं।"
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="provider_blocked_fallback",
            category="system_fallback",
            language="en",
            approved_text=REQUIRED_PROVIDER_BLOCKED_STATEMENT_EN,
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="provider_blocked_fallback",
            category="system_fallback",
            language="te",
            approved_text=(
                "మీరు చెప్పిన వివరాలను ప్రాసెస్ చేయడంలో సేతుకు అంతరాయం కలిగింది. ఈ డెమోలో నమూనా అంబులెన్స్ "
                "కేటాయింపు సృష్టించబడింది మరియు సమీక్ష కోసం ఫ్లాగ్ చేయబడింది. ఇది నిజమైన డిస్పాచ్ కాదు."
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="provider_blocked_fallback",
            category="system_fallback",
            language="hi",
            approved_text=(
                "आपने जो बताया उसे प्रोसेस करने में सेतु को कठिनाई हो रही है। इस डेमो में एक सिम्युलेटेड "
                "एम्बुलेंस आवंटन बनाया गया है और समीक्षा के लिए फ़्लैग किया गया है। यह वास्तविक डिस्पैच नहीं है।"
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # Empty / non-speech barge-in prompt (Section 18.4)
        ApprovedInstruction(
            instruction_id="didnt_catch_that",
            category="system_prompt",
            language="en",
            approved_text="Sorry, I didn't catch that. Please say that again.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="didnt_catch_that",
            category="system_prompt",
            language="te",
            approved_text="క్షమించండి, నాకు స్పష్టంగా వినిపించలేదు. దయచేసి మళ్ళీ చెప్పండి.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="didnt_catch_that",
            category="system_prompt",
            language="hi",
            approved_text="क्षमा करें, मैं समझ नहीं पाया। कृपया उसे दोबारा कहें।",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # Outage reassurance lines (Section 15.4)
        ApprovedInstruction(
            instruction_id="outage_reassurance_1",
            category="outage_reassurance",
            language="en",
            approved_text="I'm here with you. Please stay on the line while this simulation continues.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="outage_reassurance_2",
            category="outage_reassurance",
            language="en",
            approved_text="This is a prototype and is not connected to real emergency services. For a real emergency, call 112.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="outage_reassurance_1",
            category="outage_reassurance",
            language="te",
            approved_text="నేను మీతోనే ఉన్నాను. ఈ నమూనా కొనసాగుతున్నప్పుడు దయచేసి లైన్‌లో ఉండండి.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="outage_reassurance_2",
            category="outage_reassurance",
            language="te",
            approved_text="ఇది నమూనా మాత్రమే, నిజమైన అత్యవసర సేవలతో అనుసంధానించబడి లేదు. నిజమైన అత్యవసరానికి 112కి కాల్ చేయండి.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="outage_reassurance_1",
            category="outage_reassurance",
            language="hi",
            approved_text="मैं आपके साथ हूँ। कृपया इस सिम्युलेशन के दौरान लाइन पर बने रहें।",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="outage_reassurance_2",
            category="outage_reassurance",
            language="hi",
            approved_text="यह एक प्रोटोटाइप है और वास्तविक आपातकालीन सेवाओं से जुड़ा नहीं है। वास्तविक आपातकाल में 112 पर कॉल करें।",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # Speak-While-Waiting holding phrases (Section 16)
        ApprovedInstruction(
            instruction_id="holding_checking_resources",
            category="holding_phrase",
            language="en",
            approved_text="I'm checking which simulated resources and nearby hospitals are available in the directory.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="holding_checking_resources",
            category="holding_phrase",
            language="te",
            approved_text="డైరెక్టరీలో అందుబాటులో ఉన్న నమూనా వనరులు మరియు ఆసుపత్రులను పరిశీలిస్తున్నాను.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="holding_checking_resources",
            category="holding_phrase",
            language="hi",
            approved_text="मैं डायरेक्टरी में उपलब्ध सिम्युलेटेड संसाधनों और पास के अस्पतालों की जाँच कर रहा हूँ।",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # Category-specific STAY WITH ME safety instructions (Rule 4 - strictly non-medical safety guidance)
        ApprovedInstruction(
            instruction_id="safety_road_accident_01",
            category=EmergencyCategory.ROAD_ACCIDENT,
            language="en",
            approved_text="Please stay at a safe distance from moving traffic and do not stand on the active roadway.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_road_accident_01",
            category=EmergencyCategory.ROAD_ACCIDENT,
            language="te",
            approved_text="దయచేసి వాహనాల రాకపోకలకు దూరంగా సురక్షితమైన ప్రదేశంలో నిలబడండి.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_road_accident_01",
            category=EmergencyCategory.ROAD_ACCIDENT,
            language="hi",
            approved_text="कृपया चलते यातायात से सुरक्षित दूरी बनाए रखें और सड़क के बीच में खड़े न हों।",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_01",
            category=EmergencyCategory.FIRE,
            language="en",
            approved_text=(
                "Please move away from smoke and flames immediately, keep everyone at a safe distance, "
                "use stairs rather than elevators, and do not re-enter the building. For a real emergency, call 112."
            ),
            source="SETU Fire Safety Guidance (NDMA / Fire Services Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_01",
            category=EmergencyCategory.FIRE,
            language="te",
            approved_text=(
                "దయచేసి పొగ మరియు మంటలకు వెంటనే దూరంగా సురక్షిత ప్రదేశానికి వెళ్ళండి, లిఫ్ట్ వాడకండి, "
                "మళ్ళీ లోపలికి వెళ్లవద్దు. నిజమైన అత్యవసర పరిస్థితికి వెంటనే 112కి కాల్ చేయండి."
            ),
            source="SETU Fire Safety Guidance (NDMA / Fire Services Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_01",
            category=EmergencyCategory.FIRE,
            language="hi",
            approved_text=(
                "कृपया धुएँ और आग से तुरंत सुरक्षित दूरी पर हट जाएँ, लिफ्ट का उपयोग न करें और इमारत में वापस न जाएँ। "
                "वास्तविक आपातकाल के लिए कृपया 112 पर कॉल करें।"
            ),
            source="SETU Fire Safety Guidance (NDMA / Fire Services Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_gas_cylinder_01",
            category=EmergencyCategory.FIRE,
            language="en",
            approved_text=(
                "Please move to an open area well away from the hazard right away. Do not go near, touch, "
                "or try to move a burning or leaking gas cylinder, and avoid anything that can create a spark. "
                "This is a SETU demo simulation; in a real emergency, call 112 immediately."
            ),
            source="SETU LPG / Gas Cylinder Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_gas_cylinder_01",
            category=EmergencyCategory.FIRE,
            language="te",
            approved_text=(
                "దయచేసి వెంటనే మంటలకు దూరంగా బహిరంగ, సురక్షిత ప్రదేశానికి వెళ్ళండి. మండుతున్న లేదా లీక్ అవుతున్న "
                "గ్యాస్ సిలిండర్ దగ్గరికి వెళ్లవద్దు మరియు దాన్ని తాకవద్దు లేదా కదిలించవద్దు. నిజమైన అత్యవసరానికి 112కి కాల్ చేయండి."
            ),
            source="SETU LPG / Gas Cylinder Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_gas_cylinder_01",
            category=EmergencyCategory.FIRE,
            language="hi",
            approved_text=(
                "कृपया तुरंत आग और धुएँ से दूर खुले, सुरक्षित स्थान पर चले जाएँ। जलते या लीक होते गैस सिलेंडर के पास न जाएँ "
                "और उसे छूने या हटाने की कोशिश न करें। वास्तविक आपातकाल में तुरंत 112 पर कॉल करें।"
            ),
            source="SETU LPG / Gas Cylinder Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_injuries_01",
            category=EmergencyCategory.FIRE_WITH_INJURIES,
            language="en",
            approved_text=(
                "Please help move people clear of smoke and heat into fresh air only if you can do so safely without entering flames, "
                "do not try to move cylinders, and keep clear for responders. In a real emergency, call 112."
            ),
            source="SETU Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_injuries_01",
            category=EmergencyCategory.FIRE_WITH_INJURIES,
            language="te",
            approved_text=(
                "మీరు సురక్షితంగా చేయగలిగితేనే పొగ మరియు వేడికి దూరంగా స్వచ్ఛమైన గాలి ఉన్న చోటుకు వెళ్ళండి, మంటల లోపలికి వెళ్లవద్దు. "
                "నిజమైన అత్యవసర పరిస్థితికి 112కి కాల్ చేయండి."
            ),
            source="SETU Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_injuries_01",
            category=EmergencyCategory.FIRE_WITH_INJURIES,
            language="hi",
            approved_text=(
                "यदि सुरक्षित रूप से संभव हो तो ही घायलों को धुएँ और गर्मी से दूर खुली हवा में ले जाएँ, आग के पास वापस न जाएँ। "
                "वास्तविक आपातकाल में 112 पर कॉल करें।"
            ),
            source="SETU Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_medical_01",
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            language="en",
            approved_text=(
                "Please stay near the person if it is safe to do so, keep the entryway clear, "
                "keep them comfortable without giving food or water if drowsy, and call 112 in a real emergency."
            ),
            source="SETU Emergency Intake Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_medical_01",
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            language="te",
            approved_text=(
                "సురక్షితంగా ఉంటే బాధితుడి దగ్గరే ఉండండి, ప్రవేశ మార్గాన్ని ఖాళీగా ఉంచండి. "
                "నిజమైన అత్యవసర పరిస్థితికి వెంటనే 112కి కాల్ చేయండి."
            ),
            source="SETU Emergency Intake Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_medical_01",
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            language="hi",
            approved_text=(
                "यदि सुरक्षित हो तो पीड़ित के पास रहें और आने-जाने का रास्ता खुला रखें। "
                "वास्तविक आपातकाल में तुरंत 112 पर कॉल करें।"
            ),
            source="SETU Emergency Intake Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_collapse_01",
            category=EmergencyCategory.BUILDING_COLLAPSE,
            language="en",
            approved_text="Please move well away from unstable walls, debris, and overhead wires, and do not enter the damaged structure.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_collapse_01",
            category=EmergencyCategory.BUILDING_COLLAPSE,
            language="te",
            approved_text="దెబ్బతిన్న గోడలు, శిథిలాలు మరియు విద్యుత్ తీగలకు దూరంగా ఉండండి, భవనంలోకి వెళ్లవద్దు.",
            source="SETU Structural Hazard Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_collapse_01",
            category=EmergencyCategory.BUILDING_COLLAPSE,
            language="hi",
            approved_text="कमजोर दीवारों, मलबे और बिजली के तारों से दूर रहें, क्षतिग्रस्त इमारत के अंदर न जाएँ।",
            source="SETU Structural Hazard Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_attack_01",
            category=EmergencyCategory.PHYSICAL_ATTACK,
            language="en",
            approved_text="Prioritise your immediate personal safety: move to a secure location away from the threat if you safely can.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_attack_01",
            category=EmergencyCategory.PHYSICAL_ATTACK,
            language="te",
            approved_text="మీ వ్యక్తిగత భద్రతకు మొదటి ప్రాధాన్యత ఇవ్వండి: వీలైతే ప్రమాదానికి దూరంగా సురక్షిత ప్రదేశానికి వెళ్ళండి.",
            source="SETU Personal Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_attack_01",
            category=EmergencyCategory.PHYSICAL_ATTACK,
            language="hi",
            approved_text="अपनी सुरक्षा को प्राथमिकता दें: यदि सुरक्षित रूप से संभव हो तो खतरे से दूर किसी सुरक्षित स्थान पर चले जाएँ।",
            source="SETU Personal Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_gas_leak_01",
            category=EmergencyCategory.GAS_LEAK,
            language="en",
            approved_text=(
                "Move to fresh air outdoors immediately and stay at a safe distance. Do not touch electrical switches, "
                "light matches, or try to move a leaking or burning cylinder. In a real emergency, call 112."
            ),
            source="SETU Gas Leak Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_gas_leak_01",
            category=EmergencyCategory.GAS_LEAK,
            language="te",
            approved_text=(
                "వెంటనే బయట స్వచ్ఛమైన గాలి ఉన్న సురక్షిత ప్రదేశానికి వెళ్ళండి. విద్యుత్ స్విచ్‌లు, నిప్పు లేదా అగ్గిపెట్టెలు వాడవద్దు "
                "మరియు గ్యాస్ సిలిండర్ దగ్గరికి వెళ్లవద్దు. నిజమైన అత్యవసరానికి 112కి కాల్ చేయండి."
            ),
            source="SETU Gas Leak Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_gas_leak_01",
            category=EmergencyCategory.GAS_LEAK,
            language="hi",
            approved_text=(
                "तुरंत बाहर खुली हवा में सुरक्षित दूरी पर चले जाएँ। बिजली के स्विच, माचिस या आग का उपयोग न करें "
                "और लीक होते सिलेंडर के पास न जाएँ। वास्तविक आपातकाल में 112 पर कॉल करें।"
            ),
            source="SETU Gas Leak Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_explosion_01",
            category=EmergencyCategory.EXPLOSION,
            language="en",
            approved_text="Move to an open area away from damaged structures, glass, and smoke, and stay clear of potential secondary hazards.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_explosion_01",
            category=EmergencyCategory.EXPLOSION,
            language="te",
            approved_text="దెబ్బతిన్న భవనాలు, గాజు ముక్కలు మరియు పొగకు దూరంగా బహిరంగ ప్రదేశంలో ఉండండి.",
            source="SETU Hazard Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_explosion_01",
            category=EmergencyCategory.EXPLOSION,
            language="hi",
            approved_text="क्षतिग्रस्त इमारतों, टूटे काँच और धुएँ से दूर किसी खुले सुरक्षित स्थान पर चले जाएँ।",
            source="SETU Hazard Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_unknown_01",
            category=EmergencyCategory.UNKNOWN,
            language="en",
            approved_text="Please remain in a safe spot and stay on the line with me while we review the details in this simulation.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_unknown_01",
            category=EmergencyCategory.UNKNOWN,
            language="te",
            approved_text="దయచేసి సురక్షిత ప్రదేశంలో ఉండండి మరియు నేను వివరాలు నమోదు చేస్తున్నప్పుడు లైన్‌లో ఉండండి.",
            source="SETU Intake Reassurance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_unknown_01",
            category=EmergencyCategory.UNKNOWN,
            language="hi",
            approved_text="कृपया किसी सुरक्षित स्थान पर रहें और जब तक मैं विवरण दर्ज करूँ, लाइन पर बने रहें।",
            source="SETU Intake Reassurance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
    ]
    return tuple(items)


class InstructionLibrary:
    """
    Versioned instruction library and pre-synthesized audio manifest manager (Section 15).
    """

    def __init__(
        self,
        instructions: tuple[ApprovedInstruction, ...] | None = None,
        *,
        default_voice_by_lang: dict[SupportedLanguage, str] | None = None,
        timbre_version: str = "v2.5",
        timbre_gate_0_2_verified: bool = False,
    ) -> None:
        self._instructions = instructions or _build_default_instructions()
        self._voices: dict[SupportedLanguage, str] = default_voice_by_lang or {
            "en": "timbre-v2.5-en-in-std",
            "te": "timbre-v2.5-te-in-std",
            "hi": "timbre-v2.5-hi-in-std",
        }
        self._timbre_version = timbre_version
        self._timbre_gate_0_2_verified = timbre_gate_0_2_verified
        self._manifest: dict[str, AudioAssetManifestEntry] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._initialize_presynthesized_manifest()

    def _initialize_presynthesized_manifest(self) -> None:
        from pathlib import Path

        now = datetime.now(timezone.utc)
        status = "ready" if self._timbre_gate_0_2_verified else "development_placeholder_pending_gate_0_2"
        for inst in self._instructions:
            voice_id = self._voices[inst.language]
            rel_path = f"backend-media/instructions/{inst.language}/{inst.instruction_id}/{inst.version}.wav"
            wav_file = Path(rel_path)
            if wav_file.exists():
                digest = hashlib.sha256(wav_file.read_bytes()).hexdigest()
            else:
                digest = hashlib.sha256(
                    f"{inst.instruction_id}:{inst.version}:{inst.language}:{voice_id}:{self._timbre_version}:{inst.approved_text}".encode(
                        "utf-8"
                    )
                ).hexdigest()
            entry = AudioAssetManifestEntry(
                instruction_id=inst.instruction_id,
                language=inst.language,
                voice_id=voice_id,
                timbre_version=self._timbre_version,
                text_version=inst.version,
                audio_format="pcm_s16le_16000",
                file_path=rel_path,
                checksum_sha256=digest,
                generated_at=now,
                generation_status=status,
            )
            self._manifest[f"{inst.instruction_id}:{inst.language}"] = entry

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    def get_eligible_instructions(
        self,
        category: EmergencyCategory,
        language: SupportedLanguage,
    ) -> list[ApprovedInstruction]:
        """Return backend-computed eligible instructions for the category and language (Section 15.1)."""
        exact = [
            i
            for i in self._instructions
            if i.category == category and i.language == language
        ]
        if exact:
            return exact
        # Fallback to English approved safety instruction for that category if localized safety line not present
        return [
            i
            for i in self._instructions
            if i.category == category and i.language == "en"
        ]

    def get_by_id(
        self,
        instruction_id: str,
        language: SupportedLanguage,
    ) -> ApprovedInstruction | None:
        for i in self._instructions:
            if i.instruction_id == instruction_id and i.language == language:
                return i
        for i in self._instructions:
            if i.instruction_id == instruction_id and i.language == "en":
                return i
        return None

    def resolve_cached_asset(
        self,
        instruction_id: str,
        language: SupportedLanguage,
    ) -> tuple[ApprovedInstruction, AudioAssetManifestEntry | None]:
        """Look up pre-synthesized cached asset and record cache hit/miss metrics (Section 15.3)."""
        inst = self.get_by_id(instruction_id, language)
        if inst is None:
            raise KeyError(f"Unknown instruction_id: {instruction_id}")
        key = f"{inst.instruction_id}:{inst.language}"
        manifest_entry = self._manifest.get(key)
        if manifest_entry is not None and manifest_entry.generation_status in (
            "ready",
            "development_placeholder_pending_gate_0_2",
        ):
            self._cache_hits += 1
            return inst, manifest_entry
        self._cache_misses += 1
        return inst, None

    def get_spoken_disclosure(self, language: SupportedLanguage) -> ApprovedInstruction:
        inst = self.get_by_id("disclosure_session_start", language)
        assert inst is not None
        return inst

    def get_unknown_fallback_statement(self, language: SupportedLanguage) -> ApprovedInstruction:
        inst = self.get_by_id("unknown_emergency_fallback", language)
        assert inst is not None
        return inst

    def get_provider_blocked_reask_prompt(self, language: SupportedLanguage) -> ApprovedInstruction:
        inst = self.get_by_id("provider_blocked_reask", language)
        assert inst is not None
        return inst

    def get_provider_blocked_fallback_statement(self, language: SupportedLanguage) -> ApprovedInstruction:
        inst = self.get_by_id("provider_blocked_fallback", language)
        assert inst is not None
        return inst

    def get_didnt_catch_that_phrase(self, language: SupportedLanguage) -> ApprovedInstruction:
        inst = self.get_by_id("didnt_catch_that", language)
        assert inst is not None
        return inst

    def get_holding_phrase(self, language: SupportedLanguage) -> ApprovedInstruction:
        inst = self.get_by_id("holding_checking_resources", language)
        assert inst is not None
        return inst

    def get_safety_guidance_for_category(
        self,
        category: EmergencyCategory,
        language: SupportedLanguage,
        *,
        context_text: str = "",
    ) -> ApprovedInstruction | None:
        """Select the most appropriate approved safety instruction for category, hazard context, and language."""
        lower = context_text.lower()
        if any(kw in lower for kw in ("cylinder", "lpg", "gas cylinder", "సిలిండర్", "सिलेंडर")):
            inst = self.get_by_id("safety_fire_gas_cylinder_01", language)
            if inst is not None:
                return inst
        mapping = {
            EmergencyCategory.FIRE: "safety_fire_01",
            EmergencyCategory.FIRE_WITH_INJURIES: "safety_fire_injuries_01",
            EmergencyCategory.ROAD_ACCIDENT: "safety_road_accident_01",
            EmergencyCategory.MEDICAL_EMERGENCY: "safety_medical_01",
            EmergencyCategory.BUILDING_COLLAPSE: "safety_collapse_01",
            EmergencyCategory.PHYSICAL_ATTACK: "safety_attack_01",
            EmergencyCategory.GAS_LEAK: "safety_gas_leak_01",
            EmergencyCategory.EXPLOSION: "safety_explosion_01",
            EmergencyCategory.UNKNOWN: "safety_unknown_01",
        }
        inst_id = mapping.get(category, "safety_unknown_01")
        return self.get_by_id(inst_id, language)

    def get_outage_reassurance_sequence(
        self,
        language: SupportedLanguage,
    ) -> tuple[ApprovedInstruction, ...]:
        r1 = self.get_by_id("outage_reassurance_1", language)
        r2 = self.get_by_id("outage_reassurance_2", language)
        assert r1 is not None and r2 is not None
        return (r1, r2)
