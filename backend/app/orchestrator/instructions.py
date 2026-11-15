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
        # =====================================================================
        # Seven-Language Expansion: Reviewed Kannada (kn), Tamil (ta),
        # Malayalam (ml), and Bengali (bn) Instructions (18 × 4 = 72 entries)
        # =====================================================================
        # 1. disclosure_session_start
        ApprovedInstruction(
            instruction_id="disclosure_session_start",
            category="system_disclosure",
            language="kn",
            approved_text=(
                "ಸೇತು (SETU) ತುರ್ತು ಸಮನ್ವಯಕ್ಕಾಗಿ ರೂಪಿಸಲಾದ ಮಾದರಿ (prototype) ಆಗಿದೆ. ಇದು ಭಾರತದ ಅಧಿಕೃತ 112 "
                "ಸೇವೆಗೆ ಪರ್ಯಾಯವಲ್ಲ ಮತ್ತು 112ಗೆ ಸಂಪರ್ಕಗೊಂಡಿಲ್ಲ. ಈ ಡೆಮೊದಲ್ಲಿನ ಎಲ್ಲಾ ಸಂಪನ್ಮೂಲಗಳು ಮತ್ತು ನಿಯೋಜನೆಗಳು "
                "ಮಾದರಿ ಮಾತ್ರ. ನಿಜವಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿಗಾಗಿ ದಯವಿಟ್ಟು 112ಗೆ ಕರೆ ಮಾಡಿ."
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="disclosure_session_start",
            category="system_disclosure",
            language="ta",
            approved_text=(
                "சேது (SETU) என்பது அவசரகால ஒருங்கிணைப்புக்கான ஒரு மாதிரி (prototype) ஆகும். இது இந்தியாவின் அதிகாரப்பூர்வ "
                "112 சேவைக்கு மாற்றல்ல மற்றும் 112 உடன் இணைக்கப்படவில்லை. இந்த டெமோவில் உள்ள அனைத்து வாகனங்களும் "
                "ஒதுக்கீடுகளும் உருவகப்படுத்தப்பட்டவை. உண்மையான அவசரநிலைக்கு உடனே 112 ஐ அழைக்கவும்."
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="disclosure_session_start",
            category="system_disclosure",
            language="ml",
            approved_text=(
                "സേതു (SETU) അടിയന്തര ഏകോപനത്തിനുള്ള ഒരു പ്രോട്ടോടൈപ്പ് ആണ്. ഇത് ഇന്ത്യയുടെ ഔദ്യോഗിക 112 "
                "സേവനത്തിന് പകരമല്ല, 112-മായി ബന്ധിപ്പിച്ചിട്ടുമില്ല. ഈ ഡെമോയിലെ എല്ലാ വിഭവങ്ങളും അസൈൻമെന്റുകളും "
                "സിമുലേറ്റഡ് മാത്രമാണ്. യഥാർത്ഥ അടിയന്തര സാഹചര്യത്തിന് 112-ൽ വിളിക്കുക."
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="disclosure_session_start",
            category="system_disclosure",
            language="bn",
            approved_text=(
                "সেতু (SETU) জরুরি সমন্বয়ের একটি প্রোটোটাইপ। এটি ভারতের সরকারি ১১২ জরুরি পরিষেবার বিকল্প নয় "
                "এবং ১১২-এর সাথে যুক্ত নয়। এই ডেমোতে সমস্ত রিসোর্স ও বরাদ্দ সিমুলেটেড। প্রকৃত জরুরি অবস্থায় অনুগ্রহ করে ১১২-এ কল করুন।"
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # 2. unknown_emergency_fallback
        ApprovedInstruction(
            instruction_id="unknown_emergency_fallback",
            category="system_fallback",
            language="kn",
            approved_text=(
                "ಸೇತುಗೆ ತುರ್ತು ಪರಿಸ್ಥಿತಿಯ ಪ್ರಕಾರವನ್ನು ಖಚಿತವಾಗಿ ಗುರುತಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ಈ ಡೆಮೊದಲ್ಲಿ மாದರಿ ಆಂಬ್ಯುಲೆನ್ಸ್ "
                "ನಿಯೋಜನೆಯನ್ನು ರಚಿಸಲಾಗಿದೆ ಮತ್ತು ಪರಿಶೀಲನೆಗೆ ಫ್ಲ್ಯಾಗ್ ಮಾಡಲಾಗಿದೆ. ಇದು ನಿಜವಾದ ರವಾನೆಯಲ್ಲ."
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="unknown_emergency_fallback",
            category="system_fallback",
            language="ta",
            approved_text=(
                "சேதுவால் அவசரநிலையின் வகையை உறுதியாக அடையாளம் காண முடியவில்லை. இந்த டெமோவில் மாதிரி ஆம்புலன்ஸ் "
                "ஒதுக்கீடு உருவாக்கப்பட்டு ஆய்வுக்குக் குறிக்கப்பட்டுள்ளது. இது உண்மையான அனுப்புதல் அல்ல."
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="unknown_emergency_fallback",
            category="system_fallback",
            language="ml",
            approved_text=(
                "സേതുവിന് അടിയന്തരാവസ്ഥയുടെ സ്വഭാവം കൃത്യമായി തിരിച്ചറിയാൻ കഴിഞ്ഞില്ല. ഈ ഡെമോയിൽ ഒരു സിമുലേറ്റഡ് "
                "ആംബുലൻസ് അസൈൻമെന്റ് സൃഷ്ടിക്കുകയും പരിശോധനയ്ക്കായി അടയാളപ്പെടുത്തുകയും ചെയ്തിട്ടുണ്ട്. ഇത് യഥാർത്ഥ ഡിസ്പാച്ച് അല്ല."
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="unknown_emergency_fallback",
            category="system_fallback",
            language="bn",
            approved_text=(
                "সেতু জরুরি অবস্থার ধরনটি নিশ্চিতভাবে শনাক্ত করতে পারেনি। এই ডেমোতে একটি সিমুলেটেড অ্যাম্বুলেন্স "
                "বরাদ্দ তৈরি করা হয়েছে এবং পর্যালোচনার জন্য চিহ্নিত করা হয়েছে। এটি প্রকৃত ডিসপ্যাচ নয়।"
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # 3. provider_blocked_reask
        ApprovedInstruction(
            instruction_id="provider_blocked_reask",
            category="system_fallback",
            language="kn",
            approved_text="ನಾನು ಈ ಸಿಮ్యుಲೇಷನ್‌ನಲ್ಲಿ ನಿಮ್ಮೊಂದಿಗಿದ್ದೇನೆ. ದಯವಿಟ್ಟು ಅಲ್ಲಿ ಏನಾಗುತ್ತಿದೆ ಎಂದು ನಿಮ್ಮ ಮಾತುಗಳಲ್ಲಿ ಮತ್ತೊಮ್ಮೆ ಹೇಳುವಿರಾ?",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="provider_blocked_reask",
            category="system_fallback",
            language="ta",
            approved_text="நான் உங்களுடன் இணைப்பில் இருக்கிறேன். தயவுசெய்து அங்கு என்ன நடக்கிறது என்பதை உங்கள் வார்த்தைகளில் மீண்டும் கூறுவீர்களா?",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="provider_blocked_reask",
            category="system_fallback",
            language="ml",
            approved_text="ഞാൻ ഈ സിമുലേഷനിൽ നിങ്ങളോടൊപ്പമുണ്ട്. ദയവായി അവിടെ എന്താണ് സംഭവിക്കുന്നതെന്ന് നിങ്ങളുടെ വാക്കുകളിൽ ഒരിക്കൽ കൂടി പറയാമോ?",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="provider_blocked_reask",
            category="system_fallback",
            language="bn",
            approved_text="আমি এই সিমুলেশনে আপনার সাথেই আছি। অনুগ্রহ করে পরিস্থিতিটি আপনার নিজের ভাষায় আরেকবার বলবেন?",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # 4. provider_blocked_fallback
        ApprovedInstruction(
            instruction_id="provider_blocked_fallback",
            category="system_fallback",
            language="kn",
            approved_text=(
                "ನೀವು ವಿವರಿಸಿದ ಮಾಹಿತಿಯನ್ನು ಪ್ರಕ್ರಿಯೆಗೊಳಿಸುವಲ್ಲಿ ಸೇತುಗೆ ತೊಂದರೆಯಾಗಿದೆ. ಈ ಡೆಮೊದಲ್ಲಿ ಮಾದರಿ ಆಂಬ್ಯುಲೆನ್ಸ್ "
                "ನಿಯೋಜನೆಯನ್ನು ರಚಿಸಲಾಗಿದೆ ಮತ್ತು ಪರಿಶೀಲನೆಗೆ ಫ್ಲ್ಯಾಗ್ ಮಾಡಲಾಗಿದೆ. ಇದು ನಿಜವಾದ ರವಾನೆಯಲ್ಲ."
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="provider_blocked_fallback",
            category="system_fallback",
            language="ta",
            approved_text=(
                "நீங்கள் விவரித்த தகவலைச் செயல்படுத்துவதில் சேதுவுக்குச் சிரமம் ஏற்பட்டுள்ளது. இந்த டெமோவில் மாதிரி ஆம்புலன்ஸ் "
                "ஒதுக்கீடு உருவாக்கப்பட்டு ஆய்வுக்குக் குறிக்கப்பட்டுள்ளது. இது உண்மையான அனுப்புதல் அல்ல."
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="provider_blocked_fallback",
            category="system_fallback",
            language="ml",
            approved_text=(
                "നിങ്ങൾ പറഞ്ഞ വിവരങ്ങൾ പ്രോസസ്സ് ചെയ്യുന്നതിൽ സേതുവിന് തടസ്സം നേരിട്ടു. ഈ ഡെമോയിൽ ഒരു സിമുലേറ്റഡ് ആംബുലൻസ് "
                "അസൈൻമെന്റ് സൃഷ്ടിക്കുകയും പരിശോധനയ്ക്കായി അടയാളപ്പെടുത്തുകയും ചെയ്തിട്ടുണ്ട്. ഇത് യഥാർത്ഥ ഡിസ്പാച്ച് അല്ല."
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="provider_blocked_fallback",
            category="system_fallback",
            language="bn",
            approved_text=(
                "আপনার বর্ণিত তথ্য প্রক্রিয়া করতে সেতুর সমস্যা হচ্ছে। এই ডেমোতে একটি সিমুলেটেড অ্যাম্বুলেন্স বরাদ্দ তৈরি করা হয়েছে "
                "এবং পর্যালোচনার জন্য চিহ্নিত করা হয়েছে। এটি প্রকৃত ডিসপ্যাচ নয়।"
            ),
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # 5. didnt_catch_that
        ApprovedInstruction(
            instruction_id="didnt_catch_that",
            category="system_prompt",
            language="kn",
            approved_text="ಕ್ಷಮಿಸಿ, ಅದು ನನಗೆ ಸ್ಪಷ್ಟವಾಗಿ ಕೇಳಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಹೇಳಿ.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="didnt_catch_that",
            category="system_prompt",
            language="ta",
            approved_text="மன்னிக்கவும், எனக்குத் தெளிவாகக் கேட்கவில்லை. தயவுசெய்து மீண்டும் கூறுங்கள்.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="didnt_catch_that",
            category="system_prompt",
            language="ml",
            approved_text="ക്ഷമിക്കണം, എനിക്ക് അത് വ്യക്തമായി കേട്ടില്ല. ദയവായി ഒരിക്കൽ കൂടി പറയൂ.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="didnt_catch_that",
            category="system_prompt",
            language="bn",
            approved_text="দুঃখিত, আমি স্পষ্ট বুঝতে পারিনি। অনুগ্রহ করে আরেকবার বলুন।",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # 6 & 7. outage_reassurance_1 & outage_reassurance_2
        ApprovedInstruction(
            instruction_id="outage_reassurance_1",
            category="outage_reassurance",
            language="kn",
            approved_text="ನಾನು ನಿಮ್ಮೊಂದಿಗಿದ್ದೇನೆ. ಈ ಸಿಮ್ಯುಲೇಷನ್ ಮುಂದುವರಿಯುತ್ತಿರುವಾಗ ದಯವಿಟ್ಟು ಲೈನ್‌ನಲ್ಲಿ ಇರಿ.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="outage_reassurance_2",
            category="outage_reassurance",
            language="kn",
            approved_text="ಇದು ಮಾದರಿ ಮಾತ್ರ ಮತ್ತು ನಿಜವಾದ ತುರ್ತು ಸೇವೆಗಳಿಗೆ ಸಂಪರ್ಕಗೊಂಡಿಲ್ಲ. ನಿಜವಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿಗಾಗಿ 112ಗೆ ಕರೆ ಮಾಡಿ.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="outage_reassurance_1",
            category="outage_reassurance",
            language="ta",
            approved_text="நான் உங்களுடன் இருக்கிறேன். இந்த உருவகப்படுத்துதல் தொடரும்போது தயவுசெய்து இணைப்பில் இருங்கள்.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="outage_reassurance_2",
            category="outage_reassurance",
            language="ta",
            approved_text="இது ஒரு மாதிரி மட்டுமே, உண்மையான அவசர சேவைகளுடன் இணைக்கப்படவில்லை. உண்மையான அவசரநிலைக்கு 112 ஐ அழைக்கவும்.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="outage_reassurance_1",
            category="outage_reassurance",
            language="ml",
            approved_text="ഞാൻ നിങ്ങളോടൊപ്പമുണ്ട്. ഈ സിമുലേഷൻ തുടരുമ്പോൾ ദയവായി ലൈനിൽ തുടരുക.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="outage_reassurance_2",
            category="outage_reassurance",
            language="ml",
            approved_text="ഇത് ഒരു പ്രോട്ടോടൈപ്പ് മാത്രമാണ്, യഥാർത്ഥ അടിയന്തര സേവനങ്ങളുമായി ബന്ധിപ്പിച്ചിട്ടില്ല. യഥാർത്ഥ അടിയന്തരാവസ്ഥയ്ക്ക് 112-ൽ വിളിക്കുക.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="outage_reassurance_1",
            category="outage_reassurance",
            language="bn",
            approved_text="আমি আপনার সাথেই আছি। এই সিমুলেশন চলাকালীন অনুগ্রহ করে লাইনে থাকুন।",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="outage_reassurance_2",
            category="outage_reassurance",
            language="bn",
            approved_text="এটি একটি প্রোটোটাইপ এবং প্রকৃত জরুরি পরিষেবার সাথে যুক্ত নয়। প্রকৃত জরুরি অবস্থায় ১১২-এ কল করুন।",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # 8. holding_checking_resources
        ApprovedInstruction(
            instruction_id="holding_checking_resources",
            category="holding_phrase",
            language="kn",
            approved_text="ಡೈರೆಕ್ಟರಿಯಲ್ಲಿ ಯಾವ ಮಾದರಿ ಸಂಪನ್ಮೂಲಗಳು ಮತ್ತು ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆಗಳು ಲಭ್ಯವಿವೆ ಎಂದು ನಾನು ಪರಿಶೀಲಿಸುತ್ತಿದ್ದೇನೆ.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="holding_checking_resources",
            category="holding_phrase",
            language="ta",
            approved_text="டைரக்டரியில் உள்ள மாதிரி வாகனங்கள் மற்றும் அருகிலுள்ள மருத்துவமனைகளின் இருப்பை நான் சரிபார்க்கிறேன்.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="holding_checking_resources",
            category="holding_phrase",
            language="ml",
            approved_text="ഡയറക്ടറിയിൽ ലഭ്യമായ സിമുലേറ്റഡ് വിഭവങ്ങളും അടുത്തുള്ള ആശുപത്രികളും ഞാൻ പരിശോധിക്കുകയാണ്.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="holding_checking_resources",
            category="holding_phrase",
            language="bn",
            approved_text="ডিরেক্টরিতে কোন সিমুলেটেড রিসোর্স এবং কাছের হাসপাতাল উপলব্ধ আছে তা আমি পরীক্ষা করছি।",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # 9. safety_road_accident_01
        ApprovedInstruction(
            instruction_id="safety_road_accident_01",
            category=EmergencyCategory.ROAD_ACCIDENT,
            language="kn",
            approved_text="ದಯವಿಟ್ಟು ಚಲಿಸುವ ವಾಹನಗಳಿಂದ ಸುರಕ್ಷಿತ ದೂರದಲ್ಲಿ ಇರಿ ಮತ್ತು ರಸ್ತೆಯ ಮಧ್ಯದಲ್ಲಿ ನಿಲ್ಲಬೇಡಿ.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_road_accident_01",
            category=EmergencyCategory.ROAD_ACCIDENT,
            language="ta",
            approved_text="தயவுசெய்து செல்லும் வாகனங்களிலிருந்து பாதுகாப்பான தூரத்தில் இருங்கள், சாலையின் நடுவே நிற்க வேண்டாம்.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_road_accident_01",
            category=EmergencyCategory.ROAD_ACCIDENT,
            language="ml",
            approved_text="ദയവായി ഓടുന്ന വാഹനങ്ങളിൽ നിന്ന് സുരക്ഷിതമായ അകലം പാലിക്കുക, റോഡിന് നടുവിൽ നിൽക്കരുത്.",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_road_accident_01",
            category=EmergencyCategory.ROAD_ACCIDENT,
            language="bn",
            approved_text="অনুগ্রহ করে চলমান যানবাহন থেকে নিরাপদ দূরত্বে থাকুন এবং মূল রাস্তার ওপর দাঁড়াবেন না।",
            source=src,
            review_date=rev_date,
            version="v1.0",
        ),
        # 10. safety_fire_01
        ApprovedInstruction(
            instruction_id="safety_fire_01",
            category=EmergencyCategory.FIRE,
            language="kn",
            approved_text="ದಯವಿಟ್ಟು ಹೊಗೆ ಮತ್ತು ಬೆಂಕಿಯಿಂದ ತಕ್ಷಣ ಸುರಕ್ಷಿತ ದೂರಕ್ಕೆ ಸರಿಯಿರಿ, ಲಿಫ್ಟ್ ಬಳಸಬೇಡಿ ಮತ್ತು ಕಟ್ಟಡದ ಒಳಗೆ ಮತ್ತೆ ಹೋಗಬೇಡಿ. ನಿಜವಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ 112ಗೆ ಕರೆ ಮಾಡಿ.",
            source="SETU Fire Safety Guidance (NDMA / Fire Services Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_01",
            category=EmergencyCategory.FIRE,
            language="ta",
            approved_text="தயவுசெய்து புகை மற்றும் தீயிலிருந்து உடனடியாக விலகிப் பாதுகாப்பான இடத்திற்குச் செல்லுங்கள், லிஃப்ட் பயன்படுத்த வேண்டாம், கட்டிடத்திற்குள் மீண்டும் செல்ல வேண்டாம். உண்மையான அவசரநிலைக்கு 112 ஐ அழைக்கவும்.",
            source="SETU Fire Safety Guidance (NDMA / Fire Services Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_01",
            category=EmergencyCategory.FIRE,
            language="ml",
            approved_text="ദയവായി പുകയിൽ നിന്നും തീയിൽ നിന്നും ഉടൻ സുരക്ഷിതമായ അകലത്തിലേക്ക് മാറുക, ലിഫ്റ്റ് ഉപയോഗിക്കരുത്, കെട്ടിടത്തിനുള്ളിലേക്ക് തിരികെ കയറരുത്. യഥാർത്ഥ അടിയന്തരാവസ്ഥയ്ക്ക് 112-ൽ വിളിക്കുക.",
            source="SETU Fire Safety Guidance (NDMA / Fire Services Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_01",
            category=EmergencyCategory.FIRE,
            language="bn",
            approved_text="অনুগ্রহ করে ধোঁয়া ও আগুন থেকে অবিলম্বে নিরাপদ দূরত্বে সরে যান, লিফট ব্যবহার করবেন না এবং ভবনের ভেতরে পুনরায় প্রবেশ করবেন না। প্রকৃত জরুরি অবস্থায় ১১২-এ কল করুন।",
            source="SETU Fire Safety Guidance (NDMA / Fire Services Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        # 11. safety_fire_gas_cylinder_01
        ApprovedInstruction(
            instruction_id="safety_fire_gas_cylinder_01",
            category=EmergencyCategory.FIRE,
            language="kn",
            approved_text="ದಯವಿಟ್ಟು ತಕ್ಷಣ ತೆರೆದ ಸುರಕ್ಷಿತ ಪ್ರದೇಶಕ್ಕೆ ಸರಿಯಿರಿ. ಉರಿಯುತ್ತಿರುವ ಅಥವಾ ಸೋರಿಕೆಯಾಗುತ್ತಿರುವ ಗ್ಯಾಸ್ ಸಿಲಿಂಡರ್ ಹತ್ತಿರ ಹೋಗಬೇಡಿ ಅಥವಾ ಅದನ್ನು ಮುಟ್ಟಬೇಡಿ. ನಿಜವಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ 112ಗೆ ಕರೆ ಮಾಡಿ.",
            source="SETU LPG / Gas Cylinder Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_gas_cylinder_01",
            category=EmergencyCategory.FIRE,
            language="ta",
            approved_text="தயவுசெய்து உடனடியாகத் திறந்தவெளிப் பாதுகாப்பான இடத்திற்குச் செல்லுங்கள். எரியும் அல்லது கசியும் எரிவாயு சிலிண்டர் அருகில் செல்லவோ தொடவோ வேண்டாம். உண்மையான அவசரநிலைக்கு 112 ஐ அழைக்கவும்.",
            source="SETU LPG / Gas Cylinder Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_gas_cylinder_01",
            category=EmergencyCategory.FIRE,
            language="ml",
            approved_text="ദയവായി ഉടൻ തന്നെ തുറസ്സായ സുരക്ഷിത സ്ഥലത്തേക്ക് മാറുക. കത്തുന്നതോ ചോരുന്നതോ ആയ ഗ്യാസ് സിലിണ്ടറിന് അടുത്തേക്ക് പോകുകയോ തൊടുകയോ ചെയ്യരുത്. യഥാർത്ഥ അടിയന്തരാവസ്ഥയ്ക്ക് 112-ൽ വിളിക്കുക.",
            source="SETU LPG / Gas Cylinder Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_gas_cylinder_01",
            category=EmergencyCategory.FIRE,
            language="bn",
            approved_text="অনুগ্রহ করে অবিলম্বে খোলা নিরাপদ স্থানে সরে যান। জ্বলন্ত বা লিক হওয়া গ্যাস সিলিন্ডারের কাছে যাবেন না বা স্পর্শ করবেন না। প্রকৃত জরুরি অবস্থায় ১১২-এ কল করুন।",
            source="SETU LPG / Gas Cylinder Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        # 12. safety_fire_injuries_01
        ApprovedInstruction(
            instruction_id="safety_fire_injuries_01",
            category=EmergencyCategory.FIRE_WITH_INJURIES,
            language="kn",
            approved_text="ನೀವು ಸುರಕ್ಷಿತವಾಗಿ ಮಾಡಬಹುದಾದರೆ ಮಾತ್ರ ಗಾಯಾಳುಗಳನ್ನು ಹೊಗೆ ಮತ್ತು ಶಾಖದಿಂದ ದೂರ ತಾಜಾ ಗಾಳಿಗೆ ಕರೆತನ್ನಿ, ಬೆಂಕಿಯ ಒಳಗೆ ಹೋಗಬೇಡಿ. ನಿಜವಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿಗಾಗಿ 112ಗೆ ಕರೆ ಮಾಡಿ.",
            source="SETU Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_injuries_01",
            category=EmergencyCategory.FIRE_WITH_INJURIES,
            language="ta",
            approved_text="பாதுகாப்பாகச் செய்ய முடிந்தால் மட்டுமே காயமடைந்தவர்களைப் புகை மற்றும் வெப்பத்திலிருந்து விலக்கித் திறந்தவெளிக்குக் கொண்டு வாருங்கள், தீக்குள் செல்ல வேண்டாம். உண்மையான அவசரநிலைக்கு 112 ஐ அழைக்கவும்.",
            source="SETU Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_injuries_01",
            category=EmergencyCategory.FIRE_WITH_INJURIES,
            language="ml",
            approved_text="സുരക്ഷിതമായി ചെയ്യാൻ കഴിയുമെങ്കിൽ മാത്രം പരിക്കേറ്റവരെ പുകയിൽ നിന്നും ചൂടിൽ നിന്നും ശുദ്ധവായുവിലേക്ക് മാറ്റുക, തീയിലേക്ക് കയറരുത്. യഥാർത്ഥ അടിയന്തരാവസ്ഥയ്ക്ക് 112-ൽ വിളിക്കുക.",
            source="SETU Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_fire_injuries_01",
            category=EmergencyCategory.FIRE_WITH_INJURIES,
            language="bn",
            approved_text="যদি নিরাপদে সম্ভব হয় তবেই আহতদের ধোঁয়া ও তাপ থেকে দূরে খোলা বাতাসে সরিয়ে আনুন, আগুনের ভেতরে যাবেন না। প্রকৃত জরুরি অবস্থায় ১১২-এ কল করুন।",
            source="SETU Fire Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        # 13. safety_medical_01
        ApprovedInstruction(
            instruction_id="safety_medical_01",
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            language="kn",
            approved_text="ಸುರಕ್ಷಿತವಾಗಿದ್ದರೆ ವ್ಯಕ್ತಿಯ ಹತ್ತಿರವೇ ಇರಿ, ಪ್ರವೇಶ ಮಾರ್ಗವನ್ನು ಮುಕ್ತವಾಗಿಡಿ. ನಿಜವಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿಗಾಗಿ ತಕ್ಷಣ 112ಗೆ ಕರೆ ಮಾಡಿ.",
            source="SETU Emergency Intake Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_medical_01",
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            language="ta",
            approved_text="பாதுகாப்பாக இருந்தால் பாதிக்கப்பட்டவரின் அருகிலேயே இருங்கள், நுழைவு வழியைத் தடையின்றி வையுங்கள். உண்மையான அவசரநிலைக்கு உடனே 112 ஐ அழைக்கவும்.",
            source="SETU Emergency Intake Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_medical_01",
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            language="ml",
            approved_text="സുരക്ഷിതമാണെങ്കിൽ രോഗിയുടെ അടുത്തുതന്നെ തുടരുക, പ്രവേശന പാത തടസ്സമില്ലാതെ സൂക്ഷിക്കുക. യഥാർത്ഥ അടിയന്തരാവസ്ഥയ്ക്ക് ഉടൻ 112-ൽ വിളിക്കുക.",
            source="SETU Emergency Intake Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_medical_01",
            category=EmergencyCategory.MEDICAL_EMERGENCY,
            language="bn",
            approved_text="নিরাপদ হলে অসুস্থ ব্যক্তির কাছেই থাকুন এবং প্রবেশপথ খোলা রাখুন। প্রকৃত জরুরি অবস্থায় অবিলম্বে ১১২-এ কল করুন।",
            source="SETU Emergency Intake Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        # 14. safety_collapse_01
        ApprovedInstruction(
            instruction_id="safety_collapse_01",
            category=EmergencyCategory.BUILDING_COLLAPSE,
            language="kn",
            approved_text="ದುರ್ಬಲ ಗೋಡೆಗಳು, ಅವಶೇಷಗಳು ಮತ್ತು ವಿದ್ಯುತ್ ತಂತಿಗಳಿಂದ ದೂರವಿರಿ, ಹಾನಿಗೊಳಗಾದ ಕಟ್ಟಡದ ಒಳಗೆ ಹೋಗಬೇಡಿ.",
            source="SETU Structural Hazard Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_collapse_01",
            category=EmergencyCategory.BUILDING_COLLAPSE,
            language="ta",
            approved_text="பலவீனமான சுவர்கள், இடிபாடுகள் மற்றும் மின் கம்பிகளிலிருந்து விலகி இருங்கள், சேதமடைந்த கட்டிடத்திற்குள் செல்ல வேண்டாம்.",
            source="SETU Structural Hazard Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_collapse_01",
            category=EmergencyCategory.BUILDING_COLLAPSE,
            language="ml",
            approved_text="ബലക്ഷയമുള്ള മതിലുകൾ, അവശിഷ്ടങ്ങൾ, വൈദ്യുതി ലൈനുകൾ എന്നിവയിൽ നിന്ന് മാറിനിൽക്കുക, തകർന്ന കെട്ടിടത്തിനുള്ളിൽ പ്രവേശിക്കരുത്.",
            source="SETU Structural Hazard Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_collapse_01",
            category=EmergencyCategory.BUILDING_COLLAPSE,
            language="bn",
            approved_text="দুর্বল দেয়াল, ধ্বংসস্তূপ ও বৈদ্যুতিক তার থেকে দূরে থাকুন এবং ক্ষতিগ্রস্ত ভবনের ভেতরে প্রবেশ করবেন না।",
            source="SETU Structural Hazard Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        # 15. safety_attack_01
        ApprovedInstruction(
            instruction_id="safety_attack_01",
            category=EmergencyCategory.PHYSICAL_ATTACK,
            language="kn",
            approved_text="ನಿಮ್ಮ ವೈಯಕ್ತಿಕ ಸುರಕ್ಷತೆಗೆ ಮೊದಲ ಆದ್ಯತೆ ನೀಡಿ: ಸಾಧ್ಯವಾದರೆ ಅಪಾಯದಿಂದ ದೂರ ಸುರಕ್ಷಿತ ಸ್ಥಳಕ್ಕೆ ತೆರಳಿ.",
            source="SETU Personal Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_attack_01",
            category=EmergencyCategory.PHYSICAL_ATTACK,
            language="ta",
            approved_text="உங்கள் தனிப்பட்ட பாதுகாப்பிற்கு முன்னுரிமை அளியுங்கள்: முடிந்தால் ஆபத்திலிருந்து விலகிப் பாதுகாப்பான இடத்திற்குச் செல்லுங்கள்.",
            source="SETU Personal Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_attack_01",
            category=EmergencyCategory.PHYSICAL_ATTACK,
            language="ml",
            approved_text="നിങ്ങളുടെ വ്യക്തിഗത സുരക്ഷയ്ക്ക് മുൻഗണന നൽകുക: സുരക്ഷിതമായി സാധിക്കുമെങ്കിൽ അപകടത്തിൽ നിന്ന് മാറി സുരക്ഷിത സ്ഥാനത്തേക്ക് മാറുക.",
            source="SETU Personal Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_attack_01",
            category=EmergencyCategory.PHYSICAL_ATTACK,
            language="bn",
            approved_text="আপনার ব্যক্তিগত নিরাপত্তাকে সর্বোচ্চ অগ্রাধিকার দিন: সম্ভব হলে বিপদ থেকে দূরে নিরাপদ স্থানে সরে যান।",
            source="SETU Personal Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        # 16. safety_gas_leak_01
        ApprovedInstruction(
            instruction_id="safety_gas_leak_01",
            category=EmergencyCategory.GAS_LEAK,
            language="kn",
            approved_text="ತಕ್ಷಣ ಹೊರಗೆ ತಾಜಾ ಗಾಳಿಯಿರುವ ಸುರಕ್ಷಿತ ಸ್ಥಳಕ್ಕೆ ಹೋಗಿ. ವಿದ್ಯುತ್ ಸ್ವಿಚ್‌ಗಳು ಅಥವಾ ಬೆಂಕಿಕಡ್ಡಿ ಬಳಸಬೇಡಿ ಮತ್ತು ಸೋರಿಕೆಯಾಗುತ್ತಿರುವ ಸಿಲಿಂಡರ್ ಹತ್ತಿರ ಹೋಗಬೇಡಿ. ನಿಜವಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿಗೆ 112ಗೆ ಕರೆ ಮಾಡಿ.",
            source="SETU Gas Leak Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_gas_leak_01",
            category=EmergencyCategory.GAS_LEAK,
            language="ta",
            approved_text="உடனடியாக வெளியே திறந்தவெளிக்குச் செல்லுங்கள். மின் சுவிட்சுகள் அல்லது தீக்குச்சிகளைப் பயன்படுத்த வேண்டாம், கசியும் சிலிண்டர் அருகில் செல்ல வேண்டாம். உண்மையான அவசரநிலைக்கு 112 ஐ அழைக்கவும்.",
            source="SETU Gas Leak Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_gas_leak_01",
            category=EmergencyCategory.GAS_LEAK,
            language="ml",
            approved_text="ഉടൻ പുറത്ത് ശുദ്ധവായുവുള്ള സുരക്ഷിത സ്ഥലത്തേക്ക് മാറുക. വൈദ്യുത സ്വിച്ചുകളോ തീപ്പെട്ടിയോ ഉപയോഗിക്കരുത്, ചോരുന്ന സിലിണ്ടറിന് അടുത്തേക്ക് പോകരുത്. യഥാർത്ഥ അടിയന്തരാവസ്ഥയ്ക്ക് 112-ൽ വിളിക്കുക.",
            source="SETU Gas Leak Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_gas_leak_01",
            category=EmergencyCategory.GAS_LEAK,
            language="bn",
            approved_text="অবিলম্বে বাইরে খোলা বাতাসে নিরাপদ দূরত্বে চলে যান। বৈদ্যুতিক সুইচ বা দেশলাই স্পর্শ করবেন না এবং লিক হওয়া সিলিন্ডারের কাছে যাবেন না। প্রকৃত জরুরি অবস্থায় ১১২-এ কল করুন।",
            source="SETU Gas Leak Safety Guidance (NDMA Reviewed)",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        # 17. safety_explosion_01
        ApprovedInstruction(
            instruction_id="safety_explosion_01",
            category=EmergencyCategory.EXPLOSION,
            language="kn",
            approved_text="ಹಾನಿಗೊಳಗಾದ ಕಟ್ಟಡಗಳು, ಗಾಜಿನ ಚೂರುಗಳು ಮತ್ತು ಹೊಗೆಯಿಂದ ದೂರ ತೆರೆದ ಸುರಕ್ಷಿತ ಪ್ರದೇಶದಲ್ಲಿ ಇರಿ.",
            source="SETU Hazard Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_explosion_01",
            category=EmergencyCategory.EXPLOSION,
            language="ta",
            approved_text="சேதமடைந்த கட்டிடங்கள், கண்ணாடித் துண்டுகள் மற்றும் புகையிலிருந்து விலகித் திறந்தவெளிப் பாதுகாப்பான இடத்தில் இருங்கள்.",
            source="SETU Hazard Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_explosion_01",
            category=EmergencyCategory.EXPLOSION,
            language="ml",
            approved_text="തകർന്ന കെട്ടിടങ്ങൾ, ചില്ലുകൾ, പുക എന്നിവയിൽ നിന്ന് അകന്ന് തുറസ്സായ സുരക്ഷിത സ്ഥലത്ത് തുടരുക.",
            source="SETU Hazard Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_explosion_01",
            category=EmergencyCategory.EXPLOSION,
            language="bn",
            approved_text="ক্ষতিগ্রস্ত ভবন, ভাঙা কাঁচ ও ধোঁয়া থেকে দূরে খোলা নিরাপদ স্থানে সরে যান।",
            source="SETU Hazard Safety Guidance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        # 18. safety_unknown_01
        ApprovedInstruction(
            instruction_id="safety_unknown_01",
            category=EmergencyCategory.UNKNOWN,
            language="kn",
            approved_text="ದಯವಿಟ್ಟು ಸುರಕ್ಷಿತ ಸ್ಥಳದಲ್ಲಿ ಇರಿ ಮತ್ತು ನಾನು ವಿವರಗಳನ್ನು ದಾಖಲಿಸುವಾಗ ಲೈನ್‌ನಲ್ಲಿ ಮುಂದುವರಿಯಿರಿ.",
            source="SETU Intake Reassurance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_unknown_01",
            category=EmergencyCategory.UNKNOWN,
            language="ta",
            approved_text="தயவுசெய்து பாதுகாப்பான இடத்தில் இருங்கள், நான் விவரங்களைப் பதிவு செய்யும்போது இணைப்பில் தொடருங்கள்.",
            source="SETU Intake Reassurance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_unknown_01",
            category=EmergencyCategory.UNKNOWN,
            language="ml",
            approved_text="ദയവായി സുരക്ഷിതമായ ഒരിടത്ത് തുടരുക, ഞാൻ വിവരങ്ങൾ രേഖപ്പെടുത്തുമ്പോൾ ലൈനിൽ തുടരുക.",
            source="SETU Intake Reassurance",
            review_date=date(2026, 11, 10),
            version="v1.0",
        ),
        ApprovedInstruction(
            instruction_id="safety_unknown_01",
            category=EmergencyCategory.UNKNOWN,
            language="bn",
            approved_text="অনুগ্রহ করে একটি নিরাপদ স্থানে থাকুন এবং আমি তথ্য নথিভুক্ত করার সময় লাইনে থাকুন।",
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
            "kn": "timbre-v2.5-kn-in-std",
            "ta": "timbre-v2.5-ta-in-std",
            "ml": "timbre-v2.5-ml-in-std",
            "bn": "timbre-v2.5-bn-in-std",
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
        if any(
            kw in lower
            for kw in (
                "cylinder",
                "lpg",
                "gas cylinder",
                "సిలిండర్",
                "सिलेंडर",
                "ಸಿಲಿಂಡರ್",
                "சிலிண்டர்",
                "സിലിണ്ടർ",
                "সিলিন্ডার",
            )
        ):
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
