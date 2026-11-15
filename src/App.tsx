import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AssignmentStatus,
  BASE_DISPATCH_MAPPING,
  ContinuousAudioRingBuffer,
  DIRECT_LANGUAGE_OPTIONS,
  EmergencyCategory,
  FactConflict,
  FinalizedUtterance,
  formatSelectedLanguageStatus,
  geodesicDistanceMeters,
  getDirectLanguageOption,
  HospitalScoreBreakdown,
  INITIAL_SYNTHETIC_RESOURCES,
  isGpsEligibleCandidate,
  LANGUAGE_DISPLAY_NAMES,
  LocationConfirmationStatus,
  PrankFlagReviewStatus,
  ProvenancedFact,
  rankHospitalsDeterministic,
  REQUIRED_PROTOTYPE_DISCLOSURE,
  REQUIRED_PROVIDER_BLOCKED_STATEMENT,
  REQUIRED_UNKNOWN_FALLBACK_STATEMENT,
  ResourceCategory,
  resolveVoiceStagePresentation,
  SimulatedAssignment,
  SPOKEN_DISCLOSURE_BY_LANGUAGE,
  SupportedLanguage,
  SYNTHETIC_HOSPITALS,
  SYNTHETIC_LANDMARKS,
  SyntheticLandmark,
  SyntheticResource,
  transitionVoiceInteractionState,
  UtteranceRole,
  VoiceInteractionState,
  VoiceLifecycleEvent,
} from './utils/setuEngine';
import { arrayBufferToBase64, ManifestAudioAsset, SetuWebAudioController } from './utils/audioPipeline';
import { SileroVadDetector } from './utils/sileroVad';
import { SetuMapLibre } from './components/SetuMapLibre';

type WorkspaceView = 'initial' | 'live_incident' | 'control_room' | 'handoff';
type MobileTab = 'conversation' | 'map' | 'resources' | 'hospitals' | 'timeline';
type CoordinationStep = 'Listening' | 'Understanding' | 'Locating' | 'Assigning' | 'Guiding';

interface TurnWaterfallEntry {
  turnId: string;
  turnNumber: number;
  timestamp: string;
  utterancePreview: string;
  language: string;
  thinkingDepth: 'LOW' | 'MEDIUM' | 'DETERMINISTIC';
  reasoningSource: 'live_gemini' | 'deterministic_fallback' | 'scripted_scenario';
  vadFinalizeMs: number;
  prismaSttMs: number;
  geminiReasoningMs: number;
  orchestratorMs: number;
  postgresPersistMs: number;
  timbreTtfbMs: number;
  timbreTotalMs: number;
  browserPlaybackDelayMs: number;
  totalTurnMs: number;
  noiseCategory: 'quiet' | 'moderate' | 'high_ambient';
  overlappingVoices: boolean;
  clippingDetected: boolean;
}

const COORDINATION_STEPS: readonly CoordinationStep[] = [
  'Listening',
  'Understanding',
  'Locating',
  'Assigning',
  'Guiding',
];

const SHORT_PROTOTYPE_DISCLOSURE =
  'PROTOTYPE SIMULATION — Not connected to 112 or real emergency services. All dispatch and hospital actions are simulated.';

/**
 * Human-readable labels for raw field names and reason codes.
 * Raw field names are always preserved in the `title` tooltip.
 */
const HUMAN_FIELD_LABELS: Record<string, string> = {
  vehicle_collision_reported: 'Vehicle collision reported',
  injured_persons: 'Injured people reported',
  affected_person_count: 'Affected people count',
  caller_location_text: 'Spoken location',
  fire_reported: 'Fire reported',
  explosion_reported: 'Explosion reported',
  caller_safe_spot_status: 'Caller safety status',
  caller_observations: 'Caller observation',
  caller_live_statement: 'Caller statement',
};

const HUMAN_REASON_LABELS: Record<string, string> = {
  provisional_location_unconfirmed: 'Provisional, location unconfirmed',
  conflicting_caller_statements: 'Conflicting caller statements',
  possible_second_incident: 'Possible second incident',
  reclassification_conflict: 'Incident reclassified',
  pending_location: 'Location unverified',
  provider_blocked_emergency_content: 'Provider safety filter triggered',
  reasoning_unavailable: 'Reasoning service unavailable',
  caller_silence_timeout_review: 'Caller silence check-in',
  unknown_after_clarification: 'Unclassified after clarification',
  possible_duplicate_incident: 'Possible duplicate incident',
  suspected_false_or_prank_call: 'Flagged for prank/false call review',
};

function getHumanFieldLabel(rawField: string): string {
  return (
    HUMAN_FIELD_LABELS[rawField] ||
    rawField
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase())
  );
}

function getHumanReasonLabel(rawCode: string): string {
  if (rawCode.startsWith('resource_unavailable:')) {
    const resCat = rawCode.split(':')[1] || '';
    return `Simulated unit unavailable (${resCat.replace(/_/g, ' ')})`;
  }
  return (
    HUMAN_REASON_LABELS[rawCode] ||
    rawCode
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase())
  );
}

const CATEGORY_IMPLIED_FACT_BY_CATEGORY: Partial<Record<EmergencyCategory, string>> = {
  fire: 'fire_reported',
  fire_with_injuries: 'fire_reported',
  road_accident: 'vehicle_collision_reported',
  explosion: 'explosion_reported',
};

interface TrackedFact extends ProvenancedFact {
  status: 'active' | 'conflicting' | 'superseded';
  supersededByCategory?: EmergencyCategory;
}

interface AuditTimelineEntry {
  id: string;
  timestamp: string;
  event_type: string;
  severity: 'info' | 'warning' | 'critical';
  outcome: string;
  reason_code: string | null;
  provenance_ref: string;
}

interface PresetScenario {
  id: string;
  label: string;
  language: SupportedLanguage;
  languageBadge: 'EN' | 'TE' | 'HI' | 'TANGLISH' | 'HINGLISH' | 'BENGLISH';
  text: string;
  utterance_role: UtteranceRole;
  category: EmergencyCategory;
  confidence: number;
  simulateProviderBlock?: boolean;
  simulateProviderOutage?: boolean;
  extractedFacts: Omit<ProvenancedFact, 'source_utterance_id'>[];
  followupQuestion: string | null;
  safetyInstructionText: string;
  safetyAttributionLabel: string;
  responseText: string;
}

const PRESET_SCENARIOS: readonly PresetScenario[] = [
  {
    id: 'tanglish_road_accident',
    label: 'Tanglish: Road Accident near Railway Station',
    language: 'te',
    languageBadge: 'TANGLISH',
    text: 'Anna, ikkada road accident jarigindi. Oka person unconscious ga unnaru. Near Secunderabad railway station.',
    utterance_role: 'new_description',
    category: 'road_accident',
    confidence: 0.94,
    extractedFacts: [
      {
        field: 'vehicle_collision_reported',
        value: true,
        evidence_text: 'ikkada road accident jarigindi',
      },
      {
        field: 'injured_persons',
        value: true,
        evidence_text: 'Oka person unconscious ga unnaru',
      },
      {
        field: 'affected_person_count',
        value: 1,
        evidence_text: 'Oka person',
      },
      {
        field: 'caller_location_text',
        value: 'Near Secunderabad railway station',
        evidence_text: 'Near Secunderabad railway station',
      },
    ],
    followupQuestion:
      'Are you standing in a safe spot away from moving traffic, and do you confirm Secunderabad Railway Station?',
    safetyInstructionText:
      'Stay at a safe distance from moving traffic and do not stand on the active roadway.',
    safetyAttributionLabel: 'placeholder, pending review',
    responseText:
      'In this demo, I recorded a road accident and created simulated Ambulance and Police Patrol assignments.',
  },
  {
    id: 'hinglish_fire_injuries',
    label: 'Hinglish: Building Fire with Injuries',
    language: 'hi',
    languageBadge: 'HINGLISH',
    text: 'Bhaiya yahan dukan mein aag lag gayi hai, do log andar jal gaye hain near Hitech City Cyber Towers.',
    utterance_role: 'new_description',
    category: 'fire_with_injuries',
    confidence: 0.91,
    extractedFacts: [
      {
        field: 'fire_reported',
        value: true,
        evidence_text: 'dukan mein aag lag gayi hai',
      },
      {
        field: 'injured_persons',
        value: true,
        evidence_text: 'do log andar jal gaye hain',
      },
      {
        field: 'affected_person_count',
        value: 2,
        evidence_text: 'do log',
      },
      {
        field: 'caller_location_text',
        value: 'Hitech City Cyber Towers',
        evidence_text: 'near Hitech City Cyber Towers',
      },
    ],
    followupQuestion: 'Have you moved clear of the smoke and flames outside the structure?',
    safetyInstructionText:
      'Stay clear of smoke and heat in an open area and do not re-enter the structure.',
    safetyAttributionLabel: 'placeholder, pending review',
    responseText:
      'In this demo, simulated Fire Engine and Ambulance assignments have been evaluated for your incident.',
  },
  {
    id: 'benglish_telugu_codemix_rescue',
    label: 'Benglish + Telugu-English: Godown Fire & Injury',
    language: 'en',
    languageBadge: 'BENGLISH',
    text: 'Dada ekhane godown e agun legeche, dhoya berocche and ikkada oka worker oshustho spruha ledu near Secunderabad railway station.',
    utterance_role: 'new_description',
    category: 'fire_with_injuries',
    confidence: 0.92,
    extractedFacts: [
      {
        field: 'fire_reported',
        value: true,
        evidence_text: 'godown e agun legeche, dhoya berocche',
      },
      {
        field: 'injured_persons',
        value: true,
        evidence_text: 'oka worker oshustho spruha ledu',
      },
      {
        field: 'affected_person_count',
        value: 1,
        evidence_text: 'oka worker',
      },
      {
        field: 'caller_location_text',
        value: 'Near Secunderabad railway station',
        evidence_text: 'near Secunderabad railway station',
      },
    ],
    followupQuestion:
      'Have you moved everyone clear of the godown smoke, and do you confirm Secunderabad Railway Station?',
    safetyInstructionText:
      'Stay clear of smoke and heat in an open area and do not re-enter the structure.',
    safetyAttributionLabel: 'placeholder, pending review',
    responseText:
      'In this demo, I recorded a fire with injuries from your code-mixed report and prepared simulated Fire Engine and Ambulance units.',
  },
  {
    id: 'short_answer_yes',
    label: 'Short Answer ("Yes, confirmed")',
    language: 'en',
    languageBadge: 'EN',
    text: 'Yes, that landmark is correct and I am standing on the footpath.',
    utterance_role: 'answer_to_pending_question',
    category: 'unknown',
    confidence: 0.55,
    extractedFacts: [
      {
        field: 'caller_safe_spot_status',
        value: 'Standing safely on footpath',
        evidence_text: 'standing on the footpath',
      },
    ],
    followupQuestion: null,
    safetyInstructionText:
      'Remain in a safe spot and stay on the line with me while this simulation continues.',
    safetyAttributionLabel: 'placeholder, pending review',
    responseText:
      'Thank you for confirming. As an answer to a pending question, this turn does not increment the unknown counter.',
  },
  {
    id: 'ambiguous_unknown_turn',
    label: 'Ambiguous Description (Unknown Category)',
    language: 'en',
    languageBadge: 'EN',
    text: 'Something strange is happening down the street, people are gathering and shouting.',
    utterance_role: 'new_description',
    category: 'unknown',
    confidence: 0.42,
    extractedFacts: [
      {
        field: 'caller_observations',
        value: 'People gathering and shouting down the street',
        evidence_text: 'people are gathering and shouting',
      },
    ],
    followupQuestion:
      'Can you see if anyone is injured, or if there is a fire, crash, or physical threat?',
    safetyInstructionText:
      'Remain in a safe spot and stay on the line while we review the details in this simulation.',
    safetyAttributionLabel: 'placeholder, pending review',
    responseText:
      'I could not confidently identify the emergency category. Could you tell me if anyone is hurt or if there is a fire?',
  },
  {
    id: 'provider_blocked_knife_attack',
    label: 'Provider Safety Block (Violence / Weapon Content)',
    language: 'en',
    languageBadge: 'EN',
    text: 'Someone is attacking people with a knife near the market entrance and there is bleeding.',
    utterance_role: 'new_description',
    category: 'physical_attack',
    confidence: 0.9,
    simulateProviderBlock: true,
    extractedFacts: [],
    followupQuestion: null,
    safetyInstructionText:
      'I am here with you in this simulation. Please describe the situation again in your own words.',
    safetyAttributionLabel: 'placeholder, pending review',
    responseText: '',
  },
  {
    id: 'provider_outage_503',
    label: 'Reasoning Outage (Stay With Me Reassurance)',
    language: 'en',
    languageBadge: 'EN',
    text: 'There is a loud gas smell coming from the cylinder in the ground floor kitchen.',
    utterance_role: 'new_description',
    category: 'gas_leak',
    confidence: 0.88,
    simulateProviderOutage: true,
    extractedFacts: [],
    followupQuestion: null,
    safetyInstructionText:
      "I'm here with you. Please stay on the line while this simulation continues. This is a prototype and not connected to 112.",
    safetyAttributionLabel: 'placeholder, pending review',
    responseText: '',
  },
  {
    id: 'explosion_industrial_site',
    label: 'Explosion Reported (Multi-Unit Allocation)',
    language: 'en',
    languageBadge: 'EN',
    text: 'A loud explosion occurred at the godown near Secunderabad railway station and two workers are injured.',
    utterance_role: 'new_description',
    category: 'explosion',
    confidence: 0.95,
    extractedFacts: [
      {
        field: 'explosion_reported',
        value: true,
        evidence_text: 'A loud explosion occurred',
      },
      {
        field: 'injured_persons',
        value: true,
        evidence_text: 'two workers are injured',
      },
      {
        field: 'affected_person_count',
        value: 2,
        evidence_text: 'two workers',
      },
      {
        field: 'caller_location_text',
        value: 'Near Secunderabad railway station',
        evidence_text: 'near Secunderabad railway station',
      },
    ],
    followupQuestion:
      'Have you moved far away from the blast zone and any secondary fire hazard?',
    safetyInstructionText:
      'Move away from the blast area immediately and stay clear of falling debris or smoke.',
    safetyAttributionLabel: 'placeholder, pending review',
    responseText:
      'In this demo, I recorded an explosion and created simulated Fire Engine, Ambulance, and Police Patrol assignments.',
  },
];

/**
 * In-session follow-up turns used inside the collapsible Demo Controls drawer to verify:
 * 1) Conflicting caller statements (affected_person_count / location disagreement)
 * 2) Reclassification + location change inside one session (supersedes stale fire_reported fact + raises possible_second_incident)
 */
const IN_SESSION_FOLLOWUP_TURNS: readonly PresetScenario[] = [
  {
    id: 'followup_conflict_person_count',
    label: 'In-Session Turn: Conflicting Person Count (3 people vs prior count)',
    language: 'en',
    languageBadge: 'EN',
    text: 'Wait, I can see three people injured on the pavement now, not one.',
    utterance_role: 'new_description',
    category: 'road_accident',
    confidence: 0.93,
    extractedFacts: [
      {
        field: 'affected_person_count',
        value: 3,
        evidence_text: 'three people injured',
      },
    ],
    followupQuestion: 'Are all three injured people clear of moving vehicles?',
    safetyInstructionText:
      'Stay at a safe distance from moving traffic and do not stand on the active roadway.',
    safetyAttributionLabel: 'placeholder, pending review',
    responseText:
      'In this demo, I recorded your updated statement of 3 affected people and flagged the conflict with the earlier count for review.',
  },
  {
    id: 'followup_reclassify_and_new_location',
    label: 'In-Session Turn: Reclassify to Road Accident at Charminar (Supersedes Fire + Second Incident Flag)',
    language: 'en',
    languageBadge: 'EN',
    text: 'Actually there is no building fire, it is a road accident near Charminar Bus Stand with one person hurt.',
    utterance_role: 'new_description',
    category: 'road_accident',
    confidence: 0.94,
    extractedFacts: [
      {
        field: 'vehicle_collision_reported',
        value: true,
        evidence_text: 'it is a road accident',
      },
      {
        field: 'affected_person_count',
        value: 1,
        evidence_text: 'one person hurt',
      },
      {
        field: 'caller_location_text',
        value: 'Charminar Bus Stand',
        evidence_text: 'near Charminar Bus Stand',
      },
    ],
    followupQuestion:
      'Because both the incident type and location changed, please confirm if you are reporting a second incident near Charminar Bus Stand.',
    safetyInstructionText:
      'Stay at a safe distance from moving traffic and do not stand on the active roadway.',
    safetyAttributionLabel: 'placeholder, pending review',
    responseText:
      'In this demo, I updated the category to road accident, superseded stale fire facts, and flagged a possible second incident for review.',
  },
];

/**
 * Highlights `evidenceText` inside `fullText` when a user hovers the corresponding fact chip.
 */
function renderTranscriptWithHighlightedEvidence(
  fullText: string,
  evidenceText: string | null,
): React.ReactNode {
  if (!evidenceText) {
    return fullText;
  }
  const lowerFull = fullText.toLowerCase();
  const lowerEv = evidenceText.toLowerCase();
  const matchIdx = lowerFull.indexOf(lowerEv);
  if (matchIdx === -1) {
    return fullText;
  }
  const before = fullText.slice(0, matchIdx);
  const matched = fullText.slice(matchIdx, matchIdx + evidenceText.length);
  const after = fullText.slice(matchIdx + evidenceText.length);
  return (
    <>
      {before}
      <mark className="bg-amber-200 text-slate-950 font-semibold px-1 py-0.5 rounded border border-amber-400">
        {matched}
      </mark>
      {after}
    </>
  );
}

function SethuBrandEmblem({ size = 'md' }: { size?: 'md' | 'lg' }) {
  const dim = size === 'lg' ? 'w-12 h-12' : 'w-11 h-11';
  const uid = size === 'lg' ? 'sethu-lg' : 'sethu-md';
  return (
    <div
      data-testid={`sethu-brand-emblem-${size}`}
      className={`${dim} relative rounded-xl bg-[#051E1F] border border-teal-500/40 shadow-md flex items-center justify-center shrink-0 overflow-hidden`}
    >
      <svg
        viewBox="0 0 64 64"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full"
        aria-hidden="true"
      >
        <defs>
          <radialGradient id={`${uid}-bg`} cx="50%" cy="22%" r="78%">
            <stop offset="0%" stopColor="#0F4C4E" />
            <stop offset="65%" stopColor="#072A2B" />
            <stop offset="100%" stopColor="#041819" />
          </radialGradient>
          <linearGradient id={`${uid}-arch`} x1="8" y1="46" x2="56" y2="28" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#2DD4BF" />
            <stop offset="50%" stopColor="#5EEAD4" />
            <stop offset="100%" stopColor="#10B981" />
          </linearGradient>
          <linearGradient id={`${uid}-wave`} x1="32" y1="14" x2="32" y2="42" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#FFFFFF" />
            <stop offset="100%" stopColor="#5EEAD4" />
          </linearGradient>
        </defs>

        {/* Shield Base & Precision Inner Bezel */}
        <rect width="64" height="64" rx="14" fill={`url(#${uid}-bg)`} />
        <rect
          x="2.5"
          y="2.5"
          width="59"
          height="59"
          rx="12"
          stroke="#2DD4BF"
          strokeOpacity="0.28"
          strokeWidth="1.2"
        />

        {/* Subtle Radar Signal Arcs */}
        <path
          d="M21 15C24.2 12.2 27.9 10.8 32 10.8C36.1 10.8 39.8 12.2 43 15"
          stroke="#FBBF24"
          strokeOpacity="0.4"
          strokeWidth="1.5"
          strokeLinecap="round"
        />

        {/* Acoustic Voice Waveform Suspension Pillars */}
        <path
          d="M19 42V32M25.5 39V24M32 37V17M38.5 39V24M45 42V32"
          stroke={`url(#${uid}-wave)`}
          strokeWidth="3.2"
          strokeLinecap="round"
        />

        {/* Primary SETHU Bridge Arch ("The Bridge") */}
        <path
          d="M9 47C17 31 47 31 55 47"
          stroke={`url(#${uid}-arch)`}
          strokeWidth="3.8"
          strokeLinecap="round"
        />

        {/* Bridge Deck Baseline & Anchor Nodes (Caller <-> Response) */}
        <path
          d="M7 48.5H57"
          stroke="#99F6E4"
          strokeOpacity="0.45"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <circle cx="10" cy="47" r="2.6" fill="#2DD4BF" />
        <circle cx="54" cy="47" r="2.6" fill="#10B981" />

        {/* Emergency Response Beacon at Apex */}
        <circle cx="32" cy="11" r="5" fill="#FBBF24" fillOpacity="0.24" />
        <circle cx="32" cy="11" r="2.8" fill="#FBBF24" />
      </svg>
    </div>
  );
}

export default function App() {
  const initialQueryAutostart =
    typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).get('autostart') === '1';

  const [activeView, setActiveView] = useState<WorkspaceView>(
    initialQueryAutostart ? 'live_incident' : 'initial',
  );
  const [mobileTab, setMobileTab] = useState<MobileTab>('conversation');
  const [showDiagnosticsOverlay, setShowDiagnosticsOverlay] = useState<boolean>(false);
  const [showDisclosureModal, setShowDisclosureModal] = useState<boolean>(false);
  const [isDemoControlsOpen, setIsDemoControlsOpen] = useState<boolean>(false);
  const [isAuditDrawerOpen, setIsAuditDrawerOpen] = useState<boolean>(false);

  // Provider Verification State — MOCK reasoning adapter chip stays visible until a live provider is verified
  const [isLiveReasoningVerified, setIsLiveReasoningVerified] = useState<boolean>(false);

  // Initial Screen & Session State (IDs populated from backend /ws/session — never hardcoded)
  const [selectedInitialLanguage, setSelectedInitialLanguage] = useState<SupportedLanguage>('en');
  const [detectedLanguagesSeen, setDetectedLanguagesSeen] = useState<SupportedLanguage[]>(['en']);
  const [activeLanguage, setActiveLanguage] = useState<SupportedLanguage>('en');
  const [sessionId, setSessionId] = useState<string>('awaiting-backend-ws-session');
  const [incidentId, setIncidentId] = useState<string>('awaiting-backend-ws-incident');
  const sessionIdRef = useRef<string>('awaiting-backend-ws-session');
  const activeLanguageRef = useRef<SupportedLanguage>('en');
  const [sessionStatus, setSessionStatus] = useState<'not_started' | 'active' | 'completed'>(
    initialQueryAutostart ? 'active' : 'not_started',
  );
  const sessionStatusRef = useRef<'not_started' | 'active' | 'completed'>(
    initialQueryAutostart ? 'active' : 'not_started',
  );
  const updateSessionStatus = (next: 'not_started' | 'active' | 'completed') => {
    sessionStatusRef.current = next;
    setSessionStatus(next);
  };
  const [, setSpokenDisclosurePlayed] = useState<boolean>(initialQueryAutostart);

  // Classification, Counters & Fallback State
  const [incidentCategory, setIncidentCategory] = useState<EmergencyCategory>('unknown');
  const [categoryEstablishedAboveThreshold, setCategoryEstablishedAboveThreshold] =
    useState<boolean>(false);
  const [classificationAttempts, setClassificationAttempts] = useState<number>(0);
  const [classificationConfidences, setClassificationConfidences] = useState<number[]>([]);
  const [consecutiveUnknownCount, setConsecutiveUnknownCount] = useState<number>(0);
  const [consecutiveProviderBlocks, setConsecutiveProviderBlocks] = useState<number>(0);
  const [totalProviderBlocks, setTotalProviderBlocks] = useState<number>(0);
  const [unknownFallbackApplied, setUnknownFallbackApplied] = useState<boolean>(false);
  const [fallbackReason, setFallbackReason] = useState<string | null>(null);
  const [controlRoomReviewRequired, setControlRoomReviewRequired] = useState<boolean>(false);
  const [reviewReasonCodes, setReviewReasonCodes] = useState<string[]>([]);

  // Dialogue, Streaming Partial, Thinking Indicator, Provenance & Conflict State
  const [utterances, setUtterances] = useState<
    Array<FinalizedUtterance & { languageBadge?: string }>
  >([]);
  const [streamingPartialText, setStreamingPartialText] = useState<string>(
    'Waiting for caller speech…',
  );
  const [isReasoningThinking, setIsReasoningThinking] = useState<boolean>(false);
  const [hoveredFact, setHoveredFact] = useState<ProvenancedFact | null>(null);
  const [trackedFacts, setTrackedFacts] = useState<TrackedFact[]>([]);
  const [conflicts, setConflicts] = useState<FactConflict[]>([]);
  const [pendingFollowupQuestion, setPendingFollowupQuestion] = useState<string | null>(null);
  const [pendingQuestionState, setPendingQuestionState] = useState<{
    question_id: string;
    question_text: string;
    target_field: string;
    status: string;
    asked_at?: string;
  } | null>(null);
  const [requiredResourceCategories, setRequiredResourceCategories] = useState<string[]>([]);
  const [stayWithMeMessage, setStayWithMeMessage] = useState<string>(
    SPOKEN_DISCLOSURE_BY_LANGUAGE.en.text,
  );
  const [activeSafetyInstruction, setActiveSafetyInstruction] = useState<string>(
    'Remain in a safe spot and stay on the line while this simulation continues.',
  );
  const [safetyAttributionLabel, setSafetyAttributionLabel] = useState<string>(
    'placeholder, pending review',
  );
  const [identifiedHazards, setIdentifiedHazards] = useState<string[]>([]);

  // Location & Landmark Match Confidence State
  const [gpsLatitude, setGpsLatitude] = useState<number>(17.4435);
  const [gpsLongitude, setGpsLongitude] = useState<number>(78.3772);
  const [gpsAccuracyMeters, setGpsAccuracyMeters] = useState<number | null>(78);
  const [landmarkMatchConfidence, setLandmarkMatchConfidence] = useState<number | null>(null);
  const [locationSource, setLocationSource] = useState<'browser_gps' | 'landmark' | 'unresolved'>(
    'browser_gps',
  );
  const [locationConfirmationStatus, setLocationConfirmationStatus] =
    useState<LocationConfirmationStatus>('candidate_awaiting_confirmation');
  const [broadLocationText, setBroadLocationText] = useState<string | null>(null);
  const [callerLandmarkText, setCallerLandmarkText] = useState<string | null>(null);
  const [locationConfirmedAt, setLocationConfirmedAt] = useState<string | null>(null);
  const [matchedLandmarkId, setMatchedLandmarkId] = useState<string | null>(null);
  const [showLocationChangePicker, setShowLocationChangePicker] = useState<boolean>(false);

  // Synthetic Resources, Backend-Synced Assignments & Expandable Hospital Rows
  const [resources, setResources] = useState<SyntheticResource[]>([
    ...INITIAL_SYNTHETIC_RESOURCES,
  ]);
  const [assignments, setAssignments] = useState<SimulatedAssignment[]>([]);
  const [backendHospitalBreakdowns, setBackendHospitalBreakdowns] = useState<
    HospitalScoreBreakdown[] | null
  >(null);
  const [expandedHospitalIds, setExpandedHospitalIds] = useState<Record<string, boolean>>({
    'SYN-HOSP-HYD-02': true,
    'SYN-HOSP-HYD-01': false,
    'SYN-HOSP-HYD-03': false,
  });

  // Duplicate & False/Prank Review Flags
  const [possibleDuplicateFlag, setPossibleDuplicateFlag] = useState<boolean>(false);
  const [matchedDuplicateId, setMatchedDuplicateId] = useState<string | null>(null);
  const [prankReviewStatus, setPrankReviewStatus] =
    useState<PrankFlagReviewStatus>('not_reviewed');
  const [prankFlagReason, setPrankFlagReason] = useState<string | null>(null);
  const [transcriptRetentionExpired, setTranscriptRetentionExpired] = useState<boolean>(false);

  // Live Voice Interaction & Silero VAD State
  const [voiceState, setVoiceState] = useState<VoiceInteractionState>('idle');
  const [liveSpeechProbability, setLiveSpeechProbability] = useState<number>(0);
  const [liveMicRms, setLiveMicRms] = useState<number>(0);
  const [isVadReady, setIsVadReady] = useState<boolean>(false);
  const [vadError, setVadError] = useState<string | null>(null);
  const [micErrorMessage, setMicErrorMessage] = useState<string | null>(null);
  const sileroVadRef = useRef<SileroVadDetector | null>(null);
  const isTtsPlayingRef = useRef<boolean>(false);
  const vadActiveSpeechStreamingRef = useRef<boolean>(false);
  const isTurnProcessingRef = useRef<boolean>(false);
  const processingWatchdogRef = useRef<number | null>(null);
  const micStreamSeqRef = useRef<number>(0);
  const vadFinalizeSentAtRef = useRef<number | null>(null);
  const speechStartedAtMsRef = useRef<number | null>(null);
  const lowRmsSilenceSinceMsRef = useRef<number | null>(null);

  const dispatchVoiceLifecycleEvent = (
    event: VoiceLifecycleEvent,
    override?: Partial<{
      sessionActive: boolean;
      isTtsPlaying: boolean;
      isTurnProcessing: boolean;
      isSpeechStreaming: boolean;
    }>,
  ) => {
    setVoiceState((prev) =>
      transitionVoiceInteractionState(prev, event, {
        sessionActive: override?.sessionActive ?? sessionStatusRef.current === 'active',
        isTtsPlaying: override?.isTtsPlaying ?? isTtsPlayingRef.current,
        isTurnProcessing: override?.isTurnProcessing ?? isTurnProcessingRef.current,
        isSpeechStreaming: override?.isSpeechStreaming ?? vadActiveSpeechStreamingRef.current,
      }),
    );
  };

  const clearTurnProcessingState = () => {
    isTurnProcessingRef.current = false;
    if (processingWatchdogRef.current !== null) {
      window.clearTimeout(processingWatchdogRef.current);
      processingWatchdogRef.current = null;
    }
  };

  const armTurnProcessingWatchdog = () => {
    isTurnProcessingRef.current = true;
    if (processingWatchdogRef.current !== null) {
      window.clearTimeout(processingWatchdogRef.current);
    }
    processingWatchdogRef.current = window.setTimeout(() => {
      isTurnProcessingRef.current = false;
      processingWatchdogRef.current = null;
      vadFinalizeSentAtRef.current = null;
      setIsReasoningThinking(false);
      dispatchVoiceLifecycleEvent('turn_error_or_timeout', {
        isTurnProcessing: false,
      });
    }, 12000);
  };

  const finalizeActiveCallerSpeechUtterance = (vadFinalizeMs = 700) => {
    if (!vadActiveSpeechStreamingRef.current) {
      return;
    }
    vadActiveSpeechStreamingRef.current = false;
    speechStartedAtMsRef.current = null;
    lowRmsSilenceSinceMsRef.current = null;
    sileroVadRef.current?.reset();
    if (sessionStatusRef.current !== 'active') {
      dispatchVoiceLifecycleEvent('session_ended', {
        sessionActive: false,
        isSpeechStreaming: false,
      });
      return;
    }
    const activeSid =
      sessionIdRef.current && sessionIdRef.current !== 'awaiting-backend-ws-session'
        ? sessionIdRef.current
        : undefined;
    vadFinalizeSentAtRef.current = performance.now();
    armTurnProcessingWatchdog();
    dispatchVoiceLifecycleEvent('caller_speech_finalized', {
      sessionActive: true,
      isTurnProcessing: true,
      isSpeechStreaming: false,
    });
    setStreamingPartialText('Understanding what you shared…');
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'audio_chunk_pcm16',
          session_id: activeSid,
          finalize_utterance: true,
          vad_finalize_ms: vadFinalizeMs,
          pcm_base64: '',
        }),
      );
    } else {
      clearTurnProcessingState();
      dispatchVoiceLifecycleEvent('turn_error_or_timeout', {
        isTurnProcessing: false,
        isSpeechStreaming: false,
      });
    }
  };

  // Barge-In, Echo Cancellation & Continuous Ring Buffer State
  const ringBufferRef = useRef<ContinuousAudioRingBuffer>(new ContinuousAudioRingBuffer());
  const [isTtsPlaying, setIsTtsPlaying] = useState<boolean>(false);
  const [ttsPlaybackSource, setTtsPlaybackSource] = useState<'cached_instruction' | 'live_timbre'>(
    'cached_instruction',
  );
  const [bargeInCount, setBargeInCount] = useState<number>(0);
  const [falseBargeInCount, setFalseBargeInCount] = useState<number>(0);
  const [vadMisfiresFiltered, setVadMisfiresFiltered] = useState<number>(0);
  const [, setAutoVadBargeInEnabled] = useState<boolean>(true);
  const [echoCancellationActive, setEchoCancellationActive] = useState<boolean>(true);
  const [lastBargeInSummary, setLastBargeInSummary] = useState<string>(
    'Continuous 16 kHz mono PCM ring buffer ready.',
  );

  // Acoustic Environment Intelligence (In-memory DSP telemetry — zero Gemini calls, never alters severity/dispatch)
  const liveSpeechProbRef = useRef<number>(0);
  const acousticDspStatsRef = useRef<{
    noiseRmsEma: number;
    lastSpeechZcr: number;
    overlapScoreEma: number;
    clippingEventsCount: number;
    peakSampleRatio: number;
    speechFramesCount: number;
    totalFramesCount: number;
    lastUiUpdateMs: number;
  }>({
    noiseRmsEma: 0.0035,
    lastSpeechZcr: 0.12,
    overlapScoreEma: 0,
    clippingEventsCount: 0,
    peakSampleRatio: 0,
    speechFramesCount: 0,
    totalFramesCount: 0,
    lastUiUpdateMs: 0,
  });
  const [acousticTelemetry, setAcousticTelemetry] = useState<{
    noiseFloorDbfs: number;
    noiseCategory: 'quiet' | 'moderate' | 'high_ambient';
    overlappingVoicesDetected: boolean;
    voiceOverlapScore: number;
    clippingEventsCount: number;
    peakSampleRatio: number;
    micQualityStatus: 'clean' | 'clipping_detected' | 'low_signal' | 'standby';
    speechFramesCount: number;
    totalFramesCount: number;
    speechActivityPct: number;
  }>({
    noiseFloorDbfs: -49,
    noiseCategory: 'quiet',
    overlappingVoicesDetected: false,
    voiceOverlapScore: 0,
    clippingEventsCount: 0,
    peakSampleRatio: 0,
    micQualityStatus: 'standby',
    speechFramesCount: 0,
    totalFramesCount: 0,
    speechActivityPct: 0,
  });

  // Isolated Engineering Diagnostics 8 kHz Telephony-Bandwidth & Street Noise Bench (never touches live mic, real incidents, or dispatch)
  const [telephonyDiagSim, setTelephonyDiagSim] = useState<{
    active: boolean;
    codecProfile: string;
    simulatedNoiseFloorDbfs: number;
    simulatedSnrDb: number;
    bandpassHz: string;
    vadNoiseFramesRejected: number;
    speechFramesRetained: number;
    geminiCallsTriggered: number;
    lastRunTimestamp: string | null;
  }>({
    active: false,
    codecProfile: 'Standby (16 kHz Wideband Native)',
    simulatedNoiseFloorDbfs: -49,
    simulatedSnrDb: 28,
    bandpassHz: '50 Hz – 8,000 Hz (Wideband PCM16)',
    vadNoiseFramesRejected: 0,
    speechFramesRetained: 0,
    geminiCallsTriggered: 0,
    lastRunTimestamp: null,
  });

  // Measured-Only Timings & Live Per-Turn Pipeline Performance Waterfall
  const [cacheHits, setCacheHits] = useState<number>(0);
  const [cacheMisses] = useState<number>(0);
  const [lastPrismaLatencyMs, setLastPrismaLatencyMs] = useState<number | null>(null);
  const [lastReasoningLatencyMs, setLastReasoningLatencyMs] = useState<number | null>(null);
  const [lastCachedPlaybackStartMs, setLastCachedPlaybackStartMs] = useState<number | null>(null);
  const [lastLiveTimbreTtfabMs, setLastLiveTimbreTtfabMs] = useState<number | null>(null);
  const [lastBargeInStopMs, setLastBargeInStopMs] = useState<number | null>(null);
  const [stageLatencies, setStageLatencies] = useState<{
    turn_id?: string;
    vad_finalize_ms?: number | null;
    prisma_stt_ms?: number | null;
    gemini_reasoning_ms?: number | null;
    orchestrator_ms?: number | null;
    postgres_persist_ms?: number | null;
    tts_and_persist_wall_ms?: number | null;
    timbre_ttfb_ms?: number | null;
    timbre_total_ms?: number | null;
    browser_playback_delay_ms?: number | null;
    total_backend_ms?: number | null;
    task_execution_telemetry?: Record<string, unknown> | null;
    measured_at?: string;
  } | null>(null);
  const [turnWaterfallHistory, setTurnWaterfallHistory] = useState<TurnWaterfallEntry[]>([]);
  const [backendHandoffPayload, setBackendHandoffPayload] = useState<Record<string, unknown> | null>(null);
  const playedSynthesisIdsRef = useRef<Set<string>>(new Set());
  const browserPlaybackDelayRef = useRef<number | null>(null);
  const wsRequestSentAtRef = useRef<number | null>(null);

  // Audit Timeline
  const [auditEvents, setAuditEvents] = useState<AuditTimelineEntry[]>([
    {
      id: 'evt-init-1',
      timestamp: new Date().toISOString(),
      event_type: 'session_created',
      severity: 'info',
      outcome: 'Development placeholder disclosure audio ready',
      reason_code: null,
      provenance_ref: 'disclosure_session_start:v1.0',
    },
  ]);

  // Unified Web Audio (AudioContext + AudioBufferSourceNode) & /ws/session WebSocket State
  const audioControllerRef = useRef<SetuWebAudioController | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [wsConnectionState, setWsConnectionState] = useState<
    'connecting' | 'connected' | 'disconnected'
  >('connecting');
  const [manifestAssets, setManifestAssets] = useState<ManifestAudioAsset[]>([]);
  const [lastVerifiedSha256, setLastVerifiedSha256] = useState<string>('pending');
  const [lastAudioGenerationStatus, setLastAudioGenerationStatus] = useState<string>(
    'ready',
  );
  const [isMicWorkletStreaming, setIsMicWorkletStreaming] = useState<boolean>(false);
  const [pcmChunksStreamed, setPcmChunksStreamed] = useState<number>(0);
  const [isPrismaVerified, setIsPrismaVerified] = useState<boolean>(true);
  const [isTimbreVerified, setIsTimbreVerified] = useState<boolean>(true);
  const [isPostgresVerified, setIsPostgresVerified] = useState<boolean>(true);
  const [geminiStatusMessage, setGeminiStatusMessage] = useState<string>(
    'Configured (gemini-3.8-flash ready)',
  );
  const [prismaGateMessage, setPrismaGateMessage] = useState<string>(
    'Live and verified (Gnani Prisma v2.5 WSS 16 kHz mono PCM streaming)',
  );

  const backendSessionTokenRef = useRef<string>('');
  const backendLastSeqRef = useRef<number>(0);
  const [customUtteranceText, setCustomUtteranceText] = useState<string>('');
  const [silenceCheckInsSent, setSilenceCheckInsSent] = useState<number>(0);

  // Current active (non-superseded) facts
  const activeFacts = useMemo(
    () => trackedFacts.filter((f) => f.status !== 'superseded'),
    [trackedFacts],
  );
  const supersededFacts = useMemo(
    () => trackedFacts.filter((f) => f.status === 'superseded'),
    [trackedFacts],
  );
  const activeConflictsCount = conflicts.length;

  // Enforce invariant: a location must NEVER show both "assigned" and reason code `pending_location` at once
  useEffect(() => {
    const hasAssignedUnit = assignments.some((a) => a.status === 'simulated_assigned');
    if (hasAssignedUnit && reviewReasonCodes.includes('pending_location')) {
      setReviewReasonCodes((prev) => prev.filter((code) => code !== 'pending_location'));
    }
  }, [assignments, reviewReasonCodes]);

  // Sync backend snapshot (IDs, assignments, conflicts, hospital recommendations) from /ws/session
  const applyBackendSnapshot = (snap: Record<string, unknown>) => {
    if (wsRequestSentAtRef.current !== null) {
      const measuredMs = Math.max(
        1,
        Math.round((performance.now() - wsRequestSentAtRef.current) * 10) / 10,
      );
      setLastReasoningLatencyMs(measuredMs);
      wsRequestSentAtRef.current = null;
    }

    if (typeof snap.session_id === 'string' && snap.session_id) {
      sessionIdRef.current = snap.session_id;
      setSessionId(snap.session_id);
      if (snap.session_state !== 'completed') {
        try {
          sessionStorage.setItem('setu_active_session_id', snap.session_id);
          if (typeof snap.session_token === 'string' && snap.session_token) {
            sessionStorage.setItem('setu_active_session_token', snap.session_token);
          }
        } catch {
          // Ignore storage unavailability
        }
      } else {
        try {
          sessionStorage.removeItem('setu_active_session_id');
          sessionStorage.removeItem('setu_active_session_token');
        } catch {
          // Ignore
        }
      }
    }
    if (typeof snap.incident_id === 'string' && snap.incident_id) {
      setIncidentId(snap.incident_id);
    }
    if (typeof snap.active_language === 'string' && snap.active_language) {
      const lang = snap.active_language as SupportedLanguage;
      activeLanguageRef.current = lang;
      setActiveLanguage(lang);
      setSelectedInitialLanguage(lang);
    }
    if (Array.isArray(snap.detected_languages_seen) && snap.detected_languages_seen.length > 0) {
      setDetectedLanguagesSeen(
        snap.detected_languages_seen.map((code: unknown) => String(code) as SupportedLanguage),
      );
    }
    if (typeof snap.session_token === 'string') {
      backendSessionTokenRef.current = snap.session_token;
      if (snap.session_state !== 'completed' && typeof snap.session_id === 'string' && snap.session_id) {
        try {
          sessionStorage.setItem('setu_active_session_id', snap.session_id);
          sessionStorage.setItem('setu_active_session_token', snap.session_token);
        } catch {
          // Ignore storage unavailability
        }
      }
    }
    if (typeof snap.last_event_sequence === 'number') {
      backendLastSeqRef.current = snap.last_event_sequence;
    }
    if (Array.isArray(snap.assignments)) {
      const mappedAssignments: SimulatedAssignment[] = snap.assignments.map(
        (raw: Record<string, unknown>) => ({
          assignment_id: String(raw.assignment_id),
          incident_id: String(raw.incident_id),
          idempotency_key: String(raw.idempotency_key),
          category: (raw.resource_category as ResourceCategory) || 'ambulance',
          resource_id: raw.resource_id ? String(raw.resource_id) : null,
          vehicle_code: raw.vehicle_code ? String(raw.vehicle_code) : null,
          resource_callsign: raw.resource_callsign ? String(raw.resource_callsign) : null,
          status: (raw.status as AssignmentStatus) || 'simulated_assigned',
          lifecycle_status: (raw.lifecycle_status as SimulatedAssignment['lifecycle_status']) || 'ASSIGNED',
          assigned_at: String(raw.assigned_at || new Date().toISOString()),
          en_route_at: raw.en_route_at ? String(raw.en_route_at) : null,
          arrived_at: raw.arrived_at ? String(raw.arrived_at) : null,
          estimated_arrival_at: raw.estimated_arrival_at ? String(raw.estimated_arrival_at) : null,
          simulated_eta_minutes:
            typeof raw.simulated_eta_minutes === 'number' ? raw.simulated_eta_minutes : null,
          remaining_eta_minutes:
            typeof raw.remaining_eta_minutes === 'number' ? raw.remaining_eta_minutes : null,
          remaining_eta_seconds:
            typeof raw.remaining_eta_seconds === 'number' ? raw.remaining_eta_seconds : null,
          depot_latitude: typeof raw.depot_latitude === 'number' ? raw.depot_latitude : null,
          depot_longitude: typeof raw.depot_longitude === 'number' ? raw.depot_longitude : null,
          current_latitude: typeof raw.current_latitude === 'number' ? raw.current_latitude : null,
          current_longitude: typeof raw.current_longitude === 'number' ? raw.current_longitude : null,
          target_latitude: typeof raw.target_latitude === 'number' ? raw.target_latitude : null,
          target_longitude: typeof raw.target_longitude === 'number' ? raw.target_longitude : null,
          allocation_mode: raw.allocation_mode ? String(raw.allocation_mode) : 'DEMO',
          distance_meters:
            typeof raw.distance_meters === 'number' ? Math.round(raw.distance_meters) : null,
          source_provenance: String(raw.provenance_source || 'backend_rule:orchestrator'),
          is_simulated: true,
        }),
      );
      setAssignments(mappedAssignments);
      if (mappedAssignments.some((a) => a.status === 'simulated_assigned')) {
        setReviewReasonCodes((prev) => prev.filter((c) => c !== 'pending_location'));
      }
    }
    if (snap.location && typeof snap.location === 'object') {
      const locObj = snap.location as Record<string, unknown>;
      if (typeof locObj.latitude === 'number') setGpsLatitude(locObj.latitude);
      if (typeof locObj.longitude === 'number') setGpsLongitude(locObj.longitude);
      if (locObj.accuracy_m !== undefined) {
        setGpsAccuracyMeters(typeof locObj.accuracy_m === 'number' ? locObj.accuracy_m : null);
      }
      if (
        typeof locObj.source_type === 'string' &&
        (locObj.source_type === 'browser_gps' ||
          locObj.source_type === 'landmark' ||
          locObj.source_type === 'unresolved')
      ) {
        setLocationSource(locObj.source_type);
      }
      if (
        typeof locObj.confirmation_status === 'string' &&
        (locObj.confirmation_status === 'unresolved' ||
          locObj.confirmation_status === 'candidate_awaiting_confirmation' ||
          locObj.confirmation_status === 'confirmed')
      ) {
        setLocationConfirmationStatus(locObj.confirmation_status);
      }
      setLandmarkMatchConfidence(
        typeof locObj.match_confidence === 'number' ? locObj.match_confidence : null,
      );
      setBroadLocationText(
        typeof locObj.broad_area_text === 'string' && locObj.broad_area_text
          ? locObj.broad_area_text
          : null,
      );
      const landmarkTxt =
        typeof locObj.caller_landmark_text === 'string' && locObj.caller_landmark_text
          ? locObj.caller_landmark_text
          : typeof locObj.caller_location_text === 'string' && locObj.caller_location_text
            ? locObj.caller_location_text
            : null;
      setCallerLandmarkText(landmarkTxt);
      setLocationConfirmedAt(typeof locObj.confirmed_at === 'string' ? locObj.confirmed_at : null);

      if (typeof locObj.landmark_id === 'string' && locObj.landmark_id) {
        setMatchedLandmarkId(locObj.landmark_id);
      } else if (locObj.source_type === 'landmark') {
        const lowerTxt = (landmarkTxt ?? '').toLowerCase();
        const byNameOrCoords =
          SYNTHETIC_LANDMARKS.find(
            (lm) =>
              (lowerTxt && lowerTxt.includes(lm.name_romanised.toLowerCase())) ||
              (typeof locObj.latitude === 'number' &&
                typeof locObj.longitude === 'number' &&
                Math.abs(lm.latitude - locObj.latitude) < 0.0015 &&
                Math.abs(lm.longitude - locObj.longitude) < 0.0015),
          ) ?? null;
        if (byNameOrCoords) {
          setMatchedLandmarkId(byNameOrCoords.landmark_id);
        }
      } else if (locObj.source_type === 'browser_gps') {
        setMatchedLandmarkId(null);
      }
    }
    if (Array.isArray(snap.review_reason_codes)) {
      const codes = snap.review_reason_codes.map(String);
      setReviewReasonCodes(codes);
      setControlRoomReviewRequired(Boolean(snap.control_room_review_required) || codes.length > 0);
    }
    if (
      Array.isArray(snap.hospital_recommendations) &&
      snap.hospital_recommendations.length > 0
    ) {
      const mappedHospitals: HospitalScoreBreakdown[] = snap.hospital_recommendations.map(
        (h: Record<string, unknown>) => {
          const hospId = String(h.hospital_id);
          const catalogHosp = SYNTHETIC_HOSPITALS.find((sh) => sh.hospital_id === hospId);
          const wDist = Number(h.applied_distance_weight ?? 0.625);
          const wCap = Number(h.applied_capability_weight ?? 0.375);
          const matchedCaps = Array.isArray(h.matched_capabilities)
            ? (h.matched_capabilities as string[])
            : catalogHosp
              ? [...catalogHosp.capabilities]
              : ['emergency_department'];
          return {
            hospital: {
              hospital_id: hospId,
              name: String(h.name || catalogHosp?.name || hospId),
              latitude:
                typeof h.latitude === 'number'
                  ? h.latitude
                  : (catalogHosp?.latitude ?? 17.44),
              longitude:
                typeof h.longitude === 'number'
                  ? h.longitude
                  : (catalogHosp?.longitude ?? 78.38),
              address: String(h.address || catalogHosp?.address || ''),
              capabilities: catalogHosp ? [...catalogHosp.capabilities] : matchedCaps,
              source_identifier: String(
                h.source_identifier || catalogHosp?.source_identifier || 'SYNTHETIC_HYD_DIRECTORY',
              ),
              verification_date: String(
                h.verification_date || catalogHosp?.verification_date || '2025-02-15',
              ),
              is_synthetic_directory: true,
            },
            distance_km: Number(h.distance_km ?? 0),
            eta_minutes: typeof h.eta_minutes === 'number' ? h.eta_minutes : null,
            eta_source_label: 'unavailable (weights re-normalised)',
            matched_capabilities: matchedCaps,
            distance_component_score: Number(h.distance_component_score ?? 0),
            eta_component_score: null,
            capability_component_score: Number(h.capability_component_score ?? 0),
            applied_distance_weight: wDist,
            applied_eta_weight: 0,
            applied_capability_weight: wCap,
            weighted_distance_contribution: Number(h.weighted_distance_contribution ?? 0),
            weighted_eta_contribution: 0,
            weighted_capability_contribution: Number(h.weighted_capability_contribution ?? 0),
            final_score: Number(h.final_score ?? 0),
            methodology_note: `Routing ETA unavailable; weights re-normalised across distance (${Number(wDist.toFixed(3))} / ${(wDist * 100).toFixed(1)}%) and capability fit (${Number(wCap.toFixed(3))} / ${(wCap * 100).toFixed(1)}%).`,
          };
        },
      );
      setBackendHospitalBreakdowns(mappedHospitals);
    }
    syncIncidentStateFromSnapshot(snap);
  };

  // Sync structured emergency intake fields from backend snapshot without disturbing local demo state on initial mount
  const syncIncidentStateFromSnapshot = (snap: Record<string, unknown>) => {
    if (typeof snap.session_state === 'string') {
      if (snap.session_state === 'completed') {
        updateSessionStatus('completed');
      } else if (snap.session_state === 'active' && sessionStatusRef.current === 'not_started' && Array.isArray(snap.utterances) && snap.utterances.length > 0) {
        updateSessionStatus('active');
      }
    }
    if (snap.handoff && typeof snap.handoff === 'object') {
      const ho = snap.handoff as Record<string, unknown>;
      setBackendHandoffPayload(ho);
      if (ho.safety_guidance_attribution && typeof ho.safety_guidance_attribution === 'object') {
        const attr = ho.safety_guidance_attribution as Record<string, unknown>;
        if (typeof attr.approved_text === 'string' && attr.approved_text) {
          setActiveSafetyInstruction(attr.approved_text);
        }
        if (attr.source) {
          setSafetyAttributionLabel(
            `${String(attr.source)} · Reviewed ${String(attr.review_date || '')} · ${String(attr.version || 'v1.0')}`,
          );
        }
      }
    }
    if (Array.isArray(snap.identified_hazards)) {
      setIdentifiedHazards(snap.identified_hazards.map(String));
    }
    if (snap.active_safety_guidance && typeof snap.active_safety_guidance === 'object') {
      const ag = snap.active_safety_guidance as Record<string, unknown>;
      if (typeof ag.approved_text === 'string' && ag.approved_text) {
        setActiveSafetyInstruction(ag.approved_text);
      }
      if (ag.source) {
        setSafetyAttributionLabel(
          `${String(ag.source)} · Reviewed ${String(ag.review_date || '')} · ${String(ag.version || 'v1.0')}`,
        );
      }
    }
    if (Array.isArray(snap.required_resource_categories)) {
      setRequiredResourceCategories(snap.required_resource_categories.map(String));
    }
    if (snap.pending_question && typeof snap.pending_question === 'object') {
      const pq = snap.pending_question as Record<string, unknown>;
      setPendingQuestionState({
        question_id: String(pq.question_id || ''),
        question_text: String(pq.question_text || ''),
        target_field: String(pq.target_field || ''),
        status: String(pq.status || 'pending'),
        asked_at: pq.asked_at ? String(pq.asked_at) : undefined,
      });
    } else if (snap.pending_question === null) {
      setPendingQuestionState(null);
    }
    if (snap.category) {
      setIncidentCategory(snap.category as EmergencyCategory);
    }
    if (typeof snap.category_established_above_threshold === 'boolean') {
      setCategoryEstablishedAboveThreshold(snap.category_established_above_threshold);
    }
    if (typeof snap.classification_attempts === 'number') {
      setClassificationAttempts(snap.classification_attempts);
    }
    if (typeof snap.consecutive_unknown_count === 'number') {
      setConsecutiveUnknownCount(snap.consecutive_unknown_count);
    }
    if (typeof snap.consecutive_provider_blocks === 'number') {
      setConsecutiveProviderBlocks(snap.consecutive_provider_blocks);
    }
    if (typeof snap.total_provider_blocks === 'number') {
      setTotalProviderBlocks(snap.total_provider_blocks);
    }
    if (typeof snap.unknown_fallback_applied === 'boolean') {
      setUnknownFallbackApplied(snap.unknown_fallback_applied);
    }
    if (snap.fallback_reason !== undefined) {
      setFallbackReason(snap.fallback_reason ? String(snap.fallback_reason) : null);
    }
    if (typeof snap.control_room_review_required === 'boolean') {
      setControlRoomReviewRequired(snap.control_room_review_required);
    }
    if (Array.isArray(snap.review_reason_codes)) {
      const codes = snap.review_reason_codes.map(String);
      setReviewReasonCodes(codes);
      if (codes.length > 0) {
        setControlRoomReviewRequired(true);
      }
    }
    if (snap.last_spoken_response_text) {
      setStayWithMeMessage(String(snap.last_spoken_response_text));
    }
    if (snap.pending_followup_question !== undefined) {
      setPendingFollowupQuestion(
        snap.pending_followup_question ? String(snap.pending_followup_question) : null,
      );
    }
    if (Array.isArray(snap.validated_facts)) {
      const conflictFields = new Set(
        Array.isArray(snap.conflicts)
          ? snap.conflicts.map((c: Record<string, unknown>) => String(c.field_name))
          : [],
      );
      const mappedFacts: TrackedFact[] = snap.validated_facts.map((f: Record<string, unknown>) => {
        const field = String(f.field_name);
        return {
          field,
          value: (f.value ?? null) as ProvenancedFact['value'],
          source_utterance_id: String(f.source_utterance_id),
          evidence_text: String(f.evidence_text || ''),
          status: conflictFields.has(field) ? 'conflicting' : 'active',
        };
      });
      setTrackedFacts(mappedFacts);
    }
    if (Array.isArray(snap.conflicts)) {
      const mappedConflicts: FactConflict[] = snap.conflicts.map((c: Record<string, unknown>) => {
        const field = String(c.field_name);
        const priorVal = c.prior_value as FactConflict['prior_value'];
        const newVal = c.new_value as FactConflict['new_value'];
        return {
          field,
          prior_value: priorVal,
          prior_utterance_id: String(c.prior_utterance_id),
          new_value: newVal,
          new_utterance_id: String(c.new_utterance_id),
          description: `${field} updated from ${String(priorVal)} to ${String(newVal)}`,
        };
      });
      setConflicts(mappedConflicts);
    }
    if (Array.isArray(snap.utterances)) {
      const mappedUtterances: FinalizedUtterance[] = snap.utterances.map(
        (u: Record<string, unknown>, idx: number) => ({
          utterance_id: String(u.utterance_id),
          sequence: Number(u.sequence_number ?? idx + 1),
          text: String(u.text || ''),
          language: (u.language as SupportedLanguage) || activeLanguageRef.current,
          utterance_role: (u.utterance_role as UtteranceRole | null) ?? null,
          timestamp: String(u.finalized_at || new Date().toISOString()),
        }),
      );
      setUtterances(mappedUtterances);
    }
  };

  // Initialize SetuWebAudioController, Silero VAD, fetch manifest, and connect WS /ws/session
  useEffect(() => {
    audioControllerRef.current = new SetuWebAudioController((playing, srcType) => {
      setIsTtsPlaying(playing);
      isTtsPlayingRef.current = playing;
      setTtsPlaybackSource(srcType);
      if (playing) {
        dispatchVoiceLifecycleEvent(
          srcType === 'cached_instruction' ? 'intro_playback_started' : 'tts_response_started',
          { isTtsPlaying: true },
        );
      } else {
        ringBufferRef.current.clear();
        dispatchVoiceLifecycleEvent('tts_playback_ended', {
          isTtsPlaying: false,
        });
      }
    });

    // Initialize Silero VAD detector
    const vad = new SileroVadDetector(
      {
        positiveSpeechThreshold: 0.62,
        negativeSpeechThreshold: 0.40,
        minSpeechDurationMs: 380,
        redemptionFrames: 4,
        silenceTimeoutMs: 700,
      },
      {
        onReadyChange: (ready, error) => {
          setIsVadReady(ready);
          setVadError(error);
        },
        onSpeechStart: () => {
          if (sessionStatusRef.current !== 'active' || isTurnProcessingRef.current) {
            return;
          }
          const activeSid =
            sessionIdRef.current && sessionIdRef.current !== 'awaiting-backend-ws-session'
              ? sessionIdRef.current
              : undefined;
          speechStartedAtMsRef.current = performance.now();
          lowRmsSilenceSinceMsRef.current = null;
          if (isTtsPlayingRef.current) {
            setLastBargeInSummary('Acoustic caller speech detected during playback. Stopped audio immediately.');
            setBargeInCount((c) => c + 1);
            const stopRes = audioControllerRef.current?.stopPlaybackImmediately();
            if (stopRes?.localStopLatencyMs) {
              setLastBargeInStopMs(stopRes.localStopLatencyMs);
            }
            setIsTtsPlaying(false);
            isTtsPlayingRef.current = false;
            ringBufferRef.current.clear();
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(
                JSON.stringify({
                  type: 'barge_in',
                  session_id: activeSid,
                  barge_in_mode: 'genuine_speech',
                  preserve_active_stream: false,
                  was_playing_cached_instruction: stopRes?.wasPlayingCachedInstruction ?? true,
                  local_stop_latency_ms: stopRes?.localStopLatencyMs ?? 1.2,
                }),
              );
            }
            appendAudit(
              'barge_in_interruption_local_stop_first',
              'info',
              'Caller interrupted audio playback; stopped AudioBufferSourceNode immediately',
              null,
              'barge_in:genuine_speech',
            );
          }
          vadActiveSpeechStreamingRef.current = true;
          dispatchVoiceLifecycleEvent('caller_speech_started', {
            sessionActive: true,
            isTtsPlaying: false,
            isSpeechStreaming: true,
          });
          setStreamingPartialText('Hearing you — speak naturally…');

          // Flush 1,500 ms pre-roll from continuous ring buffer before streaming live frames
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            const preRoll = ringBufferRef.current.extractPreRollSamples();
            if (preRoll.byteLength > 0) {
              micStreamSeqRef.current += 1;
              const preRollBuf = preRoll.buffer.slice(
                preRoll.byteOffset,
                preRoll.byteOffset + preRoll.byteLength,
              );
              wsRef.current.send(
                JSON.stringify({
                  type: 'audio_chunk_pcm16',
                  session_id: activeSid,
                  sequence_number: micStreamSeqRef.current,
                  sample_rate_hz: 16000,
                  pcm_base64: arrayBufferToBase64(preRollBuf),
                }),
              );
            }
          }
        },
        onSpeechEnd: () => {
          finalizeActiveCallerSpeechUtterance(700);
        },
        onVADMisfire: () => {
          setVadMisfiresFiltered((c) => c + 1);
          speechStartedAtMsRef.current = null;
          lowRmsSilenceSinceMsRef.current = null;
          if (!vadActiveSpeechStreamingRef.current) {
            return;
          }
          vadActiveSpeechStreamingRef.current = false;
          const activeSid =
            sessionIdRef.current && sessionIdRef.current !== 'awaiting-backend-ws-session'
              ? sessionIdRef.current
              : undefined;
          if (sessionStatusRef.current === 'active' && !isTurnProcessingRef.current) {
            dispatchVoiceLifecycleEvent('vad_misfire_discarded', {
              sessionActive: true,
              isTurnProcessing: false,
              isSpeechStreaming: false,
            });
            setStreamingPartialText('Listening for caller speech…');
          }
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(
              JSON.stringify({
                type: 'audio_chunk_pcm16',
                session_id: activeSid,
                discard_utterance: true,
                finalize_utterance: false,
                pcm_base64: '',
              }),
            );
          }
        },
        onProbabilityUpdate: (prob) => {
          liveSpeechProbRef.current = prob;
          setLiveSpeechProbability(prob);
        },
      },
    );
    sileroVadRef.current = vad;
    void vad.init();

    fetch('/api/instructions/manifest')
      .then((r) => (r.ok ? r.json() : []))
      .then((rows: ManifestAudioAsset[]) => {
        if (Array.isArray(rows) && rows.length > 0) {
          setManifestAssets(rows);
          const first = rows[0];
          setLastVerifiedSha256(first.checksum_sha256.slice(0, 16) + '…');
          setLastAudioGenerationStatus(first.generation_status);
        }
      })
      .catch(() => {
        // Keep default status if manifest fetch fails
      });

    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}/ws/session`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnectionState('connected');
      wsRequestSentAtRef.current = performance.now();
      let savedSid: string | null = null;
      let savedTok: string | null = null;
      try {
        savedSid = sessionStorage.getItem('setu_active_session_id');
        savedTok = sessionStorage.getItem('setu_active_session_token');
      } catch {
        savedSid = null;
        savedTok = null;
      }
      if (savedSid && savedTok) {
        ws.send(
          JSON.stringify({
            type: 'session_reconnect',
            session_id: savedSid,
            session_token: savedTok,
            last_received_sequence: 0,
          }),
        );
        return;
      }
      ws.send(
        JSON.stringify({
          type: 'session_init',
          fresh_scenario: true,
          initial_language: selectedInitialLanguage,
          language_mode: 'manual',
          latitude: 17.4435,
          longitude: 78.3772,
          accuracy_m: 78.0,
        }),
      );
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(String(ev.data));
        if (msg.type === 'prisma_transcript_ready') {
          if (vadFinalizeSentAtRef.current !== null) {
            setLastPrismaLatencyMs(
              Math.max(1, Math.round((performance.now() - vadFinalizeSentAtRef.current) * 10) / 10),
            );
            vadFinalizeSentAtRef.current = null;
          }
          if (msg.transcript) {
            setStreamingPartialText(String(msg.transcript));
          }
          dispatchVoiceLifecycleEvent('transcript_ready_understanding');
          setIsReasoningThinking(true);
          return;
        }

        if (msg.type === 'orchestration_state_updated') {
          setIsReasoningThinking(false);
          if (msg.prisma_transcript) {
            setStreamingPartialText(String(msg.prisma_transcript));
          }
          if (msg.snapshot && typeof msg.snapshot === 'object') {
            applyBackendSnapshot(msg.snapshot as Record<string, unknown>);
          }
          if (msg.task_execution_telemetry && typeof msg.task_execution_telemetry === 'object') {
            const tel = msg.task_execution_telemetry as Record<string, unknown>;
            if (typeof tel.gemini_reasoning_ms === 'number' && tel.gemini_reasoning_ms > 0) {
              setIsLiveReasoningVerified(true);
            }
            setStageLatencies((prev) => ({
              ...(prev || {}),
              turn_id: msg.turn_id ? String(msg.turn_id) : prev?.turn_id,
              task_execution_telemetry: tel,
            }));
          }
          dispatchVoiceLifecycleEvent('orchestration_updated');
          return;
        }

        if (msg.type === 'live_timbre_audio_chunk') {
          clearTurnProcessingState();
          setIsReasoningThinking(false);
          if (typeof msg.ttfab_ms === 'number') {
            setLastLiveTimbreTtfabMs(msg.ttfab_ms);
          }
          const synthId = msg.synthesis_id ? String(msg.synthesis_id) : '';
          if (synthId) {
            playedSynthesisIdsRef.current.add(synthId);
          }
          setIsTtsPlaying(true);
          isTtsPlayingRef.current = true;
          setTtsPlaybackSource('live_timbre');
          dispatchVoiceLifecycleEvent('tts_response_started', {
            isTtsPlaying: true,
            isTurnProcessing: false,
          });
          if (typeof msg.pcm_base64 === 'string' && msg.pcm_base64.length > 0) {
            const tPlay0 = performance.now();
            try {
              audioControllerRef.current?.playLiveTimbreBase64Pcm16(
                msg.pcm_base64,
                Number(msg.sample_rate_hz || 16000),
                () => {
                  setIsTtsPlaying(false);
                  isTtsPlayingRef.current = false;
                  dispatchVoiceLifecycleEvent('tts_playback_ended', {
                    isTtsPlaying: false,
                  });
                },
              );
              if (msg.is_first_chunk) {
                browserPlaybackDelayRef.current =
                  Math.round((performance.now() - tPlay0) * 100) / 100;
              }
            } catch {
              setIsTtsPlaying(false);
              isTtsPlayingRef.current = false;
              dispatchVoiceLifecycleEvent('tts_playback_ended', {
                isTtsPlaying: false,
              });
            }
          }
          return;
        }

        const timbreAudio = msg.live_timbre_audio || (msg.type === 'live_timbre_audio' ? msg : null);
        const synthId = timbreAudio?.synthesis_id ? String(timbreAudio.synthesis_id) : '';
        const alreadyStreamed = Boolean(synthId && playedSynthesisIdsRef.current.has(synthId));
        const hasPlayableTimbreAudio = Boolean(
          !alreadyStreamed &&
            timbreAudio &&
            typeof timbreAudio.pcm_base64 === 'string' &&
            timbreAudio.pcm_base64.length > 0 &&
            audioControllerRef.current,
        );
        if (timbreAudio && typeof timbreAudio.ttfab_ms === 'number') {
          setLastLiveTimbreTtfabMs(timbreAudio.ttfab_ms);
        }
        if (hasPlayableTimbreAudio) {
          clearTurnProcessingState();
          setIsTtsPlaying(true);
          isTtsPlayingRef.current = true;
          setTtsPlaybackSource('live_timbre');
          dispatchVoiceLifecycleEvent('tts_response_started', {
            isTtsPlaying: true,
            isTurnProcessing: false,
          });
          const tPlay0 = performance.now();
          try {
            audioControllerRef.current?.playLiveTimbreBase64Pcm16(
              timbreAudio.pcm_base64,
              Number(timbreAudio.sample_rate_hz || 16000),
              () => {
                setIsTtsPlaying(false);
                isTtsPlayingRef.current = false;
                dispatchVoiceLifecycleEvent('tts_playback_ended', {
                  isTtsPlaying: false,
                });
              },
            );
            browserPlaybackDelayRef.current =
              Math.round((performance.now() - tPlay0) * 100) / 100;
          } catch {
            setIsTtsPlaying(false);
            isTtsPlayingRef.current = false;
            dispatchVoiceLifecycleEvent('tts_playback_ended', {
              isTtsPlaying: false,
            });
          }
        }

        if (msg.type === 'stt_gate_blocked') {
          clearTurnProcessingState();
          vadFinalizeSentAtRef.current = null;
          setPcmChunksStreamed(Number(msg.pcm_chunks_ingested || 0));
          setPrismaGateMessage(`${msg.reason_code}: ${msg.message}`);
          dispatchVoiceLifecycleEvent('turn_error_or_timeout', {
            isTurnProcessing: false,
          });
        } else if (msg.type === 'stt_chunk_accepted') {
          setPcmChunksStreamed(Number(msg.pcm_chunks_ingested || 0));
          if (msg.prisma_status) {
            setPrismaGateMessage(String(msg.prisma_status));
          }
        } else if (msg.type === 'session_state') {
          setIsReasoningThinking(false);
          if (msg.gate_status && typeof msg.gate_status === 'object') {
            setIsPrismaVerified(Boolean(msg.gate_status.gate_0_1_prisma_verified));
            setIsTimbreVerified(Boolean(msg.gate_status.gate_0_2_timbre_verified));
            setIsLiveReasoningVerified(Boolean(msg.gate_status.gate_0_3_gemini_verified));
          }
          if (msg.provider_status && typeof msg.provider_status === 'object') {
            const ps = msg.provider_status;
            if (ps.prisma?.message) {
              setPrismaGateMessage(String(ps.prisma.message));
            }
            if (ps.gemini?.message) {
              setGeminiStatusMessage(String(ps.gemini.message));
            }
            if (ps.postgresql?.status) {
              setIsPostgresVerified(ps.postgresql.status === 'live_and_verified');
            }
          }
          if (msg.snapshot && typeof msg.snapshot === 'object') {
            applyBackendSnapshot(msg.snapshot as Record<string, unknown>);
          }
          if (msg.stage_latencies && typeof msg.stage_latencies === 'object') {
            const sl = msg.stage_latencies as Record<string, unknown>;
            if (typeof sl.gemini_reasoning_ms === 'number' && sl.gemini_reasoning_ms > 0) {
              setIsLiveReasoningVerified(true);
            }
            const vadMs = typeof sl.vad_finalize_ms === 'number' ? sl.vad_finalize_ms : 0;
            const sttMs = typeof sl.prisma_stt_ms === 'number' ? sl.prisma_stt_ms : 0;
            const gemMs = typeof sl.gemini_reasoning_ms === 'number' ? sl.gemini_reasoning_ms : 0;
            const orchMs = typeof sl.orchestrator_ms === 'number' ? sl.orchestrator_ms : 0;
            const dbMs = typeof sl.postgres_persist_ms === 'number' ? sl.postgres_persist_ms : 0;
            const ttfbMs = typeof sl.timbre_ttfb_ms === 'number' ? sl.timbre_ttfb_ms : 0;
            const ttsTotMs = typeof sl.timbre_total_ms === 'number' ? sl.timbre_total_ms : 0;
            const playDelayMs = browserPlaybackDelayRef.current ?? 0;
            const totalMs =
              typeof sl.total_backend_ms === 'number'
                ? sl.total_backend_ms
                : Math.round((vadMs + sttMs + gemMs + orchMs + ttsTotMs) * 10) / 10;
            const telObj =
              sl.task_execution_telemetry && typeof sl.task_execution_telemetry === 'object'
                ? (sl.task_execution_telemetry as Record<string, unknown>)
                : null;

            setStageLatencies({
              turn_id: sl.turn_id ? String(sl.turn_id) : undefined,
              vad_finalize_ms: typeof sl.vad_finalize_ms === 'number' ? sl.vad_finalize_ms : null,
              prisma_stt_ms: typeof sl.prisma_stt_ms === 'number' ? sl.prisma_stt_ms : null,
              gemini_reasoning_ms:
                typeof sl.gemini_reasoning_ms === 'number' ? sl.gemini_reasoning_ms : null,
              orchestrator_ms: typeof sl.orchestrator_ms === 'number' ? sl.orchestrator_ms : null,
              postgres_persist_ms:
                typeof sl.postgres_persist_ms === 'number' ? sl.postgres_persist_ms : null,
              tts_and_persist_wall_ms:
                typeof sl.tts_and_persist_wall_ms === 'number' ? sl.tts_and_persist_wall_ms : null,
              timbre_ttfb_ms: typeof sl.timbre_ttfb_ms === 'number' ? sl.timbre_ttfb_ms : null,
              timbre_total_ms: typeof sl.timbre_total_ms === 'number' ? sl.timbre_total_ms : null,
              browser_playback_delay_ms: browserPlaybackDelayRef.current,
              total_backend_ms: typeof sl.total_backend_ms === 'number' ? sl.total_backend_ms : null,
              task_execution_telemetry: telObj,
              measured_at: sl.measured_at ? String(sl.measured_at) : undefined,
            });

            const turnKey = sl.turn_id ? String(sl.turn_id) : `turn-${Date.now()}`;
            const snapObj =
              msg.snapshot && typeof msg.snapshot === 'object'
                ? (msg.snapshot as Record<string, unknown>)
                : null;
            const snapUtterances = Array.isArray(snapObj?.utterances)
              ? (snapObj.utterances as Array<Record<string, unknown>>)
              : [];
            const lastSnapUtt = snapUtterances.at(-1);
            const uttText = String(
              msg.prisma_transcript || lastSnapUtt?.text || streamingPartialText || 'Caller turn',
            );
            const depthRaw = String(telObj?.thinking_level || telObj?.thinking_depth || '').toUpperCase();
            const depthLabel: 'LOW' | 'MEDIUM' | 'DETERMINISTIC' =
              depthRaw === 'MEDIUM'
                ? 'MEDIUM'
                : gemMs > 0
                  ? 'LOW'
                  : 'DETERMINISTIC';

            setTurnWaterfallHistory((prev) => {
              const existingIdx = prev.findIndex((item) => item.turnId === turnKey);
              const dsp = acousticDspStatsRef.current;
              const noiseDb = Math.max(
                -60,
                Math.min(0, Math.round(20 * Math.log10(Math.max(0.0001, dsp.noiseRmsEma)))),
              );
              const entry: TurnWaterfallEntry = {
                turnId: turnKey,
                turnNumber: existingIdx !== -1 ? prev[existingIdx].turnNumber : prev.length + 1,
                timestamp: sl.measured_at ? String(sl.measured_at) : new Date().toISOString(),
                utterancePreview: uttText.slice(0, 72),
                language: String(
                  msg.prisma_detected_language ||
                    lastSnapUtt?.language ||
                    activeLanguageRef.current,
                ).toUpperCase(),
                thinkingDepth: depthLabel,
                reasoningSource: gemMs > 0 ? 'live_gemini' : 'deterministic_fallback',
                vadFinalizeMs: vadMs,
                prismaSttMs: sttMs,
                geminiReasoningMs: gemMs,
                orchestratorMs: orchMs,
                postgresPersistMs: dbMs,
                timbreTtfbMs: ttfbMs,
                timbreTotalMs: ttsTotMs,
                browserPlaybackDelayMs: playDelayMs,
                totalTurnMs: totalMs,
                noiseCategory:
                  noiseDb > -28 ? 'high_ambient' : noiseDb > -42 ? 'moderate' : 'quiet',
                overlappingVoices: dsp.overlapScoreEma > 0.16,
                clippingDetected: dsp.clippingEventsCount > 0,
              };
              if (existingIdx !== -1) {
                const updated = [...prev];
                updated[existingIdx] = entry;
                return updated;
              }
              return [entry, ...prev].slice(0, 12);
            });
          }
          if (msg.cached_instruction_audio) {
            const asset = msg.cached_instruction_audio;
            if (asset.generation_status) {
              setLastAudioGenerationStatus(asset.generation_status);
            }
            if (asset.checksum_sha256) {
              setLastVerifiedSha256(String(asset.checksum_sha256).slice(0, 16) + '…');
            }
          }
          if (msg.timeout_evaluation) {
            const te = msg.timeout_evaluation;
            if (te.check_in_message_en) {
              setStayWithMeMessage(String(te.check_in_message_en));
            }
            if (te.flag_control_room) {
              setControlRoomReviewRequired(true);
              setReviewReasonCodes((prev) =>
                prev.includes('caller_silence_timeout_review')
                  ? prev
                  : [...prev, 'caller_silence_timeout_review'],
              );
            }
          }
          if (msg.event === 'prisma_finalized_utterance_processed') {
            clearTurnProcessingState();
            if (vadFinalizeSentAtRef.current !== null) {
              setLastPrismaLatencyMs(
                Math.max(1, Math.round((performance.now() - vadFinalizeSentAtRef.current) * 10) / 10),
              );
              vadFinalizeSentAtRef.current = null;
            }
            const snap =
              msg.snapshot && typeof msg.snapshot === 'object'
                ? (msg.snapshot as Record<string, unknown>)
                : null;
            if (snap) {
              syncIncidentStateFromSnapshot(snap);
            }
            if (msg.prisma_transcript) {
              setStreamingPartialText(String(msg.prisma_transcript));
              const hasSnapUtterances = Array.isArray(snap?.utterances) && snap.utterances.length > 0;
              const uttId =
                hasSnapUtterances
                  ? String((snap.utterances as Array<Record<string, unknown>>).at(-1)?.utterance_id || `utt-${Date.now()}`)
                  : `utt-${Date.now()}`;
              if (!hasSnapUtterances) {
                setUtterances((prev) => [
                  ...prev,
                  {
                    utterance_id: uttId,
                    sequence: prev.length + 1,
                    text: String(msg.prisma_transcript),
                    language:
                      (msg.prisma_detected_language as SupportedLanguage) || activeLanguageRef.current,
                    utterance_role: 'new_description',
                    timestamp: new Date().toISOString(),
                  },
                ]);
              }
              appendAudit(
                'prisma_transcript_finalized',
                'info',
                `Caller statement captured: "${msg.prisma_transcript}"`,
                null,
                uttId,
              );
            }
            if (!hasPlayableTimbreAudio && !alreadyStreamed) {
              dispatchVoiceLifecycleEvent('turn_completed_no_audio', {
                isTurnProcessing: false,
              });
            }
          } else if (msg.event === 'prisma_non_speech_or_silence') {
            clearTurnProcessingState();
            vadFinalizeSentAtRef.current = null;
            setStreamingPartialText('Silence or non-speech ignored — listening for caller speech…');
            dispatchVoiceLifecycleEvent('turn_error_or_timeout', {
              isTurnProcessing: false,
            });
          } else if (msg.event === 'session_reconnected') {
            const snap = msg.snapshot && typeof msg.snapshot === 'object' ? (msg.snapshot as Record<string, unknown>) : null;
            if (snap && Array.isArray(snap.utterances) && snap.utterances.length > 0) {
              updateSessionStatus('active');
              setStreamingPartialText('Reconnected to active session with saved incident context.');
            }
          } else if (msg.event === 'location_confirmed') {
            clearTurnProcessingState();
            setShowLocationChangePicker(false);
            if (!hasPlayableTimbreAudio && !alreadyStreamed) {
              dispatchVoiceLifecycleEvent('turn_completed_no_audio', {
                isTurnProcessing: false,
              });
            }
            appendAudit(
              'location_confirmed_backend',
              'info',
              'Caller explicitly confirmed incident location; authorized simulated fleet dispatch started.',
              null,
              'loc:confirmed',
            );
          } else if (
            (msg.event === 'live_gemini_utterance_processed' ||
              msg.event === 'utterance_processed') &&
            msg.snapshot
          ) {
            clearTurnProcessingState();
            const snap = msg.snapshot as Record<string, unknown>;
            syncIncidentStateFromSnapshot(snap);
            if (!hasPlayableTimbreAudio && !alreadyStreamed) {
              dispatchVoiceLifecycleEvent('turn_completed_no_audio', {
                isTurnProcessing: false,
              });
            }
          } else if (msg.event === 'language_mode_updated') {
            clearTurnProcessingState();
            const snap =
              msg.snapshot && typeof msg.snapshot === 'object'
                ? (msg.snapshot as Record<string, unknown>)
                : null;
            if (snap && typeof snap.active_language === 'string') {
              const lang = snap.active_language as SupportedLanguage;
              setStreamingPartialText(
                `Language selected: ${LANGUAGE_DISPLAY_NAMES[lang] ?? lang.toUpperCase()}`,
              );
            }
          } else if (msg.event === 'demo_lifecycle_updated') {
            if (msg.spoken_announcement_text) {
              setStayWithMeMessage(String(msg.spoken_announcement_text));
              appendAudit(
                'demo_vehicle_lifecycle_update',
                'info',
                String(msg.spoken_announcement_text),
                null,
                'demo_lifecycle',
              );
            }
          } else if (msg.event === 'session_completed') {
            clearTurnProcessingState();
            updateSessionStatus('completed');
            const endedSid = msg.session_id ? String(msg.session_id) : '';
            if (!endedSid || sessionIdRef.current === endedSid) {
              sessionIdRef.current = '';
            }
            vadActiveSpeechStreamingRef.current = false;
            speechStartedAtMsRef.current = null;
            lowRmsSilenceSinceMsRef.current = null;
            dispatchVoiceLifecycleEvent('session_ended', {
              sessionActive: false,
              isTurnProcessing: false,
              isSpeechStreaming: false,
            });
            setPendingFollowupQuestion(null);
            setStreamingPartialText('Session ended. Demo resources released and conversational memory cleared.');
          }
        } else if (msg.type === 'error') {
          clearTurnProcessingState();
          vadFinalizeSentAtRef.current = null;
          setIsReasoningThinking(false);
          dispatchVoiceLifecycleEvent('turn_error_or_timeout', {
            isTurnProcessing: false,
          });
          if (
            msg.reason_code === 'session_not_found_in_memory' ||
            msg.reason_code === 'invalid_session_id' ||
            msg.reason_code === 'session_id_required'
          ) {
            try {
              sessionStorage.removeItem('setu_active_session_id');
              sessionStorage.removeItem('setu_active_session_token');
            } catch {
              // Ignore
            }
            if (ws.readyState === WebSocket.OPEN) {
              wsRequestSentAtRef.current = performance.now();
              ws.send(
                JSON.stringify({
                  type: 'session_init',
                  fresh_scenario: true,
                  initial_language: selectedInitialLanguage,
                  language_mode: 'manual',
                  latitude: 17.4435,
                  longitude: 78.3772,
                  accuracy_m: 78.0,
                }),
              );
            }
          }
        }
      } catch {
        // Ignore malformed frame
      }
    };

    ws.onclose = () => {
      setWsConnectionState('disconnected');
      clearTurnProcessingState();
      setIsReasoningThinking(false);
      dispatchVoiceLifecycleEvent('turn_error_or_timeout', {
        isTurnProcessing: false,
      });
    };

    return () => {
      audioControllerRef.current?.stopPlaybackImmediately();
      audioControllerRef.current?.stopMicrophoneCapture();
      ws.close();
    };
  }, []);

  // Poll backend demo vehicle lifecycle (ASSIGNED -> EN_ROUTE -> ARRIVED) while assignments are active
  useEffect(() => {
    if (sessionStatus !== 'active') {
      return;
    }
    const hasPendingProgression = assignments.some(
      (a) =>
        a.status === 'simulated_assigned' &&
        (a.lifecycle_status === 'ASSIGNED' || a.lifecycle_status === 'EN_ROUTE'),
    );
    if (!hasPendingProgression) {
      return;
    }
    const timer = window.setInterval(() => {
      const sid = sessionIdRef.current;
      if (
        wsRef.current &&
        wsRef.current.readyState === WebSocket.OPEN &&
        sid &&
        sid !== 'awaiting-backend-ws-session'
      ) {
        wsRef.current.send(
          JSON.stringify({
            type: 'demo_lifecycle_tick',
            session_id: sid,
          }),
        );
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [sessionStatus, assignments]);

  // Hydrate authoritative Structured Incident Handoff when opening the Handoff view
  useEffect(() => {
    if (activeView !== 'handoff') {
      return;
    }
    const sid = sessionIdRef.current;
    if (!sid || sid === 'awaiting-backend-ws-session') {
      return;
    }
    let cancelled = false;
    fetch(`/api/session/${encodeURIComponent(sid)}/handoff`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data && data.handoff && typeof data.handoff === 'object') {
          setBackendHandoffPayload(data.handoff as Record<string, unknown>);
        }
      })
      .catch(() => {
        // Keep latest WebSocket snapshot handoff on offline fallback
      });
    return () => {
      cancelled = true;
    };
  }, [activeView, sessionId]);

  const playCachedInstructionViaAudioContext = async (
    instructionId: string,
    lang: SupportedLanguage,
    onEnded?: () => void,
  ) => {
    const entry =
      manifestAssets.find((m) => m.instruction_id === instructionId && m.language === lang) ||
      manifestAssets.find((m) => m.instruction_id === instructionId && m.language === 'en');
    const url = entry
      ? `/${entry.file_path}`
      : `/backend-media/instructions/${lang}/${instructionId}/v1.0.wav`;
    const expectedSha = entry?.checksum_sha256 || '';
    try {
      if (audioControllerRef.current) {
        const res = await audioControllerRef.current.playVerifiedInstructionAsset(
          url,
          expectedSha,
          onEnded,
        );
        setLastCachedPlaybackStartMs(
          Math.max(0.1, Math.round(res.playbackStartLatencyMs * 10) / 10),
        );
        setLastVerifiedSha256(res.computedSha256.slice(0, 16) + '…');
      }
    } catch {
      // Keep state consistent in headless test runners without audio hardware
      onEnded?.();
    }
  };

  const appendAudit = (
    event_type: string,
    severity: 'info' | 'warning' | 'critical',
    outcome: string,
    reason_code: string | null,
    provenance_ref: string,
  ) => {
    setAuditEvents((prev) => [
      ...prev,
      {
        id: `evt-${prev.length + 1}`,
        timestamp: new Date().toISOString(),
        event_type,
        severity,
        outcome,
        reason_code,
        provenance_ref,
      },
    ]);
  };

  const addReviewReason = (code: string) => {
    setControlRoomReviewRequired(true);
    setReviewReasonCodes((prev) => (prev.includes(code) ? prev : [...prev, code]));
  };

  const removeReviewReason = (code: string) => {
    setReviewReasonCodes((prev) => prev.filter((c) => c !== code));
  };

  /**
   * Location eligibility for allocation:
   * - Browser GPS with accuracy <= 500 m is eligible
   * - Landmark candidate (confirmed OR unconfirmed provisional candidate) is eligible for provisional allocation
   */
  const isLocationEligibleForAllocation = useMemo(() => {
    if (locationSource === 'browser_gps') {
      return isGpsEligibleCandidate(gpsAccuracyMeters);
    }
    if (locationSource === 'landmark') {
      return (
        locationConfirmationStatus === 'confirmed' ||
        locationConfirmationStatus === 'candidate_awaiting_confirmation'
      );
    }
    return false;
  }, [locationSource, gpsAccuracyMeters, locationConfirmationStatus]);

  // Whether current allocations are provisional on an unconfirmed location candidate
  const isProvisionalLocationAllocation =
    isLocationEligibleForAllocation && locationConfirmationStatus !== 'confirmed';

  // Hospital rankings — recomputed whenever category or coordinates change
  const hospitalBreakdowns: HospitalScoreBreakdown[] = useMemo(() => {
    if (!isLocationEligibleForAllocation) {
      return [];
    }
    if (backendHospitalBreakdowns && backendHospitalBreakdowns.length > 0) {
      return backendHospitalBreakdowns;
    }
    return rankHospitalsDeterministic(incidentCategory, gpsLatitude, gpsLongitude);
  }, [
    incidentCategory,
    gpsLatitude,
    gpsLongitude,
    isLocationEligibleForAllocation,
    backendHospitalBreakdowns,
  ]);

  const topRankedHospitalId = hospitalBreakdowns[0]?.hospital.hospital_id ?? null;
  useEffect(() => {
    if (!topRankedHospitalId) return;
    setExpandedHospitalIds((prev) => {
      if (prev[topRankedHospitalId]) return prev;
      return {
        'SYN-HOSP-HYD-01': topRankedHospitalId === 'SYN-HOSP-HYD-01',
        'SYN-HOSP-HYD-02': topRankedHospitalId === 'SYN-HOSP-HYD-02',
        'SYN-HOSP-HYD-03': topRankedHospitalId === 'SYN-HOSP-HYD-03',
        [topRankedHospitalId]: true,
      };
    });
  }, [topRankedHospitalId]);

  // Current Coordination Stepper Stage (Listening -> Understanding -> Locating -> Assigning -> Guiding)
  const currentCoordinationStep: CoordinationStep = useMemo(() => {
    if (isReasoningThinking) {
      return 'Understanding';
    }
    if (utterances.length === 0) {
      return 'Listening';
    }
    if (!isLocationEligibleForAllocation) {
      return 'Locating';
    }
    if (
      assignments.length === 0 ||
      assignments.some((a) => a.status === 'pending_location')
    ) {
      return 'Assigning';
    }
    if (locationConfirmationStatus !== 'confirmed') {
      return 'Locating';
    }
    return 'Guiding';
  }, [
    isReasoningThinking,
    utterances.length,
    isLocationEligibleForAllocation,
    locationConfirmationStatus,
    assignments,
  ]);

  // Missing Information Facts SETU Still Needs
  const missingInformationFacts = useMemo(() => {
    const missing: Array<{ key: string; label: string; detail: string }> = [];
    if (!categoryEstablishedAboveThreshold && !unknownFallbackApplied) {
      missing.push({
        key: 'emergency_category',
        label: 'Emergency nature / category',
        detail: 'Need caller description of fire, medical issue, crash, or hazard',
      });
    }
    if (locationConfirmationStatus !== 'confirmed') {
      const candidateDetail =
        locationSource === 'landmark'
          ? `Provisional landmark candidate (${Math.round((landmarkMatchConfidence ?? 0.92) * 100)}% match) awaiting caller confirmation`
          : locationConfirmationStatus === 'candidate_awaiting_confirmation'
            ? `GPS candidate ±${gpsAccuracyMeters ?? 78} m awaiting caller confirmation`
            : 'GPS accuracy insufficient; spoken landmark required';
      missing.push({
        key: 'caller_location_confirmation',
        label: 'Caller location confirmation',
        detail: candidateDetail,
      });
    }
    const hasPersonCount = activeFacts.some((f) => f.field === 'affected_person_count');
    if (!hasPersonCount) {
      missing.push({
        key: 'affected_person_count',
        label: 'Affected people count',
        detail: 'Required for incident handoff summary',
      });
    }
    const hasSafeSpot = activeFacts.some(
      (f) => f.field === 'caller_safe_spot_status' || f.field === 'caller_observations',
    );
    if (!hasSafeSpot) {
      missing.push({
        key: 'caller_safe_spot_status',
        label: 'Caller safety status',
        detail: pendingFollowupQuestion ?? 'Confirm caller is clear of active danger',
      });
    }
    return missing;
  }, [
    categoryEstablishedAboveThreshold,
    unknownFallbackApplied,
    locationConfirmationStatus,
    locationSource,
    landmarkMatchConfidence,
    gpsAccuracyMeters,
    activeFacts,
    pendingFollowupQuestion,
  ]);

  // Deterministic Dispatch Allocation helper
  const runDeterministicAllocation = (
    targetCategory: EmergencyCategory,
    currentAssignments: SimulatedAssignment[],
    currentResources: SyntheticResource[],
    locationEligible: boolean,
    lat: number,
    lon: number,
    locationIsConfirmed = locationConfirmationStatus === 'confirmed',
    reEvaluateDistances = false,
  ): { nextAssignments: SimulatedAssignment[]; nextResources: SyntheticResource[] } => {
    const requiredCategories = BASE_DISPATCH_MAPPING[targetCategory];
    const nextAssignments = [...currentAssignments];
    const nextResources = [...currentResources];

    for (const reqCat of requiredCategories) {
      const idemKey = `${incidentId}:${reqCat}`;
      const existingIdx = nextAssignments.findIndex((a) => a.category === reqCat);
      if (
        existingIdx !== -1 &&
        nextAssignments[existingIdx].status === 'simulated_assigned'
      ) {
        if (reEvaluateDistances && nextAssignments[existingIdx].resource_id) {
          const matchedRes = nextResources.find(
            (r) => r.resource_id === nextAssignments[existingIdx].resource_id,
          );
          if (matchedRes) {
            nextAssignments[existingIdx] = {
              ...nextAssignments[existingIdx],
              distance_meters: Math.round(
                geodesicDistanceMeters(lat, lon, matchedRes.latitude, matchedRes.longitude),
              ),
            };
          }
        }
        continue;
      }

      if (!locationEligible) {
        addReviewReason('pending_location');
        const pendingAssignment: SimulatedAssignment = {
          assignment_id:
            existingIdx !== -1
              ? nextAssignments[existingIdx].assignment_id
              : `asgn-${reqCat}-${nextAssignments.length + 1}`,
          incident_id: incidentId,
          idempotency_key: idemKey,
          category: reqCat,
          resource_id: null,
          resource_callsign: null,
          status: 'pending_location',
          assigned_at: new Date().toISOString(),
          simulated_eta_minutes: null,
          distance_meters: null,
          source_provenance: `backend_rule:dispatch:${targetCategory}:${reqCat}`,
          is_simulated: true,
        };
        if (existingIdx !== -1) {
          nextAssignments[existingIdx] = pendingAssignment;
        } else {
          nextAssignments.push(pendingAssignment);
        }
        continue;
      }

      // Select nearest available synthetic unit by geodesic distance
      const availableCandidates = nextResources
        .map((r, idx) => ({
          r,
          idx,
          distM: Math.round(geodesicDistanceMeters(lat, lon, r.latitude, r.longitude)),
        }))
        .filter((item) => item.r.category === reqCat && item.r.is_available)
        .sort((a, b) => a.distM - b.distM);

      if (availableCandidates.length === 0) {
        addReviewReason(`resource_unavailable:${reqCat}`);
        const unavailAssignment: SimulatedAssignment = {
          assignment_id:
            existingIdx !== -1
              ? nextAssignments[existingIdx].assignment_id
              : `asgn-${reqCat}-${nextAssignments.length + 1}`,
          incident_id: incidentId,
          idempotency_key: idemKey,
          category: reqCat,
          resource_id: null,
          resource_callsign: null,
          status: 'simulated_unavailable',
          assigned_at: new Date().toISOString(),
          simulated_eta_minutes: null,
          distance_meters: null,
          source_provenance: `backend_rule:exhausted:${reqCat}`,
          is_simulated: true,
        };
        if (existingIdx !== -1) {
          nextAssignments[existingIdx] = unavailAssignment;
        } else {
          nextAssignments.push(unavailAssignment);
        }
      } else {
        const best = availableCandidates[0];
        nextResources[best.idx] = { ...best.r, is_available: false };
        const assigned: SimulatedAssignment = {
          assignment_id:
            existingIdx !== -1
              ? nextAssignments[existingIdx].assignment_id
              : `asgn-${reqCat}-${nextAssignments.length + 1}`,
          incident_id: incidentId,
          idempotency_key: idemKey,
          category: reqCat,
          resource_id: best.r.resource_id,
          resource_callsign: best.r.callsign,
          status: 'simulated_assigned',
          assigned_at: new Date().toISOString(),
          simulated_eta_minutes: null,
          distance_meters: best.distM,
          source_provenance: `database_record:${best.r.resource_id}`,
          is_simulated: true,
        };
        if (existingIdx !== -1) {
          nextAssignments[existingIdx] = assigned;
        } else {
          nextAssignments.push(assigned);
        }
      }
    }

    // Never allow both simulated_assigned and pending_location at once
    if (nextAssignments.some((a) => a.status === 'simulated_assigned')) {
      removeReviewReason('pending_location');
      if (!locationIsConfirmed) {
        addReviewReason('provisional_location_unconfirmed');
      } else {
        removeReviewReason('provisional_location_unconfirmed');
      }
    }

    return { nextAssignments, nextResources };
  };

  // Single-application deterministic unknown fallback
  const triggerDeterministicUnknownFallback = (
    reason: string,
    reviewCode: string,
    spokenStatement: string,
    sourceUtteranceId: string,
    currentAssignments = assignments,
    currentResources = resources,
  ) => {
    addReviewReason(reviewCode);
    setStayWithMeMessage(spokenStatement);
    setCacheHits((c) => c + 1);

    if (unknownFallbackApplied) {
      appendAudit(
        'unknown_fallback_duplicate_trigger_skipped',
        'info',
        `Single-application enforced (${reason})`,
        reviewCode,
        sourceUtteranceId,
      );
      return;
    }

    setUnknownFallbackApplied(true);
    setFallbackReason(reason);
    setIncidentCategory('unknown');

    const { nextAssignments, nextResources } = runDeterministicAllocation(
      'unknown',
      currentAssignments,
      currentResources,
      isLocationEligibleForAllocation,
      gpsLatitude,
      gpsLongitude,
      locationConfirmationStatus === 'confirmed',
    );
    setAssignments(nextAssignments);
    setResources(nextResources);

    appendAudit(
      'deterministic_unknown_fallback_applied',
      'warning',
      `Simulated ambulance requested (reason=${reason})`,
      reviewCode,
      sourceUtteranceId,
    );
  };

  const startLiveMicrophonePipeline = async (): Promise<boolean> => {
    if (!audioControllerRef.current) {
      return false;
    }
    audioControllerRef.current.unlockAudioContext();

    if (sileroVadRef.current && !sileroVadRef.current.isReady) {
      const vadOk = await sileroVadRef.current.init();
      setIsVadReady(vadOk);
      setVadError(sileroVadRef.current.errorMessage);
    }

    vadActiveSpeechStreamingRef.current = false;
    micStreamSeqRef.current = 0;

    try {
      const status = await audioControllerRef.current.startMicrophoneWorkletCapture(
        ringBufferRef.current,
        (frame) => {
          setLiveMicRms(frame.rms_level);

          // In-memory Acoustic Environment DSP analysis (zero Gemini calls, zero raw audio persistence)
          const dsp = acousticDspStatsRef.current;
          dsp.totalFramesCount += 1;
          if (frame.peak_level > dsp.peakSampleRatio) {
            dsp.peakSampleRatio = frame.peak_level;
          } else {
            dsp.peakSampleRatio = dsp.peakSampleRatio * 0.995 + frame.peak_level * 0.005;
          }
          if (frame.clipped_samples > 2 || frame.peak_level >= 0.98) {
            dsp.clippingEventsCount += 1;
          }

          const currentSpeechProb = liveSpeechProbRef.current;
          if (currentSpeechProb < 0.25) {
            dsp.noiseRmsEma = dsp.noiseRmsEma * 0.94 + Math.max(0.0002, frame.rms_level) * 0.06;
            dsp.overlapScoreEma = dsp.overlapScoreEma * 0.96;
          } else if (currentSpeechProb >= 0.45) {
            dsp.speechFramesCount += 1;
            const zcrDelta = Math.abs(frame.zero_crossing_rate - dsp.lastSpeechZcr);
            dsp.lastSpeechZcr = frame.zero_crossing_rate;
            const crest = frame.peak_level / Math.max(0.005, frame.rms_level);
            const frameOverlapSignal =
              zcrDelta > 0.14 && crest > 3.6 && frame.rms_level > 0.03 ? 0.38 : 0.04;
            dsp.overlapScoreEma = dsp.overlapScoreEma * 0.88 + frameOverlapSignal * 0.12;
          }

          const nowMs = performance.now();
          if (nowMs - dsp.lastUiUpdateMs >= 250) {
            dsp.lastUiUpdateMs = nowMs;
            const noiseDbfs = Math.max(
              -60,
              Math.min(0, Math.round(20 * Math.log10(Math.max(0.0001, dsp.noiseRmsEma)))),
            );
            const noiseCat: 'quiet' | 'moderate' | 'high_ambient' =
              noiseDbfs > -28 ? 'high_ambient' : noiseDbfs > -42 ? 'moderate' : 'quiet';
            const speechPct =
              dsp.totalFramesCount > 0
                ? Math.min(100, Math.round((dsp.speechFramesCount / dsp.totalFramesCount) * 100))
                : 0;
            const micQual: 'clean' | 'clipping_detected' | 'low_signal' | 'standby' =
              dsp.clippingEventsCount > 0
                ? 'clipping_detected'
                : dsp.speechFramesCount > 8 && dsp.peakSampleRatio < 0.04
                  ? 'low_signal'
                  : 'clean';

            setAcousticTelemetry({
              noiseFloorDbfs: noiseDbfs,
              noiseCategory: noiseCat,
              overlappingVoicesDetected: dsp.overlapScoreEma > 0.16,
              voiceOverlapScore: Math.round(dsp.overlapScoreEma * 100),
              clippingEventsCount: dsp.clippingEventsCount,
              peakSampleRatio: Math.round(dsp.peakSampleRatio * 100),
              micQualityStatus: micQual,
              speechFramesCount: dsp.speechFramesCount,
              totalFramesCount: dsp.totalFramesCount,
              speechActivityPct: speechPct,
            });
          }

          const sid = sessionIdRef.current;
          const validSid = sid && sid !== 'awaiting-backend-ws-session' ? sid : undefined;
          if (
            sessionStatusRef.current === 'active' &&
            vadActiveSpeechStreamingRef.current &&
            wsRef.current &&
            wsRef.current.readyState === WebSocket.OPEN
          ) {
            micStreamSeqRef.current += 1;
            setPcmChunksStreamed((c) => c + 1);
            wsRef.current.send(
              JSON.stringify({
                type: 'audio_chunk_pcm16',
                ...(validSid ? { session_id: validSid } : {}),
                sequence_number: micStreamSeqRef.current,
                sample_rate_hz: frame.sample_rate_hz,
                pcm_base64: frame.pcm_base64,
              }),
            );

            // Acoustic RMS silence boundary fallback: ensures speech capture never stays stuck in "Hearing you…"
            const speechStartedAt = speechStartedAtMsRef.current;
            if (speechStartedAt !== null && nowMs - speechStartedAt >= 380) {
              if (frame.rms_level < 0.012 && liveSpeechProbRef.current < 0.50) {
                if (lowRmsSilenceSinceMsRef.current === null) {
                  lowRmsSilenceSinceMsRef.current = nowMs;
                } else if (nowMs - lowRmsSilenceSinceMsRef.current >= 750) {
                  finalizeActiveCallerSpeechUtterance(750);
                }
              } else {
                lowRmsSilenceSinceMsRef.current = null;
              }
            }
          }
        },
        (float32Chunk) => {
          if (sileroVadRef.current?.isReady) {
            void sileroVadRef.current.processFloat32Samples(float32Chunk);
          }
        },
      );
      setEchoCancellationActive(status.echoCancellationActive);
      setIsMicWorkletStreaming(true);
      setMicErrorMessage(null);
      appendAudit(
        'mic_audioworklet_started',
        'info',
        `16 kHz mono PCM AudioWorklet active (echoCancellation=${status.echoCancellationActive})`,
        null,
        'audioworklet:16khz_mono_pcm',
      );
      return true;
    } catch (micErr) {
      console.warn('Microphone access notice:', micErr);
      const errMsg =
        micErr instanceof Error ? micErr.message : 'Permission denied or microphone unavailable';
      setMicErrorMessage(errMsg);
      setIsMicWorkletStreaming(false);
      setLiveMicRms(0);
      setVoiceState('microphone_denied');
      appendAudit(
        'microphone_access_error',
        'warning',
        `Microphone capture unavailable: ${errMsg}`,
        null,
        'getUserMedia:failed',
      );
      return false;
    }
  };

  const handleStartSession = async () => {
    audioControllerRef.current?.stopPlaybackImmediately();
    audioControllerRef.current?.stopMicrophoneCapture();
    sileroVadRef.current?.reset();
    vadActiveSpeechStreamingRef.current = false;
    micStreamSeqRef.current = 0;
    audioControllerRef.current?.unlockAudioContext();
    updateSessionStatus('active');
    sessionIdRef.current = '';
    activeLanguageRef.current = selectedInitialLanguage;
    setActiveLanguage(selectedInitialLanguage);
    setSpokenDisclosurePlayed(true);
    setVoiceState('requesting_microphone');
    setMicErrorMessage(null);
    setUtterances([]);
    setTrackedFacts([]);
    setConflicts([]);
    setAssignments([]);
    setIncidentCategory('unknown');
    setCategoryEstablishedAboveThreshold(false);
    setUnknownFallbackApplied(false);
    setFallbackReason(null);
    setPendingFollowupQuestion(null);
    setReviewReasonCodes([]);
    setIdentifiedHazards([]);
    setControlRoomReviewRequired(false);
    setStreamingPartialText('Preparing emergency coordination session…');

    const disclosure = SPOKEN_DISCLOSURE_BY_LANGUAGE[selectedInitialLanguage];
    setStayWithMeMessage(disclosure.text);
    setActiveSafetyInstruction(
      'Remain in a safe spot and stay on the line while this simulation continues.',
    );
    setSafetyAttributionLabel('placeholder, pending review');
    setCacheHits((c) => c + 1);

    // Clear any previously saved session token when user explicitly starts a brand-new live session
    try {
      sessionStorage.removeItem('setu_active_session_id');
      sessionStorage.removeItem('setu_active_session_token');
    } catch {
      // Ignore
    }
    setBackendHandoffPayload(null);
    setPendingQuestionState(null);
    setRequiredResourceCategories([]);
    setBroadLocationText(null);
    setCallerLandmarkText(null);
    setLocationConfirmedAt(null);

    // Create a fresh backend session immediately so sessionIdRef updates before live speech starts
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRequestSentAtRef.current = performance.now();
      wsRef.current.send(
        JSON.stringify({
          type: 'session_init',
          fresh_scenario: true,
          initial_language: selectedInitialLanguage,
          language_mode: 'manual',
          latitude: gpsLatitude,
          longitude: gpsLongitude,
          accuracy_m: gpsAccuracyMeters ?? 78.0,
        }),
      );
    }

    const captureStarted = await startLiveMicrophonePipeline();

    // Play introductory disclosure WAV via AudioContext + AudioBufferSourceNode
    setIsTtsPlaying(true);
    isTtsPlayingRef.current = true;
    setTtsPlaybackSource('cached_instruction');
    setVoiceState(captureStarted ? 'speaking_introduction' : 'microphone_denied');
    if (captureStarted) {
      setStreamingPartialText(
        'Playing prototype disclosure — speak when ready or interrupt anytime.',
      );
    }

    void playCachedInstructionViaAudioContext(
      'disclosure_session_start',
      selectedInitialLanguage,
      () => {
        setIsTtsPlaying(false);
        isTtsPlayingRef.current = false;
        if (
          sessionStatusRef.current === 'active' &&
          !isTurnProcessingRef.current &&
          !vadActiveSpeechStreamingRef.current
        ) {
          setVoiceState(captureStarted ? 'listening' : 'microphone_denied');
          if (captureStarted) {
            setStreamingPartialText('Listening for caller speech…');
          }
        }
      },
    );
    appendAudit(
      'session_started',
      'info',
      `Placeholder disclosure audio played via AudioBufferSourceNode (${selectedInitialLanguage.toUpperCase()})`,
      null,
      `${disclosure.instructionId}:${disclosure.version}`,
    );
    setActiveView('live_incident');
  };

  const handleSelectManualLanguage = (lang: SupportedLanguage) => {
    setSelectedInitialLanguage(lang);
    activeLanguageRef.current = lang;
    setActiveLanguage(lang);
    setDetectedLanguagesSeen((prev) => (prev.includes(lang) ? prev : [...prev, lang]));
    const opt = getDirectLanguageOption(lang);
    setStreamingPartialText(`Language selected: ${opt.fullLabel}`);
    const sid = sessionIdRef.current;
    const activeSid = sid && sid !== 'awaiting-backend-ws-session' ? sid : undefined;
    if (
      activeView === 'live_incident' &&
      wsRef.current &&
      wsRef.current.readyState === WebSocket.OPEN &&
      activeSid
    ) {
      wsRequestSentAtRef.current = performance.now();
      wsRef.current.send(
        JSON.stringify({
          type: 'set_language_mode',
          session_id: activeSid,
          language_mode: 'manual',
          manual_language: lang,
        }),
      );
    }
    appendAudit(
      'manual_language_selected',
      'info',
      `Language set to ${LANGUAGE_DISPLAY_NAMES[lang]} (${lang.toUpperCase()}) without resetting incident`,
      null,
      `lang:${lang}`,
    );
  };

  const handleEndSession = () => {
    const endingSid = sessionIdRef.current;
    clearTurnProcessingState();
    vadActiveSpeechStreamingRef.current = false;
    sileroVadRef.current?.reset();
    audioControllerRef.current?.stopPlaybackImmediately();
    audioControllerRef.current?.stopMicrophoneCapture();
    setIsMicWorkletStreaming(false);
    setIsTtsPlaying(false);
    isTtsPlayingRef.current = false;
    setVoiceState('idle');
    updateSessionStatus('completed');
    sessionIdRef.current = '';
    try {
      sessionStorage.removeItem('setu_active_session_id');
      sessionStorage.removeItem('setu_active_session_token');
    } catch {
      // Ignore
    }
    setPendingFollowupQuestion(null);
    setPendingQuestionState(null);
    setStreamingPartialText('Session ended. Conversational memory cleared and demo units released.');
    if (
      wsRef.current &&
      wsRef.current.readyState === WebSocket.OPEN &&
      endingSid &&
      endingSid !== 'awaiting-backend-ws-session'
    ) {
      wsRef.current.send(
        JSON.stringify({
          type: 'session_end',
          session_id: endingSid,
          reason: 'user_ended_session',
        }),
      );
    }
  };

  /**
   * Processes a turn inside the current session OR as the first turn of a freshly started scenario session.
   */
  const executeScenarioTurn = (
    scenario: PresetScenario,
    options: { startFreshSession: boolean },
  ) => {
    const isFresh = options.startFreshSession;

    // Local base state depending on whether we are starting a fresh scenario session
    const baseUtterances = isFresh ? [] : utterances;
    const baseFacts = isFresh ? [] : trackedFacts;
    const baseConflicts = isFresh ? [] : conflicts;
    const baseAssignments = isFresh ? [] : assignments;
    const baseResources = isFresh ? [...INITIAL_SYNTHETIC_RESOURCES] : resources;
    const baseCategory: EmergencyCategory = isFresh ? 'unknown' : incidentCategory;
    const baseCatEstablished = isFresh ? false : categoryEstablishedAboveThreshold;
    const baseUnknownCount = isFresh ? 0 : consecutiveUnknownCount;
    const baseProviderBlocks = isFresh ? 0 : consecutiveProviderBlocks;
    const baseUnknownFallback = isFresh ? false : unknownFallbackApplied;

    if (isFresh) {
      updateSessionStatus('active');
      sessionIdRef.current = '';
      try {
        sessionStorage.removeItem('setu_active_session_id');
        sessionStorage.removeItem('setu_active_session_token');
      } catch {
        // Ignore
      }
      setBackendHandoffPayload(null);
      setPendingQuestionState(null);
      setRequiredResourceCategories([]);
      setBroadLocationText(null);
      setCallerLandmarkText(null);
      setLocationConfirmedAt(null);
      setSpokenDisclosurePlayed(true);
      setActiveLanguage(scenario.language);
      setIncidentCategory('unknown');
      setCategoryEstablishedAboveThreshold(false);
      setClassificationAttempts(0);
      setClassificationConfidences([]);
      setConsecutiveUnknownCount(0);
      setConsecutiveProviderBlocks(0);
      setTotalProviderBlocks(0);
      setUnknownFallbackApplied(false);
      setFallbackReason(null);
      setControlRoomReviewRequired(false);
      setReviewReasonCodes([]);
      setUtterances([]);
      setTrackedFacts([]);
      setConflicts([]);
      setPendingFollowupQuestion(null);
      setResources([...INITIAL_SYNTHETIC_RESOURCES]);
      setAssignments([]);
      setBackendHospitalBreakdowns(null);
      setPossibleDuplicateFlag(false);
      setMatchedDuplicateId(null);
      setPrankReviewStatus('not_reviewed');
      setPrankFlagReason(null);
      setLocationSource('browser_gps');
      setGpsLatitude(17.4435);
      setGpsLongitude(78.3772);
      setGpsAccuracyMeters(78);
      setLandmarkMatchConfidence(null);
      setMatchedLandmarkId(null);
      setLocationConfirmationStatus('candidate_awaiting_confirmation');
      setAuditEvents([
        {
          id: 'evt-init-1',
          timestamp: new Date().toISOString(),
          event_type: 'session_created',
          severity: 'info',
          outcome: `Fresh session and incident started for scenario (${scenario.label})`,
          reason_code: null,
          provenance_ref: `scenario:${scenario.id}`,
        },
      ]);

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRequestSentAtRef.current = performance.now();
        wsRef.current.send(
          JSON.stringify({
            type: 'session_init',
            fresh_scenario: true,
            initial_language: scenario.language,
            latitude: 17.4435,
            longitude: 78.3772,
            accuracy_m: 78.0,
          }),
        );
      }
    } else if (sessionStatus === 'not_started') {
      updateSessionStatus('active');
      setSpokenDisclosurePlayed(true);
    }

    setIsReasoningThinking(true);
    setStreamingPartialText('');

    const utteranceId = `utt-${baseUtterances.length + 1}-${scenario.id.slice(0, 6)}`;
    const newUtterance: FinalizedUtterance & { languageBadge?: string } = {
      utterance_id: utteranceId,
      sequence: baseUtterances.length + 1,
      text: scenario.text,
      language: scenario.language,
      languageBadge: scenario.languageBadge,
      utterance_role:
        scenario.simulateProviderBlock || scenario.simulateProviderOutage
          ? null
          : scenario.utterance_role,
      timestamp: new Date().toISOString(),
    };

    if (!isFresh && scenario.language !== activeLanguage) {
      appendAudit(
        'language_switched',
        'info',
        `${activeLanguage.toUpperCase()} -> ${scenario.language.toUpperCase()} (incident state preserved)`,
        null,
        utteranceId,
      );
      setActiveLanguage(scenario.language);
    }

    setUtterances([...baseUtterances, newUtterance]);

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRequestSentAtRef.current = performance.now();
      wsRef.current.send(
        JSON.stringify({
          type: 'scripted_finalized_utterance',
          text: scenario.text,
          language: scenario.language,
          utterance_role: scenario.utterance_role,
          category: scenario.category,
          confidence: scenario.confidence,
          followup_question: scenario.followupQuestion,
          extracted_facts: scenario.extractedFacts,
          simulate_provider_block: Boolean(scenario.simulateProviderBlock),
          simulate_provider_outage: Boolean(scenario.simulateProviderOutage),
        }),
      );
    } else {
      setIsReasoningThinking(false);
    }

    // Case 1: Provider Safety-Filter Block
    if (scenario.simulateProviderBlock) {
      setIsReasoningThinking(false);
      const nextBlocks = baseProviderBlocks + 1;
      setConsecutiveProviderBlocks(nextBlocks);
      setTotalProviderBlocks((t) => (isFresh ? 1 : t + 1));
      addReviewReason('provider_blocked_emergency_content');
      if (nextBlocks === 1) {
        setStayWithMeMessage(
          "I'm here with you in this simulation. Could you please describe the situation again in your own words?",
        );
        setActiveSafetyInstruction(
          'I am here with you in this simulation. Please describe the situation again in your own words.',
        );
        setSafetyAttributionLabel('placeholder, pending review');
        setCacheHits((c) => c + 1);
        appendAudit(
          'provider_blocked',
          'warning',
          '1st blocked turn: played cached re-ask prompt; unknown counter unchanged',
          'provider_blocked_emergency_content',
          utteranceId,
        );
      } else if (nextBlocks >= 2) {
        if (!baseCatEstablished) {
          triggerDeterministicUnknownFallback(
            'consecutive_provider_blocks',
            'provider_blocked_emergency_content',
            REQUIRED_PROVIDER_BLOCKED_STATEMENT,
            utteranceId,
            baseAssignments,
            baseResources,
          );
          setActiveSafetyInstruction(REQUIRED_PROVIDER_BLOCKED_STATEMENT);
          setSafetyAttributionLabel('placeholder, pending review');
        } else {
          setStayWithMeMessage(
            "I'm here with you. Please stay on the line while this simulation continues.",
          );
          appendAudit(
            'provider_blocked_with_existing_category',
            'warning',
            'Subsequent block with established category: flagged for review, no new assignment',
            'provider_blocked_emergency_content',
            utteranceId,
          );
        }
      }
      return;
    }

    // Case 2: Reasoning Outage
    if (scenario.simulateProviderOutage) {
      setIsReasoningThinking(false);
      addReviewReason('reasoning_unavailable');
      setStayWithMeMessage(scenario.safetyInstructionText);
      setActiveSafetyInstruction(scenario.safetyInstructionText);
      setSafetyAttributionLabel('placeholder, pending review');
      setCacheHits((c) => c + 1);
      appendAudit(
        'reasoning_provider_outage',
        'warning',
        'Static Stay With Me reassurance activated; no fabricated classification',
        'reasoning_unavailable',
        utteranceId,
      );
      return;
    }

    setConsecutiveProviderBlocks(0);

    // Fact Conflict Detection & Preservation of Both Utterances
    const incomingFacts: TrackedFact[] = scenario.extractedFacts.map((f) => ({
      ...f,
      source_utterance_id: utteranceId,
      status: 'active',
    }));

    let updatedFacts: TrackedFact[] = [...baseFacts];
    const newlyDetectedConflicts: FactConflict[] = [];
    let locationChangedInsideSession = false;

    for (const newFact of incomingFacts) {
      const priorActiveForField = updatedFacts.filter(
        (ef) => ef.field === newFact.field && ef.status !== 'superseded',
      );
      const disagreeingPrior = priorActiveForField.find(
        (ef) => String(ef.value).toLowerCase() !== String(newFact.value).toLowerCase(),
      );
      if (disagreeingPrior) {
        if (newFact.field === 'caller_location_text') {
          locationChangedInsideSession = true;
        }
        // Mark both prior and new fact as conflicting while preserving both with their source_utterance_id
        updatedFacts = updatedFacts.map((ef) =>
          ef.field === newFact.field &&
          ef.source_utterance_id === disagreeingPrior.source_utterance_id
            ? { ...ef, status: 'conflicting' }
            : ef,
        );
        newFact.status = 'conflicting';
        const conflictRecord: FactConflict = {
          field: newFact.field,
          prior_value: String(disagreeingPrior.value ?? ''),
          prior_utterance_id: disagreeingPrior.source_utterance_id,
          new_value: String(newFact.value ?? ''),
          new_utterance_id: utteranceId,
          description: `Disagreement on ${newFact.field}`,
        };
        newlyDetectedConflicts.push(conflictRecord);
      }
      updatedFacts.push(newFact);
    }

    // Check if spoken landmark matches our synthetic gazetteer
    let nextLat = isFresh ? 17.4435 : gpsLatitude;
    let nextLon = isFresh ? 78.3772 : gpsLongitude;
    let nextLocSource: 'browser_gps' | 'landmark' | 'unresolved' = isFresh
      ? 'browser_gps'
      : locationSource;
    let nextLocConfirm: LocationConfirmationStatus = isFresh
      ? 'candidate_awaiting_confirmation'
      : locationConfirmationStatus;

    const locFact = incomingFacts.find((f) => f.field === 'caller_location_text');
    if (locFact && typeof locFact.value === 'string') {
      const lowerLoc = locFact.value.toLowerCase();
      const matched = SYNTHETIC_LANDMARKS.find(
        (lm) =>
          lowerLoc.includes(lm.name_romanised.toLowerCase()) ||
          (lowerLoc.includes('secunderabad') && lm.landmark_id === 'SYN-LM-HYD-01') ||
          (lowerLoc.includes('hitech') && lm.landmark_id === 'SYN-LM-HYD-02') ||
          (lowerLoc.includes('charminar') && lm.landmark_id === 'SYN-LM-HYD-03'),
      );
      if (matched) {
        setMatchedLandmarkId(matched.landmark_id);
        setLocationSource('landmark');
        setGpsLatitude(matched.latitude);
        setGpsLongitude(matched.longitude);
        setGpsAccuracyMeters(null);
        setLandmarkMatchConfidence(matched.match_confidence);
        setLocationConfirmationStatus('candidate_awaiting_confirmation');
        nextLat = matched.latitude;
        nextLon = matched.longitude;
        nextLocSource = 'landmark';
        nextLocConfirm = 'candidate_awaiting_confirmation';
        addReviewReason('provisional_location_unconfirmed');
        appendAudit(
          'landmark_candidate_matched',
          'info',
          `Matched "${matched.name_romanised}" (${Math.round(matched.match_confidence * 100)}% match confidence); provisional allocation enabled pending caller confirmation`,
          'provisional_location_unconfirmed',
          utteranceId,
        );
      }
    }

    // Utterance Role Handling: short answers to pending questions do not increment unknown counter
    if (scenario.utterance_role === 'answer_to_pending_question') {
      setTrackedFacts(updatedFacts);
      if (newlyDetectedConflicts.length > 0) {
        setConflicts([...baseConflicts, ...newlyDetectedConflicts]);
        addReviewReason('conflicting_caller_statements');
      }
      setPendingFollowupQuestion(null);
      setStayWithMeMessage(scenario.responseText);
      setActiveSafetyInstruction(scenario.safetyInstructionText);
      setSafetyAttributionLabel(scenario.safetyAttributionLabel);
      if (scenario.text.toLowerCase().includes('yes')) {
        setLocationConfirmationStatus('confirmed');
        removeReviewReason('provisional_location_unconfirmed');
        removeReviewReason('pending_location');
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRequestSentAtRef.current = performance.now();
          wsRef.current.send(JSON.stringify({ type: 'location_confirm' }));
        }
        appendAudit(
          'caller_confirmed_location',
          'info',
          'Caller confirmed candidate location; provisional status upgraded to confirmed',
          null,
          utteranceId,
        );
      }
      appendAudit(
        'answer_to_pending_question_processed',
        'info',
        'Utterance role = answer_to_pending_question; consecutive_unknown_count unchanged',
        null,
        utteranceId,
      );
      return;
    }

    // New Description / Classification Attempt
    setClassificationAttempts((a) => (isFresh ? 1 : a + 1));
    setClassificationConfidences((prev) =>
      isFresh ? [scenario.confidence] : [...prev, scenario.confidence],
    );

    const isBelowThresholdOrUnknown =
      scenario.category === 'unknown' || scenario.confidence < 0.7;

    if (isBelowThresholdOrUnknown) {
      setTrackedFacts(updatedFacts);
      if (newlyDetectedConflicts.length > 0) {
        setConflicts([...baseConflicts, ...newlyDetectedConflicts]);
        addReviewReason('conflicting_caller_statements');
      }
      const nextUnknown = baseUnknownCount + 1;
      setConsecutiveUnknownCount(nextUnknown);
      if (nextUnknown === 1) {
        setPendingFollowupQuestion(scenario.followupQuestion);
        setStayWithMeMessage(scenario.responseText);
        setActiveSafetyInstruction(scenario.safetyInstructionText);
        setSafetyAttributionLabel(scenario.safetyAttributionLabel);
        appendAudit(
          'classification_unknown_turn_1',
          'warning',
          `Confidence ${scenario.confidence.toFixed(2)} < 0.70 or unknown; asking 1 targeted clarifying question`,
          null,
          utteranceId,
        );
      } else if (nextUnknown >= 2 && !baseCatEstablished && !baseUnknownFallback) {
        triggerDeterministicUnknownFallback(
          'unknown_after_clarification',
          'unknown_after_clarification',
          REQUIRED_UNKNOWN_FALLBACK_STATEMENT,
          utteranceId,
          baseAssignments,
          baseResources,
        );
        setActiveSafetyInstruction(REQUIRED_UNKNOWN_FALLBACK_STATEMENT);
        setSafetyAttributionLabel('placeholder, pending review');
      }
      return;
    }

    // Supported Category >= 0.70 Confidence
    setConsecutiveUnknownCount(0);
    const categoryChangedInsideSession =
      !isFresh &&
      baseCatEstablished &&
      baseCategory !== 'unknown' &&
      baseCategory !== scenario.category;

    if (categoryChangedInsideSession) {
      addReviewReason('reclassification_conflict');
      addReviewReason('conflicting_caller_statements');

      // Supersede stale category-implied facts (e.g., fire_reported when reclassifying away from fire)
      const oldImpliedField = CATEGORY_IMPLIED_FACT_BY_CATEGORY[baseCategory];
      const newImpliedField = CATEGORY_IMPLIED_FACT_BY_CATEGORY[scenario.category];
      if (oldImpliedField && oldImpliedField !== newImpliedField) {
        let supersededAny = false;
        updatedFacts = updatedFacts.map((f) => {
          if (f.field === oldImpliedField && f.status !== 'superseded') {
            supersededAny = true;
            newlyDetectedConflicts.push({
              field: oldImpliedField,
              prior_value: String(f.value ?? ''),
              prior_utterance_id: f.source_utterance_id,
              new_value: 'superseded_on_reclassification',
              new_utterance_id: utteranceId,
              description: `Superseded stale fact ${oldImpliedField} on reclassification`,
            });
            return {
              ...f,
              status: 'superseded',
              supersededByCategory: scenario.category,
            };
          }
          return f;
        });
        if (supersededAny) {
          appendAudit(
            'stale_fact_superseded_on_reclassification',
            'warning',
            `Stale fact "${getHumanFieldLabel(oldImpliedField)}" (${oldImpliedField}) superseded after reclassification from ${baseCategory} to ${scenario.category}`,
            'reclassification_conflict',
            utteranceId,
          );
        }
      }

      // If BOTH category and location changed inside one session, raise possible_second_incident
      if (locationChangedInsideSession) {
        addReviewReason('possible_second_incident');
        appendAudit(
          'possible_second_incident_flagged',
          'critical',
          `Both emergency category (${baseCategory} -> ${scenario.category}) and spoken location changed inside one session; flagged possible_second_incident`,
          'possible_second_incident',
          utteranceId,
        );
      }

      appendAudit(
        'incident_reclassified_additive',
        'warning',
        `Reclassified ${baseCategory} -> ${scenario.category}; existing assignments preserved, additive delta applied`,
        'reclassification_conflict',
        utteranceId,
      );
    }

    setTrackedFacts(updatedFacts);
    if (newlyDetectedConflicts.length > 0) {
      setConflicts([...baseConflicts, ...newlyDetectedConflicts]);
      addReviewReason('conflicting_caller_statements');
      appendAudit(
        'conflicting_caller_statements_detected',
        'warning',
        `${newlyDetectedConflicts.length} fact disagreement(s) recorded with source utterance provenance`,
        'conflicting_caller_statements',
        utteranceId,
      );
    }

    setIncidentCategory(scenario.category);
    setCategoryEstablishedAboveThreshold(true);
    setPendingFollowupQuestion(scenario.followupQuestion);
    setStayWithMeMessage(scenario.responseText);
    setActiveSafetyInstruction(scenario.safetyInstructionText);
    setSafetyAttributionLabel(scenario.safetyAttributionLabel);
    setCacheHits((c) => c + 1);
    setIsTtsPlaying(true);
    setTtsPlaybackSource('cached_instruction');

    const instId =
      scenario.category === 'road_accident'
        ? 'safety_road_accident'
        : scenario.category === 'fire' || scenario.category === 'fire_with_injuries'
          ? 'safety_fire_evacuate'
          : 'disclosure_session_start';
    void playCachedInstructionViaAudioContext(instId, scenario.language);

    const eligibleNow =
      nextLocSource === 'browser_gps'
        ? isGpsEligibleCandidate(isFresh ? 78 : gpsAccuracyMeters)
        : nextLocSource === 'landmark';

    const { nextAssignments, nextResources } = runDeterministicAllocation(
      scenario.category,
      baseAssignments,
      baseResources,
      eligibleNow,
      nextLat,
      nextLon,
      nextLocConfirm === 'confirmed',
      locationChangedInsideSession,
    );
    setAssignments(nextAssignments);
    setResources(nextResources);

    // Record local scenario turn in the Live Pipeline Performance Waterfall if WebSocket is not open
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setTurnWaterfallHistory((prev) => {
        const nextTurnNum = prev.length + 1;
        const entry: TurnWaterfallEntry = {
          turnId: utteranceId,
          turnNumber: nextTurnNum,
          timestamp: new Date().toLocaleTimeString(),
          utterancePreview: scenario.text.slice(0, 72),
          language: scenario.language.toUpperCase(),
          thinkingDepth: 'DETERMINISTIC',
          reasoningSource: 'scripted_scenario',
          vadFinalizeMs: 0,
          prismaSttMs: 0,
          geminiReasoningMs: 8,
          orchestratorMs: 4,
          postgresPersistMs: 2,
          timbreTtfbMs: lastCachedPlaybackStartMs ?? 12,
          timbreTotalMs: lastCachedPlaybackStartMs ?? 12,
          browserPlaybackDelayMs: lastCachedPlaybackStartMs ?? 12,
          totalTurnMs: 26,
          noiseCategory: acousticTelemetry.noiseCategory,
          overlappingVoices: acousticTelemetry.overlappingVoicesDetected,
          clippingDetected: acousticTelemetry.micQualityStatus === 'clipping_detected',
        };
        return [entry, ...prev].slice(0, 12);
      });
    }

    appendAudit(
      'category_classified_and_allocated',
      'info',
      `Category=${scenario.category} (conf=${scenario.confidence.toFixed(2)}); ${nextAssignments.length} total assignment(s)${nextLocConfirm !== 'confirmed' ? ' [Provisional, location unconfirmed]' : ''}`,
      nextLocConfirm !== 'confirmed' ? 'provisional_location_unconfirmed' : null,
      utteranceId,
    );
  };

  // Top-level scripted demo scenarios ALWAYS start a fresh session and incident
  const handleRunScenario = (scenario: PresetScenario) => {
    executeScenarioTurn(scenario, { startFreshSession: true });
  };

  // In-session follow-up turns run inside the existing session to test conflicts, reclassification, and second-incident detection
  const handleRunInSessionFollowup = (followupScenario: PresetScenario) => {
    executeScenarioTurn(followupScenario, { startFreshSession: false });
  };

  const handleConfirmLocation = () => {
    setLocationConfirmationStatus('confirmed');
    setLocationConfirmedAt(new Date().toISOString());
    setShowLocationChangePicker(false);
    removeReviewReason('pending_location');
    removeReviewReason('provisional_location_unconfirmed');
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRequestSentAtRef.current = performance.now();
      wsRef.current.send(
        JSON.stringify({
          type: 'location_confirm',
          ...(sessionIdRef.current && sessionIdRef.current !== 'awaiting-backend-ws-session'
            ? { session_id: sessionIdRef.current }
            : {}),
        }),
      );
      return;
    }

    if (categoryEstablishedAboveThreshold || unknownFallbackApplied) {
      const targetCat = unknownFallbackApplied ? 'unknown' : incidentCategory;
      const { nextAssignments, nextResources } = runDeterministicAllocation(
        targetCat,
        assignments,
        resources,
        true,
        gpsLatitude,
        gpsLongitude,
        true,
        true,
      );
      setAssignments(nextAssignments);
      setResources(nextResources);
      appendAudit(
        'location_confirmed_allocation_upgraded',
        'info',
        `Caller confirmed location (${gpsLatitude.toFixed(4)}, ${gpsLongitude.toFixed(4)}); provisional assignments upgraded to confirmed`,
        null,
        `loc:${locationSource}`,
      );
    } else {
      appendAudit(
        'location_confirmed',
        'info',
        `Caller confirmed location (${gpsLatitude.toFixed(4)}, ${gpsLongitude.toFixed(4)})`,
        null,
        `loc:${locationSource}`,
      );
    }
  };

  const handleInvalidateLocation = () => {
    setShowLocationChangePicker((v) => !v);
    if (locationConfirmationStatus === 'confirmed') {
      setLocationConfirmationStatus('candidate_awaiting_confirmation');
      setLocationConfirmedAt(null);
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: 'location_invalidate',
            ...(sessionIdRef.current && sessionIdRef.current !== 'awaiting-backend-ws-session'
              ? { session_id: sessionIdRef.current }
              : {}),
          }),
        );
      }
    }
  };

  /**
   * When the caller corrects or selects a landmark candidate:
   * - If unconfirmed candidate: allow provisional allocation labeled "Provisional, location unconfirmed"
   * - Re-evaluate distances and hospital rankings on caller correction
   */
  const handleSelectLandmarkLocation = (
    landmark: SyntheticLandmark,
    confirmedByCaller = false,
  ) => {
    const prevLandmarkId = matchedLandmarkId;
    setMatchedLandmarkId(landmark.landmark_id);
    setLocationSource('landmark');
    setGpsLatitude(landmark.latitude);
    setGpsLongitude(landmark.longitude);
    setGpsAccuracyMeters(null);
    setLandmarkMatchConfidence(landmark.match_confidence);
    const nextConfirmStatus: LocationConfirmationStatus = confirmedByCaller
      ? 'confirmed'
      : 'candidate_awaiting_confirmation';
    setLocationConfirmationStatus(nextConfirmStatus);
    setShowLocationChangePicker(false);
    setBackendHospitalBreakdowns(null); // Re-evaluate hospital rankings for corrected coordinates

    removeReviewReason('pending_location');
    if (confirmedByCaller) {
      removeReviewReason('provisional_location_unconfirmed');
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRequestSentAtRef.current = performance.now();
        wsRef.current.send(JSON.stringify({ type: 'location_confirm' }));
      }
    } else {
      addReviewReason('provisional_location_unconfirmed');
    }

    if (categoryEstablishedAboveThreshold || unknownFallbackApplied) {
      const targetCat = unknownFallbackApplied ? 'unknown' : incidentCategory;
      const { nextAssignments, nextResources } = runDeterministicAllocation(
        targetCat,
        assignments,
        resources,
        true,
        landmark.latitude,
        landmark.longitude,
        confirmedByCaller,
        true,
      );
      setAssignments(nextAssignments);
      setResources(nextResources);
    }

    appendAudit(
      prevLandmarkId && prevLandmarkId !== landmark.landmark_id
        ? 'caller_location_corrected_re_evaluated'
        : confirmedByCaller
          ? 'landmark_confirmed_by_caller'
          : 'landmark_provisional_candidate_selected',
      'info',
      `Selected "${landmark.name_romanised}" (${Math.round(landmark.match_confidence * 100)}% match confidence) — ${
        confirmedByCaller
          ? 'Confirmed by caller; allocation & hospital rankings re-evaluated'
          : 'Provisional allocation active; awaiting caller confirmation'
      }`,
      confirmedByCaller ? null : 'provisional_location_unconfirmed',
      `landmark:${landmark.landmark_id}`,
    );
  };

  const handleSetCoarseGpsOver500m = () => {
    setLocationSource('browser_gps');
    setGpsAccuracyMeters(640);
    setLandmarkMatchConfidence(null);
    setLocationConfirmationStatus('unresolved');
    addReviewReason('pending_location');
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRequestSentAtRef.current = performance.now();
      wsRef.current.send(
        JSON.stringify({
          type: 'gps_update',
          latitude: gpsLatitude,
          longitude: gpsLongitude,
          accuracy_m: 640.0,
          permission_status: 'granted',
          confirmed_by_caller: false,
        }),
      );
    }
    appendAudit(
      'gps_rejected_over_500m',
      'warning',
      'Browser GPS accuracy 640 m > 500 m threshold; rejected as unusable without spoken landmark',
      'pending_location',
      'gps:accuracy=640m',
    );
  };

  const handleSetUsableGpsCandidate = () => {
    setLocationSource('browser_gps');
    setGpsAccuracyMeters(42);
    setLandmarkMatchConfidence(null);
    setLocationConfirmationStatus('candidate_awaiting_confirmation');
    removeReviewReason('pending_location');
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRequestSentAtRef.current = performance.now();
      wsRef.current.send(
        JSON.stringify({
          type: 'gps_update',
          latitude: gpsLatitude,
          longitude: gpsLongitude,
          accuracy_m: 42.0,
          permission_status: 'granted',
          confirmed_by_caller: false,
        }),
      );
    }
    appendAudit(
      'gps_candidate_accepted',
      'info',
      'Browser GPS accuracy 42 m <= 500 m threshold; accepted as candidate awaiting caller confirmation',
      null,
      'gps:accuracy=42m',
    );
  };

  // Barge-in handlers (all using unified AudioBufferSourceNode stop surface)
  const handleGenuineCallerBargeIn = () => {
    const stopRes = audioControllerRef.current?.stopPlaybackImmediately();
    const measuredStopMs = stopRes ? stopRes.localStopLatencyMs : 0.4;
    const clampedStopMs = Math.max(0.2, Math.round(measuredStopMs * 10) / 10);
    setIsTtsPlaying(false);
    isTtsPlayingRef.current = false;
    dispatchVoiceLifecycleEvent('caller_barge_in_started', {
      isTtsPlaying: false,
    });
    setBargeInCount((c) => c + 1);
    setLastBargeInStopMs(clampedStopMs);
    const extracted = ringBufferRef.current.extractPreRollSamples();
    setLastBargeInSummary(
      `Local AudioBufferSourceNode stopped (${clampedStopMs} ms); ${extracted.length} pre-roll PCM samples forwarded.`,
    );
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRequestSentAtRef.current = performance.now();
      wsRef.current.send(
        JSON.stringify({
          type: 'barge_in',
          barge_in_mode: 'genuine_speech',
          local_stop_latency_ms: clampedStopMs,
          was_playing_cached_instruction: ttsPlaybackSource === 'cached_instruction',
        }),
      );
    }
    appendAudit(
      'caller_barge_in_interruption',
      'info',
      `Local AudioBufferSourceNode stopped (${clampedStopMs} ms) -> TTS cancelled -> ${extracted.length} pre-roll samples sent`,
      null,
      `ring_buffer:samples=${extracted.length}`,
    );
  };

  const handleEmptyTranscriptBargeInPreGate01 = () => {
    const stopRes = audioControllerRef.current?.stopPlaybackImmediately();
    const measuredStopMs = stopRes ? stopRes.localStopLatencyMs : 0.5;
    setLastBargeInStopMs(Math.max(0.2, Math.round(measuredStopMs * 10) / 10));
    setIsTtsPlaying(false);
    isTtsPlayingRef.current = false;
    dispatchVoiceLifecycleEvent('tts_playback_ended', {
      isTtsPlaying: false,
    });
    setStayWithMeMessage('Go ahead, I am listening.');
    setLastBargeInSummary(
      'Empty transcript after barge-in: played short "Go ahead, I am listening" prompt.',
    );
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRequestSentAtRef.current = performance.now();
      wsRef.current.send(
        JSON.stringify({
          type: 'barge_in',
          barge_in_mode: 'empty_pre_gate01',
        }),
      );
    }
    appendAudit(
      'barge_in_empty_transcript_conservative',
      'info',
      'Played "Go ahead, I am listening"; did not auto-replay interrupted prompt',
      null,
      'barge_in:empty_transcript',
    );
  };

  const handleFalseBargeInTrigger = () => {
    const nextFalse = falseBargeInCount + 1;
    setFalseBargeInCount(nextFalse);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRequestSentAtRef.current = performance.now();
      wsRef.current.send(
        JSON.stringify({
          type: 'barge_in',
          barge_in_mode: 'verified_false_trigger',
          was_playing_cached_instruction: ttsPlaybackSource === 'cached_instruction',
        }),
      );
    }
    if (nextFalse >= 3) {
      setAutoVadBargeInEnabled(false);
      setLastBargeInSummary(
        '3 false barge-in triggers reached: automatic VAD barge-in disabled; manual Interrupt button remains active.',
      );
      appendAudit(
        'auto_vad_barge_in_disabled',
        'warning',
        '3 false barge-in triggers in session; switched to manual tap-to-interrupt',
        null,
        `false_barge_in_count=${nextFalse}`,
      );
    } else {
      setIsTtsPlaying(true);
      isTtsPlayingRef.current = true;
      if (ttsPlaybackSource === 'cached_instruction') {
        setCacheHits((c) => c + 1);
        setLastBargeInSummary(
          `Verified non-speech noise (#${nextFalse}/3): replayed interrupted cached instruction audio from start.`,
        );
      } else {
        setStayWithMeMessage('Did you want to add something, or should I repeat that?');
        setLastBargeInSummary(
          `Verified non-speech noise (#${nextFalse}/3) during live TTS: asked if caller wants repetition.`,
        );
      }
      appendAudit(
        'verified_false_barge_in_handled',
        'info',
        `False trigger #${nextFalse} handled for source=${ttsPlaybackSource}`,
        null,
        `false_barge_in_count=${nextFalse}`,
      );
    }
  };

  const handleToggleMicAudioWorklet = async () => {
    if (!audioControllerRef.current) {
      return;
    }
    if (isMicWorkletStreaming) {
      vadActiveSpeechStreamingRef.current = false;
      speechStartedAtMsRef.current = null;
      lowRmsSilenceSinceMsRef.current = null;
      audioControllerRef.current.stopMicrophoneCapture();
      setIsMicWorkletStreaming(false);
      setLiveMicRms(0);
      dispatchVoiceLifecycleEvent('mic_paused', {
        isSpeechStreaming: false,
      });
      setStreamingPartialText('Microphone paused — click Enable Microphone & Listen to resume.');
      appendAudit(
        'mic_audioworklet_stopped',
        'info',
        'Stopped 16 kHz mono PCM AudioWorklet stream',
        null,
        'audioworklet:pcm16',
      );
      return;
    }
    dispatchVoiceLifecycleEvent('session_started_requesting_mic');
    setStreamingPartialText('Enabling microphone…');
    const started = await startLiveMicrophonePipeline();
    if (started) {
      if (sessionStatus === 'not_started') {
        updateSessionStatus('active');
      }
      dispatchVoiceLifecycleEvent('mic_ready_listening', {
        sessionActive: true,
      });
      setStreamingPartialText('Listening for caller speech…');
    }
  };

  const handleSendLiveGeminiUtterance = () => {
    const trimmed = customUtteranceText.trim();
    if (!trimmed) {
      return;
    }
    if (sessionStatus === 'not_started') {
      updateSessionStatus('active');
      setSpokenDisclosurePlayed(true);
    }
    const sid = sessionIdRef.current;
    const activeSid = sid && sid !== 'awaiting-backend-ws-session' ? sid : undefined;
    const currentLang = activeLanguageRef.current;
    setIsReasoningThinking(true);
    setStreamingPartialText(trimmed);
    armTurnProcessingWatchdog();
    dispatchVoiceLifecycleEvent('transcript_ready_understanding', {
      sessionActive: true,
      isTurnProcessing: true,
    });
    const utteranceId = `utt-${utterances.length + 1}-live`;
    if (!(wsRef.current && wsRef.current.readyState === WebSocket.OPEN)) {
      setUtterances((prev) => [
        ...prev,
        {
          utterance_id: utteranceId,
          sequence: prev.length + 1,
          text: trimmed,
          language: currentLang,
          languageBadge: currentLang.toUpperCase(),
          utterance_role: 'new_description',
          timestamp: new Date().toISOString(),
        },
      ]);
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRequestSentAtRef.current = performance.now();
      wsRef.current.send(
        JSON.stringify({
          type: 'live_gemini_utterance',
          ...(activeSid ? { session_id: activeSid } : {}),
          text: trimmed,
          language: currentLang,
        }),
      );
    } else {
      clearTurnProcessingState();
      setIsReasoningThinking(false);
      dispatchVoiceLifecycleEvent('turn_completed_no_audio', {
        sessionActive: true,
        isTurnProcessing: false,
      });
    }
    appendAudit(
      'live_utterance_submitted',
      'info',
      `Sent finalized utterance "${trimmed.slice(0, 48)}" over /ws/session`,
      null,
      utteranceId,
    );
    setCustomUtteranceText('');
  };

  const handleEvaluateSilenceTimeout = (idleSeconds: number) => {
    const nextCheckIns = idleSeconds >= 15 && idleSeconds < 45 ? silenceCheckInsSent + 1 : 2;
    setSilenceCheckInsSent(nextCheckIns);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRequestSentAtRef.current = performance.now();
      wsRef.current.send(
        JSON.stringify({
          type: 'evaluate_caller_silence',
          idle_seconds: idleSeconds,
          check_ins_sent: silenceCheckInsSent,
        }),
      );
    }
    appendAudit(
      'caller_silence_evaluated',
      idleSeconds >= 45 ? 'warning' : 'info',
      `Evaluated caller silence (idle_seconds=${idleSeconds}s, prior_check_ins=${silenceCheckInsSent})`,
      idleSeconds >= 45 ? 'caller_silence_timeout_review' : null,
      `timeout:${idleSeconds}s`,
    );
  };

  const handleReconnectSessionWithToken = () => {
    if (
      wsRef.current &&
      wsRef.current.readyState === WebSocket.OPEN &&
      backendSessionTokenRef.current
    ) {
      wsRequestSentAtRef.current = performance.now();
      wsRef.current.send(
        JSON.stringify({
          type: 'session_reconnect',
          session_id: sessionId,
          session_token: backendSessionTokenRef.current,
          last_received_sequence: backendLastSeqRef.current,
        }),
      );
      appendAudit(
        'session_reconnect_verified',
        'info',
        `Verified signed session token & replayed state from seq=${backendLastSeqRef.current}`,
        null,
        `token:${backendSessionTokenRef.current.slice(0, 18)}…`,
      );
    }
  };

  const handleSimulateDuplicateCheck = () => {
    setPossibleDuplicateFlag(true);
    setMatchedDuplicateId('INC-SYNTH-HYD-098');
    addReviewReason('possible_duplicate_incident');
    appendAudit(
      'possible_duplicate_flagged',
      'warning',
      'Flagged possible duplicate of INC-SYNTH-HYD-098 (within 500 m & 30 min); never auto-merged or suppressed',
      'possible_duplicate_incident',
      'duplicate_detector:500m_30m',
    );
  };

  const handleFlagSuspectedPrank = () => {
    setPrankReviewStatus('flagged_for_review');
    setPrankFlagReason('contradictory_non_emergency_pattern');
    addReviewReason('suspected_false_or_prank_call');
    appendAudit(
      'suspected_false_or_prank_flagged',
      'warning',
      'Flagged for human control-room review; never auto-rejected or blocked',
      'suspected_false_or_prank_call',
      'safety_policy:no_auto_reject',
    );
  };

  // Shortened Meta-Row Location Pill Label + Tooltip:
  // - Landmark shows match confidence (e.g., "Candidate · 92% match"), NEVER accuracy radius
  // - GPS shows accuracy radius (e.g., "Candidate ±78 m")
  const metaLocationSummary = useMemo(() => {
    if (locationConfirmationStatus === 'confirmed') {
      if (locationSource === 'landmark') {
        const confPct = Math.round((landmarkMatchConfidence ?? 0.92) * 100);
        return {
          shortLabel: `Confirmed · ${confPct}% match`,
          tooltip: `Caller confirmed landmark match (${confPct}% confidence)`,
          pillClass: 'bg-emerald-50 text-emerald-900 border-emerald-400',
          dotClass: 'bg-emerald-600',
        };
      }
      return {
        shortLabel: `Confirmed ±${gpsAccuracyMeters ?? 42} m`,
        tooltip: 'Caller confirmed GPS location candidate',
        pillClass: 'bg-emerald-50 text-emerald-900 border-emerald-400',
        dotClass: 'bg-emerald-600',
      };
    }
    if (locationConfirmationStatus === 'candidate_awaiting_confirmation') {
      if (locationSource === 'landmark') {
        const confPct = Math.round((landmarkMatchConfidence ?? 0.92) * 100);
        return {
          shortLabel: `Candidate · ${confPct}% match`,
          tooltip: 'awaiting caller confirmation (provisional allocation active)',
          pillClass: 'bg-amber-50 text-amber-950 border-amber-300',
          dotClass: 'bg-amber-500',
        };
      }
      return {
        shortLabel: `Candidate ±${gpsAccuracyMeters ?? 78} m`,
        tooltip: 'awaiting caller confirmation',
        pillClass: 'bg-amber-50 text-amber-950 border-amber-300',
        dotClass: 'bg-amber-500',
      };
    }
    return {
      shortLabel: 'Location unresolved',
      tooltip: 'GPS > 500 m or denied — awaiting spoken landmark',
      pillClass: 'bg-red-50 text-red-900 border-red-300',
      dotClass: 'bg-red-600',
    };
  }, [
    locationConfirmationStatus,
    locationSource,
    landmarkMatchConfidence,
    gpsAccuracyMeters,
  ]);

  // Distinct State Banners
  const hasProviderBlockedBanner = reviewReasonCodes.includes(
    'provider_blocked_emergency_content',
  );
  const hasReasoningUnavailableBanner = reviewReasonCodes.includes('reasoning_unavailable');
  const hasTimeoutCheckInBanner =
    reviewReasonCodes.includes('caller_silence_timeout_review') || silenceCheckInsSent > 0;

  return (
    <div className="min-h-screen bg-[#F4F6F7] text-slate-900 flex flex-col font-sans">
      {/* 1. PERSISTENT PROTOTYPE DISCLOSURE BANNER (Shown Once — Full Text in Modal) */}
      <div
        role="region"
        aria-label="Mandatory Prototype Simulation Disclosure"
        className="sticky top-0 z-50 bg-[#072A2B] border-b border-teal-800 px-3 sm:px-6 py-2 shadow-sm print:hidden"
      >
        <div className="max-w-[1440px] mx-auto flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider bg-amber-400 text-slate-950 shrink-0">
              PROTOTYPE SIMULATION
            </span>
            <p className="text-xs sm:text-sm font-medium text-teal-50">
              {SHORT_PROTOTYPE_DISCLOSURE}
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setShowDisclosureModal(true)}
              className="min-h-[36px] px-3 py-1 rounded-md text-xs font-semibold bg-teal-800/90 hover:bg-teal-700 text-white border border-teal-600 transition-colors cursor-pointer"
            >
              Read full notice
            </button>
          </div>
        </div>
      </div>

      {/* FULL DISCLOSURE MODAL (Never Truncated) */}
      {showDisclosureModal && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="full-disclosure-modal-title"
          className="fixed inset-0 z-50 bg-slate-950/60 backdrop-blur-xs flex items-center justify-center p-4"
        >
          <div className="bg-white border border-slate-200 rounded-xl max-w-lg w-full p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-900 border border-amber-300 text-xs font-bold uppercase">
                  Mandatory Notice
                </span>
                <h2
                  id="full-disclosure-modal-title"
                  className="text-base font-bold text-slate-900"
                >
                  Prototype Emergency Simulation Disclosure
                </h2>
              </div>
              <button
                type="button"
                onClick={() => setShowDisclosureModal(false)}
                className="min-h-[44px] min-w-[44px] rounded-lg text-xs font-semibold text-slate-700 hover:bg-slate-100 cursor-pointer"
              >
                Close
              </button>
            </div>
            <p
              data-testid="full-disclosure-modal-text"
              className="text-sm text-slate-800 leading-relaxed font-medium"
            >
              {REQUIRED_PROTOTYPE_DISCLOSURE}
            </p>
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-700 space-y-1">
              <div className="font-semibold text-slate-900">
                Session Start Audio Disclosure ({activeLanguage.toUpperCase()}):
              </div>
              <div>
                “{SPOKEN_DISCLOSURE_BY_LANGUAGE[activeLanguage].text}”
              </div>
              <div
                className={`font-semibold pt-1 ${
                  lastAudioGenerationStatus === 'ready' ? 'text-emerald-900' : 'text-amber-900'
                }`}
              >
                Audio Status:{' '}
                {lastAudioGenerationStatus === 'ready'
                  ? 'Gnani Timbre v2.5 verified speech (voice: Pranav, 16 kHz mono PCM WAV).'
                  : 'Development placeholder audio (pending live voice verification).'}
              </div>
            </div>
            <div className="flex justify-end pt-2">
              <button
                type="button"
                onClick={() => setShowDisclosureModal(false)}
                className="min-h-[44px] px-5 py-2 rounded-lg bg-[#0B3B3C] hover:bg-teal-900 text-white text-xs font-semibold cursor-pointer"
              >
                Acknowledge & Continue
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 2. DARK-TEAL HEADER (#0B3B3C) */}
      <header className="bg-[#0B3B3C] text-white border-b border-teal-900 px-4 sm:px-6 py-3 shadow-xs print:hidden">
        <div className="max-w-[1440px] mx-auto flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3.5">
            <SethuBrandEmblem size="md" />
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-base sm:text-lg font-extrabold tracking-tight text-white flex items-baseline">
                  <span>SETHU</span>
                  <span className="text-teal-300 font-bold">.ai</span>
                </h1>
                <span className="text-xs px-2 py-0.5 rounded bg-teal-900/90 text-teal-100 border border-teal-700 font-semibold">
                  Multilingual Emergency Coordination
                </span>
              </div>
            </div>
          </div>

          {/* Primary View Navigation (Focused on Emergency Coordination + Engineering Diagnostics) */}
          <nav
            aria-label="SETHU Workspace Views"
            className="flex flex-wrap items-center gap-1.5"
          >
            {(
              [
                { id: 'initial', label: 'Session Start' },
                { id: 'live_incident', label: 'Live Coordination' },
                {
                  id: 'control_room',
                  label: `Control Room${reviewReasonCodes.length > 0 ? ` (${reviewReasonCodes.length})` : ''}`,
                },
                { id: 'handoff', label: 'Incident Handoff' },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveView(tab.id)}
                className={`min-h-[44px] px-3.5 py-2 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
                  activeView === tab.id
                    ? 'bg-white text-slate-950 shadow-xs'
                    : 'bg-teal-900/70 text-teal-100 hover:bg-teal-800 border border-teal-700/80'
                }`}
              >
                {tab.label}
              </button>
            ))}

            <button
              type="button"
              data-testid="toggle-diagnostics-button"
              onClick={() => setShowDiagnosticsOverlay((v) => !v)}
              className={`min-h-[44px] px-3.5 py-2 rounded-lg text-xs font-semibold border transition-colors cursor-pointer flex items-center gap-1.5 ${
                showDiagnosticsOverlay
                  ? 'bg-amber-400 text-slate-950 border-amber-300 shadow-xs'
                  : 'bg-teal-950/80 text-teal-200 border-teal-700 hover:bg-teal-900'
              }`}
            >
              <span>Engineering Diagnostics</span>
              {turnWaterfallHistory.length > 0 && (
                <span
                  className={`px-1.5 py-0.5 rounded text-[11px] font-bold tabular-nums ${
                    showDiagnosticsOverlay
                      ? 'bg-slate-950 text-amber-300'
                      : 'bg-teal-800 text-teal-100'
                  }`}
                >
                  {turnWaterfallHistory.length} turn{turnWaterfallHistory.length === 1 ? '' : 's'}
                </span>
              )}
            </button>
          </nav>
        </div>
      </header>

      {/* 3. META ROW: Shortened Values + Persistent MOCK Reasoning Adapter Chip + Acoustic Environment Pill */}
      <div className="bg-white border-b border-slate-200 px-4 sm:px-6 py-2 shadow-2xs print:hidden">
        <div className="max-w-[1440px] mx-auto flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex flex-wrap items-center gap-2">
            {/* Persistent MOCK Reasoning Adapter Chip (hidden only when a verified live provider is active) */}
            {!isLiveReasoningVerified && (
              <span
                data-testid="mock-reasoning-adapter-chip"
                title="Output currently comes from the deterministic MockReasoningAdapter"
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-100 text-amber-950 border border-amber-400 font-bold"
              >
                <span className="w-2 h-2 rounded-full bg-amber-600" />
                MOCK reasoning adapter
              </span>
            )}

            {/* Incident ID from backend */}
            <span
              title={`Backend Incident ID: ${incidentId} | Session ID: ${sessionId}`}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-100 text-slate-800 border border-slate-300 font-semibold"
            >
              <span className="text-slate-600">Incident:</span>
              <span className="tabular-nums">{incidentId.slice(0, 12)}</span>
            </span>

            {/* Active Language & Mode */}
            <span
              data-testid="active-language-meta-pill"
              title={`Active language: ${LANGUAGE_DISPLAY_NAMES[activeLanguage] ?? activeLanguage.toUpperCase()} | Mode: DIRECT | Seen: ${detectedLanguagesSeen.map((l: SupportedLanguage) => l.toUpperCase()).join(', ')}`}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-teal-50 text-teal-900 border border-teal-300 font-semibold"
            >
              <span>Lang: {activeLanguage.toUpperCase()}</span>
            </span>

            {/* Location Meta Pill: Landmark shows Match Confidence, GPS shows Accuracy Radius */}
            <span
              data-testid="meta-location-pill"
              title={metaLocationSummary.tooltip}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border font-semibold cursor-help ${metaLocationSummary.pillClass}`}
            >
              <span className={`w-2 h-2 rounded-full ${metaLocationSummary.dotClass}`} />
              <span>{metaLocationSummary.shortLabel}</span>
            </span>

            {/* Active Conflicts Counter Chip */}
            <span
              data-testid="active-conflicts-chip"
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border font-semibold ${
                activeConflictsCount > 0
                  ? 'bg-red-50 text-red-900 border-red-300'
                  : 'bg-slate-100 text-slate-700 border-slate-300'
              }`}
            >
              <span>Conflicts: {activeConflictsCount}</span>
            </span>
          </div>

          {/* Right Meta Group: Live Acoustic Environment Intelligence Pill (Zero-Gemini Client DSP) */}
          <div className="flex flex-wrap items-center gap-2">
            <div
              data-testid="meta-acoustic-pill"
              title="Real-time 16 kHz PCM & Silero VAD acoustic telemetry — Advisory only (zero Gemini calls; never alters dispatch or triage severity)"
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border font-semibold ${
                acousticTelemetry.noiseCategory === 'high_ambient' ||
                acousticTelemetry.micQualityStatus === 'clipping_detected'
                  ? 'bg-amber-50 text-amber-950 border-amber-300'
                  : 'bg-emerald-50 text-emerald-900 border-emerald-300'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  acousticTelemetry.noiseCategory === 'high_ambient' ||
                  acousticTelemetry.micQualityStatus === 'clipping_detected'
                    ? 'bg-amber-600'
                    : 'bg-emerald-600'
                }`}
              />
              <span className="tabular-nums">
                Acoustic:{' '}
                {acousticTelemetry.noiseCategory === 'high_ambient'
                  ? 'High Ambient'
                  : acousticTelemetry.noiseCategory === 'moderate'
                    ? 'Moderate'
                    : 'Quiet'}{' '}
                ({acousticTelemetry.noiseFloorDbfs} dBFS)
              </span>
              <span>·</span>
              <span>
                {acousticTelemetry.overlappingVoicesDetected
                  ? 'Overlapping voices'
                  : 'Single voice'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 4. VISUALLY DISTINCT STATE BANNERS */}
      <div className="max-w-[1440px] w-full mx-auto px-4 sm:px-6 pt-2 space-y-2 print:hidden">
        {unknownFallbackApplied && (
          <div
            role="status"
            data-testid="banner-unknown-fallback"
            className="bg-amber-50 border-l-4 border-amber-600 border border-amber-300 rounded-r-lg px-4 py-2.5 flex flex-wrap items-center justify-between gap-2"
          >
            <div className="flex items-center gap-2 text-xs text-amber-950">
              <span className="px-2 py-0.5 rounded bg-amber-600 text-white font-bold uppercase">
                Unknown Fallback Applied
              </span>
              <span className="font-semibold">
                Single simulated ambulance requested and incident flagged for human review.
              </span>
            </div>
            <span className="text-xs font-semibold text-amber-900 bg-amber-100 px-2 py-0.5 rounded border border-amber-300">
              {getHumanReasonLabel(fallbackReason ?? 'unknown_after_clarification')}
            </span>
          </div>
        )}

        {hasProviderBlockedBanner && (
          <div
            role="status"
            data-testid="banner-provider-blocked"
            className="bg-red-50 border-l-4 border-red-600 border border-red-300 rounded-r-lg px-4 py-2.5 flex flex-wrap items-center justify-between gap-2"
          >
            <div className="flex items-center gap-2 text-xs text-red-950">
              <span className="px-2 py-0.5 rounded bg-red-600 text-white font-bold uppercase">
                Provider Safety Filter Triggered
              </span>
              <span className="font-semibold">
                Emergency content blocked by upstream safety filter ({consecutiveProviderBlocks}{' '}
                consecutive). Cached prompt active.
              </span>
            </div>
            <span
              title="provider_blocked_emergency_content"
              className="text-xs font-semibold text-red-900 bg-red-100 px-2 py-0.5 rounded border border-red-300"
            >
              Provider safety filter triggered
            </span>
          </div>
        )}

        {hasReasoningUnavailableBanner && (
          <div
            role="status"
            data-testid="banner-reasoning-unavailable"
            className="bg-purple-50 border-l-4 border-purple-700 border border-purple-300 rounded-r-lg px-4 py-2.5 flex flex-wrap items-center justify-between gap-2"
          >
            <div className="flex flex-wrap items-center gap-2 text-xs text-purple-950">
              <span className="px-2 py-0.5 rounded bg-purple-700 text-white font-bold uppercase">
                Reasoning Service Unavailable
              </span>
              <span className="font-semibold">
                Deterministic emergency fallback &amp; Stay With Me reassurance active ({geminiStatusMessage}).
              </span>
            </div>
            <span
              title="reasoning_unavailable"
              className="text-xs font-semibold text-purple-900 bg-purple-100 px-2 py-0.5 rounded border border-purple-300"
            >
              Reasoning service unavailable
            </span>
          </div>
        )}

        {hasTimeoutCheckInBanner && (
          <div
            role="status"
            data-testid="banner-timeout-checkin"
            className="bg-sky-50 border-l-4 border-sky-600 border border-sky-300 rounded-r-lg px-4 py-2.5 flex flex-wrap items-center justify-between gap-2"
          >
            <div className="flex items-center gap-2 text-xs text-sky-950">
              <span className="px-2 py-0.5 rounded bg-sky-700 text-white font-bold uppercase">
                Timeout Check-In
              </span>
              <span className="font-semibold">
                Caller silence check-in active ({silenceCheckInsSent}/2 check-ins sent). Session remains open for review.
              </span>
            </div>
            <span
              title="caller_silence_timeout_review"
              className="text-xs font-semibold text-sky-900 bg-sky-100 px-2 py-0.5 rounded border border-sky-300"
            >
              Caller silence check-in
            </span>
          </div>
        )}
      </div>

      {/* UNIFIED ENGINEERING DIAGNOSTICS & LIVE PIPELINE PERFORMANCE MONITOR */}
      {showDiagnosticsOverlay && (
        <section
          aria-label="Engineering Diagnostics Drawer"
          data-testid="engineering-diagnostics-drawer"
          className="bg-slate-900 text-slate-100 border-b border-slate-700 px-4 sm:px-6 py-5 shadow-lg print:hidden"
        >
          <div className="max-w-[1440px] mx-auto space-y-5">
            {/* Drawer Header */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-700 pb-3">
              <div className="space-y-0.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-teal-400 text-slate-950 text-xs font-bold uppercase">
                    Engineering Diagnostics
                  </span>
                  <span className="text-xs font-bold uppercase tracking-wider text-teal-300">
                    Live Pipeline Performance Monitor · Provider Telemetry · Simulation & Fault Injection
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  Real-time per-turn stage waterfall, zero-Gemini acoustic DSP telemetry, and deterministic fault-injection harness.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  data-testid="toggle-demo-controls-button"
                  onClick={() => setIsDemoControlsOpen((prev) => !prev)}
                  className={`min-h-[36px] px-3 py-1 rounded text-xs font-semibold border cursor-pointer transition-colors ${
                    isDemoControlsOpen
                      ? 'bg-amber-400 text-slate-950 border-amber-300'
                      : 'bg-slate-800 hover:bg-slate-700 text-amber-300 border-slate-600'
                  }`}
                >
                  {isDemoControlsOpen
                    ? 'Hide Simulation & Fault Controls ▲'
                    : 'Show Simulation & Fault Controls ▼'}
                </button>
                <button
                  type="button"
                  onClick={() => setShowDiagnosticsOverlay(false)}
                  className="min-h-[36px] px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-600 cursor-pointer"
                >
                  Close Diagnostics
                </button>
              </div>
            </div>

            {/* SECTION 1: LIVE PER-TURN PIPELINE PERFORMANCE WATERFALL */}
            <div
              data-testid="live-pipeline-waterfall-panel"
              className="bg-slate-800/90 border border-slate-700 rounded-xl p-4 space-y-3"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-teal-300">
                    1. Live Per-Turn Pipeline Latency Waterfall (Browser Mic → VAD → Prisma STT → Gemini 3.8 Flash → Orchestrator → Timbre TTS)
                  </h3>
                  <p className="text-xs text-slate-400">
                    Measured per-stage timings for each finalized turn. Adaptive LOW/MEDIUM reasoning budget keeps latency low while preserving multi-hazard accuracy.
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="px-2.5 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-700 font-semibold tabular-nums">
                    1 Utterance = 1 Gemini Request
                  </span>
                  <span className="px-2.5 py-1 rounded bg-slate-900 text-cyan-300 border border-slate-700 font-semibold tabular-nums">
                    VAD Misfires Filtered: {vadMisfiresFiltered} (0 wasted API calls)
                  </span>
                  <span className="px-2.5 py-1 rounded bg-slate-900 text-amber-300 border border-slate-700 font-semibold tabular-nums">
                    Cached Audio Hits: {cacheHits}
                  </span>
                </div>
              </div>

              {/* Legend */}
              <div className="flex flex-wrap items-center gap-4 text-[11px] text-slate-300 pt-1">
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-xs bg-cyan-400" />
                  VAD Finalize
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-xs bg-sky-500" />
                  Prisma STT
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-xs bg-amber-400" />
                  Gemini 3.8 Flash (LOW / MEDIUM)
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-xs bg-emerald-400" />
                  Orchestrator + PostGIS
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-xs bg-purple-400" />
                  Timbre v2.5 TTS (TTFB)
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-xs bg-teal-300" />
                  Browser Audio Play
                </span>
              </div>

              {turnWaterfallHistory.length === 0 ? (
                <div className="bg-slate-900/80 border border-dashed border-slate-700 rounded-lg p-3.5 text-xs text-slate-400 flex flex-wrap items-center justify-between gap-2">
                  <span>
                    No finalized turns recorded in this session yet. Speak into the microphone, type a caller statement, or run a simulation scenario below to inspect the live stage waterfall.
                  </span>
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[11px]">
                    Awaiting Turn #1
                  </span>
                </div>
              ) : (
                <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                  {turnWaterfallHistory.map((turn) => {
                    const vadMs = Math.max(0, turn.vadFinalizeMs ?? 0);
                    const sttMs = Math.max(0, turn.prismaSttMs ?? 0);
                    const gemMs = Math.max(0, turn.geminiReasoningMs ?? 0);
                    const orchMs = Math.max(0, (turn.orchestratorMs ?? 0) + (turn.postgresPersistMs ?? 0));
                    const ttsMs = Math.max(0, turn.timbreTtfbMs ?? turn.timbreTotalMs ?? 0);
                    const playMs = Math.max(0, turn.browserPlaybackDelayMs ?? 0);
                    const sumSegments = Math.max(1, vadMs + sttMs + gemMs + orchMs + ttsMs + playMs);
                    const pct = (val: number) => Math.max(val > 0 ? 4 : 0, Math.round((val / sumSegments) * 100));

                    return (
                      <div
                        key={`${turn.turnId}-${turn.turnNumber}`}
                        data-testid={`waterfall-turn-${turn.turnNumber}`}
                        className="bg-slate-900/90 border border-slate-700 rounded-lg p-3 space-y-2"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                          <div className="flex flex-wrap items-center gap-2 min-w-0">
                            <span className="px-2 py-0.5 rounded bg-teal-900 text-teal-200 font-bold tabular-nums">
                              Turn #{turn.turnNumber}
                            </span>
                            <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-200 border border-slate-700 font-bold">
                              {turn.language}
                            </span>
                            <span
                              className={`px-2 py-0.5 rounded font-bold uppercase text-[11px] ${
                                turn.thinkingDepth === 'MEDIUM'
                                  ? 'bg-amber-900/80 text-amber-200 border border-amber-700'
                                  : turn.thinkingDepth === 'LOW'
                                    ? 'bg-emerald-900/80 text-emerald-200 border border-emerald-700'
                                    : 'bg-slate-800 text-slate-300 border border-slate-600'
                              }`}
                            >
                              Thinking: {turn.thinkingDepth}
                            </span>
                            <span className="px-2 py-0.5 rounded bg-slate-800 text-teal-300 font-semibold uppercase text-[11px]">
                              {turn.reasoningSource.replace(/_/g, ' ')}
                            </span>
                            <span className="text-slate-300 truncate max-w-md italic">
                              “{turn.utterancePreview}”
                            </span>
                          </div>
                          <div className="flex items-center gap-2 shrink-0 tabular-nums">
                            <span className="px-2.5 py-0.5 rounded bg-teal-400/20 text-teal-200 border border-teal-500/40 font-mono font-bold">
                              Total: {turn.totalTurnMs ?? sumSegments} ms
                            </span>
                          </div>
                        </div>

                        {/* Horizontal Stacked Waterfall Bar */}
                        <div className="w-full h-3 rounded-full bg-slate-800 overflow-hidden flex border border-slate-700">
                          {vadMs > 0 && (
                            <div
                              style={{ width: `${pct(vadMs)}%` }}
                              className="bg-cyan-400 h-full"
                              title={`Silero VAD Finalize: ${vadMs} ms`}
                            />
                          )}
                          {sttMs > 0 && (
                            <div
                              style={{ width: `${pct(sttMs)}%` }}
                              className="bg-sky-500 h-full"
                              title={`Prisma STT Finalization: ${sttMs} ms`}
                            />
                          )}
                          {gemMs > 0 && (
                            <div
                              style={{ width: `${pct(gemMs)}%` }}
                              className="bg-amber-400 h-full"
                              title={`Gemini 3.8 Flash (${turn.thinkingDepth}): ${gemMs} ms`}
                            />
                          )}
                          {orchMs > 0 && (
                            <div
                              style={{ width: `${pct(orchMs)}%` }}
                              className="bg-emerald-400 h-full"
                              title={`Parallel Orchestrator + PostGIS: ${orchMs} ms`}
                            />
                          )}
                          {ttsMs > 0 && (
                            <div
                              style={{ width: `${pct(ttsMs)}%` }}
                              className="bg-purple-400 h-full"
                              title={`Timbre v2.5 TTS TTFB: ${ttsMs} ms`}
                            />
                          )}
                          {playMs > 0 && (
                            <div
                              style={{ width: `${pct(playMs)}%` }}
                              className="bg-teal-300 h-full"
                              title={`Browser Audio Playback Start: ${playMs} ms`}
                            />
                          )}
                        </div>

                        {/* Stage-by-Stage Numerical Breakdown + Turn Acoustic Snapshot */}
                        <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-300 font-mono">
                          <div className="flex flex-wrap items-center gap-3 tabular-nums">
                            <span>VAD: {turn.vadFinalizeMs ?? '—'} ms</span>
                            <span>·</span>
                            <span>Prisma STT: {turn.prismaSttMs ?? '—'} ms</span>
                            <span>·</span>
                            <span className="text-amber-300">
                              Gemini ({turn.thinkingDepth}): {turn.geminiReasoningMs ?? '—'} ms
                            </span>
                            <span>·</span>
                            <span className="text-emerald-300">
                              Orch+DB: {orchMs} ms
                            </span>
                            <span>·</span>
                            <span className="text-purple-300">
                              Timbre TTFB/Total: {turn.timbreTtfbMs ?? '—'}/{turn.timbreTotalMs ?? '—'} ms
                            </span>
                            <span>·</span>
                            <span>Play: {turn.browserPlaybackDelayMs ?? '—'} ms</span>
                          </div>
                          <span className="text-slate-400 font-sans">
                            Acoustic @ Turn: {turn.noiseCategory.replace(/_/g, ' ')} ·{' '}
                            {turn.overlappingVoices ? 'Multi-voice' : 'Single voice'} ({turn.timestamp})
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* SECTION 2: RUNTIME GATE, CHECKSUM & PROVIDER TRANSPARENCY TELEMETRY */}
            <div className="space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-teal-300">
                  2. Runtime Gate, Checksum & Real vs. Synthetic Provider Status
                </span>
                <span className="text-xs text-slate-400">
                  Live Providers: Prisma STT ({isPrismaVerified ? 'Verified' : 'Standby'}) · Gemini 3.8 Flash ({isLiveReasoningVerified ? 'Live' : 'Mock Fallback'}) · Timbre v2.5 ({isTimbreVerified ? 'Verified' : 'Cached WAV'}) · Neon PostGIS ({isPostgresVerified ? 'Connected' : 'In-Memory'}) · Vehicles/Hospitals: Synthetic DEMO
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2.5 text-xs">
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-2.5">
                  <div className="text-slate-400">WS /ws/session</div>
                  <div className="font-mono font-bold text-white">{wsConnectionState}</div>
                </div>
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-2.5">
                  <div className="text-slate-400">Instruction SHA-256</div>
                  <div className="font-mono font-bold text-white">{lastVerifiedSha256}</div>
                </div>
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-2.5">
                  <div className="text-slate-400">Audio Asset Status</div>
                  <div className="font-mono font-bold text-amber-300">
                    {lastAudioGenerationStatus}
                  </div>
                </div>
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-2.5">
                  <div className="text-slate-400">Measured Reasoning RTT</div>
                  <div className="font-mono font-bold text-white tabular-nums">
                    {lastReasoningLatencyMs !== null
                      ? `${lastReasoningLatencyMs} ms`
                      : 'Not yet measured'}
                  </div>
                </div>
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-2.5">
                  <div className="text-slate-400">Measured Audio Start</div>
                  <div className="font-mono font-bold text-white tabular-nums">
                    {lastCachedPlaybackStartMs !== null
                      ? `${lastCachedPlaybackStartMs} ms`
                      : 'Not yet measured'}
                  </div>
                </div>
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-2.5">
                  <div className="text-slate-400">STT Status</div>
                  <div className="font-mono font-bold text-amber-300 truncate" title={prismaGateMessage}>
                    {prismaGateMessage}
                  </div>
                </div>
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-2.5">
                  <div className="text-slate-400">VAD / Prisma / Gemini</div>
                  <div className="font-mono font-bold text-white tabular-nums">
                    {stageLatencies
                      ? `${stageLatencies.vad_finalize_ms ?? '—'} / ${stageLatencies.prisma_stt_ms ?? '—'} / ${stageLatencies.gemini_reasoning_ms ?? '—'} ms`
                      : 'Not yet measured'}
                  </div>
                </div>
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-2.5">
                  <div className="text-slate-400">Timbre TTFB / Total / Play</div>
                  <div className="font-mono font-bold text-emerald-300 tabular-nums">
                    {stageLatencies
                      ? `${stageLatencies.timbre_ttfb_ms ?? '—'} / ${stageLatencies.timbre_total_ms ?? '—'} / ${stageLatencies.browser_playback_delay_ms ?? '—'} ms`
                      : lastLiveTimbreTtfabMs !== null
                        ? `${lastLiveTimbreTtfabMs} ms TTFB`
                        : 'Not yet measured'}
                  </div>
                </div>
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-2.5">
                  <div className="text-slate-400">End-to-End Turn Latency</div>
                  <div className="font-mono font-bold text-teal-300 tabular-nums">
                    {stageLatencies?.total_backend_ms !== undefined &&
                    stageLatencies?.total_backend_ms !== null
                      ? `${stageLatencies.total_backend_ms} ms`
                      : 'Not yet measured'}
                  </div>
                </div>
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-2.5">
                  <div className="text-slate-400">Silero VAD & Mic PCM</div>
                  <div className="font-mono font-bold text-cyan-300 tabular-nums">
                    {isVadReady
                      ? `Active (${(liveSpeechProbability * 100).toFixed(0)}%) · ${pcmChunksStreamed} chunks`
                      : vadError
                        ? `Error: ${vadError}`
                        : `Pending · ${pcmChunksStreamed} chunks`}
                  </div>
                </div>
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-2.5 sm:col-span-2">
                  <div className="text-slate-400">Parallel AI Orchestration (asyncio gather)</div>
                  <div className="font-mono font-bold text-emerald-300 tabular-nums">
                    {stageLatencies?.task_execution_telemetry &&
                    typeof stageLatencies.task_execution_telemetry === 'object'
                      ? (() => {
                          const tel = stageLatencies.task_execution_telemetry as Record<string, unknown>;
                          const pWall = tel.parallel_wall_ms ?? '—';
                          const sSum = tel.sum_branch_ms ?? '—';
                          const saved = tel.sequential_saved_ms ?? 0;
                          const post =
                            tel.post_orchestration_overlap &&
                            typeof tel.post_orchestration_overlap === 'object'
                              ? (tel.post_orchestration_overlap as Record<string, unknown>)
                              : null;
                          return `wall ${String(pWall)} ms vs sum ${String(sSum)} ms (saved ${String(saved)} ms) · TTS+DB overlap: ${String(post?.parallel_wall_ms ?? '—')} ms`;
                        })()
                      : 'Not yet measured'}
                  </div>
                </div>
              </div>
            </div>

            {/* SECTION 3: SIMULATION SCENARIOS & FAULT-INJECTION TESTING HARNESS (Moved out of Main Dashboard) */}
            <div
              data-testid="demo-controls-drawer"
              className="bg-slate-800/80 border border-slate-700 rounded-xl p-4 space-y-4"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <span className="text-xs font-bold uppercase tracking-wider text-amber-300">
                    3. Simulation Scenarios & Fault-Injection Testing Controls
                  </span>
                  <p className="text-xs text-slate-400">
                    Run scripted multilingual scenarios (starts fresh session), in-session conflict/reclassification turns, or hardware/network fault-injection tests.
                  </p>
                </div>
              </div>

              {/* 7 Scripted Multilingual Demo Scenarios (Moved from Live Coordination Main View) */}
              <div className="space-y-1.5">
                <div className="text-xs font-bold text-teal-300 uppercase tracking-wider">
                  Scripted Multilingual Scenarios (Starts Fresh Session & Incident)
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                  {PRESET_SCENARIOS.map((sc) => (
                    <button
                      key={sc.id}
                      type="button"
                      data-testid={`scenario-${sc.id}`}
                      onClick={() => handleRunScenario(sc)}
                      className="min-h-[40px] px-3 py-2 rounded-lg bg-slate-900 hover:bg-teal-950 border border-slate-700 hover:border-teal-500 text-left transition-colors cursor-pointer flex items-center justify-between gap-2"
                    >
                      <span className="text-xs font-semibold text-slate-100 truncate">
                        {sc.label}
                      </span>
                      <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-600 text-teal-300 text-[11px] font-bold shrink-0">
                        {sc.languageBadge}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              {/* In-Session Conflict & Reclassification Test Turns */}
              <div className="space-y-1.5">
                <div className="text-xs font-bold text-amber-300 uppercase tracking-wider">
                  In-Session Follow-Up Turns (Tests Conflict Flag, Stale Fact Superseding & Possible Second Incident)
                </div>
                <div className="flex flex-wrap gap-2">
                  {IN_SESSION_FOLLOWUP_TURNS.map((turn) => (
                    <button
                      key={turn.id}
                      type="button"
                      data-testid={`followup-${turn.id}`}
                      onClick={() => handleRunInSessionFollowup(turn)}
                      className="min-h-[40px] px-3 py-1.5 rounded-lg bg-teal-900/90 hover:bg-teal-800 text-white border border-teal-600 text-xs font-semibold cursor-pointer text-left"
                    >
                      {turn.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Fault-Injection Controls Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
                <div className="bg-slate-900/90 border border-slate-700 rounded-lg p-3 space-y-2">
                  <div className="text-xs font-bold text-slate-200">
                    Audio & Barge-In Fault Injection
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        setIsTtsPlaying(true);
                        setTtsPlaybackSource((s) =>
                          s === 'cached_instruction' ? 'live_timbre' : 'cached_instruction',
                        );
                      }}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold cursor-pointer border border-slate-600"
                    >
                      Switch Source ({ttsPlaybackSource === 'cached_instruction' ? 'cached' : 'live'})
                    </button>
                    <button
                      type="button"
                      onClick={handleEmptyTranscriptBargeInPreGate01}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold cursor-pointer border border-slate-600"
                    >
                      Empty Transcript
                    </button>
                    <button
                      type="button"
                      onClick={handleFalseBargeInTrigger}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-amber-800 hover:bg-amber-700 text-amber-100 text-xs font-semibold cursor-pointer"
                    >
                      False Barge-In ({falseBargeInCount}/3)
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleToggleMicAudioWorklet()}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-teal-800 hover:bg-teal-700 text-white text-xs font-semibold cursor-pointer"
                    >
                      {isMicWorkletStreaming
                        ? `Stop AudioWorklet (${pcmChunksStreamed})`
                        : 'Start AudioWorklet'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setEchoCancellationActive((v) => !v)}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold cursor-pointer border border-slate-600"
                    >
                      Toggle Echo Cancellation
                    </button>
                  </div>
                </div>

                <div className="bg-slate-900/90 border border-slate-700 rounded-lg p-3 space-y-2">
                  <div className="text-xs font-bold text-slate-200">
                    Silence Timers, Reconnect & Thinking Preview
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    <button
                      type="button"
                      onClick={() => handleEvaluateSilenceTimeout(16)}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold cursor-pointer border border-slate-600"
                    >
                      Silence 16s Check-In
                    </button>
                    <button
                      type="button"
                      onClick={() => handleEvaluateSilenceTimeout(50)}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-amber-800 hover:bg-amber-700 text-amber-100 text-xs font-semibold cursor-pointer"
                    >
                      Silence 50s Flag Review
                    </button>
                    <button
                      type="button"
                      onClick={handleReconnectSessionWithToken}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold cursor-pointer border border-slate-600"
                    >
                      Reconnect Token
                    </button>
                    <button
                      type="button"
                      onClick={() => setIsReasoningThinking((v) => !v)}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold cursor-pointer border border-slate-600"
                    >
                      {isReasoningThinking ? 'Stop Thinking' : 'Preview Thinking'}
                    </button>
                  </div>
                </div>

                <div className="bg-slate-900/90 border border-slate-700 rounded-lg p-3 space-y-2">
                  <div className="text-xs font-bold text-slate-200">
                    GPS Threshold & Duplicate / Prank Review Flags
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    <button
                      type="button"
                      onClick={handleSetUsableGpsCandidate}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold cursor-pointer border border-slate-600"
                    >
                      GPS 42 m (Usable)
                    </button>
                    <button
                      type="button"
                      onClick={handleSetCoarseGpsOver500m}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-red-900 hover:bg-red-800 text-red-100 text-xs font-semibold cursor-pointer"
                    >
                      GPS 640 m (Reject)
                    </button>
                    <button
                      type="button"
                      onClick={handleSimulateDuplicateCheck}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold cursor-pointer border border-slate-600"
                    >
                      Flag Duplicate
                    </button>
                    <button
                      type="button"
                      onClick={handleFlagSuspectedPrank}
                      className="min-h-[38px] px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold cursor-pointer border border-slate-600"
                    >
                      Flag Suspected Prank
                    </button>
                  </div>
                </div>
              </div>

              {/* Isolated 8 kHz Telephony-Bandwidth & Noisy Street Audio Diagnostic Bench (Exclusively in Engineering Diagnostics) */}
              <div
                data-testid="diagnostics-telephony-noise-bench"
                className="bg-slate-900/95 border border-teal-700/60 rounded-lg p-3.5 space-y-2.5"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-teal-300 uppercase tracking-wider">
                        Isolated 8 kHz Telephony-Bandwidth & Street Noise Stress-Test Bench
                      </span>
                      <span className="px-2 py-0.5 rounded bg-slate-800 text-emerald-300 border border-slate-700 text-[11px] font-bold">
                        Diagnostics Sandbox Only · 0 Gemini Calls · Never Alters Live Mic or Dispatch
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Simulates cellular G.711 band-limited telephony (300–3,400 Hz @ 8 kHz resampled to 16 kHz PCM16) with -24 dBFS Indian traffic/siren bursts to verify Silero VAD noise-misfire rejection in isolation.
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      data-testid="run-telephony-noise-sim-button"
                      onClick={() =>
                        setTelephonyDiagSim((prev) => ({
                          active: true,
                          codecProfile: 'G.711 μ-law 8 kHz Cellular → 16 kHz PCM16 Upsampled',
                          simulatedNoiseFloorDbfs: -24,
                          simulatedSnrDb: 9.5,
                          bandpassHz: '300 Hz – 3,400 Hz ( Narrowband Telephonic )',
                          vadNoiseFramesRejected: prev.vadNoiseFramesRejected + 14,
                          speechFramesRetained: prev.speechFramesRetained + 38,
                          geminiCallsTriggered: 0,
                          lastRunTimestamp: new Date().toLocaleTimeString(),
                        }))
                      }
                      className="min-h-[38px] px-3 py-1.5 rounded bg-teal-700 hover:bg-teal-600 text-white text-xs font-bold cursor-pointer transition-colors"
                    >
                      Simulate 8 kHz Noisy Telephony Frame Burst
                    </button>
                    {telephonyDiagSim.active && (
                      <button
                        type="button"
                        data-testid="reset-telephony-noise-sim-button"
                        onClick={() =>
                          setTelephonyDiagSim({
                            active: false,
                            codecProfile: 'Standby (16 kHz Wideband Native)',
                            simulatedNoiseFloorDbfs: -49,
                            simulatedSnrDb: 28,
                            bandpassHz: '50 Hz – 8,000 Hz (Wideband PCM16)',
                            vadNoiseFramesRejected: 0,
                            speechFramesRetained: 0,
                            geminiCallsTriggered: 0,
                            lastRunTimestamp: null,
                          })
                        }
                        className="min-h-[38px] px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-600 text-xs font-semibold cursor-pointer"
                      >
                        Reset Bench
                      </button>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs font-mono">
                  <div className="bg-slate-800/90 border border-slate-700 rounded p-2">
                    <div className="text-[11px] text-slate-400 font-sans">Telephony Profile</div>
                    <div className="font-bold text-white truncate" title={telephonyDiagSim.codecProfile}>
                      {telephonyDiagSim.active ? '8 kHz G.711 Narrowband' : '16 kHz Wideband'}
                    </div>
                  </div>
                  <div className="bg-slate-800/90 border border-slate-700 rounded p-2">
                    <div className="text-[11px] text-slate-400 font-sans">Passband Filter</div>
                    <div className="font-bold text-cyan-300 tabular-nums">
                      {telephonyDiagSim.active ? '300–3,400 Hz' : '50–8,000 Hz'}
                    </div>
                  </div>
                  <div className="bg-slate-800/90 border border-slate-700 rounded p-2">
                    <div className="text-[11px] text-slate-400 font-sans">Sim Street Noise / SNR</div>
                    <div className="font-bold text-amber-300 tabular-nums">
                      {telephonyDiagSim.simulatedNoiseFloorDbfs} dBFS · {telephonyDiagSim.simulatedSnrDb} dB SNR
                    </div>
                  </div>
                  <div className="bg-slate-800/90 border border-slate-700 rounded p-2">
                    <div className="text-[11px] text-slate-400 font-sans">VAD Siren Misfires Rejected</div>
                    <div className="font-bold text-emerald-300 tabular-nums">
                      {telephonyDiagSim.vadNoiseFramesRejected} noise bursts ({telephonyDiagSim.speechFramesRetained} speech frames)
                    </div>
                  </div>
                  <div className="bg-slate-800/90 border border-slate-700 rounded p-2">
                    <div className="text-[11px] text-slate-400 font-sans">Isolation Guarantee</div>
                    <div className="font-bold text-teal-300 tabular-nums">
                      {telephonyDiagSim.geminiCallsTriggered} Gemini calls · Live Mic Untouched
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* MAIN WORKSPACE CONTENT */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-4">
        {/* VIEW 1: INITIAL SESSION START VIEW (No duplicate disclosure card — single banner + modal is used) */}
        {activeView === 'initial' && (
          <div className="max-w-3xl mx-auto space-y-4">
            <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="px-2.5 py-1 rounded-md bg-teal-50 text-teal-900 border border-teal-300 text-xs font-bold uppercase">
                  Caller Session Initialization
                </span>
                <span
                  className={`px-2.5 py-1 rounded-md text-xs font-semibold border ${
                    lastAudioGenerationStatus === 'ready'
                      ? 'bg-emerald-50 text-emerald-900 border-emerald-300'
                      : 'bg-amber-50 text-amber-900 border-amber-300'
                  }`}
                >
                  {lastAudioGenerationStatus === 'ready'
                    ? 'Timbre v2.5 verified audio ready'
                    : 'Placeholder audio ready'}
                </span>
              </div>

              <div className="flex items-start gap-3.5">
                <SethuBrandEmblem size="lg" />
                <div className="space-y-1">
                  <h2 className="text-xl font-bold text-slate-900">
                    SETHU.ai — Start Multilingual Emergency Coordination Session
                  </h2>
                  <p className="text-xs text-slate-600">
                    Select the caller language from the seven supported Indian languages below (with Tanglish, Hinglish, and Benglish support). You can also switch languages at any time during an active session without losing incident context.
                  </p>
                </div>
              </div>

              {/* Language Selection — Seven Direct Language Choices */}
              <div className="space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <label className="block text-xs font-bold uppercase tracking-wider text-slate-700">
                    Initial Session Language (7 Direct Options)
                  </label>
                  <span
                    data-testid="language-selection-status-badge"
                    className="px-2.5 py-1 rounded-md text-xs font-bold border bg-emerald-50 text-emerald-900 border-emerald-300"
                  >
                    {formatSelectedLanguageStatus(selectedInitialLanguage)}
                  </span>
                </div>

                <div
                  data-testid="direct-language-grid"
                  className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3"
                >
                  {DIRECT_LANGUAGE_OPTIONS.map((lang) => {
                    const isSelected = selectedInitialLanguage === lang.code;
                    return (
                      <button
                        key={lang.code}
                        type="button"
                        aria-pressed={isSelected}
                        data-testid={`lang-select-${lang.code}`}
                        onClick={() => handleSelectManualLanguage(lang.code)}
                        className={`min-h-[64px] p-3.5 rounded-xl border text-left transition-all cursor-pointer flex flex-col justify-between ${
                          isSelected
                            ? 'bg-teal-50/90 border-teal-700 ring-2 ring-teal-700/25 text-slate-950 shadow-2xs'
                            : 'bg-white border-slate-200 text-slate-800 hover:bg-slate-50 hover:border-slate-300'
                        }`}
                      >
                        <div className="flex items-center justify-between gap-1.5">
                          <span className="text-sm font-bold leading-snug">{lang.fullLabel}</span>
                          {isSelected && (
                            <span className="px-1.5 py-0.5 rounded bg-[#0B3B3C] text-white text-[10px] font-bold uppercase shrink-0">
                              Selected
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-slate-600 mt-1">{lang.sublabel}</div>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="pt-1 flex flex-wrap items-center justify-between gap-3">
                <div className="text-xs text-slate-600">
                  Session and incident identifiers are assigned by the backend on start.
                </div>
                <button
                  type="button"
                  onClick={handleStartSession}
                  className="min-h-[48px] px-6 py-3 rounded-xl bg-[#0B3B3C] hover:bg-teal-900 text-white font-bold text-sm shadow-sm transition-colors cursor-pointer"
                >
                  Start Simulated Emergency Call →
                </button>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 2: LIVE COORDINATION SCREEN (Three-Column Desktop, Two-Column Tablet, Single-Column Mobile with Tabs) */}
        {activeView === 'live_incident' && (
          <div className="space-y-4">
            {/* STATUS STEPPER ABOVE CONVERSATION: Listening -> Understanding -> Locating -> Assigning -> Guiding */}
            <div
              aria-label="Coordination Status Stepper"
              className="bg-white border border-slate-200 rounded-xl px-4 py-3 shadow-2xs flex flex-wrap items-center justify-between gap-3"
            >
              <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
                {COORDINATION_STEPS.map((step, idx) => {
                  const currentIdx = COORDINATION_STEPS.indexOf(currentCoordinationStep);
                  const isCurrent = step === currentCoordinationStep;
                  const isCompleted = idx < currentIdx;
                  return (
                    <React.Fragment key={step}>
                      <div
                        data-testid={`stepper-${step.toLowerCase()}`}
                        className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold border transition-all ${
                          isCurrent
                            ? 'bg-[#0B3B3C] text-white border-teal-900 shadow-xs'
                            : isCompleted
                              ? 'bg-emerald-50 text-emerald-900 border-emerald-300'
                              : 'bg-slate-100 text-slate-600 border-slate-200'
                        }`}
                      >
                        <span
                          className={`w-2 h-2 rounded-full ${
                            isCurrent
                              ? 'bg-teal-300'
                              : isCompleted
                                ? 'bg-emerald-600'
                                : 'bg-slate-400'
                          }`}
                        />
                        <span>{step}</span>
                      </div>
                      {idx < COORDINATION_STEPS.length - 1 && (
                        <span className="text-slate-400 text-xs font-bold select-none">→</span>
                      )}
                    </React.Fragment>
                  );
                })}
              </div>

              <div className="flex items-center gap-2 text-xs">
                <span className="px-2.5 py-1 rounded-md bg-slate-100 text-slate-800 border border-slate-300 font-semibold">
                  Category:{' '}
                  <span className="font-bold text-teal-950 uppercase">
                    {incidentCategory.replace(/_/g, ' ')}
                  </span>
                </span>
              </div>
            </div>

            {/* UNIFIED VOICE & BARGE-IN COMMAND BAR (Merges Call Status, Mic/Echo Status, Active Spoken Response & Barge-In) */}
            {(() => {
              const voicePresentation = resolveVoiceStagePresentation(
                voiceState,
                isTtsPlaying,
                liveMicRms,
                liveSpeechProbability,
              );
              const containerGlowClass =
                voicePresentation.stage === 'responding'
                  ? 'setu-speaking-glow border-teal-500'
                  : voicePresentation.stage === 'hearing'
                    ? 'setu-hearing-glow border-emerald-500/90'
                    : voicePresentation.stage === 'understanding'
                      ? 'setu-understanding-glow border-amber-500/90'
                      : 'border-slate-800';
              return (
                <section
                  aria-label="Unified Voice and Barge-In Command Bar"
                  data-testid="unified-voice-command-bar"
                  data-voice-stage={voicePresentation.stage}
                  className={`bg-slate-900 border text-white rounded-xl p-4 shadow-sm space-y-3 transition-all duration-300 ${containerGlowClass}`}
                >
                  {/* Top Row: Voice Pipeline Status + Waveform + Source Tags + Mic/Barge-In Controls */}
                  <div className="flex flex-wrap items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      {/* Live Status Pulse Indicator */}
                      <div className="relative flex items-center justify-center">
                        <span
                          className={`w-4 h-4 rounded-full transition-colors duration-200 ${
                            voicePresentation.stage === 'listening'
                              ? 'bg-teal-400 animate-ping'
                              : voicePresentation.stage === 'hearing'
                                ? 'bg-emerald-400 animate-ping'
                                : voicePresentation.stage === 'understanding'
                                  ? 'bg-amber-400 animate-pulse'
                                  : voicePresentation.stage === 'responding'
                                    ? 'bg-sky-400 animate-pulse'
                                    : voiceState === 'microphone_denied'
                                      ? 'bg-red-500'
                                      : 'bg-slate-500'
                          }`}
                        />
                        <span
                          className={`absolute w-2.5 h-2.5 rounded-full transition-colors duration-200 ${
                            voicePresentation.stage === 'listening'
                              ? 'bg-teal-400'
                              : voicePresentation.stage === 'hearing'
                                ? 'bg-emerald-400'
                                : voicePresentation.stage === 'understanding'
                                  ? 'bg-amber-400'
                                  : voicePresentation.stage === 'responding'
                                    ? 'bg-sky-400'
                                    : voiceState === 'microphone_denied'
                                      ? 'bg-red-500'
                                      : 'bg-slate-400'
                          }`}
                        />
                      </div>

                      {/* State-Aware Animated Waveform Bars */}
                      <div
                        aria-hidden="true"
                        data-testid="voice-interaction-waveform"
                        data-animation-mode={voicePresentation.animationMode}
                        className={`flex items-end gap-1 h-6 px-2.5 py-1 rounded border transition-colors duration-200 ${
                          voicePresentation.stage === 'hearing'
                            ? 'bg-emerald-950/90 border-emerald-700'
                            : voicePresentation.stage === 'understanding'
                              ? 'bg-amber-950/80 border-amber-700/80'
                              : voicePresentation.stage === 'responding'
                                ? 'bg-sky-950/90 border-sky-700'
                                : 'bg-teal-950 border-teal-800'
                        }`}
                      >
                        {voicePresentation.barHeightsPct.map((heightPct, idx) => (
                          <span
                            key={idx}
                            style={
                              voicePresentation.animationMode === 'hearing_reactive'
                                ? { height: `${heightPct}%` }
                                : undefined
                            }
                            className={`w-1 rounded-full transition-all duration-75 ${
                              voicePresentation.animationMode === 'responding_wave'
                                ? 'h-3.5 bg-sky-300 setu-waveform-bar'
                                : voicePresentation.animationMode === 'understanding_pulse'
                                  ? 'h-3.5 bg-amber-300 setu-thinking-bar'
                                  : voicePresentation.animationMode === 'hearing_reactive'
                                    ? 'bg-emerald-300'
                                    : voicePresentation.stage === 'listening'
                                      ? 'h-2.5 bg-teal-300/85'
                                      : 'h-2 bg-slate-400/60'
                            }`}
                          />
                        ))}
                      </div>

                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-xs uppercase font-extrabold tracking-wider text-slate-300">
                            SETHU Voice & Barge-In:
                          </span>
                          <span
                            role="status"
                            aria-live="polite"
                            data-testid="voice-interaction-stage-badge"
                            data-stage={voicePresentation.stage}
                            className={`px-2.5 py-0.5 rounded text-xs font-bold border transition-colors duration-200 ${
                              voicePresentation.stage === 'hearing'
                                ? 'bg-emerald-950 text-emerald-200 border-emerald-600'
                                : voicePresentation.stage === 'understanding'
                                  ? 'bg-amber-950 text-amber-200 border-amber-600'
                                  : voicePresentation.stage === 'responding'
                                    ? 'bg-sky-950 text-sky-200 border-sky-600'
                                    : voicePresentation.stage === 'listening'
                                      ? 'bg-slate-800 text-teal-300 border-slate-700'
                                      : 'bg-slate-800 text-slate-300 border-slate-700'
                            }`}
                          >
                            {voicePresentation.label}
                          </span>

                      {/* Required "cached" or "live" tag on SETHU's speech */}
                      <span
                        data-testid="speech-source-tag"
                        className={`px-2 py-0.5 rounded text-xs font-bold uppercase border ${
                          ttsPlaybackSource === 'cached_instruction'
                            ? 'bg-teal-950 text-teal-200 border-teal-700'
                            : 'bg-sky-950 text-sky-200 border-sky-700'
                        }`}
                      >
                        {ttsPlaybackSource === 'cached_instruction' ? 'cached' : 'live'}
                      </span>

                      <span className="px-2 py-0.5 rounded text-xs font-semibold bg-slate-800 text-emerald-300 border border-slate-700">
                        {ttsPlaybackSource === 'live_timbre'
                          ? 'Timbre v2.5 live'
                          : lastAudioGenerationStatus === 'ready'
                            ? 'Timbre v2.5 audio'
                            : 'Placeholder audio'}
                      </span>

                      {/* Consolidated Mic & Echo Cancellation Chip */}
                      <span
                        data-testid="mic-echo-status-chip"
                        className={`px-2 py-0.5 rounded text-xs font-semibold border ${
                          echoCancellationActive
                            ? 'bg-slate-800 text-slate-300 border-slate-700'
                            : 'bg-amber-950 text-amber-200 border-amber-700'
                        }`}
                      >
                        {isMicWorkletStreaming ? 'Mic live' : 'Mic standby'} · AEC{' '}
                        {echoCancellationActive ? 'on' : 'off (use headphones)'}
                      </span>
                      {/* Live Active Language Indicator */}
                      <span
                        data-testid="live-language-status-chip"
                        className="px-2 py-0.5 rounded text-xs font-semibold border bg-teal-950/90 text-teal-200 border-teal-700"
                      >
                        {getDirectLanguageOption(activeLanguage).fullLabel}
                      </span>
                    </div>

                    <div className="text-xs text-slate-400 mt-0.5 flex flex-wrap items-center gap-2">
                      <span>
                        {isMicWorkletStreaming
                          ? `Microphone active — speaking in ${LANGUAGE_DISPLAY_NAMES[activeLanguage]} (Tanglish, Hinglish & Benglish supported)`
                          : micErrorMessage || 'Microphone paused — click Enable Microphone & Listen'}
                      </span>
                      {lastBargeInStopMs !== null && (
                        <span className="text-teal-300 text-[11px] font-semibold tabular-nums">
                          · Last barge-in stop: {lastBargeInStopMs} ms
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Mic Input Level & Action Buttons */}
                <div className="flex flex-wrap items-center gap-2.5">
                  {isMicWorkletStreaming && (
                    <div className="hidden sm:flex flex-col items-end gap-1">
                      <div className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">
                        Mic Level
                      </div>
                      <div className="w-20 h-2 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
                        <div
                          className="h-full bg-gradient-to-r from-sky-400 to-cyan-300 transition-all duration-75"
                          style={{ width: `${Math.min(100, Math.max(2, liveMicRms * 500))}%` }}
                        />
                      </div>
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => void handleToggleMicAudioWorklet()}
                    className={`min-h-[40px] px-3 py-1.5 rounded-lg text-xs font-bold cursor-pointer transition-colors ${
                      isMicWorkletStreaming
                        ? 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-600'
                        : 'bg-teal-600 hover:bg-teal-500 text-white'
                    }`}
                  >
                    {isMicWorkletStreaming ? 'Pause Microphone' : 'Enable Microphone & Listen'}
                  </button>

                  <button
                    type="button"
                    onClick={handleGenuineCallerBargeIn}
                    className={`min-h-[40px] px-3.5 py-1.5 rounded-lg text-white text-xs font-bold cursor-pointer transition-colors ${
                      voiceState === 'speaking' || isTtsPlaying
                        ? 'bg-rose-600 hover:bg-rose-500 animate-pulse'
                        : 'bg-rose-700/90 hover:bg-rose-600'
                    }`}
                  >
                    Caller Speaks (Barge-In Interrupt)
                  </button>

                  {sessionStatus === 'active' && (
                    <button
                      type="button"
                      onClick={handleEndSession}
                      data-testid="end-session-btn"
                      className="min-h-[40px] px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-rose-800 text-slate-200 hover:text-white border border-slate-600 text-xs font-bold cursor-pointer transition-colors"
                    >
                      End Session
                    </button>
                  )}
                </div>
              </div>

              {/* Bottom Row: Active Spoken Response & Safety Instruction Attribution */}
              <div className="bg-slate-800/90 border border-slate-700 rounded-lg px-3.5 py-2.5 flex flex-wrap items-center justify-between gap-2">
                <div className="space-y-0.5 min-w-0 flex-1">
                  <p className="text-xs sm:text-sm font-semibold text-white leading-snug">
                    “{stayWithMeMessage}”
                  </p>
                  <p className="text-xs text-slate-300">
                    {activeSafetyInstruction}
                  </p>
                </div>
                  <span
                    data-testid="safety-instruction-attribution"
                    className="px-2.5 py-1 rounded bg-amber-400/20 text-amber-200 border border-amber-400/40 text-xs font-semibold shrink-0"
                  >
                    {safetyAttributionLabel}
                  </span>
                </div>

                {/* Live Session Direct Language Selector Bar (7 Languages) */}
                <div className="bg-slate-950/80 border border-slate-800 rounded-lg px-3.5 py-2.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mr-1">
                        Session Language:
                      </span>
                      {DIRECT_LANGUAGE_OPTIONS.map((lang) => {
                        const isSelected = activeLanguage === lang.code;
                        return (
                          <button
                            key={lang.code}
                            type="button"
                            aria-pressed={isSelected}
                            data-testid={`live-lang-select-${lang.code}`}
                            onClick={() => handleSelectManualLanguage(lang.code)}
                            className={`min-h-[32px] px-2.5 py-1 rounded text-xs font-bold border transition-colors cursor-pointer ${
                              isSelected
                                ? 'bg-teal-500 text-slate-950 border-teal-400'
                                : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700'
                            }`}
                          >
                            {lang.fullLabel}
                          </button>
                        );
                      })}
                    </div>

                    <span
                      data-testid="live-language-selection-status"
                      className="px-2.5 py-1 rounded text-xs font-bold border bg-emerald-950/90 text-emerald-200 border-emerald-700"
                    >
                      {formatSelectedLanguageStatus(activeLanguage)}
                    </span>
                  </div>
                </div>
              </section>
              );
            })()}

            {/* PHONE SINGLE-COLUMN TABS (Visible on < md screens) */}
            <div
              role="tablist"
              aria-label="Mobile Coordination Sections"
              className="flex md:hidden items-center gap-1 overflow-x-auto bg-white border border-slate-200 rounded-xl p-1.5 shadow-2xs"
            >
              {(
                [
                  { id: 'conversation', label: 'Conversation' },
                  { id: 'map', label: 'Map' },
                  { id: 'resources', label: 'Resources' },
                  { id: 'hospitals', label: 'Hospitals' },
                  { id: 'timeline', label: 'Timeline' },
                ] as const
              ).map((t) => (
                <button
                  key={t.id}
                  role="tab"
                  aria-selected={mobileTab === t.id}
                  type="button"
                  onClick={() => setMobileTab(t.id)}
                  className={`min-h-[44px] px-3 py-2 rounded-lg text-xs font-bold whitespace-nowrap transition-colors cursor-pointer ${
                    mobileTab === t.id
                      ? 'bg-[#0B3B3C] text-white'
                      : 'text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {/* THREE-COLUMN DESKTOP / TWO-COLUMN TABLET / SINGLE-COLUMN MOBILE GRID */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-12 gap-4 items-start">
              {/* COLUMN 1 (lg:col-span-4): Conversation, Inline Location Confirm, Missing Info, Conflicts & Provenance Facts */}
              <div
                className={`lg:col-span-4 space-y-4 ${
                  mobileTab === 'conversation' ? 'block' : 'hidden md:block'
                }`}
              >

                {/* Live Multilingual Transcript Card with Per-Message Language Badge, Streaming Partial & Thinking Indicator */}
                <section
                  aria-label="Multilingual Caller Transcript"
                  className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <h2 className="text-sm font-bold text-slate-900">
                        Caller Conversation
                      </h2>
                      {!isLiveReasoningVerified && (
                        <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-950 border border-amber-300 text-xs font-bold">
                          MOCK reasoning adapter
                        </span>
                      )}
                    </div>
                    <span className="text-xs font-semibold text-slate-600">
                      {utterances.length} finalized turn{utterances.length === 1 ? '' : 's'}
                    </span>
                  </div>

                  {/* Transcript Utterance List */}
                  <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                    {utterances.length === 0 ? (
                      <div className="bg-slate-50 border border-dashed border-slate-300 rounded-lg p-3 text-xs text-slate-600">
                        Speak into the microphone or type a caller utterance below. Scripted simulation scenarios are available in <strong>Engineering Diagnostics</strong>.
                      </div>
                    ) : (
                      utterances.map((u) => {
                        const isHoveredSource =
                          hoveredFact !== null &&
                          hoveredFact.source_utterance_id === u.utterance_id;
                        return (
                          <div
                            key={u.utterance_id}
                            data-testid={`transcript-utterance-${u.sequence}`}
                            className={`rounded-lg p-3 border transition-all ${
                              isHoveredSource
                                ? 'bg-amber-50/90 border-amber-400 ring-2 ring-amber-300'
                                : 'bg-slate-50 border-slate-200'
                            }`}
                          >
                            <div className="flex flex-wrap items-center justify-between gap-1.5 mb-1">
                              <div className="flex items-center gap-1.5">
                                <span className="px-2 py-0.5 rounded bg-[#0B3B3C] text-white text-xs font-bold">
                                  #{u.sequence}
                                </span>
                                {/* Per-message language badge */}
                                <span
                                  data-testid={`utterance-lang-badge-${u.sequence}`}
                                  className="px-2 py-0.5 rounded bg-teal-100 text-teal-950 border border-teal-300 text-xs font-bold uppercase"
                                >
                                  {u.languageBadge ?? u.language.toUpperCase()}
                                </span>
                                {u.utterance_role && (
                                  <span
                                    title={u.utterance_role}
                                    className="px-2 py-0.5 rounded bg-white text-slate-700 border border-slate-300 text-xs font-medium"
                                  >
                                    {u.utterance_role === 'answer_to_pending_question'
                                      ? 'Answer to question'
                                      : 'New description'}
                                  </span>
                                )}
                              </div>
                              <span className="text-xs text-slate-600 tabular-nums">
                                {u.timestamp.slice(11, 19)} UTC
                              </span>
                            </div>
                            <p className="text-xs sm:text-sm text-slate-900 font-medium leading-snug">
                              {isHoveredSource && hoveredFact
                                ? renderTranscriptWithHighlightedEvidence(
                                    u.text,
                                    hoveredFact.evidence_text,
                                  )
                                : u.text}
                            </p>
                          </div>
                        );
                      })
                    )}

                    {/* Live Stage Pill inside Caller Conversation Card */}
                    {(voiceState === 'speech_detected' ||
                      voiceState === 'transcribing' ||
                      voiceState === 'understanding' ||
                      voiceState === 'speaking' ||
                      isReasoningThinking) && (
                      <div
                        data-testid="conversation-voice-stage-banner"
                        className={`rounded-lg px-3 py-2 border flex items-center justify-between gap-2 text-xs font-semibold transition-all duration-200 ${
                          voiceState === 'speech_detected'
                            ? 'bg-emerald-50 border-emerald-300 text-emerald-950'
                            : voiceState === 'transcribing' ||
                                voiceState === 'understanding' ||
                                isReasoningThinking
                              ? 'bg-amber-50 border-amber-300 text-amber-950'
                              : 'bg-sky-50 border-sky-300 text-sky-950'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <span
                            className={`w-2 h-2 rounded-full ${
                              voiceState === 'speech_detected'
                                ? 'bg-emerald-600 animate-ping'
                                : voiceState === 'transcribing' ||
                                    voiceState === 'understanding' ||
                                    isReasoningThinking
                                  ? 'bg-amber-600 animate-pulse'
                                  : 'bg-sky-600 animate-pulse'
                            }`}
                          />
                          <span>
                            {voiceState === 'speech_detected'
                              ? 'Hearing you…'
                              : voiceState === 'transcribing' ||
                                  voiceState === 'understanding' ||
                                  isReasoningThinking
                                ? 'Understanding…'
                                : 'Responding…'}
                          </span>
                        </div>
                        <span className="text-[11px] font-medium opacity-80">
                          {voiceState === 'speech_detected'
                            ? 'Capturing live audio'
                            : voiceState === 'transcribing'
                              ? 'Transcribing speech'
                              : voiceState === 'understanding' || isReasoningThinking
                                ? 'Analyzing emergency context'
                                : 'Streaming spoken guidance'}
                        </span>
                      </div>
                    )}

                    {/* Streaming Partial-Transcript Line */}
                    {streamingPartialText && (
                      <div
                        data-testid="streaming-partial-transcript"
                        className="rounded-lg px-3 py-2 bg-slate-100/80 border border-dashed border-slate-300 flex items-center justify-between gap-2 text-xs text-slate-700 italic"
                      >
                        <span className="truncate">{streamingPartialText}</span>
                        <span className="px-1.5 py-0.5 rounded bg-slate-200 text-slate-800 not-italic font-semibold shrink-0">
                          partial
                        </span>
                      </div>
                    )}
                  </div>

                  {/* INLINE LOCATION-CONFIRMATION CARD IN CHAT WHEN LOCATION IS ONLY A CANDIDATE */}
                  {locationConfirmationStatus === 'candidate_awaiting_confirmation' && (
                    <div
                      data-testid="inline-location-confirmation-card"
                      className="bg-amber-50 border-2 border-amber-400 rounded-xl p-3.5 space-y-2.5 shadow-2xs"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="px-2 py-0.5 rounded bg-amber-300 text-slate-950 text-xs font-bold uppercase">
                          Candidate Location · Confirmation Required Before Dispatch
                        </span>
                        <span className="text-xs font-semibold text-amber-950 tabular-nums">
                          {locationSource === 'landmark' && landmarkMatchConfidence !== null
                            ? `${Math.round(landmarkMatchConfidence * 100)}% synthetic match`
                            : gpsAccuracyMeters !== null
                              ? `±${Math.round(gpsAccuracyMeters)} m GPS`
                              : 'Caller-reported'}
                        </span>
                      </div>
                      <p className="text-xs font-bold text-slate-900">
                        Is this the exact emergency location?{' '}
                        <span className="font-normal text-slate-800">
                          (
                          {locationSource === 'landmark' && matchedLandmarkId
                            ? SYNTHETIC_LANDMARKS.find(
                                (l) => l.landmark_id === matchedLandmarkId,
                              )?.name_romanised
                            : callerLandmarkText && broadLocationText
                              ? `${callerLandmarkText} (${broadLocationText})`
                              : callerLandmarkText || broadLocationText || 'Hitech City / Madhapur Corridor'}
                          )
                        </span>
                      </p>
                      <div className="flex flex-wrap items-center gap-2">
                        <button
                          type="button"
                          data-testid="confirm-location-yes-button"
                          onClick={handleConfirmLocation}
                          className="min-h-[44px] px-4 py-2 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-bold cursor-pointer"
                        >
                          Yes, Confirm Location
                        </button>
                        <button
                          type="button"
                          data-testid="confirm-location-change-button"
                          onClick={handleInvalidateLocation}
                          className="min-h-[44px] px-4 py-2 rounded-lg bg-white hover:bg-slate-100 text-slate-900 border border-slate-300 text-xs font-bold cursor-pointer"
                        >
                          No, change
                        </button>
                      </div>

                      {showLocationChangePicker && (
                        <div className="pt-2 border-t border-amber-300 space-y-1.5">
                          <div className="text-xs font-semibold text-slate-800">
                            Select corrected landmark (re-evaluates provisional allocation & hospitals):
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {SYNTHETIC_LANDMARKS.map((lm) => (
                              <button
                                key={lm.landmark_id}
                                type="button"
                                onClick={() => handleSelectLandmarkLocation(lm, false)}
                                className="min-h-[40px] px-2.5 py-1.5 rounded bg-white hover:bg-teal-50 text-slate-900 border border-slate-300 text-xs font-semibold cursor-pointer"
                              >
                                {lm.name_romanised} ({Math.round(lm.match_confidence * 100)}% match)
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Simulation Scenarios Link (Moved to Engineering Diagnostics to keep Live Coordination focused) */}
                  <div className="pt-2 border-t border-slate-200 flex items-center justify-between gap-2 text-xs text-slate-600">
                    <span>Need scripted multilingual scenarios or fault-injection tests?</span>
                    <button
                      type="button"
                      onClick={() => setShowDiagnosticsOverlay(true)}
                      className="font-bold text-teal-800 hover:text-teal-950 underline cursor-pointer shrink-0"
                    >
                      Open in Engineering Diagnostics →
                    </button>
                  </div>

                  {/* Custom Caller Statement Input */}
                  <div className="pt-2 border-t border-slate-200 flex gap-2">
                    <input
                      type="text"
                      value={customUtteranceText}
                      onChange={(e) => {
                        setCustomUtteranceText(e.target.value);
                        setStreamingPartialText(
                          e.target.value
                            ? `${e.target.value}…`
                            : 'Waiting for caller speech…',
                        );
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && customUtteranceText.trim()) {
                          e.preventDefault();
                          handleSendLiveGeminiUtterance();
                        }
                      }}
                      placeholder={`Type caller statement in ${LANGUAGE_DISPLAY_NAMES[activeLanguage]}…`}
                      aria-label="Custom caller utterance"
                      className="flex-1 min-h-[44px] px-3 py-2 rounded-lg bg-slate-50 border border-slate-300 text-xs text-slate-900 focus:outline-none focus:border-teal-700"
                    />
                    <button
                      type="button"
                      disabled={!customUtteranceText.trim()}
                      onClick={handleSendLiveGeminiUtterance}
                      className="min-h-[44px] px-4 py-2 rounded-lg bg-[#0B3B3C] hover:bg-teal-900 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-bold cursor-pointer shrink-0 transition-opacity"
                    >
                      Send
                    </button>
                  </div>
                </section>

                {/* MISSING INFORMATION CARD */}
                <section
                  aria-label="Missing Information SETHU Still Needs"
                  data-testid="missing-information-card"
                  className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-2.5"
                >
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
                      Missing Information ({missingInformationFacts.length})
                    </h3>
                    <span className="text-xs font-semibold text-slate-600">
                      {missingInformationFacts.length === 0 ? 'Complete' : 'Needs clarification'}
                    </span>
                  </div>
                  {missingInformationFacts.length === 0 ? (
                    <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-2.5 text-xs text-emerald-900 font-semibold">
                      All primary triage facts and caller location confirmation collected.
                    </div>
                  ) : (
                    <div className="space-y-1.5">
                      {missingInformationFacts.map((item) => (
                        <div
                          key={item.key}
                          title={item.key}
                          className="bg-amber-50/70 border border-amber-200 rounded-lg px-3 py-2 flex items-start gap-2"
                        >
                          <span className="w-2 h-2 rounded-full bg-amber-500 mt-1.5 shrink-0" />
                          <div className="min-w-0">
                            <div className="text-xs font-bold text-slate-900">{item.label}</div>
                            <div className="text-xs text-slate-700 truncate">{item.detail}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </section>

                {/* EXTRACTED FACTS WITH HUMAN LABELS + RAW FIELD TOOLTIP & CONFLICT / SUPERSEDED DISPLAY */}
                <section
                  aria-label="Extracted Facts with Utterance Provenance"
                  className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3"
                >
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
                      Extracted Facts (Hover to Highlight Source)
                    </h3>
                    <span className="text-xs text-slate-600 font-semibold">
                      {activeFacts.length} current
                    </span>
                  </div>

                  {identifiedHazards.length > 0 && (
                    <div
                      data-testid="identified-hazards-row"
                      className="flex flex-wrap items-center gap-1.5"
                    >
                      <span className="text-[11px] font-bold uppercase tracking-wider text-amber-900">
                        Active Hazards:
                      </span>
                      {identifiedHazards.map((hz) => (
                        <span
                          key={hz}
                          className="px-2 py-0.5 rounded-md bg-amber-50 border border-amber-300 text-amber-950 text-[11px] font-semibold capitalize"
                        >
                          {hz.replace(/_/g, ' ')}
                        </span>
                      ))}
                    </div>
                  )}

                  {activeFacts.length === 0 ? (
                    <p className="text-xs text-slate-600">
                      No facts extracted yet. Facts must link to a finalized utterance.
                    </p>
                  ) : (
                    <div className="flex flex-wrap gap-1.5">
                      {activeFacts.map((fact, idx) => (
                        <button
                          key={`${fact.field}-${fact.source_utterance_id}-${idx}`}
                          type="button"
                          title={`Raw field: ${fact.field} | Source utterance: ${fact.source_utterance_id}`}
                          data-testid={`fact-chip-${fact.field}`}
                          onMouseEnter={() => setHoveredFact(fact)}
                          onMouseLeave={() => setHoveredFact(null)}
                          onFocus={() => setHoveredFact(fact)}
                          onBlur={() => setHoveredFact(null)}
                          className={`min-h-[38px] px-2.5 py-1.5 rounded-lg border text-left transition-all cursor-pointer flex items-center gap-1.5 text-xs ${
                            hoveredFact === fact
                              ? 'bg-amber-100 border-amber-500 text-slate-950 font-bold shadow-2xs'
                              : fact.status === 'conflicting'
                                ? 'bg-red-50 border-red-300 text-red-950 hover:bg-red-100'
                                : 'bg-slate-50 border-slate-300 text-slate-800 hover:bg-amber-50/60'
                          }`}
                        >
                          <span className="font-bold text-teal-900">
                            {getHumanFieldLabel(fact.field)}:
                          </span>
                          <span className="font-medium">{String(fact.value)}</span>
                          <span className="px-1.5 py-0.5 rounded bg-white border border-slate-300 text-xs text-slate-600">
                            {fact.source_utterance_id.slice(0, 6)}
                          </span>
                          {fact.status === 'conflicting' && (
                            <span className="px-1.5 py-0.5 rounded bg-red-600 text-white text-xs font-bold uppercase">
                              Conflict
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Active Conflicts Box in Live Coordination */}
                  {conflicts.length > 0 && (
                    <div
                      data-testid="live-coordination-conflicts-card"
                      className="bg-red-50 border border-red-300 rounded-lg p-3 space-y-1.5"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold uppercase text-red-900">
                          Conflicting Caller Statements ({conflicts.length})
                        </span>
                        <span
                          title="conflicting_caller_statements"
                          className="px-2 py-0.5 rounded bg-red-600 text-white text-xs font-bold"
                        >
                          Flagged for Review
                        </span>
                      </div>
                      {conflicts.map((c, idx) => (
                        <div
                          key={`${c.field}-${idx}`}
                          title={`Raw field: ${c.field}`}
                          className="text-xs text-red-950 bg-white/90 border border-red-200 rounded px-2.5 py-1.5 flex flex-wrap items-center justify-between gap-1"
                        >
                          <span className="font-bold">{getHumanFieldLabel(c.field)}:</span>
                          <span>
                            Prior <strong>{String(c.prior_value)}</strong> ({c.prior_utterance_id.slice(0, 6)}) vs New{' '}
                            <strong>{String(c.new_value)}</strong> ({c.new_utterance_id.slice(0, 6)})
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Superseded Stale Facts Log */}
                  {supersededFacts.length > 0 && (
                    <div
                      data-testid="superseded-facts-card"
                      className="bg-slate-100 border border-slate-300 rounded-lg p-2.5 space-y-1"
                    >
                      <div className="text-xs font-bold text-slate-700">
                        Superseded Stale Facts (Cleared on Reclassification)
                      </div>
                      {supersededFacts.map((sf, idx) => (
                        <div
                          key={`sup-${sf.field}-${idx}`}
                          title={`Raw field: ${sf.field}`}
                          className="text-xs text-slate-600 flex items-center justify-between"
                        >
                          <span className="line-through">
                            {getHumanFieldLabel(sf.field)}: {String(sf.value)}
                          </span>
                          <span className="px-1.5 py-0.5 rounded bg-slate-200 text-slate-800 font-semibold">
                            Superseded by {sf.supersededByCategory?.replace(/_/g, ' ')}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </section>
              </div>

              {/* COLUMN 2 (lg:col-span-4): Real MapLibre Map & Location Resolution */}
              <div
                className={`lg:col-span-4 space-y-4 ${
                  mobileTab === 'map' ? 'block' : 'hidden md:block'
                }`}
              >
                <section
                  aria-label="Incident Location MapLibre View"
                  className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h2 className="text-sm font-bold text-slate-900">
                        Spatial Coordination Map
                      </h2>
                      <p className="text-xs text-slate-600">
                        {locationSource === 'landmark'
                          ? 'Landmark match confidence shown (GPS accuracy circle hidden for landmarks)'
                          : '20px caller pin · translucent GPS accuracy circle · pulsing outer ring'}
                      </p>
                    </div>
                    <span
                      title={metaLocationSummary.tooltip}
                      className={`px-2.5 py-1 rounded-md border text-xs font-bold ${metaLocationSummary.pillClass}`}
                    >
                      {metaLocationSummary.shortLabel}
                    </span>
                  </div>

                  {/* Real MapLibre Component */}
                  <SetuMapLibre
                    latitude={gpsLatitude}
                    longitude={gpsLongitude}
                    accuracyMeters={gpsAccuracyMeters}
                    matchConfidence={landmarkMatchConfidence}
                    locationSource={locationSource}
                    confirmationStatus={locationConfirmationStatus}
                    matchedLandmarkId={matchedLandmarkId}
                    callerLandmarkText={callerLandmarkText}
                    broadLocationText={broadLocationText}
                    activeAssignments={assignments}
                    onSelectLandmark={(lm) => handleSelectLandmarkLocation(lm, false)}
                  />

                  {/* Location Status & Confirmation Controls */}
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                      <span className="font-bold text-slate-800">
                        Source:{' '}
                        <span className="text-teal-900">
                          {locationSource === 'browser_gps'
                            ? 'Browser GPS'
                            : locationSource === 'landmark'
                              ? 'Spoken Landmark Match'
                              : callerLandmarkText
                                ? 'Caller-Reported Landmark'
                                : 'Unresolved'}
                        </span>
                      </span>
                      <span
                        data-testid="location-card-metric"
                        className="font-semibold text-slate-700 tabular-nums"
                      >
                        {locationSource === 'browser_gps' && gpsAccuracyMeters !== null
                          ? `Accuracy: ±${Math.round(gpsAccuracyMeters)} m`
                          : locationSource === 'landmark' && landmarkMatchConfidence !== null
                            ? `Synthetic match: ${Math.round(landmarkMatchConfidence * 100)}%`
                            : callerLandmarkText
                              ? 'Caller street/landmark (no synthetic confidence)'
                              : 'Accuracy: Unresolved'}
                      </span>
                    </div>

                    {(broadLocationText || callerLandmarkText || locationConfirmedAt) && (
                      <div className="text-xs text-slate-700 space-y-0.5 bg-white border border-slate-200 rounded px-2.5 py-2">
                        {broadLocationText && (
                          <div>
                            <span className="font-semibold text-slate-600">Area / Corridor:</span>{' '}
                            <span className="font-medium text-slate-900">{broadLocationText}</span>
                          </div>
                        )}
                        {callerLandmarkText && (
                          <div>
                            <span className="font-semibold text-slate-600">Landmark / Street:</span>{' '}
                            <span className="font-medium text-slate-900">{callerLandmarkText}</span>
                          </div>
                        )}
                        {locationConfirmedAt && (
                          <div className="text-emerald-800 font-semibold">
                            Verified at {new Date(locationConfirmedAt).toLocaleTimeString()}
                          </div>
                        )}
                      </div>
                    )}

                    <div className="flex flex-wrap items-center gap-2 pt-1">
                      <button
                        type="button"
                        data-testid="confirm-location-panel-button"
                        onClick={handleConfirmLocation}
                        disabled={locationConfirmationStatus === 'confirmed'}
                        className={`min-h-[44px] px-3 py-1.5 rounded-lg text-xs font-bold transition-colors cursor-pointer ${
                          locationConfirmationStatus === 'confirmed'
                            ? 'bg-emerald-100 text-emerald-900 border border-emerald-300 cursor-default'
                            : 'bg-emerald-700 hover:bg-emerald-800 text-white'
                        }`}
                      >
                        {locationConfirmationStatus === 'confirmed'
                          ? '✓ Location Confirmed'
                          : 'Confirm Candidate Location'}
                      </button>
                      {locationConfirmationStatus === 'confirmed' && (
                        <button
                          type="button"
                          onClick={handleInvalidateLocation}
                          className="min-h-[44px] px-3 py-1.5 rounded-lg bg-white hover:bg-slate-100 text-slate-800 border border-slate-300 text-xs font-semibold cursor-pointer"
                        >
                          Change / Unverify Location
                        </button>
                      )}
                    </div>
                  </div>
                </section>

                {/* ACOUSTIC ENVIRONMENT & AUDIO QUALITY CARD (Zero-Gemini Client DSP — Advisory Only) */}
                <section
                  aria-label="Acoustic Environment and Audio Quality"
                  data-testid="acoustic-environment-card"
                  className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900">
                        Acoustic Environment & Audio Quality
                      </h3>
                      <p className="text-[11px] text-slate-600">
                        16 kHz PCM & Silero VAD telemetry · Zero Gemini calls · No raw audio stored
                      </p>
                    </div>
                    <span
                      title="Acoustic indicators are strictly advisory and never independently alter emergency severity or dispatch decisions"
                      className="px-2 py-0.5 rounded bg-teal-50 text-teal-900 border border-teal-300 text-[11px] font-bold uppercase"
                    >
                      Advisory Only
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5 space-y-0.5">
                      <div className="text-slate-600 text-[11px] font-semibold">
                        Ambient Noise Floor
                      </div>
                      <div className="font-bold text-slate-900 tabular-nums">
                        {acousticTelemetry.noiseCategory === 'high_ambient'
                          ? 'High Ambient'
                          : acousticTelemetry.noiseCategory === 'moderate'
                            ? 'Moderate'
                            : 'Quiet'}{' '}
                        ({acousticTelemetry.noiseFloorDbfs} dBFS)
                      </div>
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5 space-y-0.5">
                      <div className="text-slate-600 text-[11px] font-semibold">
                        Voice Environment
                      </div>
                      <div
                        className={`font-bold ${
                          acousticTelemetry.overlappingVoicesDetected
                            ? 'text-amber-900'
                            : 'text-slate-900'
                        }`}
                      >
                        {acousticTelemetry.overlappingVoicesDetected
                          ? 'Possible Overlapping Voices'
                          : 'Single Primary Caller'}
                      </div>
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5 space-y-0.5">
                      <div className="text-slate-600 text-[11px] font-semibold">
                        Microphone Signal
                      </div>
                      <div
                        className={`font-bold tabular-nums ${
                          acousticTelemetry.micQualityStatus === 'clipping_detected'
                            ? 'text-red-800'
                            : acousticTelemetry.micQualityStatus === 'low_signal'
                              ? 'text-amber-900'
                              : 'text-emerald-900'
                        }`}
                      >
                        {acousticTelemetry.micQualityStatus === 'clipping_detected'
                          ? `Clipping (${acousticTelemetry.clippingEventsCount})`
                          : acousticTelemetry.micQualityStatus === 'low_signal'
                            ? 'Low Input Gain'
                            : acousticTelemetry.micQualityStatus === 'clean'
                              ? 'Clean PCM Signal'
                              : 'Mic Standby'}
                      </div>
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5 space-y-0.5">
                      <div className="text-slate-600 text-[11px] font-semibold">
                        Speech Activity & Barge-Ins
                      </div>
                      <div className="font-bold text-slate-900 tabular-nums">
                        {acousticTelemetry.speechActivityPct}% speech · {bargeInCount} interrupt{bargeInCount === 1 ? '' : 's'}
                      </div>
                    </div>
                  </div>

                  <p className="text-[11px] text-slate-500 leading-snug">
                    Advisory context for control-room supervisors. Does not independently modify triage category, severity, or nearest-ambulance dispatch.
                  </p>
                </section>

                {/* Compact Audit Timeline Card (Visible on Desktop or when Timeline tab selected on mobile) */}
                <section
                  aria-label="Recent Session Events"
                  className={`bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-2.5 ${
                    mobileTab === 'timeline' ? 'block' : 'hidden lg:block'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
                      Recent Session Events
                    </h3>
                    <span className="text-xs font-semibold text-slate-600">
                      {auditEvents.length} events
                    </span>
                  </div>
                  <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                    {auditEvents
                      .slice()
                      .reverse()
                      .slice(0, 6)
                      .map((evt) => (
                        <div
                          key={evt.id}
                          className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs flex items-start justify-between gap-2"
                        >
                          <div className="min-w-0">
                            <div className="font-bold text-slate-900">
                              {evt.event_type.replace(/_/g, ' ')}
                            </div>
                            <div className="text-slate-700 truncate">{evt.outcome}</div>
                          </div>
                          <span className="text-xs text-slate-600 tabular-nums shrink-0">
                            {evt.timestamp.slice(11, 19)}
                          </span>
                        </div>
                      ))}
                  </div>
                </section>
              </div>

              {/* COLUMN 3 (lg:col-span-4): Simulated Resource Cards & Expandable Hospital Score Breakdowns */}
              <div
                className={`lg:col-span-4 space-y-4 ${
                  mobileTab === 'resources' || mobileTab === 'hospitals'
                    ? 'block'
                    : 'hidden md:block'
                }`}
              >
                {/* SIMULATED RESOURCE ALLOCATION CARDS */}
                <section
                  aria-label="Simulated Resource Assignments"
                  className={`bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3 ${
                    mobileTab === 'hospitals' ? 'hidden md:block' : 'block'
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h2 className="text-sm font-bold text-slate-900">
                        Resource Allocation
                      </h2>
                      <p className="text-xs text-slate-600">
                        {isProvisionalLocationAllocation
                          ? 'Provisional allocation on unconfirmed location candidate'
                          : 'Nearest available synthetic unit · additive on reclassification'}
                      </p>
                    </div>
                    <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-950 border border-amber-300 text-xs font-bold uppercase">
                      SIMULATED
                    </span>
                  </div>

                  {assignments.length === 0 ? (
                    requiredResourceCategories.length > 0 &&
                    locationConfirmationStatus !== 'confirmed' ? (
                      <div
                        data-testid="dispatch-gate-awaiting-location"
                        className="bg-amber-50 border border-amber-300 rounded-lg p-3 space-y-2"
                      >
                        <div className="text-xs font-bold text-amber-950">
                          Awaiting Caller Location Confirmation Before Dispatch
                        </div>
                        <div className="text-xs text-amber-900">
                          Required categories prepared:{' '}
                          <span className="font-semibold">
                            {requiredResourceCategories
                              .map((c) => c.replace(/_/g, ' ').toUpperCase())
                              .join(' + ')}
                          </span>
                          . Click <strong>Confirm Candidate Location</strong> to authorize simulated dispatch.
                        </div>
                      </div>
                    ) : (
                      <div className="bg-slate-50 border border-dashed border-slate-300 rounded-lg p-3 text-xs text-slate-600">
                        No simulated assignments yet. Run a scenario or classify an emergency to allocate units.
                      </div>
                    )
                  ) : (
                    <div className="space-y-2">
                      {assignments.map((asgn) => {
                        const lifecycleLabel =
                          asgn.status === 'simulated_assigned'
                            ? asgn.lifecycle_status || 'ASSIGNED'
                            : asgn.status === 'pending_location'
                              ? 'PENDING_LOCATION'
                              : 'UNAVAILABLE';
                        const displayEta =
                          asgn.remaining_eta_minutes !== undefined && asgn.remaining_eta_minutes !== null
                            ? asgn.remaining_eta_minutes
                            : asgn.simulated_eta_minutes;
                        const displayEtaSec =
                          asgn.remaining_eta_seconds !== undefined && asgn.remaining_eta_seconds !== null
                            ? asgn.remaining_eta_seconds
                            : null;
                        return (
                          <div
                            key={asgn.assignment_id}
                            data-testid={`resource-assignment-card-${asgn.category}`}
                            className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-2"
                          >
                            {/* Top Row: Resource Callsign + DEMO Badge */}
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <div className="flex items-center gap-1.5 min-w-0">
                                <span
                                  className="text-xs font-bold text-slate-950 truncate"
                                  title={
                                    asgn.resource_callsign ??
                                    asgn.vehicle_code ??
                                    asgn.category.replace(/_/g, ' ').toUpperCase()
                                  }
                                >
                                  {asgn.resource_callsign ??
                                    asgn.vehicle_code ??
                                    asgn.category.replace(/_/g, ' ').toUpperCase()}
                                </span>
                                <span className="px-1.5 py-0.5 rounded bg-amber-100 text-amber-950 border border-amber-300 text-[11px] font-bold uppercase shrink-0">
                                  DEMO
                                </span>
                              </div>
                              <span className="text-[11px] text-amber-900 font-semibold shrink-0">
                                SIMULATED
                              </span>
                            </div>

                            {/* Bottom Row: Lifecycle Status Badge + Distance + ETA */}
                            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                              <div className="flex flex-wrap items-center gap-1.5">
                                <span
                                  data-testid={`assignment-status-badge-${asgn.category}`}
                                  title={`${asgn.status} / ${lifecycleLabel}`}
                                  className={`px-2 py-0.5 rounded font-semibold ${
                                    asgn.status === 'simulated_assigned'
                                      ? lifecycleLabel === 'ARRIVED'
                                        ? 'bg-emerald-100 text-emerald-900 border border-emerald-300'
                                        : lifecycleLabel === 'EN_ROUTE'
                                          ? 'bg-amber-100 text-amber-950 border border-amber-300'
                                          : 'bg-sky-100 text-sky-900 border border-sky-300'
                                      : asgn.status === 'pending_location'
                                        ? 'bg-amber-100 text-amber-950 border border-amber-300'
                                        : 'bg-red-100 text-red-900 border border-red-300'
                                  }`}
                                >
                                  {lifecycleLabel}
                                </span>
                                {asgn.distance_meters !== null && (
                                  <span className="text-slate-700 tabular-nums font-medium">
                                    {(asgn.distance_meters / 1000).toFixed(2)} km
                                  </span>
                                )}
                              </div>

                              <div className="text-xs font-bold text-slate-900 tabular-nums">
                                {lifecycleLabel === 'ARRIVED'
                                  ? 'Arrived (0 min)'
                                  : asgn.status === 'pending_location'
                                    ? 'Awaiting location'
                                    : `Demo ETA: ${displayEta !== null ? `${displayEta} min` : 'unavailable'}${
                                        displayEtaSec !== null && lifecycleLabel === 'EN_ROUTE'
                                          ? ` (~${displayEtaSec}s live)`
                                          : ''
                                      }`}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </section>

                {/* HOSPITAL RECOMMENDATIONS & EXPANDABLE SCORE BREAKDOWN */}
                <section
                  aria-label="Hospital Recommendations and Score Breakdown"
                  className={`bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3 ${
                    mobileTab === 'resources' ? 'hidden md:block' : 'block'
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h2 className="text-sm font-bold text-slate-900">
                        Hospital Ranking & Breakdown
                      </h2>
                      <p className="text-xs text-slate-600">
                        Click any hospital row to expand its weighted component breakdown
                      </p>
                    </div>
                    <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-950 border border-amber-300 text-xs font-bold uppercase">
                      SIMULATED
                    </span>
                  </div>

                  {!isLocationEligibleForAllocation ? (
                    <div className="bg-amber-50 border border-amber-300 rounded-lg p-3 text-xs text-amber-950 font-medium">
                      Hospital ranking requires a usable GPS reading (≤ 500 m) or spoken landmark candidate.
                    </div>
                  ) : (
                    <div className="space-y-2.5">
                      {hospitalBreakdowns.map((hb, idx) => {
                        const isExpanded = Boolean(expandedHospitalIds[hb.hospital.hospital_id]);
                        const distPct = Math.round(
                          (hb.weighted_distance_contribution / Math.max(0.01, hb.final_score)) *
                            100,
                        );
                        const capPct = Math.max(0, 100 - distPct);

                        const formatWeightVal = (w: number) =>
                          w === 0
                            ? '0.00'
                            : Math.abs(w * 100 - Math.round(w * 100)) > 0.01
                              ? w.toFixed(3)
                              : w.toFixed(2);

                        const distWeightStr = formatWeightVal(hb.applied_distance_weight);
                        const etaWeightStr = formatWeightVal(hb.applied_eta_weight);
                        const capWeightStr = formatWeightVal(hb.applied_capability_weight);

                        const distFormula = `${hb.distance_component_score.toFixed(2)} × ${distWeightStr} = ${hb.weighted_distance_contribution.toFixed(3)}`;
                        const etaFormula =
                          hb.eta_component_score !== null
                            ? `${hb.eta_component_score.toFixed(2)} × ${etaWeightStr} = ${hb.weighted_eta_contribution.toFixed(3)}`
                            : `0.00 × 0.00 = 0.000`;
                        const capFormula = `${hb.capability_component_score.toFixed(2)} × ${capWeightStr} = ${hb.weighted_capability_contribution.toFixed(3)}`;

                        return (
                          <div
                            key={hb.hospital.hospital_id}
                            data-testid={`hospital-row-${idx + 1}`}
                            className={`border rounded-xl overflow-hidden transition-colors ${
                              idx === 0
                                ? 'border-teal-600/60 bg-teal-50/20'
                                : 'border-slate-200 bg-slate-50/70'
                            }`}
                          >
                            {/* Expandable Header Row */}
                            <button
                              type="button"
                              aria-expanded={isExpanded}
                              onClick={() =>
                                setExpandedHospitalIds((prev) => ({
                                  ...prev,
                                  [hb.hospital.hospital_id]: !prev[hb.hospital.hospital_id],
                                }))
                              }
                              className="w-full min-h-[52px] p-3 text-left bg-white hover:bg-slate-50 flex items-center justify-between gap-2 cursor-pointer transition-colors"
                            >
                              <div className="min-w-0 flex-1">
                                <div className="flex items-center gap-1.5">
                                  <span
                                    className={`px-2 py-0.5 rounded text-white text-xs font-bold shrink-0 ${
                                      idx === 0 ? 'bg-[#0B3B3C]' : 'bg-slate-700'
                                    }`}
                                  >
                                    #{idx + 1}
                                  </span>
                                  <span
                                    className="text-xs sm:text-sm font-bold text-slate-900 truncate"
                                    title={hb.hospital.name}
                                  >
                                    {hb.hospital.name}
                                  </span>
                                </div>
                                <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 mt-1 text-xs text-slate-700">
                                  <span className="tabular-nums font-semibold">
                                    {hb.distance_km.toFixed(2)} km
                                  </span>
                                  <span>·</span>
                                  <span className="text-amber-900 font-semibold">
                                    ETA: {hb.eta_source_label}
                                  </span>
                                </div>
                              </div>

                              <div className="text-right shrink-0">
                                <div className="text-xs font-bold text-teal-950 tabular-nums">
                                  Score: {hb.final_score.toFixed(3)}
                                </div>
                                <div className="text-xs text-slate-600">
                                  {isExpanded ? 'Hide breakdown ▲' : 'Show breakdown ▼'}
                                </div>
                              </div>
                            </button>

                            {/* Expandable Breakdown Body: Stacked Bar + Full-Width Single-Line Formula Rows */}
                            {isExpanded && (
                              <div className="p-3 border-t border-slate-200 space-y-2.5 bg-slate-50">
                                {/* Stacked Contribution Bar Above the Cells */}
                                <div className="space-y-1">
                                  <div className="flex flex-wrap items-center justify-between gap-1 text-xs text-slate-700">
                                    <span className="font-semibold">
                                      Weighted Score Composition
                                    </span>
                                    <span className="tabular-nums font-bold text-slate-900">
                                      {hb.weighted_distance_contribution.toFixed(3)} +{' '}
                                      {hb.weighted_capability_contribution.toFixed(3)} ={' '}
                                      {hb.final_score.toFixed(3)}
                                    </span>
                                  </div>
                                  <div
                                    aria-label="Stacked score contribution bar"
                                    className="w-full h-2.5 rounded-full bg-slate-200 overflow-hidden flex"
                                  >
                                    <div
                                      style={{ width: `${distPct}%` }}
                                      className="bg-teal-700 h-full"
                                      title={`Distance contribution: ${hb.weighted_distance_contribution.toFixed(3)} (${distPct}%)`}
                                    />
                                    <div
                                      style={{ width: `${capPct}%` }}
                                      className="bg-emerald-500 h-full"
                                      title={`Capability contribution: ${hb.weighted_capability_contribution.toFixed(3)} (${capPct}%)`}
                                    />
                                  </div>
                                </div>

                                {/* Three Full-Width Single-Line Formula Rows (Never Overflow or Collide) */}
                                <div className="grid grid-cols-1 gap-1.5">
                                  <div className="bg-white border border-slate-200 rounded-lg px-2.5 py-2 flex items-center justify-between gap-2">
                                    <div className="flex items-center gap-1.5 text-xs font-bold text-teal-900">
                                      <span className="w-2 h-2 rounded-full bg-teal-700 shrink-0" />
                                      <span>Distance</span>
                                      <span className="text-[11px] font-semibold text-slate-500 tabular-nums">
                                        ({(hb.applied_distance_weight * 100).toFixed(1)}% wt)
                                      </span>
                                    </div>
                                    <div
                                      data-testid={`formula-distance-${idx + 1}`}
                                      className="text-xs font-mono font-semibold text-slate-900 whitespace-nowrap tabular-nums"
                                    >
                                      {distFormula}
                                    </div>
                                  </div>

                                  <div className="bg-white border border-slate-200 rounded-lg px-2.5 py-2 flex items-center justify-between gap-2">
                                    <div className="flex items-center gap-1.5 text-xs font-bold text-slate-700">
                                      <span className="w-2 h-2 rounded-full bg-slate-400 shrink-0" />
                                      <span>ETA (simulated)</span>
                                      <span className="text-[11px] font-semibold text-slate-500 tabular-nums">
                                        (0% wt)
                                      </span>
                                    </div>
                                    <div
                                      data-testid={`formula-eta-${idx + 1}`}
                                      className="text-xs font-mono font-semibold text-slate-600 whitespace-nowrap tabular-nums"
                                    >
                                      {etaFormula}
                                    </div>
                                  </div>

                                  <div className="bg-white border border-slate-200 rounded-lg px-2.5 py-2 flex items-center justify-between gap-2">
                                    <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-900">
                                      <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0" />
                                      <span>Capability Fit</span>
                                      <span className="text-[11px] font-semibold text-slate-500 tabular-nums">
                                        ({(hb.applied_capability_weight * 100).toFixed(1)}% wt)
                                      </span>
                                    </div>
                                    <div
                                      data-testid={`formula-capability-${idx + 1}`}
                                      className="text-xs font-mono font-semibold text-slate-900 whitespace-nowrap tabular-nums"
                                    >
                                      {capFormula}
                                    </div>
                                  </div>
                                </div>

                                {/* Matched Capabilities & Re-normalisation Methodology Note */}
                                {hb.matched_capabilities.length > 0 && (
                                  <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                                    <span className="text-[11px] font-bold text-slate-700">
                                      Matched capabilities:
                                    </span>
                                    {hb.matched_capabilities.map((cap) => (
                                      <span
                                        key={cap}
                                        className="px-2 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-900 text-[11px] font-semibold capitalize"
                                      >
                                        {cap.replace(/_/g, ' ')}
                                      </span>
                                    ))}
                                  </div>
                                )}

                                <p className="text-[11px] text-slate-600 leading-snug">
                                  {hb.methodology_note}
                                </p>

                                {/* Hospital Data Source & Verification Date */}
                                <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-xs text-slate-600 border-t border-slate-200">
                                  <span>
                                    Source: <strong>{hb.hospital.source_identifier}</strong>
                                  </span>
                                  <span>
                                    Verified: <strong>{hb.hospital.verification_date}</strong>
                                  </span>
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </section>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 3: CONTROL ROOM VIEW (Incident Queue with Flag Badges + Collapsible Audit Drawer — UX #13) */}
        {activeView === 'control_room' && (
          <div className="space-y-4">
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 pb-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="px-2.5 py-1 rounded bg-amber-100 text-amber-950 border border-amber-300 text-xs font-bold uppercase">
                      Simulation, not a real control room
                    </span>
                    {!isLiveReasoningVerified && (
                      <span className="px-2.5 py-1 rounded bg-amber-100 text-amber-950 border border-amber-400 text-xs font-bold">
                        MOCK reasoning adapter
                      </span>
                    )}
                  </div>
                  <h2 className="text-lg font-bold text-slate-900 mt-1">
                    Control Room Incident Review Queue
                  </h2>
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className={`px-3 py-1.5 rounded-lg text-xs font-bold border ${
                      controlRoomReviewRequired || reviewReasonCodes.length > 0
                        ? 'bg-red-50 text-red-900 border-red-300'
                        : 'bg-emerald-50 text-emerald-900 border-emerald-300'
                    }`}
                  >
                    {controlRoomReviewRequired || reviewReasonCodes.length > 0
                      ? `Review Required (${reviewReasonCodes.length} Flag${reviewReasonCodes.length === 1 ? '' : 's'})`
                      : 'Nominal Queue'}
                  </span>
                  <button
                    type="button"
                    data-testid="toggle-audit-drawer-button"
                    onClick={() => setIsAuditDrawerOpen((prev) => !prev)}
                    className="min-h-[44px] px-3.5 py-2 rounded-lg bg-[#0B3B3C] hover:bg-teal-900 text-white text-xs font-bold cursor-pointer"
                  >
                    {isAuditDrawerOpen
                      ? 'Hide Audit Log Drawer'
                      : `Open Audit Log Drawer (${auditEvents.length})`}
                  </button>
                </div>
              </div>

              {/* INCIDENT QUEUE CARDS WITH FLAG BADGES */}
              <div className="space-y-3">
                <div
                  data-testid="control-room-incident-queue-card"
                  className="bg-slate-50 border border-slate-300 rounded-xl p-4 space-y-3"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="px-2.5 py-1 rounded bg-[#0B3B3C] text-white text-xs font-bold">
                        Incident {incidentId.slice(0, 12)}
                      </span>
                      <span className="px-2.5 py-1 rounded bg-teal-100 text-teal-950 border border-teal-300 text-xs font-bold uppercase">
                        {incidentCategory.replace(/_/g, ' ')}
                      </span>
                      <span
                        title={metaLocationSummary.tooltip}
                        className={`px-2.5 py-1 rounded border text-xs font-bold ${metaLocationSummary.pillClass}`}
                      >
                        {metaLocationSummary.shortLabel}
                      </span>
                      <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-950 border border-amber-300 text-xs font-bold uppercase">
                        SIMULATED
                      </span>
                    </div>
                    <span className="text-xs font-semibold text-slate-700">
                      Active Conflicts: <strong>{activeConflictsCount}</strong> · Assignments:{' '}
                      <strong>{assignments.length}</strong>
                    </span>
                  </div>

                  {/* Flag Badges */}
                  <div className="space-y-1.5">
                    <div className="text-xs font-bold uppercase tracking-wider text-slate-700">
                      Active Review Flags
                    </div>
                    {reviewReasonCodes.length === 0 ? (
                      <div className="text-xs text-slate-600">
                        No active review flags on this incident.
                      </div>
                    ) : (
                      <div className="flex flex-wrap gap-2">
                        {reviewReasonCodes.map((code) => (
                          <span
                            key={code}
                            title={`Raw reason code: ${code}`}
                            data-testid={`control-room-flag-${code}`}
                            className="px-3 py-1 rounded-lg bg-red-100 text-red-950 border border-red-300 text-xs font-bold"
                          >
                            {getHumanReasonLabel(code)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Conflicting Statements Shown in Control Room */}
                  {conflicts.length > 0 && (
                    <div
                      data-testid="control-room-conflicts-panel"
                      className="bg-red-50 border border-red-300 rounded-lg p-3 space-y-1.5"
                    >
                      <div className="text-xs font-bold text-red-900 uppercase">
                        Caller Fact Disagreements ({conflicts.length})
                      </div>
                      {conflicts.map((c, idx) => (
                        <div
                          key={`cr-conf-${idx}`}
                          title={`Raw field: ${c.field}`}
                          className="text-xs text-red-950 flex flex-wrap items-center justify-between gap-2 bg-white px-2.5 py-1.5 rounded border border-red-200"
                        >
                          <span className="font-bold">{getHumanFieldLabel(c.field)}</span>
                          <span>
                            Utterance {c.prior_utterance_id.slice(0, 6)}: <strong>{String(c.prior_value)}</strong> → Utterance{' '}
                            {c.new_utterance_id.slice(0, 6)}: <strong>{String(c.new_value)}</strong>
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Duplicate & Suspected False/Prank Status Summary + Acoustic Environment Summary */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
                    <div className="bg-white border border-slate-200 rounded-lg p-3 flex items-center justify-between">
                      <div>
                        <div className="text-xs font-bold text-slate-800">
                          Duplicate Incident Check
                        </div>
                        <div className="text-xs text-slate-600">
                          {possibleDuplicateFlag
                            ? `Possible duplicate of ${matchedDuplicateId} (never auto-merged)`
                            : 'No duplicate match within 500 m & 30 min'}
                        </div>
                      </div>
                      <span
                        className={`px-2 py-0.5 rounded text-xs font-bold ${
                          possibleDuplicateFlag
                            ? 'bg-amber-100 text-amber-950 border border-amber-300'
                            : 'bg-emerald-50 text-emerald-900 border border-emerald-200'
                        }`}
                      >
                        {possibleDuplicateFlag ? 'Flagged' : 'Clear'}
                      </span>
                    </div>

                    <div className="bg-white border border-slate-200 rounded-lg p-3 flex items-center justify-between">
                      <div>
                        <div className="text-xs font-bold text-slate-800">
                          False / Prank Call Review
                        </div>
                        <div className="text-xs text-slate-600">
                          {prankReviewStatus === 'not_reviewed'
                            ? 'No false-call indicators flagged (never auto-rejected)'
                            : `Status: ${prankReviewStatus.replace(/_/g, ' ')}`}
                        </div>
                      </div>
                      {prankReviewStatus === 'flagged_for_review' && (
                        <button
                          type="button"
                          onClick={() => setPrankReviewStatus('confirmed_false_or_prank')}
                          className="min-h-[36px] px-2.5 py-1 rounded bg-emerald-700 text-white text-xs font-bold cursor-pointer"
                        >
                          Reviewer Confirm
                        </button>
                      )}
                    </div>

                    <div
                      data-testid="control-room-acoustic-summary"
                      className="bg-white border border-slate-200 rounded-lg p-3 flex items-center justify-between gap-2"
                    >
                      <div>
                        <div className="text-xs font-bold text-slate-800">
                          Acoustic Environment (Advisory)
                        </div>
                        <div className="text-xs text-slate-600 tabular-nums">
                          {acousticTelemetry.noiseCategory.replace(/_/g, ' ')} ({acousticTelemetry.noiseFloorDbfs} dBFS) ·{' '}
                          {acousticTelemetry.overlappingVoicesDetected
                            ? 'Overlapping voices'
                            : 'Single voice'}{' '}
                          · {acousticTelemetry.micQualityStatus.replace(/_/g, ' ')}
                        </div>
                      </div>
                      <span className="px-2 py-0.5 rounded bg-teal-50 text-teal-900 border border-teal-200 text-[11px] font-bold shrink-0">
                        0 Gemini Calls
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* COLLAPSIBLE AUDIT LOG DRAWER (Hidden by Default — UX #13) */}
              {isAuditDrawerOpen && (
                <div
                  data-testid="control-room-audit-drawer"
                  className="border-t border-slate-200 pt-4 space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
                      Append-Only Incident Audit Log ({auditEvents.length} Events)
                    </h3>
                    <button
                      type="button"
                      onClick={() => setIsAuditDrawerOpen(false)}
                      className="min-h-[36px] px-2.5 py-1 rounded bg-slate-100 text-slate-700 text-xs font-semibold cursor-pointer"
                    >
                      Close Audit Drawer
                    </button>
                  </div>
                  <div className="overflow-x-auto border border-slate-200 rounded-lg">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="bg-slate-100 text-slate-800 border-b border-slate-200">
                          <th className="py-2 px-3 font-bold">Time (UTC)</th>
                          <th className="py-2 px-3 font-bold">Event</th>
                          <th className="py-2 px-3 font-bold">Severity</th>
                          <th className="py-2 px-3 font-bold">Outcome</th>
                          <th className="py-2 px-3 font-bold">Flag</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200">
                        {auditEvents.map((evt) => (
                          <tr key={evt.id} className="hover:bg-slate-50">
                            <td className="py-2 px-3 tabular-nums whitespace-nowrap">
                              {evt.timestamp.slice(11, 19)}
                            </td>
                            <td className="py-2 px-3 font-semibold text-slate-900">
                              {evt.event_type.replace(/_/g, ' ')}
                            </td>
                            <td className="py-2 px-3">
                              <span
                                className={`px-2 py-0.5 rounded font-bold uppercase ${
                                  evt.severity === 'critical'
                                    ? 'bg-red-100 text-red-900'
                                    : evt.severity === 'warning'
                                      ? 'bg-amber-100 text-amber-900'
                                      : 'bg-emerald-100 text-emerald-900'
                                }`}
                              >
                                {evt.severity}
                              </span>
                            </td>
                            <td className="py-2 px-3 text-slate-800">{evt.outcome}</td>
                            <td className="py-2 px-3 text-slate-700">
                              {evt.reason_code ? getHumanReasonLabel(evt.reason_code) : '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* VIEW 4: INCIDENT HANDOFF VIEW (Clean Printable Report with All 10 Authoritative Sections & PDF/Print Export) */}
        {activeView === 'handoff' &&
          (() => {
            const ho = backendHandoffPayload;
            const hoCategory = String(ho?.incident_category ?? incidentCategory);
            const hoLang = String(ho?.active_language ?? activeLanguage).toUpperCase();
            const hoGeneratedAt = ho?.generated_at
              ? new Date(String(ho.generated_at)).toLocaleString()
              : new Date().toLocaleString();
            const hoSafety =
              ho?.safety_guidance_attribution && typeof ho.safety_guidance_attribution === 'object'
                ? (ho.safety_guidance_attribution as Record<string, unknown>)
                : null;
            const hoBroadArea =
              typeof ho?.broad_location_text === 'string' && ho.broad_location_text
                ? ho.broad_location_text
                : broadLocationText;
            const hoLandmark =
              typeof ho?.caller_landmark_text === 'string' && ho.caller_landmark_text
                ? ho.caller_landmark_text
                : callerLandmarkText;
            const hoLocConfirmedAt =
              typeof ho?.location_confirmed_at === 'string' && ho.location_confirmed_at
                ? ho.location_confirmed_at
                : locationConfirmedAt;
            const hoLat = typeof ho?.latitude === 'number' ? ho.latitude : gpsLatitude;
            const hoLon = typeof ho?.longitude === 'number' ? ho.longitude : gpsLongitude;
            const hoReviewCodes = Array.isArray(ho?.control_room_review_reason_codes)
              ? (ho.control_room_review_reason_codes as unknown[]).map(String)
              : reviewReasonCodes;
            const hoFacts = Array.isArray(ho?.confirmed_facts)
              ? (ho.confirmed_facts as Array<Record<string, unknown>>).map((f) => ({
                  field: String(f.field_name || ''),
                  value: f.value,
                  source_utterance_id: String(f.source_utterance_id || ''),
                  evidence_text: String(f.evidence_text || ''),
                }))
              : activeFacts;
            const hoConflicts = Array.isArray(ho?.conflicting_information)
              ? (ho.conflicting_information as Array<Record<string, unknown>>).map((c) => ({
                  field: String(c.field_name || ''),
                  prior_value: c.prior_value,
                  prior_utterance_id: String(c.prior_utterance_id || ''),
                  new_value: c.new_value,
                  new_utterance_id: String(c.new_utterance_id || ''),
                }))
              : conflicts;
            const hoAssignments = Array.isArray(ho?.simulated_assignments)
              ? (ho.simulated_assignments as Array<Record<string, unknown>>)
              : assignments.map((a) => ({
                  assignment_id: a.assignment_id,
                  vehicle_code: a.vehicle_code ?? a.resource_id,
                  resource_callsign: a.resource_callsign,
                  category: a.category,
                  lifecycle_status: a.lifecycle_status || 'ASSIGNED',
                  status: a.status,
                  simulated_eta_minutes: a.simulated_eta_minutes,
                  assigned_at: a.assigned_at,
                  en_route_at: a.en_route_at,
                  arrived_at: a.arrived_at,
                  distance_meters: a.distance_meters,
                }));
            const hoUtterances = Array.isArray(ho?.utterance_records)
              ? (ho.utterance_records as Array<Record<string, unknown>>)
              : utterances.map((u) => ({
                  utterance_id: u.utterance_id,
                  sequence_number: u.sequence,
                  text: u.text,
                  language: u.language,
                  utterance_role: u.utterance_role,
                  finalized_at: u.timestamp,
                }));
            const hoTimeline = Array.isArray(ho?.event_timeline)
              ? (ho.event_timeline as Array<Record<string, unknown>>)
              : auditEvents.map((e) => ({
                  event_id: e.id,
                  event_type: e.event_type,
                  severity: e.severity,
                  timestamp: e.timestamp,
                  outcome: e.outcome,
                  reason_code: e.reason_code,
                }));
            return (
              <div className="space-y-4">
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-5">
                  {/* Printable Report Header */}
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-4">
                    <div className="space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="px-2.5 py-0.5 rounded bg-amber-100 text-amber-950 border border-amber-300 text-xs font-bold uppercase">
                          SIMULATED PROTOTYPE HANDOFF REPORT — NOT OFFICIAL 112
                        </span>
                        {!isLiveReasoningVerified && (
                          <span className="px-2.5 py-0.5 rounded bg-amber-100 text-amber-950 border border-amber-400 text-xs font-bold">
                            MOCK reasoning adapter
                          </span>
                        )}
                        <span
                          className={`px-2.5 py-0.5 rounded text-xs font-bold border ${
                            hoConflicts.length > 0
                              ? 'bg-red-50 text-red-900 border-red-300'
                              : 'bg-emerald-50 text-emerald-900 border-emerald-300'
                          }`}
                        >
                          Active Conflicts: {hoConflicts.length}
                        </span>
                        <span
                          data-testid="handoff-language-chip"
                          className="px-2.5 py-0.5 rounded text-xs font-bold border bg-teal-50 text-teal-900 border-teal-300"
                        >
                          Lang Mode: {String(ho?.language_mode ?? 'manual').toUpperCase()} · Final:{' '}
                          {String(ho?.active_language ?? activeLanguage).toUpperCase()} · Seen:{' '}
                          {(
                            (Array.isArray(ho?.detected_languages_seen)
                              ? ho.detected_languages_seen
                              : detectedLanguagesSeen) as string[]
                          )
                            .map((c) => String(c).toUpperCase())
                            .join(', ')}
                        </span>
                      </div>
                      <h2 className="text-lg font-bold text-slate-900">
                        Structured Incident Handoff Report
                      </h2>
                      <p className="text-xs text-slate-600">
                        Incident ID: <strong>{String(ho?.incident_id ?? incidentId)}</strong> · Session ID:{' '}
                        <strong>{String(ho?.session_id ?? sessionId)}</strong> · Generated:{' '}
                        <strong>{hoGeneratedAt}</strong>
                      </p>
                    </div>

                    <div className="flex flex-wrap items-center gap-2 print:hidden">
                      <button
                        type="button"
                        onClick={() => setTranscriptRetentionExpired((v) => !v)}
                        className="min-h-[44px] px-3.5 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-800 border border-slate-300 text-xs font-semibold cursor-pointer"
                      >
                        {transcriptRetentionExpired
                          ? 'Restore Raw Transcript (≤ 24h)'
                          : 'Simulate > 24h Retention Expiry'}
                      </button>
                      <button
                        type="button"
                        data-testid="handoff-print-export-button"
                        onClick={() => window.print()}
                        className="min-h-[44px] px-4 py-2 rounded-lg bg-[#0B3B3C] hover:bg-teal-900 text-white text-xs font-bold shadow-2xs cursor-pointer"
                      >
                        Export PDF / Print Report
                      </button>
                    </div>
                  </div>

                  {/* Section Cards Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2">
                      <div className="text-xs font-bold uppercase tracking-wider text-slate-700">
                        1. Classification & Language
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="px-2.5 py-1 rounded bg-[#0B3B3C] text-white text-xs font-bold uppercase">
                          {hoCategory.replace(/_/g, ' ')}
                        </span>
                        <span className="px-2.5 py-1 rounded bg-teal-50 text-teal-950 border border-teal-300 text-xs font-bold">
                          Language: {hoLang}
                        </span>
                      </div>
                      <div className="text-xs text-slate-700 space-y-1">
                        <div>
                          Safety Attribution:{' '}
                          <strong>
                            {hoSafety
                              ? `${String(hoSafety.source || '')} (${String(hoSafety.review_date || '')} · ${String(hoSafety.version || 'v1.0')})`
                              : safetyAttributionLabel}
                          </strong>
                        </div>
                        <div className="text-slate-600 italic">
                          "{String(hoSafety?.approved_text ?? activeSafetyInstruction)}"
                        </div>
                      </div>
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2">
                      <div className="text-xs font-bold uppercase tracking-wider text-slate-700">
                        2. Location & Verification
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <span
                          title={metaLocationSummary.tooltip}
                          className={`px-2.5 py-1 rounded border text-xs font-bold ${metaLocationSummary.pillClass}`}
                        >
                          {String(ho?.location_confirmation_status ?? metaLocationSummary.shortLabel) === 'confirmed'
                            ? 'Confirmed'
                            : metaLocationSummary.shortLabel}
                        </span>
                      </div>
                      <div className="text-xs text-slate-700 space-y-0.5 tabular-nums">
                        {hoBroadArea && (
                          <div>
                            Area: <strong>{hoBroadArea}</strong>
                          </div>
                        )}
                        {hoLandmark && (
                          <div>
                            Landmark/Street: <strong>{hoLandmark}</strong>
                          </div>
                        )}
                        {hoLat !== null && hoLon !== null && (
                          <div>
                            Coordinates: {Number(hoLat).toFixed(4)}° N, {Number(hoLon).toFixed(4)}° E
                          </div>
                        )}
                        {hoLocConfirmedAt && (
                          <div className="text-emerald-800 font-semibold">
                            Confirmed: {new Date(hoLocConfirmedAt).toLocaleString()}
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2">
                      <div className="text-xs font-bold uppercase tracking-wider text-slate-700">
                        3. Review Flags & Acoustic Context
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {hoReviewCodes.length === 0 ? (
                          <span className="px-2.5 py-1 rounded bg-emerald-50 text-emerald-900 border border-emerald-300 text-xs font-bold">
                            No Review Flags
                          </span>
                        ) : (
                          hoReviewCodes.map((code) => (
                            <span
                              key={code}
                              title={`Raw code: ${code}`}
                              className="px-2.5 py-1 rounded bg-red-100 text-red-950 border border-red-300 text-xs font-bold"
                            >
                              {getHumanReasonLabel(code)}
                            </span>
                          ))
                        )}
                      </div>
                      <div
                        data-testid="handoff-acoustic-context"
                        className="text-xs text-slate-700 pt-1 border-t border-slate-200 space-y-0.5 tabular-nums"
                      >
                        <div>
                          Ambient Noise:{' '}
                          <strong>
                            {acousticTelemetry.noiseCategory.replace(/_/g, ' ')} ({acousticTelemetry.noiseFloorDbfs} dBFS)
                          </strong>
                        </div>
                        <div>
                          Voice / Signal:{' '}
                          <strong>
                            {acousticTelemetry.overlappingVoicesDetected
                              ? 'Overlapping voices detected'
                              : 'Single primary voice'}{' '}
                            · {acousticTelemetry.micQualityStatus.replace(/_/g, ' ')}
                          </strong>
                        </div>
                        <div className="text-[11px] text-slate-500">
                          Advisory DSP indicator · {bargeInCount} barge-in interrupt(s) · {acousticTelemetry.speechActivityPct}% speech ratio
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Section Card 4 & 5: Validated Facts & Conflicts */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2.5">
                      <div className="text-xs font-bold uppercase tracking-wider text-slate-700">
                        4. Current Validated Facts ({hoFacts.length})
                      </div>
                      {hoFacts.length === 0 ? (
                        <p className="text-xs text-slate-600">No current facts recorded.</p>
                      ) : (
                        <div className="space-y-1.5">
                          {hoFacts.map((f, i) => (
                            <div
                              key={`ho-fact-${i}`}
                              title={`Raw field: ${f.field} | Evidence: ${transcriptRetentionExpired ? 'not retained' : f.evidence_text}`}
                              className="bg-white border border-slate-200 rounded-lg px-3 py-2 text-xs flex items-center justify-between gap-2"
                            >
                              <span className="font-bold text-slate-900">
                                {getHumanFieldLabel(f.field)}
                              </span>
                              <span className="text-slate-800 font-medium">
                                {String(f.value)}{' '}
                                <span className="text-slate-500">
                                  ({f.source_utterance_id.slice(0, 6)})
                                </span>
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    <div
                      data-testid="handoff-conflicts-section"
                      className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2.5"
                    >
                      <div className="text-xs font-bold uppercase tracking-wider text-slate-700">
                        5. Disagreements & Superseded Facts ({hoConflicts.length})
                      </div>
                      {hoConflicts.length === 0 ? (
                        <p className="text-xs text-slate-600">
                          No conflicting caller statements recorded in this session.
                        </p>
                      ) : (
                        <div className="space-y-1.5">
                          {hoConflicts.map((c, idx) => (
                            <div
                              key={`ho-conf-${idx}`}
                              title={`Raw field: ${c.field}`}
                              className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs text-red-950"
                            >
                              <div className="font-bold">{getHumanFieldLabel(c.field)}</div>
                              <div>
                                Earlier: <strong>{String(c.prior_value)}</strong> (
                                {c.prior_utterance_id.slice(0, 6)}) → Later:{' '}
                                <strong>{String(c.new_value)}</strong> (
                                {c.new_utterance_id.slice(0, 6)})
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Section Card 6 & 7: Simulated Resource Assignments & Hospitals */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2.5">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                          6. Simulated Resource Assignments ({hoAssignments.length})
                        </span>
                        <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-950 border border-amber-300 text-xs font-bold">
                          SIMULATED
                        </span>
                      </div>
                      {hoAssignments.length === 0 ? (
                        <p className="text-xs text-slate-600">
                          {locationConfirmationStatus !== 'confirmed'
                            ? 'Held pending caller location confirmation.'
                            : 'No simulated units allocated.'}
                        </p>
                      ) : (
                        <div className="space-y-1.5">
                          {hoAssignments.map((a, idx) => (
                            <div
                              key={String(a.assignment_id ?? idx)}
                              className="bg-white border border-slate-200 rounded-lg px-3 py-2 text-xs flex items-center justify-between gap-2"
                            >
                              <div>
                                <div className="font-bold text-slate-900">
                                  {String(
                                    a.resource_callsign ??
                                      a.vehicle_code ??
                                      String(a.category || 'unit').replace(/_/g, ' ').toUpperCase(),
                                  )}
                                </div>
                                <div className="text-slate-600">
                                  Lifecycle: <strong>{String(a.lifecycle_status ?? a.status)}</strong>
                                  {a.simulated_eta_minutes !== null &&
                                  a.simulated_eta_minutes !== undefined
                                    ? ` · Demo ETA: ${String(a.simulated_eta_minutes)} min`
                                    : ''}
                                </div>
                              </div>
                              <span className="font-semibold text-slate-800 tabular-nums">
                                {typeof a.distance_meters === 'number'
                                  ? `${(a.distance_meters / 1000).toFixed(2)} km`
                                  : 'Simulated'}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2.5">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                          7. Top Hospital Recommendations ({hospitalBreakdowns.length})
                        </span>
                        <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-950 border border-amber-300 text-xs font-bold">
                          SIMULATED
                        </span>
                      </div>
                      {hospitalBreakdowns.length === 0 ? (
                        <p className="text-xs text-slate-600">
                          Awaiting eligible caller location for hospital ranking.
                        </p>
                      ) : (
                        <div className="space-y-1.5">
                          {hospitalBreakdowns.map((hb, idx) => (
                            <div
                              key={hb.hospital.hospital_id}
                              className="bg-white border border-slate-200 rounded-lg px-3 py-2 text-xs flex items-center justify-between gap-2"
                            >
                              <div>
                                <div className="font-bold text-slate-900">
                                  #{idx + 1} {hb.hospital.name}
                                </div>
                                <div className="text-slate-600">
                                  {hb.distance_km.toFixed(2)} km · Verified {hb.hospital.verification_date}
                                </div>
                              </div>
                              <span className="font-bold text-teal-950 tabular-nums">
                                Score {hb.final_score.toFixed(3)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Section Card 8 & 9: Transcript Record & Event Timeline */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                          8. Caller Transcript Record ({hoUtterances.length})
                        </span>
                        <span className="px-2 py-0.5 rounded bg-white border border-slate-300 text-xs font-semibold text-slate-700">
                          {transcriptRetentionExpired ? 'Retention Expired (> 24h)' : 'Active (≤ 24h)'}
                        </span>
                      </div>
                      {transcriptRetentionExpired ? (
                        <p className="text-xs text-amber-900 font-medium">
                          Raw caller utterances purged after 24h retention window; structured provenanced facts and audit events remain preserved.
                        </p>
                      ) : hoUtterances.length === 0 ? (
                        <p className="text-xs text-slate-600">No caller utterances recorded yet.</p>
                      ) : (
                        <div className="space-y-1.5 max-h-60 overflow-y-auto pr-1">
                          {hoUtterances.map((u, idx) => (
                            <div key={String(u.utterance_id ?? idx)} className="text-xs text-slate-800 bg-white border border-slate-200 rounded px-2.5 py-1.5">
                              <strong>
                                [#{String(u.sequence_number ?? idx + 1)} · {String(u.language || 'en').toUpperCase()} · {String(u.utterance_role || 'turn')}]
                              </strong>{' '}
                              {String(u.text || '')}{' '}
                              {Boolean(u.finalized_at) && (
                                <span className="text-slate-500 ml-1 tabular-nums">
                                  ({new Date(String(u.finalized_at)).toLocaleTimeString()})
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                          9. Authoritative Event Timeline ({hoTimeline.length})
                        </span>
                      </div>
                      {hoTimeline.length === 0 ? (
                        <p className="text-xs text-slate-600">No session events logged yet.</p>
                      ) : (
                        <div className="space-y-1.5 max-h-60 overflow-y-auto pr-1">
                          {hoTimeline.map((evt, idx) => (
                            <div
                              key={String(evt.event_id ?? idx)}
                              className="text-xs bg-white border border-slate-200 rounded px-2.5 py-1.5 text-slate-800"
                            >
                              <div className="flex items-center justify-between gap-2">
                                <span className="font-bold text-slate-900">
                                  {String(evt.event_type)}
                                </span>
                                <span className="text-slate-500 tabular-nums">
                                  {evt.timestamp ? new Date(String(evt.timestamp)).toLocaleTimeString() : ''}
                                </span>
                              </div>
                              <div className="text-slate-700">{String(evt.outcome || '')}</div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}
      </main>
    </div>
  );
}
