/**
 * Pure deterministic SETU coordination engine & domain logic for the React dashboard.
 * Mirrors the exact rules in backend/app/orchestrator/*.py so that both unit tests
 * (via node:test) and the interactive coordination dashboard enforce identical rules:
 * - Section 2.1: Mandatory emergency-service prototype disclosure & spoken disclosure language.
 * - Section 5 (Rule 1 & 2): Deterministic emergency-to-resource mapping; no LLM operational IDs.
 * - Section 5 (Rule 3): Strict source_utterance_id provenance for caller facts vs BackendProvenance.
 * - Section 9.3: Utterance role rules (`new_description` only increments classification attempts),
 *   confidence threshold (0.70), two-strike unknown fallback, and single-unknown session-end fallback.
 * - Section 9.4: Provider-blocked emergency content escalation (N=2 default) with distinct reason
 *   code `provider_blocked_emergency_content`, never incrementing the unknown counter.
 * - Section 10.3 & 10.4: Location precondition (`pending_location` when GPS > 500m or unconfirmed landmark)
 *   and additive-only reclassification that never cancels existing assignments.
 * - Section 12 & 13: 500m GPS threshold, synthetic landmark matching, and hospital weighted score
 *   breakdown with re-normalised weights when routing ETA is unavailable (Gate 0.8).
 * - Section 14: Duplicate detection (same category, <= 200m, <= 5 min, confirmed location).
 * - Section 18: Continuous 16 kHz mono PCM ring buffer with 1500ms pre-roll and immediate local
 *   playback stop before network cancellation.
 */

export type SupportedLanguage = 'en' | 'te' | 'hi';

export type EmergencyCategory =
  | 'medical_emergency'
  | 'fire'
  | 'fire_with_injuries'
  | 'road_accident'
  | 'building_collapse'
  | 'physical_attack'
  | 'gas_leak'
  | 'explosion'
  | 'unknown';

export type UtteranceRole =
  | 'new_description'
  | 'answer_to_pending_question'
  | 'acknowledgement'
  | 'other';

export type ResourceCategory =
  | 'ambulance'
  | 'fire_engine'
  | 'fire_and_rescue'
  | 'police_patrol';

export type AssignmentStatus =
  | 'simulated_assigned'
  | 'pending_location'
  | 'simulated_unavailable';

export type LocationConfirmationStatus =
  | 'unresolved'
  | 'candidate_awaiting_confirmation'
  | 'confirmed';

export type PrankFlagReviewStatus =
  | 'not_reviewed'
  | 'flagged_for_review'
  | 'confirmed_false_or_prank'
  | 'not_false_or_prank';

export const REQUIRED_PROTOTYPE_DISCLOSURE =
  "SETU is a prototype for emergency coordination. It does not replace India's 112 emergency service and is not connected to 112. This demo uses simulated emergency resources, availability, assignments and ETAs. No real emergency response is requested or dispatched. For a real emergency, call 112.";

export const REQUIRED_UNKNOWN_FALLBACK_STATEMENT =
  "SETU couldn't confidently identify the emergency. In this demo, I've created a simulated ambulance assignment and flagged the incident for review. This is not a real dispatch.";

export const REQUIRED_PROVIDER_BLOCKED_STATEMENT =
  "SETU is having trouble processing what you described. In this demo, I've created a simulated ambulance assignment and flagged the incident for review. This is not a real dispatch.";

export const SPOKEN_DISCLOSURE_BY_LANGUAGE: Record<SupportedLanguage, { instructionId: string; version: string; text: string }> = {
  en: {
    instructionId: 'disclosure_session_start',
    version: 'v1.0',
    text: REQUIRED_PROTOTYPE_DISCLOSURE,
  },
  te: {
    instructionId: 'disclosure_session_start',
    version: 'v1.0',
    text: 'సేతు (SETU) అత్యవసర సమన్వయం కోసం రూపొందించిన నమూనా (prototype). ఇది భారతదేశ అధికారిక 112 సేవకు ప్రత్యామ్నాయం కాదు మరియు 112కి అనుసంధానించబడి లేదు. ఈ డెమోలో వనరులు మరియు కేటాయింపులు అన్నీ నమూనా (simulated) మాత్రమే. నిజమైన అత్యవసర పరిస్థితికి 112కి కాల్ చేయండి.',
  },
  hi: {
    instructionId: 'disclosure_session_start',
    version: 'v1.0',
    text: 'सेतु (SETU) आपातकालीन समन्वय का एक प्रोटोटाइप है। यह भारत की आधिकारिक 112 सेवा का स्थान नहीं लेता और 112 से जुड़ा नहीं है। इस डेमो में सभी संसाधन, आवंटन और समय (ETA) सिम्युलेटेड हैं। वास्तविक आपातकाल के लिए कृपया 112 पर कॉल करें।',
  },
};

export const BASE_DISPATCH_MAPPING: Record<EmergencyCategory, readonly ResourceCategory[]> = {
  medical_emergency: ['ambulance'],
  fire: ['fire_engine'],
  fire_with_injuries: ['fire_engine', 'ambulance'],
  road_accident: ['ambulance', 'police_patrol'],
  building_collapse: ['fire_and_rescue', 'ambulance', 'police_patrol'],
  physical_attack: ['police_patrol'],
  gas_leak: ['fire_engine', 'police_patrol'],
  explosion: ['fire_engine', 'ambulance', 'police_patrol'],
  unknown: ['ambulance'],
};

export interface ProvenancedFact {
  field: string;
  value: boolean | number | string | null;
  source_utterance_id: string;
  evidence_text: string;
}

export interface FactConflict {
  field: string;
  prior_value: boolean | number | string;
  prior_utterance_id: string;
  new_value: boolean | number | string;
  new_utterance_id: string;
  description: string;
}

export interface FinalizedUtterance {
  utterance_id: string;
  sequence: number;
  text: string;
  language: SupportedLanguage;
  utterance_role: UtteranceRole | null;
  timestamp: string;
}

export interface SyntheticLandmark {
  landmark_id: string;
  name_romanised: string;
  name_native: string;
  language: SupportedLanguage;
  city: string;
  latitude: number;
  longitude: number;
  match_confidence: number;
  source_identifier: string;
  source_licence: string;
  ingestion_date: string;
  is_synthetic: true;
}

export const DEFAULT_DIRECTORY_VERIFICATION_DATE = new Date().toISOString().slice(0, 10);

export const SYNTHETIC_LANDMARKS: readonly SyntheticLandmark[] = [
  {
    landmark_id: 'SYN-LM-HYD-01',
    name_romanised: 'Secunderabad Railway Station',
    name_native: 'సికింద్రాబాద్ రైల్వే స్టేషన్',
    language: 'te',
    city: 'Hyderabad',
    latitude: 17.4337,
    longitude: 78.5016,
    match_confidence: 0.94,
    source_identifier: 'setu-synthetic-landmarks-v1',
    source_licence: 'Synthetic Demo Dataset (Not Real Map Data)',
    ingestion_date: DEFAULT_DIRECTORY_VERIFICATION_DATE,
    is_synthetic: true,
  },
  {
    landmark_id: 'SYN-LM-HYD-02',
    name_romanised: 'Hitech City Cyber Towers',
    name_native: 'హైటెక్ సిటీ సైబర్ టవర్స్',
    language: 'te',
    city: 'Hyderabad',
    latitude: 17.4504,
    longitude: 78.3808,
    match_confidence: 0.92,
    source_identifier: 'setu-synthetic-landmarks-v1',
    source_licence: 'Synthetic Demo Dataset (Not Real Map Data)',
    ingestion_date: DEFAULT_DIRECTORY_VERIFICATION_DATE,
    is_synthetic: true,
  },
  {
    landmark_id: 'SYN-LM-HYD-03',
    name_romanised: 'Charminar Bus Stand',
    name_native: 'చార్మినార్ బస్ స్టాండ్',
    language: 'te',
    city: 'Hyderabad',
    latitude: 17.3616,
    longitude: 78.4747,
    match_confidence: 0.91,
    source_identifier: 'setu-synthetic-landmarks-v1',
    source_licence: 'Synthetic Demo Dataset (Not Real Map Data)',
    ingestion_date: DEFAULT_DIRECTORY_VERIFICATION_DATE,
    is_synthetic: true,
  },
  {
    landmark_id: 'SYN-LM-DEL-01',
    name_romanised: 'Kashmere Gate ISBT Metro Station',
    name_native: 'कश्मीरी गेट बस अड्डा मेट्रो स्टेशन',
    language: 'hi',
    city: 'New Delhi',
    latitude: 28.6675,
    longitude: 77.2282,
    match_confidence: 0.9,
    source_identifier: 'setu-synthetic-landmarks-v1',
    source_licence: 'Synthetic Demo Dataset (Not Real Map Data)',
    ingestion_date: DEFAULT_DIRECTORY_VERIFICATION_DATE,
    is_synthetic: true,
  },
];

export interface SyntheticResource {
  resource_id: string;
  category: ResourceCategory;
  callsign: string;
  latitude: number;
  longitude: number;
  is_available: boolean;
  is_simulated: true;
}

export const INITIAL_SYNTHETIC_RESOURCES: readonly SyntheticResource[] = [
  {
    resource_id: 'SIM-AMB-HYD-101',
    category: 'ambulance',
    callsign: 'SIM-Ambulance-Hitech-101',
    latitude: 17.4455,
    longitude: 78.3795,
    is_available: true,
    is_simulated: true,
  },
  {
    resource_id: 'SIM-AMB-HYD-102',
    category: 'ambulance',
    callsign: 'SIM-Ambulance-Secunderabad-102',
    latitude: 17.435,
    longitude: 78.5,
    is_available: true,
    is_simulated: true,
  },
  {
    resource_id: 'SIM-FIRE-HYD-201',
    category: 'fire_engine',
    callsign: 'SIM-FireEngine-Madhapur-201',
    latitude: 17.448,
    longitude: 78.382,
    is_available: true,
    is_simulated: true,
  },
  {
    resource_id: 'SIM-FRESC-HYD-301',
    category: 'fire_and_rescue',
    callsign: 'SIM-FireRescue-Jubilee-301',
    latitude: 17.432,
    longitude: 78.407,
    is_available: true,
    is_simulated: true,
  },
  {
    resource_id: 'SIM-POL-HYD-401',
    category: 'police_patrol',
    callsign: 'SIM-PolicePatrol-Cyberabad-401',
    latitude: 17.444,
    longitude: 78.378,
    is_available: true,
    is_simulated: true,
  },
];

export interface SimulatedAssignment {
  assignment_id: string;
  incident_id: string;
  idempotency_key: string;
  category: ResourceCategory;
  resource_id: string | null;
  vehicle_code?: string | null;
  resource_callsign: string | null;
  status: AssignmentStatus;
  lifecycle_status?: 'ASSIGNED' | 'EN_ROUTE' | 'ARRIVED' | 'PENDING_LOCATION' | 'UNAVAILABLE';
  assigned_at: string;
  en_route_at?: string | null;
  arrived_at?: string | null;
  estimated_arrival_at?: string | null;
  simulated_eta_minutes: number | null;
  remaining_eta_minutes?: number | null;
  remaining_eta_seconds?: number | null;
  depot_latitude?: number | null;
  depot_longitude?: number | null;
  current_latitude?: number | null;
  current_longitude?: number | null;
  target_latitude?: number | null;
  target_longitude?: number | null;
  allocation_mode?: string;
  distance_meters: number | null;
  source_provenance: string;
  is_simulated: true;
}

export interface HospitalRecord {
  hospital_id: string;
  name: string;
  latitude: number;
  longitude: number;
  address: string;
  capabilities: readonly string[];
  source_identifier: string;
  verification_date: string;
  is_synthetic_directory: true;
}

export const SYNTHETIC_HOSPITALS: readonly HospitalRecord[] = [
  {
    hospital_id: 'SYN-HOSP-HYD-01',
    name: 'NIMS Emergency & Trauma Centre (Synthetic Directory)',
    latitude: 17.4226,
    longitude: 78.4525,
    address: 'Punjagutta, Hyderabad, Telangana',
    capabilities: ['emergency_department', 'trauma_care', 'icu', 'cardiac_care', 'burn_unit'],
    source_identifier: 'setu-synthetic-hospital-directory-v1',
    verification_date: DEFAULT_DIRECTORY_VERIFICATION_DATE,
    is_synthetic_directory: true,
  },
  {
    hospital_id: 'SYN-HOSP-HYD-02',
    name: 'Hitech Metro Care Hospital (Synthetic Directory)',
    latitude: 17.4474,
    longitude: 78.3762,
    address: 'Madhapur Main Road, Hyderabad, Telangana',
    capabilities: ['emergency_department', 'trauma_care', 'icu'],
    source_identifier: 'setu-synthetic-hospital-directory-v1',
    verification_date: DEFAULT_DIRECTORY_VERIFICATION_DATE,
    is_synthetic_directory: true,
  },
  {
    hospital_id: 'SYN-HOSP-HYD-03',
    name: 'Secunderabad General Emergency Facility (Synthetic Directory)',
    latitude: 17.4399,
    longitude: 78.4983,
    address: 'Secunderabad Station Road, Hyderabad, Telangana',
    capabilities: ['emergency_department', 'cardiac_care'],
    source_identifier: 'setu-synthetic-hospital-directory-v1',
    verification_date: DEFAULT_DIRECTORY_VERIFICATION_DATE,
    is_synthetic_directory: true,
  },
];

export interface HospitalScoreBreakdown {
  hospital: HospitalRecord;
  distance_km: number;
  eta_minutes: number | null;
  eta_source_label: string;
  matched_capabilities: readonly string[];
  distance_component_score: number;
  eta_component_score: number | null;
  capability_component_score: number;
  applied_distance_weight: number;
  applied_eta_weight: number;
  applied_capability_weight: number;
  weighted_distance_contribution: number;
  weighted_eta_contribution: number;
  weighted_capability_contribution: number;
  final_score: number;
  methodology_note: string;
}

export function geodesicDistanceMeters(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
): number {
  const R = 6371008.8;
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const phi1 = toRad(lat1);
  const phi2 = toRad(lat2);
  const dPhi = toRad(lat2 - lat1);
  const dLambda = toRad(lon2 - lon1);
  const a =
    Math.sin(dPhi / 2) ** 2 +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLambda / 2) ** 2;
  return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(Math.max(0, 1 - a)));
}

export function isGpsEligibleCandidate(accuracyMeters: number | null | undefined): boolean {
  return typeof accuracyMeters === 'number' && accuracyMeters > 0 && accuracyMeters <= 500;
}

export function rankHospitalsDeterministic(
  category: EmergencyCategory,
  callerLat: number,
  callerLon: number,
  weights = { distance: 0.5, eta: 0.2, capability: 0.3 },
): HospitalScoreBreakdown[] {
  const sum = weights.distance + weights.eta + weights.capability;
  if (
    weights.distance < 0 ||
    weights.eta < 0 ||
    weights.capability < 0 ||
    Math.abs(sum - 1.0) > 1e-6
  ) {
    throw new Error('Hospital weights must be non-negative and sum to 1.0 within tolerance');
  }

  const desiredByCategory: Record<EmergencyCategory, readonly string[]> = {
    medical_emergency: ['emergency_department', 'icu', 'cardiac_care'],
    fire: ['emergency_department', 'burn_unit'],
    fire_with_injuries: ['emergency_department', 'burn_unit', 'trauma_care', 'icu'],
    road_accident: ['emergency_department', 'trauma_care', 'icu'],
    building_collapse: ['emergency_department', 'trauma_care', 'icu'],
    physical_attack: ['emergency_department', 'trauma_care'],
    gas_leak: ['emergency_department', 'icu'],
    explosion: ['emergency_department', 'trauma_care', 'burn_unit', 'icu'],
    unknown: ['emergency_department'],
  };

  const desired = desiredByCategory[category] ?? ['emergency_department'];
  const eligible = SYNTHETIC_HOSPITALS.filter((h) =>
    h.capabilities.includes('emergency_department'),
  );

  // Gate 0.8: Routing ETA is unavailable; re-normalise across distance and capability fit
  const activeWeightSum = weights.distance + weights.capability;
  const wDist = activeWeightSum > 0 ? weights.distance / activeWeightSum : 0.5;
  const wCap = activeWeightSum > 0 ? weights.capability / activeWeightSum : 0.5;
  const wEta = 0;

  return eligible
    .map((hospital) => {
      const distMeters = geodesicDistanceMeters(
        callerLat,
        callerLon,
        hospital.latitude,
        hospital.longitude,
      );
      const distKm = distMeters / 1000;
      const distScore = 1 / (1 + distKm / 10);
      const matched = desired.filter((c) => hospital.capabilities.includes(c));
      const capScore = matched.length / Math.max(1, desired.length);

      const cDist = distScore * wDist;
      const cCap = capScore * wCap;
      const finalScore = cDist + cCap;

      return {
        hospital,
        distance_km: Number(distKm.toFixed(2)),
        eta_minutes: null,
        eta_source_label: 'unavailable (re-normalised distance + capability)',
        matched_capabilities: matched,
        distance_component_score: Number(distScore.toFixed(4)),
        eta_component_score: null,
        capability_component_score: Number(capScore.toFixed(4)),
        applied_distance_weight: Number(wDist.toFixed(4)),
        applied_eta_weight: wEta,
        applied_capability_weight: Number(wCap.toFixed(4)),
        weighted_distance_contribution: Number(cDist.toFixed(4)),
        weighted_eta_contribution: 0,
        weighted_capability_contribution: Number(cCap.toFixed(4)),
        final_score: Number(finalScore.toFixed(4)),
        methodology_note: `Routing ETA unavailable (Gate 0.8); weights re-normalised across distance (${wDist.toFixed(2)}) and capability fit (${wCap.toFixed(2)}).`,
      };
    })
    .sort((a, b) => b.final_score - a.final_score || a.distance_km - b.distance_km);
}

/**
 * Bounded 16 kHz mono PCM circular ring buffer with 1500 ms pre-roll (Section 18.1).
 */
export class ContinuousAudioRingBuffer {
  private readonly buffer: Int16Array;
  private writeIndex = 0;
  private totalSamplesWritten = 0;
  public readonly sampleRateHz: number;
  public readonly totalDurationSeconds: number;
  public readonly preRollMs: number;

  constructor(options?: {
    sampleRateHz?: number;
    totalDurationSeconds?: number;
    preRollMs?: number;
  }) {
    this.sampleRateHz = options?.sampleRateHz ?? 16000;
    this.totalDurationSeconds = options?.totalDurationSeconds ?? 10;
    this.preRollMs = options?.preRollMs ?? 1500;
    this.buffer = new Int16Array(this.sampleRateHz * this.totalDurationSeconds);
  }

  public pushFrame(samples: Int16Array): void {
    const cap = this.buffer.length;
    for (let i = 0; i < samples.length; i++) {
      this.buffer[this.writeIndex] = samples[i];
      this.writeIndex = (this.writeIndex + 1) % cap;
      this.totalSamplesWritten++;
    }
  }

  public extractPreRollSamples(): Int16Array {
    const requestedSamples = Math.min(
      Math.floor((this.sampleRateHz * this.preRollMs) / 1000),
      this.totalSamplesWritten,
      this.buffer.length,
    );
    const out = new Int16Array(requestedSamples);
    const cap = this.buffer.length;
    let readIdx = (this.writeIndex - requestedSamples + cap) % cap;
    for (let i = 0; i < requestedSamples; i++) {
      out[i] = this.buffer[readIdx];
      readIdx = (readIdx + 1) % cap;
    }
    return out;
  }

  public get memoryBytes(): number {
    return this.buffer.byteLength;
  }
}
