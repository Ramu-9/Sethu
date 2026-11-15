import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BASE_DISPATCH_MAPPING,
  ContinuousAudioRingBuffer,
  DIRECT_LANGUAGE_OPTIONS,
  formatSelectedLanguageStatus,
  getDirectLanguageOption,
  isGpsEligibleCandidate,
  rankHospitalsDeterministic,
  REQUIRED_PROTOTYPE_DISCLOSURE,
  resolveVoiceStagePresentation,
  SPOKEN_DISCLOSURE_BY_LANGUAGE,
  transitionVoiceInteractionState,
  VoiceInteractionState,
  VoiceTransitionContext,
} from './setuEngine';
import { SileroVadDetector } from './sileroVad';

test('Section 2.1: prototype disclosure and spoken disclosures exist for en, te, hi', () => {
  assert.ok(REQUIRED_PROTOTYPE_DISCLOSURE.includes('112'));
  assert.ok(SPOKEN_DISCLOSURE_BY_LANGUAGE.en.text.includes('112'));
  assert.ok(SPOKEN_DISCLOSURE_BY_LANGUAGE.te.text.includes('112'));
  assert.ok(SPOKEN_DISCLOSURE_BY_LANGUAGE.hi.text.includes('112'));
});

test('Section 10.1: base deterministic dispatch mapping covers all 9 categories', () => {
  assert.deepEqual(BASE_DISPATCH_MAPPING.medical_emergency, ['ambulance']);
  assert.deepEqual(BASE_DISPATCH_MAPPING.fire, ['fire_engine']);
  assert.deepEqual(BASE_DISPATCH_MAPPING.fire_with_injuries, ['fire_engine', 'ambulance']);
  assert.deepEqual(BASE_DISPATCH_MAPPING.road_accident, ['ambulance', 'police_patrol']);
  assert.deepEqual(BASE_DISPATCH_MAPPING.building_collapse, [
    'fire_and_rescue',
    'ambulance',
    'police_patrol',
  ]);
  assert.deepEqual(BASE_DISPATCH_MAPPING.physical_attack, ['police_patrol']);
  assert.deepEqual(BASE_DISPATCH_MAPPING.gas_leak, ['fire_engine', 'police_patrol']);
  assert.deepEqual(BASE_DISPATCH_MAPPING.explosion, [
    'fire_engine',
    'ambulance',
    'police_patrol',
  ]);
  assert.deepEqual(BASE_DISPATCH_MAPPING.unknown, ['ambulance']);
});

test('Section 12.1: GPS threshold accepts <= 500m and rejects > 500m or null', () => {
  assert.equal(isGpsEligibleCandidate(42), true);
  assert.equal(isGpsEligibleCandidate(500), true);
  assert.equal(isGpsEligibleCandidate(500.1), false);
  assert.equal(isGpsEligibleCandidate(null), false);
});

test('Section 13.2: Hospital ranking re-normalises weights when ETA is unavailable', () => {
  const ranked = rankHospitalsDeterministic('fire_with_injuries', 17.4435, 78.3772);
  assert.equal(ranked.length, 3);
  const top = ranked[0];
  assert.equal(top.eta_minutes, null);
  assert.equal(top.applied_eta_weight, 0);
  assert.ok(Math.abs(top.applied_distance_weight + top.applied_capability_weight - 1.0) < 1e-3);
});

test('Section 18.1: 16kHz continuous ring buffer preserves 1500ms pre-roll and ~320KB bound', () => {
  const ring = new ContinuousAudioRingBuffer();
  assert.equal(ring.memoryBytes, 320000);
  const frame = new Int16Array(1600);
  for (let i = 0; i < frame.length; i++) {
    frame[i] = i + 1;
  }
  ring.pushFrame(frame);
  const extracted = ring.extractPreRollSamples();
  assert.equal(extracted.length, 1600);
  assert.equal(extracted[0], 1);
  assert.equal(extracted[1599], 1600);
});

test('Voice Lifecycle State Machine: transitions cleanly through Hearing -> Understanding -> Responding -> Listening -> Idle', () => {
  const baseCtx: VoiceTransitionContext = {
    sessionActive: true,
    isTtsPlaying: false,
    isTurnProcessing: false,
    isSpeechStreaming: false,
  };

  let state: VoiceInteractionState = 'idle';
  assert.equal(resolveVoiceStagePresentation(state, false, 0, 0).label, 'Ready');

  // 1. Microphone ready & listening
  state = transitionVoiceInteractionState(state, 'mic_ready_listening', baseCtx);
  assert.equal(state, 'listening');
  assert.equal(resolveVoiceStagePresentation(state, false, 0.005, 0.1).label, 'Listening');

  // 2. Caller begins speaking -> "Hearing you…" with audio-reactive bars
  state = transitionVoiceInteractionState(state, 'caller_speech_started', {
    ...baseCtx,
    isSpeechStreaming: true,
  });
  assert.equal(state, 'speech_detected');
  const hearingPres = resolveVoiceStagePresentation(state, false, 0.08, 0.85);
  assert.equal(hearingPres.stage, 'hearing');
  assert.equal(hearingPres.label, 'Hearing you…');
  assert.equal(hearingPres.animationMode, 'hearing_reactive');
  assert.ok(hearingPres.barHeightsPct[2] > 60);

  // 3. Caller finishes speaking -> immediately transitions to "Understanding…" (never stuck on Hearing you)
  state = transitionVoiceInteractionState(state, 'caller_speech_finalized', {
    ...baseCtx,
    isTurnProcessing: true,
    isSpeechStreaming: false,
  });
  assert.equal(state, 'transcribing');
  const transcribingPres = resolveVoiceStagePresentation(state, false, 0.002, 0.05);
  assert.equal(transcribingPres.stage, 'understanding');
  assert.equal(transcribingPres.label, 'Understanding…');
  assert.equal(transcribingPres.animationMode, 'understanding_pulse');

  // 4. Prisma transcript arrives -> remains in "Understanding…" while Gemini/deterministic reasoning runs
  state = transitionVoiceInteractionState(state, 'transcript_ready_understanding', {
    ...baseCtx,
    isTurnProcessing: true,
  });
  assert.equal(state, 'understanding');
  assert.equal(resolveVoiceStagePresentation(state, false, 0, 0).label, 'Understanding…');

  // 5. Timbre TTS streaming starts -> transitions to "Responding…"
  state = transitionVoiceInteractionState(state, 'tts_response_started', {
    ...baseCtx,
    isTtsPlaying: true,
  });
  assert.equal(state, 'speaking');
  const respondingPres = resolveVoiceStagePresentation(state, true, 0, 0);
  assert.equal(respondingPres.stage, 'responding');
  assert.equal(respondingPres.label, 'Responding…');
  assert.equal(respondingPres.animationMode, 'responding_wave');

  // 6. Playback finishes while session is active -> returns to "Listening"
  state = transitionVoiceInteractionState(state, 'tts_playback_ended', baseCtx);
  assert.equal(state, 'listening');
  assert.equal(resolveVoiceStagePresentation(state, false, 0, 0).label, 'Listening');

  // 7. Session ends -> returns to "Ready" (idle)
  state = transitionVoiceInteractionState(state, 'session_ended', {
    ...baseCtx,
    sessionActive: false,
  });
  assert.equal(state, 'idle');
  assert.equal(resolveVoiceStagePresentation(state, false, 0, 0).label, 'Ready');
});

test('Voice Lifecycle State Machine: prevents race conditions and recovers from errors/timeouts', () => {
  const activeCtx: VoiceTransitionContext = {
    sessionActive: true,
    isTtsPlaying: false,
    isTurnProcessing: false,
    isSpeechStreaming: false,
  };

  // Race 1: Stale orchestration_updated arrives after TTS already started -> must stay in 'speaking'
  const speakingState = transitionVoiceInteractionState('speaking', 'orchestration_updated', {
    ...activeCtx,
    isTtsPlaying: true,
  });
  assert.equal(speakingState, 'speaking');

  // Race 2: Stale orchestration_updated arrives while caller is actively speaking -> must stay in 'speech_detected'
  const capturingState = transitionVoiceInteractionState(
    'speech_detected',
    'orchestration_updated',
    {
      ...activeCtx,
      isSpeechStreaming: true,
    },
  );
  assert.equal(capturingState, 'speech_detected');

  // Race 3: Previous TTS playback finishes while new turn is already processing -> must stay in 'understanding'
  const overlappingTurnState = transitionVoiceInteractionState(
    'transcribing',
    'tts_playback_ended',
    {
      ...activeCtx,
      isTurnProcessing: true,
    },
  );
  assert.equal(overlappingTurnState, 'understanding');

  // Error/Timeout recovery: if STT or Gemini times out during 'transcribing' or 'understanding', returns to 'listening'
  const recoveredFromTranscribing = transitionVoiceInteractionState(
    'transcribing',
    'turn_error_or_timeout',
    activeCtx,
  );
  assert.equal(recoveredFromTranscribing, 'listening');

  const recoveredFromUnderstanding = transitionVoiceInteractionState(
    'understanding',
    'turn_error_or_timeout',
    activeCtx,
  );
  assert.equal(recoveredFromUnderstanding, 'listening');
});

test('SileroVadDetector hysteresis recovery: moderate ambient frames do not trap detector in Hearing state', () => {
  let speechStarted = 0;
  let speechEnded = 0;
  const vad = new SileroVadDetector(
    {
      positiveSpeechThreshold: 0.62,
      negativeSpeechThreshold: 0.40,
      minSpeechDurationMs: 128,
      silenceTimeoutMs: 256,
    },
    {
      onSpeechStart: () => {
        speechStarted += 1;
      },
      onSpeechEnd: () => {
        speechEnded += 1;
      },
    },
  );

  const evalProb = (vad as unknown as { evaluateProbability: (p: number) => void }).evaluateProbability.bind(vad);

  // Trigger speech start (4 frames = 128ms >= 128ms)
  for (let i = 0; i < 4; i++) {
    evalProb(0.85);
  }
  assert.equal(speechStarted, 1);
  assert.equal(speechEnded, 0);

  // Feed hysteresis-band frames (0.45 is between negative=0.40 and positive=0.62)
  // Previously these never incremented silence frames; now 16 frames * 0.5 = 8 silence frames -> onSpeechEnd
  for (let i = 0; i < 16; i++) {
    evalProb(0.45);
  }
  assert.equal(speechEnded, 1);
});

test('Seven direct language options: provides exact bilingual labels and disclosures for all 7 languages', () => {
  assert.equal(DIRECT_LANGUAGE_OPTIONS.length, 7);
  assert.deepEqual(
    DIRECT_LANGUAGE_OPTIONS.map((opt) => opt.code),
    ['en', 'te', 'hi', 'kn', 'ta', 'ml', 'bn'],
  );
  assert.deepEqual(
    DIRECT_LANGUAGE_OPTIONS.map((opt) => opt.fullLabel),
    [
      'English (IN) — English',
      'Telugu — తెలుగు',
      'Hindi — हिन्दी',
      'Kannada — ಕನ್ನಡ',
      'Tamil — தமிழ்',
      'Malayalam — മലയാളം',
      'Bengali — বাংলা',
    ],
  );

  for (const opt of DIRECT_LANGUAGE_OPTIONS) {
    assert.equal(getDirectLanguageOption(opt.code).fullLabel, opt.fullLabel);
    assert.ok(SPOKEN_DISCLOSURE_BY_LANGUAGE[opt.code].text.length > 10);
  }

  assert.equal(formatSelectedLanguageStatus('en'), 'Selected: English (IN) — English');
  assert.equal(formatSelectedLanguageStatus('te'), 'Selected: Telugu — తెలుగు');
  assert.equal(formatSelectedLanguageStatus('hi'), 'Selected: Hindi — हिन्दी');
  assert.equal(formatSelectedLanguageStatus('kn'), 'Selected: Kannada — ಕನ್ನಡ');
  assert.equal(formatSelectedLanguageStatus('ta'), 'Selected: Tamil — தமிழ்');
  assert.equal(formatSelectedLanguageStatus('ml'), 'Selected: Malayalam — മലയാളം');
  assert.equal(formatSelectedLanguageStatus('bn'), 'Selected: Bengali — বাংলা');
});


