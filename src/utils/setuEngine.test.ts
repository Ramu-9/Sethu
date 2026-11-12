import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BASE_DISPATCH_MAPPING,
  ContinuousAudioRingBuffer,
  isGpsEligibleCandidate,
  rankHospitalsDeterministic,
  REQUIRED_PROTOTYPE_DISCLOSURE,
  SPOKEN_DISCLOSURE_BY_LANGUAGE,
} from './setuEngine';

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
