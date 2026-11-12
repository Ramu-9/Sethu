/**
 * Unified Web Audio Playback & 16 kHz AudioWorklet Capture Surface for SETU
 * (Sections 2.1, 15.2, 18.1, 18.2, 18.4).
 *
 * Rules Enforced:
 * - Never uses window.speechSynthesis or HTML <audio> elements anywhere.
 * - Pre-synthesized instruction files are fetched from /backend-media/instructions/*,
 *   verified against the SHA-256 digest from GET /api/instructions/manifest using
 *   WebCrypto (crypto.subtle.digest('SHA-256')), and played via AudioContext + AudioBufferSourceNode.
 * - Live Timbre TTS 16 kHz PCM chunks use the exact same AudioContext + AudioBufferSourceNode
 *   queue so barge-in (Section 18.4) has a single synchronous stop surface:
 *     1. Stop active AudioBufferSourceNode synchronously first.
 *     2. Clear queued audio buffers second.
 *     3. Emit WebSocket cancellation to /ws/session third.
 */

import { ContinuousAudioRingBuffer } from './setuEngine';

export interface ManifestAudioAsset {
  instruction_id: string;
  language: 'en' | 'te' | 'hi';
  voice_id: string;
  timbre_version: string;
  text_version: string;
  audio_format: string;
  file_path: string;
  checksum_sha256: string;
  generated_at: string;
  generation_status: string;
  placeholder_notice?: string;
  approved_text: string;
}

export interface BargeInStopResult {
  stoppedImmediately: boolean;
  localStopLatencyMs: number;
  clearedQueueCount: number;
  wasPlayingCachedInstruction: boolean;
}

export async function computeSha256Hex(buffer: ArrayBuffer): Promise<string> {
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

export function arrayBufferToBase64(buffer: ArrayBufferLike): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

export class SetuWebAudioController {
  private audioContext: AudioContext | null = null;
  private activeSourceNode: AudioBufferSourceNode | null = null;
  private scheduledNodes: Set<AudioBufferSourceNode> = new Set();
  private nextScheduledTime = 0;
  private queuedBuffers: Array<{
    buffer: AudioBuffer;
    sourceType: 'cached_instruction' | 'live_timbre';
    onEnded?: () => void;
  }> = [];
  private activeSourceType: 'cached_instruction' | 'live_timbre' = 'cached_instruction';
  private isPlayingState = false;
  private onPlaybackStateChange?: (isPlaying: boolean, sourceType: 'cached_instruction' | 'live_timbre') => void;

  // Microphone & AudioWorklet state
  private mediaStream: MediaStream | null = null;
  private micSourceNode: MediaStreamAudioSourceNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private silentGainNode: GainNode | null = null;
  private workletLoadedCtx: AudioContext | null = null;
  private echoCancellationActive = true;

  constructor(
    onPlaybackStateChange?: (isPlaying: boolean, sourceType: 'cached_instruction' | 'live_timbre') => void,
  ) {
    this.onPlaybackStateChange = onPlaybackStateChange;
  }

  private ensureAudioContext(): AudioContext {
    if (!this.audioContext || this.audioContext.state === 'closed') {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.audioContext = new AudioCtx();
      this.workletLoadedCtx = null;
    }
    if (this.audioContext.state === 'suspended') {
      void this.audioContext.resume();
    }
    return this.audioContext;
  }

  public get isEchoCancellationActive(): boolean {
    return this.echoCancellationActive;
  }

  /**
   * Synchronously create/resume the shared AudioContext inside a user gesture handler.
   */
  public unlockAudioContext(): AudioContext {
    return this.ensureAudioContext();
  }

  /**
   * Fetch, SHA-256 verify, decode, and play a cached instruction WAV asset
   * through AudioContext + AudioBufferSourceNode.
   */
  public async playVerifiedInstructionAsset(
    assetUrl: string,
    expectedSha256: string,
    onEnded?: () => void,
  ): Promise<{
    verified: boolean;
    computedSha256: string;
    playbackStartLatencyMs: number;
  }> {
    const t0 = performance.now();
    const response = await fetch(assetUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch instruction audio from ${assetUrl} (HTTP ${response.status})`);
    }
    const rawBytes = await response.arrayBuffer();
    const computedSha256 = await computeSha256Hex(rawBytes);
    const verified =
      !expectedSha256 || computedSha256.toLowerCase() === expectedSha256.toLowerCase();

    if (!verified) {
      throw new Error(
        `SHA-256 mismatch for ${assetUrl}: expected ${expectedSha256}, computed ${computedSha256}`,
      );
    }

    const ctx = this.ensureAudioContext();
    const audioBuffer = await ctx.decodeAudioData(rawBytes.slice(0));
    this.enqueueAndPlayBuffer(audioBuffer, 'cached_instruction', onEnded);

    return {
      verified: true,
      computedSha256,
      playbackStartLatencyMs: Number((performance.now() - t0).toFixed(2)),
    };
  }

  /**
   * Enqueue a live Timbre 16 kHz 16-bit mono PCM chunk onto the same
   * AudioContext + AudioBufferSourceNode surface.
   */
  public playLiveTimbrePcm16Chunk(
    pcm16Bytes: Int16Array,
    sampleRate = 16000,
    onEnded?: () => void,
  ): void {
    if (pcm16Bytes.length === 0) {
      onEnded?.();
      return;
    }
    const ctx = this.ensureAudioContext();
    const audioBuffer = ctx.createBuffer(1, pcm16Bytes.length, sampleRate);
    const channelData = audioBuffer.getChannelData(0);
    for (let i = 0; i < pcm16Bytes.length; i++) {
      channelData[i] = pcm16Bytes[i] / 32768.0;
    }
    this.scheduleGaplessBuffer(audioBuffer, 'live_timbre', onEnded);
  }

  private scheduleGaplessBuffer(
    buffer: AudioBuffer,
    sourceType: 'cached_instruction' | 'live_timbre',
    onEnded?: () => void,
  ): void {
    const ctx = this.ensureAudioContext();
    const startAt = Math.max(ctx.currentTime + 0.015, this.nextScheduledTime);
    this.nextScheduledTime = startAt + buffer.duration;

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    this.scheduledNodes.add(source);
    this.activeSourceNode = source;
    this.activeSourceType = sourceType;
    if (!this.isPlayingState) {
      this.isPlayingState = true;
      this.onPlaybackStateChange?.(true, sourceType);
    }

    source.onended = () => {
      this.scheduledNodes.delete(source);
      if (this.activeSourceNode === source) {
        this.activeSourceNode = null;
      }
      if (this.scheduledNodes.size === 0 && this.queuedBuffers.length === 0) {
        this.isPlayingState = false;
        this.nextScheduledTime = 0;
        this.onPlaybackStateChange?.(false, this.activeSourceType);
        onEnded?.();
      }
    };

    source.start(startAt);
  }

  public playLiveTimbreBase64Pcm16(
    pcmBase64: string,
    sampleRate = 16000,
    onEnded?: () => void,
  ): void {
    if (!pcmBase64) return;
    const binary = atob(pcmBase64);
    const byteLen = binary.length - (binary.length % 2);
    const bytes = new Uint8Array(byteLen);
    for (let i = 0; i < byteLen; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    const int16 = new Int16Array(bytes.buffer);
    this.playLiveTimbrePcm16Chunk(int16, sampleRate, onEnded);
  }

  private enqueueAndPlayBuffer(
    buffer: AudioBuffer,
    sourceType: 'cached_instruction' | 'live_timbre',
    onEnded?: () => void,
  ): void {
    if (this.isPlayingState && this.activeSourceNode) {
      this.queuedBuffers.push({ buffer, sourceType, onEnded });
      return;
    }
    this.startBufferNode(buffer, sourceType, onEnded);
  }

  private startBufferNode(
    buffer: AudioBuffer,
    sourceType: 'cached_instruction' | 'live_timbre',
    onEnded?: () => void,
  ): void {
    const ctx = this.ensureAudioContext();
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    this.activeSourceNode = source;
    this.activeSourceType = sourceType;
    this.isPlayingState = true;
    this.onPlaybackStateChange?.(true, sourceType);

    source.onended = () => {
      if (this.activeSourceNode === source) {
        this.activeSourceNode = null;
        const next = this.queuedBuffers.shift();
        if (next) {
          this.startBufferNode(next.buffer, next.sourceType, next.onEnded);
        } else {
          this.isPlayingState = false;
          this.onPlaybackStateChange?.(false, this.activeSourceType);
          onEnded?.();
        }
      }
    };

    source.start(0);
  }

  /**
   * Section 18.4 Mandatory Barge-In Ordering:
   * 1. Stop active AudioBufferSourceNode synchronously FIRST.
   * 2. Clear queued audio chunks SECOND.
   * (Caller sends server WebSocket cancellation THIRD.)
   */
  public stopPlaybackImmediately(): BargeInStopResult {
    const t0 = performance.now();
    const wasPlayingCached = this.activeSourceType === 'cached_instruction';
    let clearedScheduledExtra = Math.max(0, this.scheduledNodes.size - 1);

    for (const node of this.scheduledNodes) {
      try {
        node.onended = null;
        node.stop(0);
        node.disconnect();
      } catch {
        // Ignore if already stopped
      }
    }
    this.scheduledNodes.clear();
    this.nextScheduledTime = 0;

    if (this.activeSourceNode) {
      try {
        this.activeSourceNode.onended = null;
        this.activeSourceNode.stop(0);
        this.activeSourceNode.disconnect();
      } catch {
        // Ignore if node already stopped
      }
      this.activeSourceNode = null;
    }

    const clearedCount = this.queuedBuffers.length + clearedScheduledExtra;
    this.queuedBuffers = [];
    this.isPlayingState = false;
    this.onPlaybackStateChange?.(false, this.activeSourceType);

    const elapsedMs = Math.max(0.2, Number((performance.now() - t0).toFixed(2)));
    return {
      stoppedImmediately: true,
      localStopLatencyMs: elapsedMs,
      clearedQueueCount: clearedCount,
      wasPlayingCachedInstruction: wasPlayingCached,
    };
  }

  /**
   * Start 16 kHz mono PCM microphone capture via AudioWorklet (/pcm16-worklet-processor.js),
   * inspect echoCancellation settings per Section 18.2, populate the 10s ring buffer
   * (1500 ms pre-roll), and stream PCM frames over /ws/session.
   */
  public async startMicrophoneWorkletCapture(
    ringBuffer: ContinuousAudioRingBuffer,
    onPcmFrame: (frame: {
      sequence_number: number;
      sample_rate_hz: number;
      rms_level: number;
      pcm_base64: string;
    }) => void,
    onFloat32Frame?: (samples: Float32Array) => void,
  ): Promise<{
    echoCancellationActive: boolean;
    noiseSuppressionActive: boolean;
    autoGainControlActive: boolean;
  }> {
    this.stopMicrophoneCapture();

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    this.mediaStream = stream;

    const audioTrack = stream.getAudioTracks()[0];
    const settings = audioTrack ? audioTrack.getSettings() : {};
    this.echoCancellationActive = settings.echoCancellation !== false;

    const ctx = this.ensureAudioContext();
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    if (this.workletLoadedCtx !== ctx) {
      await ctx.audioWorklet.addModule('/pcm16-worklet-processor.js');
      this.workletLoadedCtx = ctx;
    }

    const micSource = ctx.createMediaStreamSource(stream);
    const worklet = new AudioWorkletNode(ctx, 'setu-pcm16-worklet-processor');
    // Connect through a zero-gain sink to destination so Chromium's audio graph
    // actively schedules process() continuously with zero speaker passthrough.
    const silentGain = ctx.createGain();
    silentGain.gain.value = 0;

    this.micSourceNode = micSource;
    this.workletNode = worklet;
    this.silentGainNode = silentGain;

    worklet.port.onmessage = (event: MessageEvent) => {
      const data = event.data;
      if (data && data.type === 'pcm16_frame' && data.pcm16_buffer instanceof ArrayBuffer) {
        const int16 = new Int16Array(data.pcm16_buffer);
        ringBuffer.pushFrame(int16);

        if (onFloat32Frame) {
          const f32 = new Float32Array(int16.length);
          for (let i = 0; i < int16.length; i++) {
            f32[i] = int16[i] / 32768.0;
          }
          onFloat32Frame(f32);
        }

        onPcmFrame({
          sequence_number: Number(data.sequence_number),
          sample_rate_hz: Number(data.sample_rate_hz || 16000),
          rms_level: Number(data.rms_level || 0),
          pcm_base64: arrayBufferToBase64(data.pcm16_buffer),
        });
      }
    };

    micSource.connect(worklet);
    worklet.connect(silentGain);
    silentGain.connect(ctx.destination);
    return {
      echoCancellationActive: this.echoCancellationActive,
      noiseSuppressionActive: settings.noiseSuppression !== false,
      autoGainControlActive: settings.autoGainControl !== false,
    };
  }

  public stopMicrophoneCapture(): void {
    if (this.workletNode) {
      this.workletNode.port.onmessage = null;
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.micSourceNode) {
      this.micSourceNode.disconnect();
      this.micSourceNode = null;
    }
    if (this.silentGainNode) {
      this.silentGainNode.disconnect();
      this.silentGainNode = null;
    }
    if (this.mediaStream) {
      for (const track of this.mediaStream.getTracks()) {
        track.stop();
      }
      this.mediaStream = null;
    }
  }
}
