/**
 * Silero VAD (Voice Activity Detection) ONNX Runtime Web Integration.
 *
 * Runs Silero VAD v5 ONNX model in browser WebAssembly to detect speech start,
 * speech end (utterance boundaries), and acoustic caller interruptions (barge-in).
 */

import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime WebAssembly binary path
ort.env.wasm.wasmPaths = '/onnx-wasm/';
ort.env.wasm.numThreads = 1;

export interface SileroVadConfig {
  modelPath?: string;
  sampleRate?: number;
  positiveSpeechThreshold?: number;
  negativeSpeechThreshold?: number;
  redemptionFrames?: number;
  silenceTimeoutMs?: number;
  maxSpeechDurationMs?: number;
  minSpeechDurationMs?: number;
}

export interface VadCallbacks {
  onSpeechStart?: () => void;
  onSpeechEnd?: () => void;
  onVADMisfire?: () => void;
  onProbabilityUpdate?: (probability: number, isSpeaking: boolean) => void;
  onReadyChange?: (ready: boolean, error: string | null) => void;
}

export class SileroVadDetector {
  private session: ort.InferenceSession | null = null;
  private stateTensor: ort.Tensor | null = null;
  private srTensor: ort.Tensor | null = null;
  private isModelLoaded = false;
  private initPromise: Promise<boolean> | null = null;
  private lastError: string | null = null;

  // Audio buffering for 512-sample (32 ms at 16 kHz) chunks + 64-sample Silero v5 context
  private sampleBuffer: number[] = [];
  private readonly chunkSize = 512;
  private readonly contextSize = 64;
  private context: Float32Array = new Float32Array(64);
  private inferenceQueue: Promise<void> = Promise.resolve();

  // Detection thresholds & timing
  private readonly positiveThreshold: number;
  private readonly negativeThreshold: number;
  private readonly silenceTimeoutMs: number;
  private readonly minSpeechDurationMs: number;
  private readonly maxSpeechDurationMs: number;
  private readonly redemptionFrames: number;

  // Runtime tracking
  private isSpeaking = false;
  private speechStartTimestamp = 0;
  private consecutiveSpeechFrames = 0;
  private consecutiveSilenceFrames = 0;
  private silenceFramesThreshold: number;
  private resetEpoch = 0;

  private callbacks: VadCallbacks = {};

  constructor(config: SileroVadConfig = {}, callbacks: VadCallbacks = {}) {
    this.positiveThreshold = config.positiveSpeechThreshold ?? 0.5;
    this.negativeThreshold = config.negativeSpeechThreshold ?? 0.35;
    this.silenceTimeoutMs = config.silenceTimeoutMs ?? 620; // ~620ms pause = responsive utterance boundary
    this.minSpeechDurationMs = config.minSpeechDurationMs ?? 250; // At least 250ms speech
    this.maxSpeechDurationMs = config.maxSpeechDurationMs ?? 15000; // Cap at 15s
    this.redemptionFrames = config.redemptionFrames ?? 2; // ~64ms of speech to trigger speech_start

    // 512 samples = 32 ms. Number of silence chunks to reach silenceTimeoutMs:
    this.silenceFramesThreshold = Math.max(8, Math.round(this.silenceTimeoutMs / 32));
    this.callbacks = callbacks;
  }

  public setCallbacks(callbacks: VadCallbacks): void {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  public get errorMessage(): string | null {
    return this.lastError;
  }

  public async init(modelUrl = '/silero_vad.onnx'): Promise<boolean> {
    if (this.isModelLoaded && this.session) {
      return true;
    }
    if (this.initPromise) {
      return this.initPromise;
    }
    this.initPromise = (async () => {
      try {
        this.session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
        this.resetState();
        this.isModelLoaded = true;
        this.lastError = null;
        this.callbacks.onReadyChange?.(true, null);
        return true;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.warn('Silero VAD ONNX initialization notice:', err);
        this.lastError = msg;
        this.isModelLoaded = false;
        this.callbacks.onReadyChange?.(false, msg);
        this.initPromise = null;
        return false;
      }
    })();
    return this.initPromise;
  }

  public resetState(): void {
    this.resetEpoch += 1;
    if (this.stateTensor) {
      try {
        this.stateTensor.dispose();
      } catch {
        // Ignore already disposed
      }
    }
    // Reset Silero VAD v5 hidden state tensor [2, 1, 128] and 64-sample context window
    this.stateTensor = new ort.Tensor('float32', new Float32Array(2 * 1 * 128), [2, 1, 128]);
    this.context = new Float32Array(this.contextSize);
    if (!this.srTensor) {
      this.srTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(16000)]), []);
    }
    this.sampleBuffer = [];
    this.isSpeaking = false;
    this.consecutiveSpeechFrames = 0;
    this.consecutiveSilenceFrames = 0;
    this.speechStartTimestamp = 0;
  }

  public get isCurrentlySpeaking(): boolean {
    return this.isSpeaking;
  }

  public get isReady(): boolean {
    return this.isModelLoaded && this.session !== null;
  }

  /**
   * Ingest raw Float32 samples resampled to 16 kHz.
   * Serialized via promise queue because onnxruntime-web InferenceSession.run() cannot overlap.
   */
  public processFloat32Samples(samples: Float32Array): Promise<void> {
    if (!this.session || !this.isModelLoaded) {
      return Promise.resolve();
    }

    for (let i = 0; i < samples.length; i++) {
      this.sampleBuffer.push(samples[i]);
    }
    // Backpressure: cap unread sample backlog to 4 seconds (64,000 samples)
    if (this.sampleBuffer.length > 64000) {
      this.sampleBuffer.splice(0, this.sampleBuffer.length - 64000);
    }

    const chunks: Float32Array[] = [];
    while (this.sampleBuffer.length >= this.chunkSize) {
      chunks.push(new Float32Array(this.sampleBuffer.splice(0, this.chunkSize)));
    }
    if (chunks.length === 0) {
      return Promise.resolve();
    }

    const scheduledEpoch = this.resetEpoch;
    this.inferenceQueue = this.inferenceQueue.then(async () => {
      for (const chunk of chunks) {
        if (scheduledEpoch !== this.resetEpoch) {
          return;
        }
        await this.step(chunk, scheduledEpoch);
      }
    });
    return this.inferenceQueue;
  }

  private async step(chunk: Float32Array, expectedEpoch = this.resetEpoch): Promise<void> {
    if (!this.session || !this.stateTensor || !this.srTensor || expectedEpoch !== this.resetEpoch) {
      return;
    }

    let inputTensor: ort.Tensor | null = null;
    try {
      const prevState = this.stateTensor;
      // Silero VAD v5 expects [context(64) + chunk(512)] = 576 samples at 16 kHz
      const fullInput = new Float32Array(this.contextSize + this.chunkSize);
      fullInput.set(this.context, 0);
      fullInput.set(chunk, this.contextSize);
      this.context = new Float32Array(fullInput.subarray(this.chunkSize));
      inputTensor = new ort.Tensor('float32', fullInput, [1, this.contextSize + this.chunkSize]);
      const feeds: Record<string, ort.Tensor> = {
        input: inputTensor,
        state: prevState,
        sr: this.srTensor,
      };

      const results = await this.session.run(feeds);
      this.stateTensor = results.stateN;
      const prob = Number(results.output.data[0]);
      try {
        results.output.dispose();
        if (prevState !== results.stateN) {
          prevState.dispose();
        }
      } catch {
        // Ignore cleanup failures
      }

      if (expectedEpoch === this.resetEpoch) {
        this.evaluateProbability(prob);
      }
    } catch (err) {
      console.warn('Silero VAD inference step error:', err);
    } finally {
      if (inputTensor) {
        try {
          inputTensor.dispose();
        } catch {
          // Ignore
        }
      }
    }
  }

  private evaluateProbability(prob: number): void {
    const now = performance.now();
    this.callbacks.onProbabilityUpdate?.(prob, this.isSpeaking);

    if (prob >= this.positiveThreshold) {
      this.consecutiveSpeechFrames += 1;
      this.consecutiveSilenceFrames = 0;

      if (!this.isSpeaking && this.consecutiveSpeechFrames >= this.redemptionFrames) {
        this.isSpeaking = true;
        this.speechStartTimestamp = now;
        this.callbacks.onSpeechStart?.();
      } else if (this.isSpeaking && now - this.speechStartTimestamp >= this.maxSpeechDurationMs) {
        this.isSpeaking = false;
        this.consecutiveSilenceFrames = 0;
        this.consecutiveSpeechFrames = 0;
        this.callbacks.onSpeechEnd?.();
      }
    } else if (prob < this.negativeThreshold) {
      this.consecutiveSilenceFrames += 1;
      this.consecutiveSpeechFrames = 0;

      if (this.isSpeaking) {
        const speechDuration = now - this.speechStartTimestamp;
        const reachedSilenceLimit = this.consecutiveSilenceFrames >= this.silenceFramesThreshold;
        const reachedMaxDuration = speechDuration >= this.maxSpeechDurationMs;

        if (reachedSilenceLimit || reachedMaxDuration) {
          this.isSpeaking = false;
          this.consecutiveSilenceFrames = 0;
          this.consecutiveSpeechFrames = 0;

          if (speechDuration >= this.minSpeechDurationMs) {
            this.callbacks.onSpeechEnd?.();
          } else {
            this.callbacks.onVADMisfire?.();
          }
        }
      }
    } else if (this.isSpeaking && now - this.speechStartTimestamp >= this.maxSpeechDurationMs) {
      this.isSpeaking = false;
      this.consecutiveSilenceFrames = 0;
      this.consecutiveSpeechFrames = 0;
      this.callbacks.onSpeechEnd?.();
    }
  }

  public reset(): void {
    this.resetEpoch += 1;
    this.isSpeaking = false;
    this.speechStartTimestamp = 0;
    this.consecutiveSpeechFrames = 0;
    this.consecutiveSilenceFrames = 0;
    this.sampleBuffer.length = 0;
    this.context.fill(0);
    if (this.session) {
      try {
        this.stateTensor?.dispose();
        this.stateTensor = new ort.Tensor('float32', new Float32Array(2 * 1 * 128), [2, 1, 128]);
      } catch {
        // Ignore reset tensor errors when uninitialized
      }
    }
  }
}
