/**
 * SETU 16 kHz Mono PCM AudioWorkletProcessor (Section 18.1 & 18.2).
 *
 * Captures browser microphone Float32 frames, resamples to 16000 Hz 16-bit signed
 * mono PCM (Int16Array), and posts 100 ms (1600-sample / 3200-byte) frames to the
 * main thread for the continuous 10s ring buffer (1500 ms pre-roll) and /ws/session.
 */
class SetuPcm16WorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._targetSampleRate = 16000;
    this._chunkSamples = 1600; // 100 ms at 16 kHz
    this._buffer = new Int16Array(this._chunkSamples);
    this._writeIndex = 0;
    this._sequenceNumber = 0;
    this._resampleAccumulator = 0;
    this._rmsSumSquares = 0;
    this._rmsSampleCount = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }
    const channelData = input[0];
    if (!channelData || channelData.length === 0) {
      return true;
    }

    const ratio = sampleRate / this._targetSampleRate;

    for (let i = 0; i < channelData.length; i++) {
      const sample = channelData[i];
      this._resampleAccumulator += 1;
      if (this._resampleAccumulator >= ratio) {
        this._resampleAccumulator -= ratio;
        const clamped = Math.max(-1, Math.min(1, sample));
        this._rmsSumSquares += clamped * clamped;
        this._rmsSampleCount += 1;
        const pcm16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
        this._buffer[this._writeIndex++] = Math.round(pcm16);

        if (this._writeIndex >= this._chunkSamples) {
          const rms = Math.sqrt(this._rmsSumSquares / Math.max(1, this._rmsSampleCount));
          this._rmsSumSquares = 0;
          this._rmsSampleCount = 0;
          const frameCopy = new Int16Array(this._buffer);
          this.port.postMessage(
            {
              type: 'pcm16_frame',
              sequence_number: this._sequenceNumber++,
              sample_rate_hz: this._targetSampleRate,
              rms_level: rms,
              pcm16_buffer: frameCopy.buffer,
            },
            [frameCopy.buffer],
          );
          this._writeIndex = 0;
        }
      }
    }

    return true;
  }
}

registerProcessor('setu-pcm16-worklet-processor', SetuPcm16WorkletProcessor);
