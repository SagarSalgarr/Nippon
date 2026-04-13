/**
 * react-native-indicai — JS/TS public API
 *
 * This is the ONLY file the React Native app team ever imports.
 * All model downloads, ONNX inference, sha256 verification, and
 * manifest fetching happen inside the native layer — invisible here.
 *
 * Usage:
 *   import IndicAI from 'react-native-indicai';
 *   await IndicAI.init('hi');
 *
 *   // Speech input flow:
 *   const text    = await IndicAI.transcribe(base64Pcm);
 *   const english = await IndicAI.translateToEnglish(text);
 *
 *   // Text input flow:
 *   const english = await IndicAI.translateToEnglish(indicText);
 *
 *   // Response flow (EN → Indic → TTS):
 *   const { indicText, audio } = await IndicAI.respondWithSpeech(englishResponse);
 *
 *   // Or separately:
 *   const indic = await IndicAI.translateToIndic(englishResponse);
 *   const audio = await IndicAI.synthesize(indic);
 */

import {
  NativeModules,
  NativeEventEmitter,
  EventSubscription,
} from 'react-native';

// React Native / Hermes does not ship @types/node, so we declare require here.
declare function require(module: string): any;

const { IndicAIModule } = NativeModules;

if (!IndicAIModule) {
  throw new Error(
    '[react-native-indicai] Native module not linked.\n' +
    'Android: rebuild the app after adding the dependency.\n' +
    'iOS: run `cd ios && pod install`, then rebuild.\n' +
    'If using Expo: use a dev build (not Expo Go).'
  );
}

const emitter = new NativeEventEmitter(IndicAIModule);

// ── Public types ──────────────────────────────────────────────────────────────

export interface DownloadProgress {
  model: string;
  percent: number;
  downloaded: number;
  total: number;
}

export type LanguageCode =
  | 'hi' | 'mr' | 'ta' | 'te' | 'kn'
  | 'ml' | 'bn' | 'gu' | 'pa' | 'ur'
  | string;

export interface TranscribeResult {
  text: string;
  language: LanguageCode;
}

export interface RespondWithSpeechResult {
  indicText: string;
  audioPath: string;  // absolute path to WAV file on device
}

export interface IntentResult {
  intent: string;       // e.g. "check_balance", "create_gullak"
  confidence: number;   // 0.0 – 1.0
}

export interface IndicAIConfig {
  manifestUrl?: string;
  s3BaseUrl?: string;
}

// ── Main API object ───────────────────────────────────────────────────────────

const IndicAI = {

  configure(config: IndicAIConfig): void {
    IndicAIModule.configure(config.manifestUrl ?? null, config.s3BaseUrl ?? null);
  },

  /**
   * Initialise the SDK for a specific language.
   * Downloads Whisper + IndicTrans2 (both directions) models if not cached.
   */
  init(languageCode: LanguageCode): Promise<void> {
    return IndicAIModule.initialize(languageCode);
  },

  /**
   * Transcribe PCM audio to native-script text (Whisper STT).
   * @param base64Audio  Raw PCM Int16 LE, 16 kHz, mono as base64
   */
  transcribe(base64Audio: string): Promise<string> {
    return IndicAIModule.transcribe(base64Audio);
  },

  /**
   * Translate native-script text to English (IndicTrans2 Indic→EN).
   */
  translateToEnglish(text: string): Promise<string> {
    return IndicAIModule.translateToEnglish(text);
  },

  /**
   * Translate English text to the current Indic language (IndicTrans2 EN→Indic).
   */
  translateToIndic(englishText: string): Promise<string> {
    return IndicAIModule.translateToIndic(englishText);
  },

  /**
   * Synthesize speech from text using on-device MMS-TTS.
   * @returns Float32 PCM audio as base64 string, 16 kHz mono
   */
  synthesize(text: string): Promise<string> {
    return IndicAIModule.synthesize(text);
  },

  /**
   * Full response flow: translate English to Indic, then synthesize speech.
   * Combines translateToIndic + synthesize in one native call for efficiency.
   * @returns { indicText, audio } where audio is base64 Float32 PCM
   */
  respondWithSpeech(englishText: string): Promise<RespondWithSpeechResult> {
    return IndicAIModule.respondWithSpeech(englishText);
  },

  /**
   * Classify English text to an intent label using the on-device DistilBERT model.
   * The intent model must be included in the manifest for this to work.
   */
  classifyIntent(englishText: string): Promise<IntentResult> {
    return IndicAIModule.classifyIntent(englishText);
  },

  /**
   * Convenience: transcribe then translate and classify intent in one call.
   * This is the main user-input flow: speak → text → English → intent → APK.
   */
  async transcribeAndTranslate(
    base64Audio: string
  ): Promise<{ nativeText: string; englishText: string; intent?: IntentResult }> {
    const nativeText = await IndicAI.transcribe(base64Audio);
    const englishText = await IndicAI.translateToEnglish(nativeText);
    let intent: IntentResult | undefined;
    try { intent = await IndicAI.classifyIntent(englishText); } catch (_) {}
    return { nativeText, englishText, intent };
  },

  /**
   * Full round-trip for text input:
   * Indic text → English → (send to backend) → English response → Indic → TTS
   */
  async processTextRoundTrip(
    indicText: string,
    getResponse: (english: string) => Promise<string>
  ): Promise<{ english: string; response: string; indicResponse: string; audioPath: string }> {
    const english = await IndicAI.translateToEnglish(indicText);
    const response = await getResponse(english);
    const result = await IndicAI.respondWithSpeech(response);
    return {
      english,
      response,
      indicResponse: result.indicText,
      audioPath: result.audioPath,
    };
  },

  onProgress(callback: (progress: DownloadProgress) => void): () => void {
    const sub: EventSubscription = emitter.addListener(
      'IndicAI_Progress',
      callback
    );
    return () => sub.remove();
  },

  /**
   * Start native microphone recording (raw PCM 16-bit 16kHz mono).
   */
  startRecording(): Promise<void> {
    return IndicAIModule.startRecording();
  },

  /**
   * Stop recording and return base64-encoded raw PCM suitable for transcribe().
   */
  stopRecording(): Promise<string> {
    return IndicAIModule.stopRecording();
  },

  getSupportedLanguages(): Promise<LanguageCode[]> {
    return IndicAIModule.getSupportedLanguages();
  },

  isTtsCached(languageCode: LanguageCode): Promise<boolean> {
    return IndicAIModule.isTtsCached(languageCode);
  },

  release(): void {
    IndicAIModule.release();
  },
};

export default IndicAI;

// ── Auto-configure from .env (via react-native-config) ───────────────────────
// If the consuming app has react-native-config installed and sets
//   INDICAI_MANIFEST_URL=...
//   INDICAI_S3_BASE_URL=...
// in their .env file, the SDK will pick those up automatically on import.
// No manual configure() call is needed in the app.
// If react-native-config is not installed, the SDK silently falls back to
// the built-in production CDN URLs.
(function autoConfigureFromEnv() {
  try {
    const Config = require('react-native-config').default;
    const manifestUrl: string | undefined = Config?.INDICAI_MANIFEST_URL;
    const s3BaseUrl: string | undefined   = Config?.INDICAI_S3_BASE_URL;
    if (manifestUrl || s3BaseUrl) {
      IndicAI.configure({ manifestUrl, s3BaseUrl });
    }
  } catch (_) {
    // react-native-config not installed — use SDK defaults
  }
})();

// ── Audio conversion helpers ──────────────────────────────────────────────────

/** Decode a base64 string to Uint8Array without Node.js Buffer (works in Hermes) */
function b64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/** Encode a Uint8Array to base64 string without Node.js Buffer (works in Hermes) */
function bytesToB64(bytes: Uint8Array): string {
  let binary = '';
  const chunkSize = 8192;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

export async function fileToBase64Pcm(fileUri: string): Promise<string> {
  try {
    const RNFS = require('react-native-fs');
    const b64 = await RNFS.readFile(fileUri, 'base64');
    const bytes = b64ToBytes(b64);
    const isWav = bytes[0] === 0x52 && bytes[1] === 0x49; // "RI" = RIFF
    const pcmBytes = isWav ? bytes.slice(44) : bytes;
    return bytesToB64(pcmBytes);
  } catch {
    throw new Error(
      '[react-native-indicai] fileToBase64Pcm requires react-native-fs.\n' +
      'Install it: npm install react-native-fs'
    );
  }
}

export function base64PcmToWavUri(base64Float32Pcm: string, sampleRate = 16000): string {
  const pcmBytes = b64ToBytes(base64Float32Pcm);
  const numSamples = pcmBytes.length / 4;
  const int16 = new Int16Array(numSamples);

  const view = new DataView(pcmBytes.buffer);
  for (let i = 0; i < numSamples; i++) {
    const f = view.getFloat32(i * 4, true);
    int16[i] = Math.max(-32768, Math.min(32767, Math.round(f * 32767)));
  }

  const wavBuffer = new ArrayBuffer(44 + int16.byteLength);
  const wav = new DataView(wavBuffer);
  const writeStr = (off: number, s: string) =>
    s.split('').forEach((c, i) => wav.setUint8(off + i, c.charCodeAt(0)));

  writeStr(0, 'RIFF');
  wav.setUint32(4, 36 + int16.byteLength, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  wav.setUint32(16, 16, true);
  wav.setUint16(20, 1, true);
  wav.setUint16(22, 1, true);
  wav.setUint32(24, sampleRate, true);
  wav.setUint32(28, sampleRate * 2, true);
  wav.setUint16(32, 2, true);
  wav.setUint16(34, 16, true);
  writeStr(36, 'data');
  wav.setUint32(40, int16.byteLength, true);
  new Int16Array(wavBuffer, 44).set(int16);

  return `data:audio/wav;base64,${bytesToB64(new Uint8Array(wavBuffer))}`;
}
