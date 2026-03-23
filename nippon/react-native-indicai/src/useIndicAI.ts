/**
 * useIndicAI.ts
 * Drop-in React hook for the RN app team.
 * Handles init, progress, transcribe → translate → intent pipeline, and text input flow.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import IndicAI, { DownloadProgress, fileToBase64Pcm } from 'react-native-indicai';

export type SDKStatus = 'idle' | 'downloading' | 'ready' | 'processing' | 'error';

export interface IndicAIState {
  status: SDKStatus;
  progress: DownloadProgress | null;
  nativeText: string;
  englishText: string;
  intent: string;
  intentConfidence: number;
  indicResponse: string;
  ttsAudioUri: string | null;
  error: string | null;
}

export function useIndicAI(languageCode: string) {
  const [state, setState] = useState<IndicAIState>({
    status: 'idle', progress: null,
    nativeText: '', englishText: '',
    intent: '', intentConfidence: 0,
    indicResponse: '', ttsAudioUri: null, error: null,
  });

  const langRef = useRef(languageCode);

  useEffect(() => {
    langRef.current = languageCode;
    setState(s => ({
      ...s, status: 'downloading', error: null,
      nativeText: '', englishText: '', intent: '', intentConfidence: 0, indicResponse: '',
    }));

    const unsub = IndicAI.onProgress(p => {
      setState(s => ({ ...s, progress: p }));
    });

    IndicAI.init(languageCode)
      .then(() => setState(s => ({ ...s, status: 'ready', progress: null })))
      .catch(e => setState(s => ({ ...s, status: 'error', error: String(e) })));

    return () => {
      unsub();
      IndicAI.release();
    };
  }, [languageCode]);

  // ── Classify intent helper ──────────────────────────────────────────────────
  const getIntent = async (englishText: string) => {
    try {
      const result = await IndicAI.classifyIntent(englishText);
      return { intent: result.intent, intentConfidence: result.confidence };
    } catch (e) {
      console.error('[IndicAI] classifyIntent failed:', e);
      return { intent: '', intentConfidence: 0 };
    }
  };

  // ── Process audio file (speech input → STT → MT → intent) ─────────────────
  const processAudioFile = useCallback(async (fileUri: string) => {
    if (state.status !== 'ready') return;
    setState(s => ({ ...s, status: 'processing', error: null }));
    try {
      const b64Pcm = await fileToBase64Pcm(fileUri);
      const native = await IndicAI.transcribe(b64Pcm);
      const english = await IndicAI.translateToEnglish(native);
      const { intent, intentConfidence } = await getIntent(english);
      setState(s => ({ ...s, nativeText: native, englishText: english, intent, intentConfidence, status: 'ready' }));
    } catch (e) {
      setState(s => ({ ...s, status: 'error', error: String(e) }));
    }
  }, [state.status]);

  // ── Process raw base64 PCM (streaming recorder) → STT → MT → intent ───────
  const processBase64Pcm = useCallback(async (base64Pcm: string) => {
    if (state.status !== 'ready') return;
    setState(s => ({ ...s, status: 'processing', error: null }));
    try {
      const native = await IndicAI.transcribe(base64Pcm);
      const english = await IndicAI.translateToEnglish(native);
      const { intent, intentConfidence } = await getIntent(english);
      setState(s => ({ ...s, nativeText: native, englishText: english, intent, intentConfidence, status: 'ready' }));
    } catch (e) {
      setState(s => ({ ...s, status: 'error', error: String(e) }));
    }
  }, [state.status]);

  // ── Process text input (text → MT → intent) ────────────────────────────────
  const processText = useCallback(async (indicText: string) => {
    if (state.status !== 'ready') return;
    setState(s => ({ ...s, status: 'processing', error: null }));
    try {
      const english = await IndicAI.translateToEnglish(indicText);
      const { intent, intentConfidence } = await getIntent(english);
      setState(s => ({
        ...s, nativeText: indicText, englishText: english, intent, intentConfidence, status: 'ready',
      }));
    } catch (e) {
      setState(s => ({ ...s, status: 'error', error: String(e) }));
    }
  }, [state.status]);

  // ── Respond with speech (EN response → Indic → TTS) ───────────────────────
  const respondWithSpeech = useCallback(async (englishResponse: string) => {
    if (state.status !== 'ready') return null;
    setState(s => ({ ...s, status: 'processing', error: null }));
    try {
      const result = await IndicAI.respondWithSpeech(englishResponse);
      setState((s: IndicAIState) => ({
        ...s,
        indicResponse: result.indicText,
        ttsAudioUri: result.audioPath,
        status: 'ready',
      }));
      return { indicText: result.indicText, audioUri: result.audioPath };
    } catch (e) {
      setState(s => ({ ...s, status: 'error', error: String(e) }));
      return null;
    }
  }, [state.status]);

  // ── Synthesize TTS only ────────────────────────────────────────────────────
  const speak = useCallback(async (text: string) => {
    try {
      const audioPath = await IndicAI.synthesize(text);
      setState(s => ({ ...s, ttsAudioUri: audioPath }));
      return audioPath;
    } catch (e) {
      setState(s => ({ ...s, error: String(e) }));
      return null;
    }
  }, []);

  // ── Translate only (EN → Indic) ────────────────────────────────────────────
  const translateToIndic = useCallback(async (englishText: string) => {
    try {
      return await IndicAI.translateToIndic(englishText);
    } catch (e) {
      setState(s => ({ ...s, error: String(e) }));
      return null;
    }
  }, []);

  return {
    ...state,
    processAudioFile,
    processBase64Pcm,
    processText,
    respondWithSpeech,
    speak,
    translateToIndic,
  };
}
