/**
 * IndicAIScreen.tsx
 * Complete demo screen for the RN app team.
 * Recording uses the native SDK (raw PCM) — no AudioRecorderPlayer for capture.
 * Playback uses AudioRecorderPlayer.startPlayer() with the WAV path from native TTS.
 */

import React, { useState, useRef } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  ActivityIndicator, ScrollView, Platform, TextInput,
  PermissionsAndroid,
} from 'react-native';
import AudioRecorderPlayer from 'react-native-audio-recorder-player';
import IndicAI from 'react-native-indicai';
import { useIndicAI } from 'react-native-indicai/src/useIndicAI';

const LANGUAGES = [
  { code: 'hi', label: 'हिंदी' },
  { code: 'mr', label: 'मराठी' },
  { code: 'ta', label: 'தமிழ்' },
  { code: 'te', label: 'తెలుగు' },
  { code: 'kn', label: 'ಕನ್ನಡ' },
  { code: 'ml', label: 'മലയാളം' },
  { code: 'bn', label: 'বাংলা' },
  { code: 'gu', label: 'ગુજરાતી' },
  { code: 'pa', label: 'ਪੰਜਾਬੀ' },
  { code: 'ur', label: 'اردو' },
];

export default function IndicAIScreen() {
  const [selectedLang, setSelectedLang] = useState('hi');
  const [isRecording, setIsRecording] = useState(false);
  const [textInput, setTextInput] = useState('');
  const playerRef = useRef(new AudioRecorderPlayer());

  const {
    status, progress,
    nativeText, englishText, intent, intentConfidence, indicResponse,
    error,
    processBase64Pcm, processText, respondWithSpeech, speak,
  } = useIndicAI(selectedLang);

  // ── Permission helper ───────────────────────────────────────────────────────
  const requestMicPermission = async (): Promise<boolean> => {
    if (Platform.OS !== 'android') return true;
    const granted = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
      {
        title: 'Microphone Permission',
        message: 'IndicAI needs the microphone to transcribe your speech.',
        buttonPositive: 'Allow',
      },
    );
    return granted === PermissionsAndroid.RESULTS.GRANTED;
  };

  // ── Recording via native SDK (raw PCM → Whisper STT) ───────────────────────
  const startRecording = async () => {
    const ok = await requestMicPermission();
    if (!ok) return;
    try {
      await IndicAI.startRecording();
      setIsRecording(true);
    } catch (e) {
      console.error('startRecording error:', e);
    }
  };

  const stopRecording = async () => {
    try {
      const base64Pcm = await IndicAI.stopRecording();
      setIsRecording(false);
      await processBase64Pcm(base64Pcm);
    } catch (e) {
      setIsRecording(false);
      console.error('stopRecording error:', e);
    }
  };

  // ── Text input ──────────────────────────────────────────────────────────────
  const handleTextSubmit = async () => {
    if (!textInput.trim()) return;
    await processText(textInput.trim());
    setTextInput('');
  };

  // ── TTS playback helper ─────────────────────────────────────────────────────
  const playWav = async (filePath: string) => {
    try {
      // stop any current playback first
      await playerRef.current.stopPlayer().catch(() => {});
      // AudioRecorderPlayer needs file:// URI on Android
      const uri = filePath.startsWith('file://') ? filePath : `file://${filePath}`;
      await playerRef.current.startPlayer(uri);
    } catch (e) {
      console.error('playWav error:', e);
    }
  };

  // ── Respond with speech (EN → Indic TTS) ───────────────────────────────────
  const handleRespondWithSpeech = async () => {
    if (!englishText) return;
    const result = await respondWithSpeech(englishText);
    if (result?.audioUri) {
      await playWav(result.audioUri);
    }
  };

  // ── Speak text directly ─────────────────────────────────────────────────────
  const playTts = async (text: string) => {
    const filePath = await speak(text);
    if (filePath) await playWav(filePath);
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>

      <Text style={styles.title}>IndicAI SDK Demo</Text>

      <Text style={styles.sectionLabel}>Select language</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.langRow}>
        {LANGUAGES.map(lang => (
          <TouchableOpacity
            key={lang.code}
            style={[styles.langPill, selectedLang === lang.code && styles.langPillActive]}
            onPress={() => setSelectedLang(lang.code)}
          >
            <Text style={[styles.langText, selectedLang === lang.code && styles.langTextActive]}>
              {lang.label}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {status === 'downloading' && (
        <View style={styles.progressBox}>
          <ActivityIndicator size="small" color="#4F46E5" />
          <Text style={styles.progressLabel}>
            {progress ? `Downloading ${progress.model}… ${progress.percent}%` : 'Initialising…'}
          </Text>
        </View>
      )}

      {status === 'error' && (
        <View style={styles.errorBox}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {/* Text input */}
      <View style={styles.textInputRow}>
        <TextInput
          style={styles.textInput}
          placeholder="Type in your language…"
          value={textInput}
          onChangeText={setTextInput}
          editable={status === 'ready'}
          onSubmitEditing={handleTextSubmit}
        />
        <TouchableOpacity
          style={[styles.sendBtn, status !== 'ready' && styles.btnDisabled]}
          onPress={handleTextSubmit}
          disabled={status !== 'ready'}
        >
          <Text style={styles.sendBtnText}>Send</Text>
        </TouchableOpacity>
      </View>

      {/* Record button */}
      <TouchableOpacity
        style={[
          styles.recordBtn,
          isRecording && styles.recordBtnActive,
          (status !== 'ready' && !isRecording) && styles.btnDisabled,
        ]}
        onPress={isRecording ? stopRecording : startRecording}
        disabled={status !== 'ready' && !isRecording}
      >
        <Text style={styles.recordBtnText}>
          {isRecording ? '⏹ Stop recording' : '🎙 Hold to speak'}
        </Text>
      </TouchableOpacity>

      {status === 'processing' && (
        <View style={styles.progressBox}>
          <ActivityIndicator size="small" color="#4F46E5" />
          <Text style={styles.progressLabel}>Processing…</Text>
        </View>
      )}

      {/* Transcription result */}
      {nativeText.length > 0 && (
        <View style={styles.resultBox}>
          <Text style={styles.resultLabel}>Transcription</Text>
          <Text style={styles.resultText}>{nativeText}</Text>
          <TouchableOpacity style={styles.ttsBtn} onPress={() => playTts(nativeText)}>
            <Text style={styles.ttsBtnText}>▶ Play TTS</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* English translation */}
      {englishText.length > 0 && (
        <View style={styles.resultBox}>
          <Text style={styles.resultLabel}>English translation</Text>
          <Text style={styles.resultText}>{englishText}</Text>
          <TouchableOpacity style={styles.ttsBtn} onPress={handleRespondWithSpeech}>
            <Text style={styles.ttsBtnText}>🔊 Respond with speech</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Intent */}
      {intent.length > 0 && (
        <View style={styles.intentBox}>
          <Text style={styles.resultLabel}>Intent</Text>
          <View style={styles.intentRow}>
            <Text style={styles.intentText}>{intent}</Text>
            <Text style={styles.intentConf}>{Math.round(intentConfidence * 100)}%</Text>
          </View>
        </View>
      )}

      {/* Indic response */}
      {indicResponse.length > 0 && (
        <View style={styles.resultBox}>
          <Text style={styles.resultLabel}>Response in your language</Text>
          <Text style={styles.resultText}>{indicResponse}</Text>
          <TouchableOpacity style={styles.ttsBtn} onPress={() => playTts(indicResponse)}>
            <Text style={styles.ttsBtnText}>▶ Play TTS</Text>
          </TouchableOpacity>
        </View>
      )}

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container:       { flex: 1, backgroundColor: '#F9FAFB' },
  content:         { padding: 20, paddingBottom: 60 },
  title:           { fontSize: 22, fontWeight: '700', color: '#111827', marginBottom: 20 },
  sectionLabel:    { fontSize: 13, color: '#6B7280', marginBottom: 8, fontWeight: '500' },
  langRow:         { flexDirection: 'row', marginBottom: 24 },
  langPill:        { paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20, backgroundColor: '#E5E7EB', marginRight: 8 },
  langPillActive:  { backgroundColor: '#4F46E5' },
  langText:        { fontSize: 15, color: '#374151' },
  langTextActive:  { color: '#FFFFFF', fontWeight: '600' },
  progressBox:     { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: '#EEF2FF', padding: 12, borderRadius: 10, marginBottom: 16 },
  progressLabel:   { color: '#4338CA', fontSize: 14 },
  errorBox:        { backgroundColor: '#FEF2F2', padding: 12, borderRadius: 10, marginBottom: 16 },
  errorText:       { color: '#DC2626', fontSize: 14 },
  textInputRow:    { flexDirection: 'row', marginBottom: 12, gap: 8 },
  textInput:       { flex: 1, backgroundColor: '#FFFFFF', borderRadius: 12, paddingHorizontal: 16, paddingVertical: 14, fontSize: 16, borderWidth: 1, borderColor: '#E5E7EB' },
  sendBtn:         { backgroundColor: '#4F46E5', paddingHorizontal: 20, borderRadius: 12, justifyContent: 'center' },
  sendBtnText:     { color: '#FFFFFF', fontWeight: '600', fontSize: 15 },
  recordBtn:       { backgroundColor: '#4F46E5', padding: 18, borderRadius: 14, alignItems: 'center', marginBottom: 16 },
  recordBtnActive: { backgroundColor: '#DC2626' },
  btnDisabled:     { opacity: 0.5 },
  recordBtnText:   { color: '#FFFFFF', fontSize: 17, fontWeight: '600' },
  resultBox:       { backgroundColor: '#FFFFFF', padding: 16, borderRadius: 12, marginBottom: 12, shadowColor: '#000', shadowOpacity: 0.06, shadowRadius: 8, elevation: 2 },
  resultLabel:     { fontSize: 12, color: '#9CA3AF', marginBottom: 6, fontWeight: '500', textTransform: 'uppercase', letterSpacing: 0.5 },
  resultText:      { fontSize: 17, color: '#111827', lineHeight: 26 },
  ttsBtn:          { marginTop: 12, backgroundColor: '#F3F4F6', padding: 10, borderRadius: 8, alignItems: 'center' },
  ttsBtnText:      { color: '#4F46E5', fontWeight: '600', fontSize: 14 },
  intentBox:       { backgroundColor: '#F0FDF4', padding: 16, borderRadius: 12, marginBottom: 12, borderWidth: 1, borderColor: '#BBF7D0' },
  intentRow:       { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 4 },
  intentText:      { fontSize: 17, color: '#15803D', fontWeight: '600' },
  intentConf:      { fontSize: 15, color: '#16A34A', fontWeight: '500', backgroundColor: '#DCFCE7', paddingHorizontal: 10, paddingVertical: 3, borderRadius: 12 },
});
