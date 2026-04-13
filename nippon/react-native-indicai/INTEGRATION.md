# IndicAI SDK — Integration Guide

> **For the APK / iOS App Team**
> Read this once. You will know exactly what to do, what you get, and what you send back — for both text and voice input — without any assumptions.
> This guide covers both **Android** and **iOS** integration.

---

## Table of Contents

1. [What This SDK Does](#1-what-this-sdk-does)
2. [What We Share With You](#2-what-we-share-with-you)
3. [Installation — Android](#3-installation--android)
4. [Installation — iOS](#4-installation--ios)
5. [Android vs iOS: What Is Different](#5-android-vs-ios-what-is-different)
6. [First-Time Setup](#6-first-time-setup)
7. [Text Input — Full Flow](#7-text-input--full-flow)
8. [Speech Input — Full Flow](#8-speech-input--full-flow)
9. [Playing the Response Audio](#9-playing-the-response-audio)
10. [Using the Drop-In Hook](#10-using-the-drop-in-hook-recommended)
11. [All 27 Intent Labels](#11-all-27-intent-labels)
12. [Exact API Contracts](#12-exact-api-contracts)
13. [Progress Events During Download](#13-progress-events-during-download)
14. [Error Codes](#14-error-codes)
15. [Complete System Diagram](#15-complete-system-diagram)
16. [Supported Languages](#16-supported-languages)

---

## 1. What This SDK Does

Users of the SimplySave app speak or type in their native language (Hindi, Tamil, Telugu, etc.).
They do not speak English. Your backend speaks English.
This SDK sits in the middle and handles everything:

```
User speaks Hindi
      ↓
SDK converts voice → Hindi text
SDK translates Hindi → English
SDK identifies what user wants (intent)
      ↓
You receive: English text + intent label
You call your backend API with English text
Your backend returns an English reply
      ↓
You send that English reply to SDK
SDK translates English → Hindi
SDK speaks the Hindi reply out loud
      ↓
User hears the response in their language
```

Everything happens **on the device**. No cloud, no API keys, no external service.
Models are downloaded once from our CDN on first launch (~550 MB). After that, works offline.

---

## 2. What We Share With You

| # | Artifact | How to Use |
|---|---|---|
| 1 | GitHub repo — `react-native-indicai` | Install via npm (see Section 3.1 / 4.1) |
| 2 | This document | Follow step by step |
| 3 | `IndicAIDemo.apk` (Android) | Install and test on Android — see it working before you integrate |
| 4 | `IndicAIDemo.ipa` (iOS, coming soon) | Install and test on iPhone via TestFlight |
| 5 | 27 intent label strings | Listed in [Section 11](#11-all-27-intent-labels) — map them to your app screens |

**GitHub Repository:** `https://github.com/SagarSalgarr/Nippon`
SDK is located in the `nippon/react-native-indicai/` subfolder.

**You do NOT need:** AWS keys, model files, Python scripts, any server, any API key.
Everything is embedded in the SDK and downloads automatically.

**CDN (model downloads — no auth needed):**
- Manifest: `https://indicai-cdn.s3.ap-south-1.amazonaws.com/manifest.json`
- Models: `https://indicai-cdn.s3.ap-south-1.amazonaws.com/models/`

---

## 3. Installation — Android

### 3.1 Install the package

**Option A — Install directly from GitHub (recommended):**
```bash
# In your React Native project root:
npm install github:SagarSalgarr/Nippon#main
```

**Option B — Install from a shared `.tgz` file:**
```bash
npm install /path/to/react-native-indicai-1.0.0.tgz
```

**Option C — Install from local folder (if you cloned the repo):**
```bash
npm install ./react-native-indicai
```

### 3.2 Android permissions

Open `android/app/src/main/AndroidManifest.xml` and add:

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

`INTERNET` — needed to download AI models on first launch.
`RECORD_AUDIO` — needed only if you use the voice/mic feature.

### 3.3 Install peer dependency for audio playback

The SDK gives you a WAV file path. To play it, install any audio player:

```bash
npm install react-native-audio-recorder-player
```

---

## 4. Installation — iOS

### 4.1 Install the package (same as Android)

**Option A — Install directly from GitHub (recommended):**
```bash
npm install github:SagarSalgarr/Nippon#main
```

**Option B — Install from a shared `.tgz` file:**
```bash
npm install /path/to/react-native-indicai-1.0.0.tgz
```

**Option C — Install from local folder:**
```bash
npm install ./react-native-indicai
```

### 4.2 Install iOS native dependencies (CocoaPods)

First, ensure your `ios/Podfile` has the correct minimum deployment target:

```ruby
# ios/Podfile — must be 15.0 or higher
platform :ios, '15.0'
```

Then run:

```bash
cd ios
pod install
cd ..
```

This downloads `onnxruntime-objc` (~150 MB) and links the SDK's Swift code.
Run this once after `npm install`, and again whenever the SDK is updated.

**Always open `YourApp.xcworkspace` (not `.xcodeproj`) after pod install.**

### 4.3 iOS permissions — microphone

Open `ios/YourApp/Info.plist` in Xcode and add:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>This app needs the microphone to understand what you say in your language.</string>
```

Or in Xcode: click `Info.plist` → `+` button → add key `NSMicrophoneUsageDescription` → enter your description.

Without this key, iOS will crash the app when the microphone is first accessed.
`INTERNET` is not needed in `Info.plist` — iOS allows network access by default.

### 4.4 iOS does NOT need PermissionsAndroid

On Android, you call `PermissionsAndroid.request()` explicitly.
On iOS, the system automatically shows the permission dialog when `startRecording()` is called for the first time.

```typescript
// Android: you must request permission manually
if (Platform.OS === 'android') {
  await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO, { ... });
}

// iOS: nothing to do — dialog appears automatically on first startRecording() call
await IndicAI.startRecording();
```

### 4.5 Install peer dependency for audio playback (same as Android)

```bash
npm install react-native-audio-recorder-player
cd ios && pod install   # re-run pod install after adding any new npm package
```

---

## 5. Android vs iOS: What Is Different

**The JavaScript/TypeScript API is 100% identical on both platforms.**
You write the same code. The SDK handles platform differences internally.

### What is the same

- All method names: `init()`, `transcribe()`, `translateToEnglish()`, `classifyIntent()`, `respondWithSpeech()`, `startRecording()`, `stopRecording()`, `release()`
- All event names: `IndicAI_Progress`
- All error codes: `INDICAI_INIT_ERROR`, `INDICAI_TRANSCRIBE_ERROR`, etc.
- Audio format from `stopRecording()`: base64 PCM 16-bit 16kHz mono
- Return value from `respondWithSpeech()`: `{ indicText: string, audioPath: string }`
- Audio playback: add `file://` prefix to `audioPath` (same on both platforms)
- Model download: same 5 models, same sizes, same CDN

### What is different

| | Android | iOS |
|---|---|---|
| **Installation** | `npm install` only | `npm install` + `pod install` |
| **Microphone permission** | `PermissionsAndroid.request()` | Automatic (system dialog) |
| **Info.plist** | Not needed | Add `NSMicrophoneUsageDescription` |
| **AndroidManifest.xml** | Add `INTERNET` + `RECORD_AUDIO` | Not needed |
| **Audio file path** | `/data/user/0/<pkg>/cache/indicai_tts.wav` | `/tmp/indicai_tts.wav` |
| **Simulator voice** | Emulator has mic | Simulator has NO mic (use device) |
| **Build tool** | `./gradlew assembleDebug` | Xcode / `npx react-native run-ios` |
| **Build platform** | Linux / Mac / Windows | Mac only |

### Code you may need to platform-check

Only one place typically needs a platform check — requesting mic permission:

```typescript
import { Platform, PermissionsAndroid } from 'react-native';

async function requestMicPermission(): Promise<boolean> {
  if (Platform.OS === 'ios') return true; // iOS handles it automatically

  const result = await PermissionsAndroid.request(
    PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
    {
      title: 'Microphone Access Needed',
      message: 'IndicAI needs your microphone to understand what you say.',
      buttonPositive: 'Allow',
    }
  );
  return result === PermissionsAndroid.RESULTS.GRANTED;
}
```

Everything else — `init()`, `transcribe()`, `translateToEnglish()`, `classifyIntent()`, `respondWithSpeech()`, `startRecording()`, `stopRecording()` — works identically with no `Platform.OS` check.

---

## 6. CDN Configuration (Optional)

By default the SDK downloads model files from the production CDN:

```
https://indicai-cdn.s3.ap-south-1.amazonaws.com/manifest.json
https://indicai-cdn.s3.ap-south-1.amazonaws.com/models/<file>.zip
```

**You do not need to configure anything for production** — these URLs are built into the SDK.

If you want to point to a custom/staging CDN (or keep the URL out of your git history), use `.env`:

### Step 1 — Install react-native-config

```sh
npm install react-native-config
```

### Step 2 — Android: apply the dotenv Gradle plugin

In `android/app/build.gradle`, add this at the very top (above all other lines):

```gradle
apply from: project(':react-native-config').projectDir.getPath() + "/dotenv.gradle"
```

### Step 3 — iOS: set ENVFILE in Build Phase

In Xcode → your target → Build Phases → "Bundle React Native code and images", add at the top:

```sh
export ENVFILE=.env
```

### Step 4 — Create your `.env` file

Copy the `.env.example` from the SDK and place it in your **app root** (same folder as `package.json`):

```sh
# .env  (app root — never commit this file)
# Fill in the actual CDN URLs — get them from the SDK team.
# Leave blank to use the built-in production CDN.
INDICAI_MANIFEST_URL=
INDICAI_S3_BASE_URL=
```

**That's it.** The SDK reads these automatically on import — no extra code needed in your app. If the `.env` values are absent or `react-native-config` is not installed, the SDK silently uses the built-in production CDN.

> **Do not commit `.env`** — add it to `.gitignore`. Commit `.env.example` instead.

---

## 7. First-Time Setup

### 7.1 Initialize the SDK

Call this once when the user selects their language (e.g., on a language selection screen or app startup).

```typescript
import IndicAI from 'react-native-indicai';

// Initialize for Hindi
await IndicAI.init('hi');

// When this Promise resolves → SDK is fully ready, all models are loaded
// When this Promise rejects  → something failed, show error and retry
```

On first call, this will download ~550 MB of AI models in background.
On subsequent calls, models are already cached — loads in ~5 seconds.

### 7.2 Show download progress to the user

Subscribe **before** calling `init()`:

```typescript
import IndicAI from 'react-native-indicai';

// Subscribe to progress events
const unsubscribe = IndicAI.onProgress((progress) => {
  // progress.model   → which model is downloading, e.g. "Whisper"
  // progress.percent → 0 to 100
  // progress.downloaded → bytes downloaded so far
  // progress.total      → total bytes

  console.log(`Downloading ${progress.model}… ${progress.percent}%`);
  setProgressText(`Downloading ${progress.model}… ${progress.percent}%`);
});

// Now initialize
await IndicAI.init('hi');

// Stop listening once ready
unsubscribe();
```

Models downloaded (in order):

| Model | Purpose | Size |
|---|---|---|
| Whisper | Voice → text (speech-to-text) | 183 MB |
| IndicTrans2 Indic→EN | User's language → English | 177 MB |
| IndicTrans2 EN→Indic | English → User's language | 204 MB |
| MMS-TTS (selected language) | Text → spoken audio | 101 MB |
| MiniLM Intent | Detect what user wants | 80 MB |

### 7.3 Switch language

Call `init()` again with a new language code. The SDK reinitializes for the new language.
Previously downloaded shared models (Whisper, IndicTrans2) are cached and not re-downloaded.
Only the TTS model for the new language is downloaded if not already cached.

```typescript
await IndicAI.init('ta'); // switch to Tamil
```

### 7.4 Cleanup on screen unmount

```typescript
useEffect(() => {
  IndicAI.init('hi');
  return () => {
    IndicAI.release(); // frees ONNX sessions from memory
  };
}, []);
```

---

## 8. Text Input — Full Flow

User types in their language → you process it → SDK translates response back.

### Complete code:

```typescript
import IndicAI from 'react-native-indicai';

async function handleTextInput(userTypedText: string) {

  // ─────────────────────────────────────────────────────────────────
  // STEP 1: SDK translates user's text to English
  // Input:  string — the text the user typed in their language
  //         e.g. "मुझे अकाउंट बनाना है"
  // Output: string — English translation
  //         e.g. "I need to create an account"
  // ─────────────────────────────────────────────────────────────────
  const englishText = await IndicAI.translateToEnglish(userTypedText);


  // ─────────────────────────────────────────────────────────────────
  // STEP 2: SDK detects what the user wants (intent classification)
  // Input:  string — MUST be English (pass the output from step 1)
  // Output: { intent: string, confidence: number }
  //         intent     → one of 27 labels, e.g. "create_account"
  //         confidence → 0.0 to 1.0, how sure the model is
  // ─────────────────────────────────────────────────────────────────
  const intentResult = await IndicAI.classifyIntent(englishText);
  // intentResult.intent     = "create_account"
  // intentResult.confidence = 0.89


  // ─────────────────────────────────────────────────────────────────
  // YOU NOW HAVE EVERYTHING FROM THE SDK:
  //   userTypedText          → show in chat bubble as user message
  //   englishText            → send to your backend
  //   intentResult.intent    → navigate to correct screen / trigger action
  //   intentResult.confidence→ optional: show warning if < 0.5
  // ─────────────────────────────────────────────────────────────────

  // Navigate based on intent (if it's a screen navigation intent):
  if (intentResult.intent === 'create_account') {
    navigation.navigate('SignUpScreen');
    return;
  }
  if (intentResult.intent === 'check_balance') {
    navigation.navigate('BalanceScreen');
    return;
  }
  // ... see Section 9 for all 27 intents


  // ─────────────────────────────────────────────────────────────────
  // STEP 3: YOU call your backend API with the English text
  // Your backend always receives and returns English
  // ─────────────────────────────────────────────────────────────────
  const backendReply = await yourBackendAPI.chat({
    message: englishText,
    intent: intentResult.intent,
  });
  // backendReply = "Your account has been created successfully"


  // ─────────────────────────────────────────────────────────────────
  // STEP 4: SDK translates the English reply back to user's language
  //         AND generates spoken audio in one call
  // Input:  string — your backend's English reply
  // Output: {
  //           indicText: string   → translated text in user's language
  //           audioPath: string   → absolute path to WAV file on device
  //         }
  // ─────────────────────────────────────────────────────────────────
  const sdkResponse = await IndicAI.respondWithSpeech(backendReply);
  // sdkResponse.indicText  = "आपका खाता बना दिया गया है।"
  // sdkResponse.audioPath  = "/data/user/0/com.yourapp/cache/indicai_tts.wav"


  // Show the translated text in your UI:
  setChatResponse(sdkResponse.indicText);

  // Play the audio (see Section 9 for audio playback):
  await playAudio(sdkResponse.audioPath);
}
```

---

## 9. Speech Input — Full Flow

User taps mic → speaks → SDK processes → you get English + intent → call backend → SDK responds.

### 9.1 Request microphone permission (do this once)

```typescript
import { PermissionsAndroid, Platform } from 'react-native';

async function requestMicPermission(): Promise<boolean> {
  // iOS: system shows dialog automatically — nothing to do here
  if (Platform.OS === 'ios') return true;

  // Android: must request explicitly
  const result = await PermissionsAndroid.request(
    PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
    {
      title: 'Microphone Access Needed',
      message: 'IndicAI needs your microphone to understand what you say.',
      buttonPositive: 'Allow',
    }
  );
  return result === PermissionsAndroid.RESULTS.GRANTED;
}
```

### 9.2 Start recording (when user presses mic button)

```typescript
import IndicAI from 'react-native-indicai';

async function onMicButtonPress() {
  const allowed = await requestMicPermission();
  if (!allowed) {
    alert('Microphone permission is required for voice input.');
    return;
  }

  // Start recording
  // SDK captures audio at 16kHz, mono, PCM 16-bit (exactly what Whisper needs)
  // Works identically on Android and iOS
  await IndicAI.startRecording();

  setIsRecording(true);
}
```

### 9.3 Stop recording and process (when user releases mic button)

```typescript
async function onMicButtonRelease() {
  setIsRecording(false);

  // ─────────────────────────────────────────────────────────────────
  // STEP 1: Stop recording
  // Returns: string — raw audio bytes as base64 (NOT a WAV file)
  //          Pass this directly to transcribe() — do not modify it
  // ─────────────────────────────────────────────────────────────────
  const base64Pcm = await IndicAI.stopRecording();


  // ─────────────────────────────────────────────────────────────────
  // STEP 2: SDK converts voice to text (Whisper speech-to-text)
  // Input:  string — base64 PCM from stopRecording()
  // Output: string — text in user's language
  //         e.g. "मुझे अकाउंट बनाना है"
  // ─────────────────────────────────────────────────────────────────
  const nativeText = await IndicAI.transcribe(base64Pcm);
  setTranscription(nativeText); // show what user said


  // ─────────────────────────────────────────────────────────────────
  // STEP 3: SDK translates to English
  // Input:  string — text in user's language (output from step 2)
  // Output: string — English text
  //         e.g. "I need to create an account"
  // ─────────────────────────────────────────────────────────────────
  const englishText = await IndicAI.translateToEnglish(nativeText);


  // ─────────────────────────────────────────────────────────────────
  // STEP 4: SDK detects intent
  // Input:  string — MUST be English (output from step 3)
  // Output: { intent: string, confidence: number }
  // ─────────────────────────────────────────────────────────────────
  const intentResult = await IndicAI.classifyIntent(englishText);
  // intentResult.intent     = "create_account"
  // intentResult.confidence = 0.89


  // ─────────────────────────────────────────────────────────────────
  // YOU NOW HAVE FROM THE SDK:
  //   nativeText             → what user actually said (show in chat)
  //   englishText            → send to your backend
  //   intentResult.intent    → navigate to screen / trigger action
  //   intentResult.confidence→ confidence score 0.0–1.0
  // ─────────────────────────────────────────────────────────────────

  // Navigate based on intent:
  if (intentResult.intent === 'transfer_money') {
    navigation.navigate('TransferScreen');
    return;
  }


  // ─────────────────────────────────────────────────────────────────
  // STEP 5: YOU call your backend with the English text
  // ─────────────────────────────────────────────────────────────────
  const backendReply = await yourBackendAPI.chat({
    message: englishText,
    intent: intentResult.intent,
  });
  // backendReply = "Your account has been created successfully"


  // ─────────────────────────────────────────────────────────────────
  // STEP 6: SDK translates backend reply + generates spoken audio
  // Input:  string — your backend's English reply
  // Output: {
  //           indicText: string  → translated reply in user's language
  //           audioPath: string  → path to WAV audio file on device
  //         }
  // ─────────────────────────────────────────────────────────────────
  const sdkResponse = await IndicAI.respondWithSpeech(backendReply);
  // sdkResponse.indicText = "आपका खाता बना दिया गया है।"
  // sdkResponse.audioPath = "/data/user/0/com.yourapp/cache/indicai_tts.wav"


  // Show text:
  setChatResponse(sdkResponse.indicText);

  // Play audio:
  await playAudio(sdkResponse.audioPath);
}
```

---

## 10. Playing the Response Audio

The SDK returns a file path. You need to play it with an audio player.
The same code works on both Android and iOS.

```typescript
import AudioRecorderPlayer from 'react-native-audio-recorder-player';

const audioPlayer = new AudioRecorderPlayer();

async function playAudio(audioPath: string) {
  // Stop any currently playing audio first
  await audioPlayer.stopPlayer().catch(() => {});

  // Both Android and iOS require the "file://" prefix
  // Android path example: /data/user/0/com.yourapp/cache/indicai_tts.wav
  // iOS path example:     /tmp/indicai_tts.wav
  const uri = audioPath.startsWith('file://')
    ? audioPath
    : `file://${audioPath}`;

  await audioPlayer.startPlayer(uri);
}
```

---

## 11. Using the Drop-In Hook (Recommended)

Instead of calling each SDK method manually, use our `useIndicAI` hook.
It manages all state — status, progress, results, errors — automatically.

```typescript
import React, { useState, useRef } from 'react';
import { View, Text, TextInput, TouchableOpacity } from 'react-native';
import IndicAI from 'react-native-indicai';
import { useIndicAI } from 'react-native-indicai/src/useIndicAI';
import AudioRecorderPlayer from 'react-native-audio-recorder-player';

const audioPlayer = new AudioRecorderPlayer();

export default function VoiceChatScreen() {
  const [language, setLanguage] = useState('hi');
  const [textInput, setTextInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);

  const {
    // ── Status ────────────────────────────────────────────────────────
    status,
    // 'idle'        → SDK not started yet
    // 'downloading' → AI models being downloaded (first launch)
    // 'ready'       → fully loaded, ready to use
    // 'processing'  → currently running AI inference
    // 'error'       → something failed

    progress,
    // During 'downloading': { model: "Whisper", percent: 45, downloaded: ..., total: ... }
    // Otherwise: null

    // ── Results from user input ────────────────────────────────────────
    nativeText,
    // What user said/typed in their language
    // e.g. "मुझे अकाउंट बनाना है"
    // Show this in the chat bubble as user's message

    englishText,
    // English translation of nativeText
    // e.g. "I need to create an account"
    // Send this to your backend API

    intent,
    // Detected intent label
    // e.g. "create_account"
    // Use this to navigate to the right screen

    intentConfidence,
    // How confident the model is: 0.0 to 1.0
    // e.g. 0.89 means 89% confident
    // Optional: show warning to user if < 0.5

    // ── Result after respondWithSpeech() ──────────────────────────────
    indicResponse,
    // Translated response in user's language
    // e.g. "आपका खाता बना दिया गया है।"
    // Show this in the chat bubble as assistant's reply

    error,
    // Error message string if status === 'error'
    // Show to user and offer retry

    // ── Methods ───────────────────────────────────────────────────────
    processText,
    // Call with text input → runs translate + intent → updates state
    // async (indicText: string) => void

    processBase64Pcm,
    // Call with output of stopRecording() → runs transcribe + translate + intent → updates state
    // async (base64Pcm: string) => void

    respondWithSpeech,
    // Call with your backend's English reply → translates + generates audio → updates state
    // async (englishReply: string) => { indicText: string, audioUri: string } | null

  } = useIndicAI(language);


  // ── Text input handler ─────────────────────────────────────────────────────
  const handleSend = async () => {
    if (!textInput.trim() || status !== 'ready') return;

    await processText(textInput.trim());
    setTextInput('');

    // At this point: nativeText, englishText, intent, intentConfidence are set

    // Navigate based on intent:
    if (intent === 'check_balance') {
      navigation.navigate('BalanceScreen');
      return;
    }

    // Call YOUR backend:
    const backendReply = await yourBackendAPI.chat(englishText);

    // SDK translates + speaks:
    const result = await respondWithSpeech(backendReply);
    if (result?.audioUri) {
      await audioPlayer.stopPlayer().catch(() => {});
      const uri = result.audioUri.startsWith('file://')
        ? result.audioUri
        : `file://${result.audioUri}`;
      await audioPlayer.startPlayer(uri);
    }
  };


  // ── Voice input handlers ───────────────────────────────────────────────────
  const handleMicStart = async () => {
    await IndicAI.startRecording();
    setIsRecording(true);
  };

  const handleMicStop = async () => {
    setIsRecording(false);
    const base64Pcm = await IndicAI.stopRecording();

    await processBase64Pcm(base64Pcm);
    // At this point: nativeText, englishText, intent, intentConfidence are set

    // Call YOUR backend:
    const backendReply = await yourBackendAPI.chat(englishText);

    // SDK translates + speaks:
    const result = await respondWithSpeech(backendReply);
    if (result?.audioUri) {
      await audioPlayer.stopPlayer().catch(() => {});
      const uri = result.audioUri.startsWith('file://')
        ? result.audioUri
        : `file://${result.audioUri}`;
      await audioPlayer.startPlayer(uri);
    }
  };


  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <View>
      {/* Show download progress */}
      {status === 'downloading' && progress && (
        <Text>Downloading {progress.model}… {progress.percent}%</Text>
      )}

      {/* Show error */}
      {status === 'error' && (
        <Text>Error: {error}</Text>
      )}

      {/* Show what user said */}
      {nativeText.length > 0 && (
        <Text>You: {nativeText}</Text>
      )}

      {/* Show translated response */}
      {indicResponse.length > 0 && (
        <Text>Assistant: {indicResponse}</Text>
      )}

      {/* Show detected intent */}
      {intent.length > 0 && (
        <Text>Intent: {intent} ({Math.round(intentConfidence * 100)}%)</Text>
      )}

      {/* Text input */}
      <TextInput
        value={textInput}
        onChangeText={setTextInput}
        editable={status === 'ready'}
        placeholder="Type in your language…"
      />
      <TouchableOpacity
        onPress={handleSend}
        disabled={status !== 'ready'}
      >
        <Text>Send</Text>
      </TouchableOpacity>

      {/* Mic button */}
      <TouchableOpacity
        onPressIn={handleMicStart}
        onPressOut={handleMicStop}
        disabled={status !== 'ready' && !isRecording}
      >
        <Text>{isRecording ? 'Recording…' : 'Hold to Speak'}</Text>
      </TouchableOpacity>
    </View>
  );
}
```

---

## 12. All 27 Intent Labels

These are the exact strings the SDK returns in `intentResult.intent`.
Map each one to the appropriate screen or action in your app.

| Intent Label | What User Wants | Suggested Action |
|---|---|---|
| `check_balance` | See account balance | Navigate to balance screen |
| `view_transactions` | See transaction history | Navigate to transactions list |
| `transfer_money` | Send money to someone | Navigate to transfer screen |
| `add_money` | Add funds to account | Navigate to add money screen |
| `withdraw_money` | Withdraw cash | Navigate to withdrawal screen |
| `create_account` | Create a new account | Navigate to sign up screen |
| `login_signup` | Login to existing account | Navigate to login screen |
| `reset_password` | Forgot password | Navigate to reset password screen |
| `update_profile` | Edit profile information | Navigate to profile edit screen |
| `create_gullak` | Create a new piggy bank / savings pot | Navigate to create gullak screen |
| `deposit_gullak` | Add money to piggy bank | Navigate to deposit gullak screen |
| `withdraw_gullak` | Take money from piggy bank | Navigate to withdraw from gullak screen |
| `view_gullak` | See piggy bank details | Navigate to gullak detail screen |
| `set_goal` | Set a savings goal | Navigate to set goal screen |
| `check_goal` | Check progress toward a goal | Navigate to goal progress screen |
| `loan_inquiry` | Ask about loan options | Navigate to loan info screen |
| `loan_apply` | Apply for a loan | Navigate to loan application screen |
| `loan_repay` | Repay an existing loan | Navigate to loan repayment screen |
| `bill_payment` | Pay a utility bill | Navigate to bill payment screen |
| `recharge` | Mobile / DTH recharge | Navigate to recharge screen |
| `investment` | Invest money | Navigate to investment screen |
| `insurance` | Buy or check insurance | Navigate to insurance screen |
| `fd_rd` | Fixed deposit / recurring deposit | Navigate to FD/RD screen |
| `help_support` | Need customer support | Navigate to support / help screen |
| `fraud_report` | Report fraud or suspicious activity | Navigate to fraud report screen |
| `close_account` | Close the account | Navigate to account closure screen |
| `general_inquiry` | General question (does not fit any above) | Open chat / fallback screen |

**Tip on confidence score:**
If `intentConfidence < 0.5`, the model is uncertain. Consider treating it as `general_inquiry`
and asking the user to clarify.

---

## 13. Exact API Contracts

### `IndicAI.init(languageCode)`

```
Purpose:  Download + load all AI models for a language
Input:    languageCode — one of: 'hi', 'mr', 'ta', 'te', 'kn', 'ml', 'bn', 'gu', 'pa'
Output:   Promise<void> — resolves when fully ready, rejects on failure
Error:    Rejects with message describing what failed
Platform: Android + iOS (identical)
Notes:
  - Call once per language selection
  - Safe to call again with a different language
  - Must resolve before calling any other SDK method
```

### `IndicAI.translateToEnglish(text)`

```
Purpose:  Translate text from user's language to English
Input:    text — string in user's language (e.g. "मुझे पैसे भेजने हैं")
Output:   Promise<string> — English translation (e.g. "I want to send money")
Error:    Rejects with code "INDICAI_TRANSLATE_ERROR"
Platform: Android + iOS (identical)
Notes:
  - Language is determined by which language was passed to init()
  - Pass the output directly to classifyIntent()
```

### `IndicAI.classifyIntent(englishText)`

```
Purpose:  Detect what the user wants to do
Input:    englishText — MUST be English text (pass output of translateToEnglish)
Output:   Promise<{ intent: string, confidence: number }>
            intent     — one of 27 labels listed in Section 11
            confidence — float between 0.0 and 1.0
Error:    Rejects with code "INDICAI_INTENT_ERROR"
Platform: Android + iOS (identical)
Notes:
  - Always pass English, not Indic text
  - Always returns a label even if confidence is low
  - Use confidence to decide whether to trust the result
```

### `IndicAI.startRecording()`

```
Purpose:  Start capturing audio from the microphone
Input:    none
Output:   Promise<void> — resolves when recording has started
Error:    Rejects with code "INDICAI_RECORD_ERROR"
            Reasons: no microphone permission, mic hardware unavailable
Platform: Android + iOS (identical call — see Section 8.1 for permission differences)
Notes:
  - Android: requires RECORD_AUDIO permission granted before calling
  - iOS: system dialog appears automatically on first call
  - Audio is captured at 16 kHz, mono, PCM 16-bit on both platforms
  - Call stopRecording() to get the audio
```

### `IndicAI.stopRecording()`

```
Purpose:  Stop recording and return the captured audio
Input:    none
Output:   Promise<string> — base64-encoded raw PCM bytes
            This is NOT a WAV file — it is raw audio bytes
            Pass this DIRECTLY to transcribe() without modification
Error:    Rejects with code "INDICAI_RECORD_ERROR"
Platform: Android + iOS (identical)
Notes:
  - Returns empty string if recording was not started
  - Always call after startRecording()
```

### `IndicAI.transcribe(base64Pcm)`

```
Purpose:  Convert voice audio to text (speech-to-text using Whisper)
Input:    base64Pcm — base64 string from stopRecording()
Output:   Promise<string> — transcribed text in user's language
            e.g. "मुझे अकाउंट बनाना है"
Error:    Rejects with code "INDICAI_TRANSCRIBE_ERROR"
Platform: Android + iOS (identical)
Notes:
  - Pass the raw base64 from stopRecording() without any changes
  - Output language matches the language passed to init()
  - Pass output to translateToEnglish()
```

### `IndicAI.respondWithSpeech(englishText)`

```
Purpose:  Translate English response to user's language AND generate speech audio
          This is a single call that does two things:
            1. Translate English → user's language
            2. Generate spoken audio from the translated text
Input:    englishText — English string from your backend API response
Output:   Promise<{
            indicText: string   // translated text in user's language
            audioPath: string   // absolute path to WAV file on device
          }>
            indicText example:  "आपका खाता बना दिया गया है।"
            audioPath (Android): "/data/user/0/com.yourapp/cache/indicai_tts.wav"
            audioPath (iOS):     "/tmp/indicai_tts.wav"
Error:    Rejects with code "INDICAI_RESPOND_ERROR"
Platform: Android + iOS (identical call, path format differs)
Notes:
  - audioPath is a file path, not a URL
  - Add "file://" prefix to play on both Android and iOS (see Section 9)
  - The WAV file is overwritten on each call (one file, reused)
  - Display indicText in your chat UI as the assistant's reply
```

### `IndicAI.onProgress(callback)`

```
Purpose:  Listen to model download progress during init()
Input:    callback — function called with DownloadProgress object:
            {
              model:      string  // name of model being downloaded
              percent:    number  // 0 to 100
              downloaded: number  // bytes downloaded so far
              total:      number  // total bytes for this model
            }
Output:   () => void — call this function to stop listening
Platform: Android + iOS (identical)
Notes:
  - Subscribe BEFORE calling init()
  - Unsubscribe after init() resolves
  - model values: "Whisper", "IndicTrans2 Indic→EN",
                  "IndicTrans2 EN→Indic", "TTS-hi", "Intent"
```

### `IndicAI.release()`

```
Purpose:  Free AI model memory (ONNX sessions)
Input:    none
Output:   void (not a Promise)
Platform: Android + iOS (identical)
Notes:
  - Call in useEffect cleanup when your screen unmounts
  - After release(), call init() again before using the SDK
```

---

## 14. Progress Events During Download

During `IndicAI.init()`, the SDK emits progress for each model.
Same event names and structure on Android and iOS.

```
Model 1/5: "Whisper"               183 MB  — speech to text
Model 2/5: "IndicTrans2 Indic→EN"  177 MB  — your language to English
Model 3/5: "IndicTrans2 EN→Indic"  204 MB  — English to your language
Model 4/5: "TTS-hi"                101 MB  — text to speech (Hindi)
Model 5/5: "Intent"                 80 MB  — intent classification

Total first launch: ~745 MB
Subsequent launches: 0 MB (all cached)
```

Each model fires multiple progress events from 0% to 100%, then moves to the next.

---

## 15. Error Codes

Same error codes on Android and iOS.

| Code | Method | Meaning |
|---|---|---|
| `INDICAI_INIT_ERROR` | `init()` | Model download or load failed |
| `INDICAI_TRANSLATE_ERROR` | `translateToEnglish()`, `translateToIndic()` | Translation model failed |
| `INDICAI_TRANSCRIBE_ERROR` | `transcribe()` | Whisper STT failed |
| `INDICAI_INTENT_ERROR` | `classifyIntent()` | Intent model not loaded or inference failed |
| `INDICAI_TTS_ERROR` | `synthesize()` | TTS model failed |
| `INDICAI_RESPOND_ERROR` | `respondWithSpeech()` | Translation or TTS failed |
| `INDICAI_RECORD_ERROR` | `startRecording()`, `stopRecording()` | Mic permission or hardware issue |

All errors carry a human-readable `.message` property. Log it for debugging.

---

## 16. Complete System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            YOUR APP                                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    TEXT INPUT PATH                               │    │
│  │                                                                  │    │
│  │  User types: "मुझे अकाउंट बनाना है"                              │    │
│  │       │                                                          │    │
│  │       ▼                                                          │    │
│  │  IndicAI.translateToEnglish("मुझे अकाउंट बनाना है")              │    │
│  │       │  [IndicTrans2 Indic→EN model — runs on device]           │    │
│  │       ▼                                                          │    │
│  │  → "I need to create an account"                                 │    │
│  │       │                                                          │    │
│  │       ▼                                                          │    │
│  │  IndicAI.classifyIntent("I need to create an account")           │    │
│  │       │  [MiniLM model — runs on device]                         │    │
│  │       ▼                                                          │    │
│  │  → { intent: "create_account", confidence: 0.89 }               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   SPEECH INPUT PATH                              │    │
│  │                                                                  │    │
│  │  User speaks into mic                                            │    │
│  │       │                                                          │    │
│  │       ▼                                                          │    │
│  │  IndicAI.startRecording()   [SDK captures 16kHz PCM mono]        │    │
│  │  IndicAI.stopRecording()    [returns base64 PCM bytes]           │    │
│  │       │                                                          │    │
│  │       ▼                                                          │    │
│  │  IndicAI.transcribe(base64Pcm)                                   │    │
│  │       │  [Whisper model — runs on device]                        │    │
│  │       ▼                                                          │    │
│  │  → "मुझे अकाउंट बनाना है"                                        │    │
│  │       │                                                          │    │
│  │       ▼                                                          │    │
│  │  IndicAI.translateToEnglish(...)  →  "I need to create account"  │    │
│  │  IndicAI.classifyIntent(...)      →  { intent, confidence }      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ════════════════════ SDK gives YOU these values ═════════════════════  │
│                                                                         │
│     nativeText   = "मुझे अकाउंट बनाना है"    ← show in chat            │
│     englishText  = "I need to create account" ← send to your backend   │
│     intent       = "create_account"           ← navigate / act         │
│     confidence   = 0.89                       ← trust score            │
│                                                                         │
│  ════════════════ YOU call YOUR backend ══════════════════════════════  │
│                                                                         │
│     POST yourapi.com/chat                                               │
│     Body: { message: "I need to create account", intent: "create_account" }
│     ← Response: "Your account has been created successfully"           │
│                                                                         │
│  ═══════════════ YOU give reply back to SDK ══════════════════════════  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    RESPONSE PATH                                 │    │
│  │                                                                  │    │
│  │  IndicAI.respondWithSpeech("Your account has been created…")     │    │
│  │       │                                                          │    │
│  │       ├─ [IndicTrans2 EN→Indic] → "आपका खाता बना दिया गया है।"  │    │
│  │       └─ [MMS-TTS model]        → WAV audio file on device       │    │
│  │       │                                                          │    │
│  │       ▼                                                          │    │
│  │  → {                                                             │    │
│  │      indicText:  "आपका खाता बना दिया गया है।"                    │    │
│  │      audioPath:  "/data/.../cache/indicai_tts.wav"               │    │
│  │    }                                                             │    │
│  │                                                                  │    │
│  │  You show indicText in chat UI                                   │    │
│  │  You play audioPath with your audio player                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌────────────────────┐
                    │   On-Device AI     │
                    │                   │
                    │  Whisper STT      │  ← no internet after download
                    │  IndicTrans2      │  ← no internet after download
                    │  MMS-TTS          │  ← no internet after download
                    │  MiniLM Intent    │  ← no internet after download
                    └────────────────────┘

                    ┌────────────────────┐
                    │   Our CDN (S3)     │  ← only on first launch
                    │                   │
                    │  manifest.json    │  ← model index file
                    │  model .zip files │  ← downloaded once, cached forever
                    └────────────────────┘
```

---

## 17. Supported Languages

Same languages supported on Android and iOS.

| Code | Language | Script |
|---|---|---|
| `hi` | Hindi | Devanagari |
| `mr` | Marathi | Devanagari |
| `ta` | Tamil | Tamil |
| `te` | Telugu | Telugu |
| `kn` | Kannada | Kannada |
| `ml` | Malayalam | Malayalam |
| `bn` | Bengali | Bengali |
| `gu` | Gujarati | Gujarati |
| `pa` | Punjabi | Gurmukhi |

Pass the code (left column) to `IndicAI.init()` and `useIndicAI()`.

---

## Quick Reference Cheatsheet

This code works on **both Android and iOS** without modification.

```typescript
// ── Setup ──────────────────────────────────────────────────────────────────
import IndicAI from 'react-native-indicai';
import { useIndicAI } from 'react-native-indicai/src/useIndicAI';

const unsub = IndicAI.onProgress(p => showProgress(p.model, p.percent));
await IndicAI.init('hi');
unsub();

// ── Text input ─────────────────────────────────────────────────────────────
const english = await IndicAI.translateToEnglish(userText);
const { intent, confidence } = await IndicAI.classifyIntent(english);
const backendReply = await yourAPI.chat(english);
const { indicText, audioPath } = await IndicAI.respondWithSpeech(backendReply);

// ── Mic permission (Android only — iOS handles automatically) ───────────────
if (Platform.OS === 'android') {
  await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO, { ... });
}

// ── Voice input ────────────────────────────────────────────────────────────
await IndicAI.startRecording();
// ... user speaks ...
const base64Pcm = await IndicAI.stopRecording();
const nativeText = await IndicAI.transcribe(base64Pcm);
const english    = await IndicAI.translateToEnglish(nativeText);
const { intent, confidence } = await IndicAI.classifyIntent(english);
const backendReply = await yourAPI.chat(english);
const { indicText, audioPath } = await IndicAI.respondWithSpeech(backendReply);

// ── Play audio (works on both Android and iOS) ─────────────────────────────
const uri = audioPath.startsWith('file://') ? audioPath : `file://${audioPath}`;
await audioPlayer.startPlayer(uri);

// ── Cleanup ────────────────────────────────────────────────────────────────
IndicAI.release();
```

---

## For iOS Team — What You Need From Us

1. `react-native-indicai/` folder — same folder as Android team uses
2. This `INTEGRATION.md` — covers both platforms
3. Run `pod install` in your app's `ios/` folder after `npm install`
4. Add `NSMicrophoneUsageDescription` to your `Info.plist`
5. No other changes — the JS API is identical to Android

**iOS-specific contact:** If models fail to download or ONNX sessions crash on first launch, send us the Xcode console log filtered by `[IndicAI`.

---

*SDK version: 1.0.0 | Android: min API 26 (Android 8.0) | iOS: min 15.0 — IndicAI Team*
