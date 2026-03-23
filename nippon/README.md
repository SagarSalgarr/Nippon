# react-native-indicai

On-device Indic language AI for React Native.
Speech-to-text, translation, intent detection, and text-to-speech — all running locally on the device. No cloud. No API keys. No internet after first launch.

---

## What This SDK Does

```
User speaks/types in Hindi, Tamil, Telugu (etc.)
        ↓
SDK → transcribes voice → translates to English → detects intent
        ↓
You call your backend with English text + intent
        ↓
SDK → translates English reply → speaks it back in user's language
```

**Supported languages:** Hindi · Marathi · Tamil · Telugu · Kannada · Malayalam · Bengali · Gujarati · Punjabi

**Supported platforms:** Android 8.0+ · iOS 15.0+

---

## Table of Contents

- [For Integration — read INTEGRATION.md](#for-integration)
- [Part 1 — What You Need Before Starting](#part-1--what-you-need-before-starting)
- [Part 2 — Get the Code](#part-2--get-the-code)
- [Part 3 — Build the Demo APK (Android)](#part-3--build-the-demo-apk-android)
- [Part 3B — Build the Demo iOS App (Mac only)](#part-3b--build-the-demo-ios-app-mac-only)
- [Part 4 — Install APK on Your Android Phone](#part-4--install-apk-on-your-android-phone)
- [Part 4B — Run on iOS Simulator or iPhone](#part-4b--run-on-ios-simulator-or-iphone)
- [Part 5 — Test the App on Your Phone](#part-5--test-the-app-on-your-phone)
- [Part 6 — Test Without USB (Standalone)](#part-6--test-without-usb-standalone)
- [Part 7 — How to Rebuild After Code Changes](#part-7--how-to-rebuild-after-code-changes)
- [Part 8 — Run the Python Pipeline Tests](#part-8--run-the-python-pipeline-tests)
- [Part 9 — Folder Structure Explained](#part-9--folder-structure-explained)
- [Part 10 — Common Problems and Fixes](#part-10--common-problems-and-fixes)
- [Part 11 — Android vs iOS: How They Differ](#part-11--android-vs-ios-how-they-differ)
- [Part 12 — How to Introduce or Replace a Model](#part-12--how-to-introduce-or-replace-a-model)

---

## For Integration

> If you are the **APK team** and just want to use this SDK in your existing app,
> read **[INTEGRATION.md](./INTEGRATION.md)** instead of this file.
> INTEGRATION.md tells you exactly what to install, what you receive, and what to send back.

This README is for people who want to **build, test, or develop** the SDK itself.

---

## Part 1 — What You Need Before Starting

You need to install these tools on your computer first.
If you already have them, skip to Part 2.

### 1.1 Check what you already have

Open a terminal and run each command. If it prints a version number, you have it.

```bash
node --version        # should print v18.x.x or higher
npm --version         # should print 9.x.x or higher
java -version         # should print 17.x.x
adb --version         # should print Android Debug Bridge version 1.x.x
```

### 1.2 Install Node.js (if missing)

Download from: https://nodejs.org
Choose the "LTS" version (the recommended one).
Install it. Then run `node --version` to confirm.

### 1.3 Install Java 17 (if missing)

The Android build system needs Java 17 exactly.

**On Ubuntu/Linux:**
```bash
sudo apt install openjdk-17-jdk
java -version   # confirm it says "17"
```

**On Mac:**
```bash
brew install openjdk@17
java -version
```

**On Windows:**
Download from: https://adoptium.net — choose "Temurin 17"

### 1.4 Install Android SDK and ADB (if missing)

The easiest way is to install **Android Studio**:
https://developer.android.com/studio

After installing Android Studio:
1. Open it
2. Go to: Tools → SDK Manager
3. Make sure **Android SDK 34** is installed
4. Make sure **Build Tools 34.0.0** is installed

To verify `adb` works, run:
```bash
adb --version
# Android Debug Bridge version 1.0.41
```

If `adb` is not found, add it to your PATH:
```bash
# Linux/Mac — add to ~/.bashrc or ~/.zshrc:
export PATH="$PATH:$HOME/Android/Sdk/platform-tools"
```

### 1.5 Install Xcode (Mac only — needed for iOS)

iOS apps can only be built on a Mac. If you are on Linux or Windows, skip this section and use Android only.

1. Open the **App Store** on your Mac
2. Search for **Xcode** and install it (it is free, ~14 GB)
3. After installation, open Xcode once to accept the license agreement
4. Install command-line tools:
```bash
xcode-select --install
```
5. Verify:
```bash
xcodebuild -version
# Xcode 15.x
# Build version 15x.x.x
```

You need Xcode 15 or higher to build this SDK.

### 1.6 Install CocoaPods (Mac only — needed for iOS)

CocoaPods manages iOS native dependencies (like `onnxruntime-objc`).

```bash
# Check if already installed:
pod --version   # should print 1.14.x or higher

# Install if missing:
sudo gem install cocoapods
pod --version   # confirm
```

If `gem` is slow or fails, use Homebrew instead:
```bash
brew install cocoapods
pod --version
```

### 1.7 Enable Developer Mode on your Android phone

You need this so your computer can install APKs directly to the phone.

1. On your phone, go to: **Settings → About Phone**
2. Tap **Build Number** 7 times rapidly
3. You will see "You are now a developer!"
4. Go back to Settings → **Developer Options**
5. Turn on: **USB Debugging**

### 1.8 Install Python 3.10+ (only needed for pipeline tests)

```bash
python3 --version   # should print 3.10 or higher
```

If missing:
```bash
# Ubuntu/Linux:
sudo apt install python3 python3-pip python3-venv

# Mac:
brew install python3
```

---

## Part 2 — Get the Code

### 2.1 What you should have on your computer

The project folder looks like this:

```
nippon/
├── react-native-indicai/    ← the SDK (this is what we build and share)
├── IndicAIDemo/             ← the demo app that tests the SDK
└── pipeline/                ← Python scripts that build the AI models
    └── models/
        └── raw/             ← vocab/tokenizer files needed for the build
```

If you got this as a zip, extract it. If you cloned it from git:
```bash
git clone <repo-url>
cd nippon
```

### 2.2 Confirm the folder structure exists

```bash
ls nippon/
# should show: react-native-indicai  IndicAIDemo  pipeline
```

---

## Part 3 — Build the Demo APK (Android)

The demo app (`IndicAIDemo`) is a ready-made React Native app that shows every SDK feature working. You build it once and install it on your phone.

### Option A — One command (easiest)

```bash
cd nippon/IndicAIDemo
bash build_apk.sh
```

This script does everything automatically:
1. Copies tokenizer vocab files from `pipeline/models/raw/` into the SDK
2. Runs `npm install` to install all JavaScript dependencies
3. Builds the Android APK using Gradle

When it finishes you will see:
```
APK: ./android/app/build/outputs/apk/debug/app-debug.apk
```

### Option B — Step by step (if something goes wrong)

**Step 1 — Copy vocab files into SDK assets:**
```bash
cd nippon/react-native-indicai
bash setup_assets.sh
# This copies vocab/dictionary files from pipeline/models/raw/
# into android/src/main/assets/
# These files are needed at app startup for tokenization
```

**Step 2 — Install JavaScript dependencies:**
```bash
cd nippon/IndicAIDemo
npm install
# Downloads all JS packages listed in package.json
# Creates the node_modules/ folder
# Takes 1-3 minutes on first run
```

**Step 3 — Bundle the JavaScript:**
```bash
# Still inside nippon/IndicAIDemo
npx react-native bundle \
  --platform android \
  --dev false \
  --entry-file index.js \
  --bundle-output android/app/src/main/assets/index.android.bundle \
  --assets-dest android/app/src/main/res
# This compiles all your JavaScript/TypeScript into one file
# the Android app can understand
```

**Step 4 — Build the APK:**
```bash
cd android
./gradlew assembleDebug
# This compiles all Kotlin/Java code and packages everything
# into a single .apk file
# First run: 5-10 minutes (downloads Gradle and Android dependencies)
# Subsequent runs: 30-60 seconds
```

**Step 5 — Find the APK:**
```bash
find . -name "app-debug.apk"
# ./app/build/outputs/apk/debug/app-debug.apk
```

---

## Part 4 — Install APK on Your Android Phone

### 4.1 Connect your phone

1. Plug in your phone via USB cable
2. On the phone: a popup may ask "Allow USB Debugging from this computer?" → tap **Allow**

### 4.2 Check your phone is detected

```bash
adb devices
```

You should see something like:
```
List of devices attached
RZCX413K6QZ    device
```

If you see nothing or `unauthorized`:
- Unplug and replug the USB
- Check you tapped "Allow" on the phone
- Try a different USB cable (some cables are charge-only)

If you see multiple devices listed:
```bash
adb devices
# RZCX413K6QZ    device
# emulator-5554  device
```
Note your device ID (e.g. `RZCX413K6QZ`) — you will need it.

### 4.3 Install the APK

```bash
# If you have one device connected:
adb install -r nippon/IndicAIDemo/android/app/build/outputs/apk/debug/app-debug.apk

# If you have multiple devices, specify the ID:
adb -s RZCX413K6QZ install -r nippon/IndicAIDemo/android/app/build/outputs/apk/debug/app-debug.apk
```

`-r` means "replace" — it overwrites any existing version without uninstalling first.

You should see:
```
Performing Streamed Install
Success
```

### 4.4 Launch the app

```bash
adb shell am start -n com.indicaidemo/.MainActivity
```

Or just find **IndicAI Demo** in your phone's app drawer and tap it.

---

## Part 3B — Build the Demo iOS App (Mac only)

> **Requires:** Mac with Xcode 15+, CocoaPods, Node 18+.
> iOS development is not possible on Linux or Windows.

### Step 1 — Install JavaScript dependencies

```bash
cd nippon/IndicAIDemo
npm install
```

### Step 2 — Copy vocab assets into SDK (same as Android)

```bash
cd nippon/react-native-indicai
bash setup_assets.sh
# This copies vocab files into ios/assets/
```

### Step 3 — Generate the iOS project folder

The `ios/` folder is not included in the repo. You generate it with:

```bash
cd nippon/IndicAIDemo
npx react-native init IndicAIDemo --skip-install
# This creates the ios/ folder with Xcode project files
# It takes 1-2 minutes
```

> If `ios/` already exists, skip this step.

### Step 4 — Install iOS native dependencies (CocoaPods)

```bash
cd nippon/IndicAIDemo/ios
pod install
```

This downloads `onnxruntime-objc` (~150 MB) and all other iOS native dependencies.
It creates `IndicAIDemo.xcworkspace` — always open this, never the `.xcodeproj`.

First run takes 3-5 minutes. Subsequent runs are instant if nothing changed.

### Step 5 — Bundle the JavaScript

```bash
cd nippon/IndicAIDemo

npx react-native bundle \
  --platform ios \
  --dev false \
  --entry-file index.js \
  --bundle-output ios/main.jsbundle \
  --assets-dest ios
```

### Step 6 — Open in Xcode

```bash
open nippon/IndicAIDemo/ios/IndicAIDemo.xcworkspace
```

**Important:** Always open `.xcworkspace`, not `.xcodeproj`.
Opening `.xcodeproj` will fail because CocoaPods dependencies are missing.

### Step 7 — Build in Xcode

In Xcode:
1. Select your target device from the top bar (Simulator or your iPhone)
2. Press **Cmd+R** (or click the Play button) to build and run

---

## Part 4B — Run on iOS Simulator or iPhone

### Option A — Simulator (no iPhone required)

You can test the full SDK on the iOS Simulator without a physical device.
The Simulator runs on your Mac and behaves like a real iPhone.

```bash
cd nippon/IndicAIDemo

# List available simulators:
npx react-native run-ios --list-devices

# Run on the default simulator (iPhone 15 Pro):
npx react-native run-ios

# Run on a specific simulator:
npx react-native run-ios --simulator "iPhone 14"
```

**Simulator limitations vs physical device:**
- Microphone does not work in the Simulator — you cannot test voice recording
- All other features work: text input, translation, intent, TTS, model download
- TTS audio plays through your Mac's speakers

**To test voice input** you need a physical iPhone (see Option B).

### Option B — Physical iPhone

Requirements:
- Apple Developer account (free account is enough for development builds)
- iPhone running iOS 15.0 or higher
- USB cable (Lightning or USB-C)

**Step 1 — Trust your Mac on the iPhone**

Plug in the iPhone. On the phone, tap **Trust** when asked "Trust This Computer?"

**Step 2 — Sign the app in Xcode**

1. Open `IndicAIDemo.xcworkspace` in Xcode
2. Click `IndicAIDemo` in the left sidebar (the project)
3. Click the `IndicAIDemo` target
4. Go to **Signing & Capabilities** tab
5. Under **Team**, select your Apple ID (sign in at Xcode → Settings → Accounts if needed)
6. Xcode will auto-generate a provisioning profile

**Step 3 — Select your iPhone as the target**

In Xcode's top bar, click the device selector and choose your iPhone by name.

**Step 4 — Build and install**

Press **Cmd+R**. Xcode builds the app and installs it on your phone automatically.

First build: 3-5 minutes.
Subsequent builds: 30-60 seconds.

**Step 5 — Trust the developer on your iPhone**

First time only:
1. On iPhone: **Settings → General → VPN & Device Management**
2. Tap your Apple ID under "Developer App"
3. Tap **Trust**

Now open **IndicAI Demo** from the home screen.

### Viewing iOS logs

While the app is running (connected to Xcode):
- Xcode shows logs in the bottom console panel
- Filter by `IndicAI` to see SDK-specific logs

From terminal:
```bash
# Show iOS device logs (requires idevicesyslog — brew install libimobiledevice):
idevicesyslog | grep IndicAI
```

### Clear app data on iPhone (re-download models)

Delete the app from iPhone → reinstall. There is no `adb pm clear` equivalent on iOS.
Models are stored in the app's `Library/Application Support/` directory, which is wiped on delete.

---

## Part 5 — Test the App on Your Phone

### 5.1 First launch — model download

The very first time you open the app, it will download the AI models.
This takes **5-15 minutes** depending on your internet speed.
Total download: ~550 MB.

You will see a progress bar showing which model is downloading:
```
Downloading Whisper… 23%
Downloading IndicTrans2 Indic→EN… 67%
...
```

**Do not close the app during download.**
After all models are downloaded, they are stored on your phone permanently.
Next time you open the app, it loads in ~5 seconds with no download.

### 5.2 What to test — Text Input

1. Wait for the app to show the mic button (means it's ready)
2. Select a language from the top row (e.g. हिंदी for Hindi)
3. In the text box, type: `मुझे अकाउंट बनाना है`
4. Tap **Send**

You should see:
```
TRANSCRIPTION
मुझे अकाउंट बनाना है

ENGLISH TRANSLATION
I need to create an account

INTENT
create_account                    89%

RESPONSE IN YOUR LANGUAGE
मुझे एक खाता बनाना है।
```

### 5.3 What to test — Voice Input

1. Tap and hold the **🎙 Hold to speak** button
2. Say something in Hindi, e.g.: "मेरा बैलेंस क्या है"
3. Release the button

You should see the transcription appear, then the translation, then the intent.

### 5.4 What to test — Text-to-Speech

After getting a result:
- Tap **▶ Play TTS** under the transcription → it speaks the transcribed text
- Tap **🔊 Respond with speech** under the English translation → SDK translates and speaks the response
- Tap **▶ Play TTS** under the response → it speaks the Indic response

### 5.5 What each result means

| Result shown | What it means |
|---|---|
| **Transcription** | What the user said/typed in their language |
| **English Translation** | The English meaning of what they said |
| **Intent** | What they want to do + confidence % |
| **Response in your language** | The English reply translated back to their language |

---

## Part 6 — Test Without USB (Standalone)

Once the APK is installed on the phone, **you do not need USB at all**.

- Unplug the cable
- All models are cached on the phone
- The app works fully offline — no internet needed
- Open the app from the phone's app drawer like any other app

The only time you need internet again is if we release new model versions
(the app checks for updates in the background).

---

## Part 7 — How to Rebuild After Code Changes

When you make changes to the SDK code (Kotlin or TypeScript), you need to rebuild.

### If you changed TypeScript files (src/):

```bash
cd nippon/IndicAIDemo

# Rebundle the JavaScript (mandatory after any JS/TS change)
npx react-native bundle \
  --platform android \
  --dev false \
  --entry-file index.js \
  --bundle-output android/app/src/main/assets/index.android.bundle \
  --assets-dest android/app/src/main/res

# Rebuild the APK
cd android
./gradlew assembleDebug

# Install on phone
cd ..
adb install -r android/app/build/outputs/apk/debug/app-debug.apk

# Launch
adb shell am start -n com.indicaidemo/.MainActivity
```

**Important:** If you skip the bundle step and only rebuild the APK,
your TypeScript changes will NOT be included. The APK uses the pre-bundled JS file.

### If you changed Kotlin files (android/src/main/java/):

```bash
cd nippon/IndicAIDemo/android

# Just rebuild APK (no need to rebundle JS)
./gradlew assembleDebug

# Install
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.indicaidemo/.MainActivity
```

### If you changed TypeScript files (src/) — iOS:

```bash
cd nippon/IndicAIDemo

# Rebundle for iOS
npx react-native bundle \
  --platform ios \
  --dev false \
  --entry-file index.js \
  --bundle-output ios/main.jsbundle \
  --assets-dest ios

# Then rebuild in Xcode (Cmd+R) or from terminal:
npx react-native run-ios
```

### If you changed Swift/ObjC files (ios/):

```bash
# Rebuild in Xcode (Cmd+R) — Xcode automatically recompiles changed Swift files

# Or from terminal (builds and runs on the default simulator):
npx react-native run-ios
```

No JS re-bundle needed when only Swift/ObjC changes.

### If you added or changed a CocoaPod dependency:

```bash
cd nippon/IndicAIDemo/ios
pod install
# Then rebuild in Xcode
```

### Check logs while testing — Android

```bash
# Show only SDK-relevant logs while app is running:
adb logcat 2>&1 | grep "IndicAI\."

# You will see lines like:
# D IndicAI.Models:   whisper-small-int4-v1.zip — cache hit
# D IndicAI.Inference: translate result: 'I need to create an account'
# D IndicAI.Intent:   classify('I need to create an account') → create_account (sim=0.891)
# D IndicAISDK:       Ready for hi
```

### Check logs while testing — iOS

Logs appear in Xcode's console at the bottom of the window.
Filter by typing `IndicAI` in the console filter box.

You will see lines like:
```
[IndicAI.Models] whisper-small-int4-v1.zip — cache hit
[IndicAI.Inference] translate result: 'I need to create an account'
[IndicAI.Intent] classify('I need to create an account') → create_account (sim=0.891)
```

### Clear app data (start fresh, re-download models)

**Android:**
```bash
adb shell pm clear com.indicaidemo
# Wipes all cached models and app data
# Next launch will download everything again
```

**iOS:**
Delete the app from the iPhone/Simulator, then reinstall.
On Simulator: Device → Erase All Content and Settings (for clean state)

---

## Part 8 — Run the Python Pipeline Tests

The `pipeline/` folder has Python scripts that test the SDK end-to-end
by downloading models from S3 and running inference — exactly like the real app does.

### 8.1 Setup Python environment (first time only)

```bash
cd nippon/pipeline

# Create a virtual environment (keeps dependencies isolated)
python3 -m venv .venv

# Activate it
source .venv/bin/activate   # Linux/Mac
# OR on Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Takes 2-5 minutes on first run
```

### 8.2 Run the SDK simulation test

```bash
# Make sure venv is active (you see (.venv) in terminal)
source .venv/bin/activate

python3 test_sdk_from_s3.py
```

This script:
1. Downloads `manifest.json` from our S3 CDN
2. Downloads all model zip files (or uses cached versions)
3. Runs the full pipeline: 5 test sentences × 9 languages
4. Shows intent classification results for each

Expected output:
```
Downloading manifest.json...
whisper-small-int4-v1.zip — cache hit
indic-trans2-indic-en-int8-v1.zip — cache hit
intent-minilm-v1.zip — cache hit

Language: Hindi (hi)
┌─────────────────────────────────────┬──────────────────────┬─────────────┬────────┐
│ Input                               │ English              │ Intent      │ Conf   │
├─────────────────────────────────────┼──────────────────────┼─────────────┼────────┤
│ मुझे अकाउंट बनाना है                │ I need to create ... │ create_acc… │ 89.1%  │
│ मेरा बैलेंस क्या है                 │ What is my balance?  │ check_bala… │ 92.3%  │
...
```

### 8.3 Where models are cached locally

```bash
ls pipeline/.cache/
# whisper-small-int4-v1/
# indic-trans2-indic-en-int8-v1/
# intent-minilm-v1/
# manifest.json
```

To force re-download (test fresh download):
```bash
rm -rf pipeline/.cache/
python3 test_sdk_from_s3.py
```

---

## Part 9 — Folder Structure Explained

```
nippon/
│
├── react-native-indicai/              ← THE SDK PACKAGE (this is what you share with APK team)
│   │
│   ├── src/
│   │   ├── index.ts                   ← Public JS API — the only file app team imports
│   │   └── useIndicAI.ts             ← Drop-in React hook (optional but recommended)
│   │
│   ├── android/
│   │   └── src/main/java/com/indicai/
│   │       ├── rn/
│   │       │   └── IndicAIModule.kt   ← React Native bridge (connects JS ↔ Android)
│   │       └── sdk/
│   │           ├── IndicAISDK.kt      ← Main SDK class (init, transcribe, translate, TTS)
│   │           ├── InferenceEngine.kt ← Runs all ONNX models
│   │           ├── IntentEngine.kt    ← Runs MiniLM intent classification
│   │           ├── ModelManager.kt    ← Downloads, verifies (SHA-256), caches models
│   │           ├── ManifestManager.kt ← Fetches manifest.json from CDN
│   │           └── LanguageRegistry.kt← Maps language codes to config
│   │
│   ├── android/src/main/assets/       ← Vocab files (copied by setup_assets.sh)
│   │   ├── whisper_vocab.json
│   │   ├── whisper_merges.txt
│   │   ├── indic_en_src_vocab.json
│   │   ├── indic_en_tgt_vocab.json
│   │   ├── en_indic_src_vocab.json
│   │   ├── en_indic_tgt_vocab.json
│   │   └── mms_tts_hi_vocab.json (+ mr, ta, te, kn, ml, bn, gu, pa)
│   │
│   ├── setup_assets.sh                ← Copies vocab files from pipeline/models/raw/
│   ├── package.json                   ← npm package definition
│   ├── README.md                      ← This file
│   └── INTEGRATION.md                 ← APK team's integration guide
│
├── IndicAIDemo/                       ← DEMO APP (tests the SDK end-to-end)
│   ├── App.tsx                        ← Entry point (just loads IndicAIScreen)
│   ├── index.js                       ← React Native entry file
│   ├── android/
│   │   └── app/
│   │       ├── src/main/assets/
│   │       │   └── index.android.bundle  ← Pre-built JS bundle (generated by react-native bundle)
│   │       └── build/outputs/apk/debug/
│   │           └── app-debug.apk      ← Built APK file (generated by gradlew assembleDebug)
│   ├── node_modules/
│   │   └── react-native-indicai/      ← Copy of the SDK (linked via npm install)
│   ├── ios/                           ← iOS native project (generated — see Part 3B)
│   │   ├── IndicAIDemo.xcworkspace    ← Always open this (not .xcodeproj)
│   │   └── IndicAIDemo/
│   │       └── Info.plist             ← iOS app settings (mic permission goes here)
│   ├── build_apk.sh                   ← One-command Android build script
│   └── package.json
│
└── pipeline/                          ← PYTHON TOOLS (for building and testing models)
    ├── test_sdk_from_s3.py            ← End-to-end test (downloads from S3, runs inference)
    ├── requirements.txt               ← Python dependencies
    ├── .venv/                         ← Python virtual environment (created by you)
    ├── .cache/                        ← Cached model downloads (created automatically)
    ├── dist/
    │   └── manifest.json              ← The manifest file that gets uploaded to S3
    └── models/
        ├── raw/                       ← Vocab/tokenizer files (needed by setup_assets.sh)
        └── intent-minilm/             ← Built MiniLM intent model files
```

---

## Part 10 — Common Problems and Fixes

### "adb: command not found"

ADB is not in your PATH. Fix:
```bash
# Find where Android SDK is installed:
ls ~/Android/Sdk/platform-tools/    # Linux
ls ~/Library/Android/sdk/platform-tools/  # Mac

# Add to PATH (add to ~/.bashrc or ~/.zshrc):
export PATH="$PATH:$HOME/Android/Sdk/platform-tools"

# Reload:
source ~/.bashrc
adb --version   # should work now
```

---

### "no devices/emulators found" when running adb install

Your phone is not detected. Try:
```bash
adb kill-server
adb start-server
adb devices
```

If still empty:
- On your phone, look for a USB mode notification → change to **File Transfer** mode
- Check USB cable (some are charge-only, not data)
- Try `adb devices` after tapping "Allow" on phone's USB Debugging popup

---

### "INSTALL_FAILED_UPDATE_INCOMPATIBLE"

The app was previously installed with a different signature. Fix:
```bash
adb uninstall com.indicaidemo
adb install nippon/IndicAIDemo/android/app/build/outputs/apk/debug/app-debug.apk
```

---

### Build fails: "SDK location not found"

```bash
# Create the file:
echo "sdk.dir=$HOME/Android/Sdk" > nippon/IndicAIDemo/android/local.properties
```

---

### Build fails: "Unresolved reference: File" or similar Kotlin errors

The node_modules copy of SDK might be out of sync with the source. Fix:
```bash
cd nippon/IndicAIDemo
rm -rf node_modules
npm install
cd android
./gradlew clean
./gradlew assembleDebug
```

---

### App shows error: "Native module not linked"

The JS bundle was built before the SDK was installed. Fix:
```bash
cd nippon/IndicAIDemo
npm install  # make sure react-native-indicai is in node_modules

npx react-native bundle \
  --platform android \
  --dev false \
  --entry-file index.js \
  --bundle-output android/app/src/main/assets/index.android.bundle \
  --assets-dest android/app/src/main/res

cd android && ./gradlew assembleDebug
```

---

### App stuck on "Downloading…" forever

Check internet connection on the phone.
Check logs:
```bash
adb logcat 2>&1 | grep "IndicAI\."
```

Look for lines like:
```
W IndicAI.Models: Download failed for whisper-small-int4-v1.zip, retry 1/3
W IndicAI.Models: java.lang.SecurityException: SHA-256 mismatch
```

If you see SHA-256 mismatch, the manifest has wrong hashes. Contact the SDK team.

---

### Intent not showing in the app

The JS bundle may be stale. Rebuild it:
```bash
cd nippon/IndicAIDemo

npx react-native bundle \
  --platform android \
  --dev false \
  --entry-file index.js \
  --bundle-output android/app/src/main/assets/index.android.bundle \
  --assets-dest android/app/src/main/res

cd android && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

---

### "Execution failed for task ':react-native-indicai:compileDebugKotlin'"

Check the exact error:
```bash
./gradlew react-native-indicai:compileDebugKotlin 2>&1 | grep "e:"
```

Common fix — missing import in SDK Kotlin file:
```bash
# If error says "Unresolved reference: File",
# open the failing .kt file and add at the top:
# import java.io.File
```

---

### Phone has no space for models (~550 MB needed)

```bash
# Check available space on phone:
adb shell df /data | tail -1
```

Free up space by deleting unused apps on the phone, then try again.
Once downloaded, models stay cached — you only download once.

---

### iOS: "pod install" fails with "Unable to find a specification for onnxruntime-objc"

```bash
cd nippon/IndicAIDemo/ios
pod repo update   # refresh the CocoaPods spec repos
pod install
```

---

### iOS: Xcode build fails — "No such module 'React'"

You opened `.xcodeproj` instead of `.xcworkspace`. Fix:
```bash
open nippon/IndicAIDemo/ios/IndicAIDemo.xcworkspace
# Never: open nippon/IndicAIDemo/ios/IndicAIDemo.xcodeproj
```

---

### iOS: "Untrusted Developer" when opening app on iPhone

```
Settings → General → VPN & Device Management → [your Apple ID] → Trust
```

---

### iOS: App builds but crashes immediately on launch

Check Xcode logs. Common cause: JS bundle not built.
```bash
cd nippon/IndicAIDemo
npx react-native bundle \
  --platform ios --dev false \
  --entry-file index.js \
  --bundle-output ios/main.jsbundle \
  --assets-dest ios
# Then rebuild in Xcode
```

---

### iOS: Voice recording does not work in Simulator

This is expected — the iOS Simulator does not support the microphone.
You must test voice recording on a physical iPhone.
All other features (text input, translation, TTS, intent) work in the Simulator.

---

### iOS: "NSMicrophoneUsageDescription" error — app rejected at runtime

Your `Info.plist` is missing the microphone usage description. Add this:
```xml
<key>NSMicrophoneUsageDescription</key>
<string>IndicAI needs the microphone to understand your voice.</string>
```

In Xcode: click `Info.plist` → add a new row → key = `NSMicrophoneUsageDescription` → value = your description string.

---

## Summary — Commands You Will Use Most Often

### Android

```bash
# ── Build everything from scratch ────────────────────────────────────
cd nippon/IndicAIDemo && bash build_apk.sh

# ── After changing TypeScript/JS ────────────────────────────────────
cd nippon/IndicAIDemo
npx react-native bundle --platform android --dev false \
  --entry-file index.js \
  --bundle-output android/app/src/main/assets/index.android.bundle \
  --assets-dest android/app/src/main/res
cd android && ./gradlew assembleDebug

# ── After changing Kotlin ────────────────────────────────────────────
cd nippon/IndicAIDemo/android && ./gradlew assembleDebug

# ── Install on phone ─────────────────────────────────────────────────
adb install -r nippon/IndicAIDemo/android/app/build/outputs/apk/debug/app-debug.apk

# ── Launch app ───────────────────────────────────────────────────────
adb shell am start -n com.indicaidemo/.MainActivity

# ── Watch SDK logs ───────────────────────────────────────────────────
adb logcat 2>&1 | grep "IndicAI\."

# ── Fresh start (clear all downloaded models) ────────────────────────
adb shell pm clear com.indicaidemo

# ── Check connected devices ──────────────────────────────────────────
adb devices

# ── Python pipeline test ─────────────────────────────────────────────
cd nippon/pipeline && source .venv/bin/activate && python3 test_sdk_from_s3.py
```

### iOS (Mac only)

```bash
# ── First-time iOS setup ──────────────────────────────────────────────
cd nippon/IndicAIDemo/ios && pod install

# ── After changing TypeScript/JS ────────────────────────────────────
cd nippon/IndicAIDemo
npx react-native bundle --platform ios --dev false \
  --entry-file index.js \
  --bundle-output ios/main.jsbundle \
  --assets-dest ios
# Then rebuild in Xcode (Cmd+R) or:
npx react-native run-ios

# ── After changing Swift/ObjC ────────────────────────────────────────
# Just rebuild in Xcode (Cmd+R) — no bundle step needed
npx react-native run-ios

# ── Run on Simulator ─────────────────────────────────────────────────
npx react-native run-ios
npx react-native run-ios --simulator "iPhone 14"

# ── Open in Xcode ────────────────────────────────────────────────────
open nippon/IndicAIDemo/ios/IndicAIDemo.xcworkspace
```

---

## Part 11 — Android vs iOS: How They Differ

Both platforms run the same JavaScript/TypeScript API. The differences are all internal.

### Platform Requirements

| | Android | iOS |
|---|---|---|
| Build machine | Linux, Mac, Windows | **Mac only** |
| Build tool | Gradle (`./gradlew`) | Xcode (`xcodebuild`) |
| Package manager | `npm install` (no extra step) | `npm install` + `pod install` |
| IDE (optional) | Android Studio | Xcode |
| Device testing | USB + ADB | USB + Xcode trust |
| Simulator | Android Emulator (any OS) | iOS Simulator (Mac only) |

### Native Language

| | Android | iOS |
|---|---|---|
| SDK written in | Kotlin | Swift |
| Bridge to JS | `IndicAIModule.kt` | `IndicAIModule.swift` + `IndicAIModule.m` |
| Bridging header | None needed | `IndicAIModule-Bridging-Header.h` |
| Dependency format | Gradle (AAR/maven) | CocoaPods (.podspec) |

### Permissions

| Permission | Android | iOS |
|---|---|---|
| Internet (model download) | `AndroidManifest.xml` → `INTERNET` | Automatic (no config needed) |
| Microphone | `AndroidManifest.xml` → `RECORD_AUDIO` | `Info.plist` → `NSMicrophoneUsageDescription` |
| Runtime request | `PermissionsAndroid.request()` in JS | iOS shows system dialog automatically on first use |

### File Paths

| | Android | iOS |
|---|---|---|
| TTS WAV file location | `/data/user/0/<pkg>/cache/indicai_tts.wav` | `/tmp/indicai_tts.wav` (NSTemporaryDirectory) |
| Audio URI prefix | Needs `file://` prefix | Needs `file://` prefix (same) |
| Model cache dir | `filesDir/indicai_models/` | `Library/Application Support/indicai_models/` |

### Audio

| | Android | iOS |
|---|---|---|
| Recording capture | `AudioRecord` (MediaRecorder API) | `AVAudioEngine` (AVFoundation) |
| Audio format | PCM 16-bit 16kHz mono | PCM 16-bit 16kHz mono (same) |
| Audio session | Automatic | Requires `AVAudioSession.setCategory(.record)` |
| Playing WAV | Any player — `file://` URI | Any player — `file://` URI (same API) |

### SDK Flow (identical across both platforms)

The JavaScript API is **100% identical** on Android and iOS:

```typescript
// This code works the same on Android and iOS — no platform checks needed
await IndicAI.init('hi');
const english = await IndicAI.translateToEnglish(userText);
const { intent } = await IndicAI.classifyIntent(english);
const { indicText, audioPath } = await IndicAI.respondWithSpeech(backendReply);

// Only one difference: audio player URI handling
// Both platforms need file:// prefix (same code works on both)
const uri = audioPath.startsWith('file://') ? audioPath : `file://${audioPath}`;
```

### Testing Checklist

| Test | Android | iOS Simulator | Physical iPhone |
|---|---|---|---|
| Text → translate → intent | ✓ | ✓ | ✓ |
| Respond with speech (TTS) | ✓ | ✓ | ✓ |
| Model download progress | ✓ | ✓ | ✓ |
| Voice recording (mic) | ✓ | ✗ (no mic) | ✓ |
| Audio playback | ✓ | ✓ (Mac speakers) | ✓ |

---

*SDK version: 1.0.0 | React Native: 0.73.6 | Min Android: API 26 (Android 8.0) | Min iOS: 15.0*

---

## Part 12 — How to Introduce or Replace a Model

This section is for the **SDK team** — people who build, retrain, or swap the AI models.
There are four model slots: Whisper (STT), IndicTrans2 Indic→EN, IndicTrans2 EN→Indic, and Intent (MiniLM).
Each language also has its own TTS slot.

All model configuration lives in **one file: `pipeline/config.py`**.

---

### Scenario A — Replace Whisper with a newer version

Whisper is the speech-to-text model. The current one is `openai/whisper-small`.

**Step 1 — Change the model source in `config.py`:**

```python
# pipeline/config.py

# Change this:
WHISPER_MODEL_ID = "openai/whisper-small"

# To the new model, e.g. whisper-medium:
WHISPER_MODEL_ID = "openai/whisper-medium"
```

**Step 2 — Bump the version:**

```python
# pipeline/config.py
MODEL_VERSION = 2   # was 1 — increment every time you change any model
```

**Step 3 — Re-run the pipeline:**

```bash
cd pipeline
source .venv/bin/activate

python step1_download.py    # downloads new Whisper from HuggingFace
python step2_quantize.py    # exports ONNX + quantizes to INT4 + zips
python step3_eval.py        # checks quality hasn't degraded (optional but recommended)
python step4_upload.py      # uploads new zip + updates manifest on S3
```

**Step 4 — Upload:**

```bash
bash upload_all.sh   # re-uploads dist/ to S3 + pushes new manifest.json
```

Done. The app automatically downloads the new model on next launch (manifest version changed).

---

### Scenario B — Replace IndicTrans2 (translation model)

Two model IDs control translation: Indic→English and English→Indic.

**Step 1 — Change model sources in `config.py`:**

```python
# pipeline/config.py

INDIC_EN_MODEL_ID   = "ai4bharat/indictrans2-indic-en-dist-200M"   # Indic → EN
INDIC_FROM_MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"   # EN → Indic

# Example: upgrade to 1B parameter version:
INDIC_EN_MODEL_ID   = "ai4bharat/indictrans2-indic-en-1B"
INDIC_FROM_MODEL_ID = "ai4bharat/indictrans2-en-indic-1B"
```

**Step 2 — Bump `MODEL_VERSION` and re-run pipeline** (same as Scenario A, steps 2–4).

**Note on tokenizer vocab files:** IndicTrans2 models come with their own tokenizer vocab.
If you change the model family, the vocab files change too.
`step2_quantize.py` exports the new vocab automatically into `dist/`.
`setup_assets.sh` copies them into the SDK's `android/src/main/assets/` and `ios/assets/`.
Always run `setup_assets.sh` after re-running the pipeline.

```bash
cd react-native-indicai
bash setup_assets.sh   # re-copies vocab files into Android + iOS assets
```

---

### Scenario C — Replace the Intent model (MiniLM)

The intent model classifies English text into one of the 27 intents.

**Step 1 — Rebuild the intent model:**

```bash
cd pipeline
source .venv/bin/activate
python build_intent_model.py   # trains the new intent bank + exports ONNX
```

**Step 2 — Update intent labels if you added/removed/renamed any intent:**

The intent labels are defined in two places:
1. `pipeline/build_intent_model.py` — the training phrases and labels
2. `INTEGRATION.md` — Section 11 (the 27 intent labels table)

Update both if you change intent names or add new ones.

**Step 3 — Bump version and upload** (same as above).

---

### Scenario D — Add a new language

All language configuration is in `config.py` under `LANGUAGES`. Adding a new language touches only that dict.

**Step 1 — Add the language entry to `config.py`:**

```python
# pipeline/config.py
LANGUAGES = {
    # ... existing languages ...

    "or": {                        # ISO 639-1 code
        "prompt":       "ଓଡ଼ିଆରେ ଲେଖ।",  # Whisper prompt for this script
        "whisper_lang": "or",      # BCP-47 code Whisper understands
        "flores":       "ory_Orya",  # FLORES-200 code for IndicTrans2
        "mms_lang":     "ory",     # MMS-TTS language code (facebook/mms-tts-{mms_lang})
        "name":         "Odia",
    },
}
```

To find the correct codes:
- `flores` code: look in [IndicTrans2 docs](https://github.com/AI4Bharat/IndicTrans2) — format is `xxx_Xxxx`
- `mms_lang`: check [facebook/mms-tts](https://huggingface.co/facebook/mms-tts) — lowercase 3-letter code
- `whisper_lang`: check [Whisper language list](https://github.com/openai/whisper#available-models-and-languages)

**Step 2 — Re-run only the TTS download + quantize for the new language:**

```bash
cd pipeline
source .venv/bin/activate
python step1_download.py    # downloads TTS model for new language
python step2_quantize.py    # quantizes TTS + zips it
python step4_upload.py      # uploads new zip
```

**Step 3 — Copy new TTS vocab into SDK assets:**

```bash
cd react-native-indicai
bash setup_assets.sh   # copies tts/{new_lang}_vocab.json into Android + iOS
```

**Step 4 — Add language to `LanguageRegistry`:**

Open `react-native-indicai/android/src/main/java/com/indicai/sdk/LanguageRegistry.kt` and add:

```kotlin
"or" to LanguageConfig(
    code        = "or",
    whisperLang = "or",
    flores      = "ory_Orya",
    name        = "Odia"
),
```

Open `react-native-indicai/ios/LanguageRegistry.swift` and add the same entry:

```swift
"or": LanguageConfig(code: "or", whisperLang: "or", flores: "ory_Orya", name: "Odia"),
```

**Step 5 — Bump `MODEL_VERSION` and upload manifest** (same as above).

The new language is now available in `IndicAI.init('or')`.

---

### Scenario E — Add a new TTS model for an existing language

Each language's TTS model is the MMS-TTS model from Facebook.
If a better TTS model becomes available for an existing language, update it here:

**Step 1 — Change the TTS model pattern in `config.py`:**

```python
# config.py
# Current:
MMS_TTS_MODEL_PATTERN = "facebook/mms-tts-{mms_lang}"

# To use a different source per language, edit step1_download.py's TTS download section
# and update step2_quantize.py's TTS export section.
```

**Step 2 — Re-run `step1_download.py` and `step2_quantize.py` for the affected language only.**

---

### Summary — Which files change for each scenario

| Change | `config.py` | `LanguageRegistry.kt` | `LanguageRegistry.swift` | `setup_assets.sh` | Pipeline steps |
|---|---|---|---|---|---|
| Replace Whisper | `WHISPER_MODEL_ID` | — | — | — | 1→2→3→4 |
| Replace IndicTrans2 | `INDIC_*_MODEL_ID` | — | — | run it | 1→2→3→4 |
| Replace Intent | — (edit build_intent_model.py) | — | — | — | build_intent_model.py + step4 |
| Add new language | Add to `LANGUAGES` | Add entry | Add entry | run it | 1→2→4 |
| Bump version only | `MODEL_VERSION` | — | — | — | step4 + upload_all.sh |

**After any model change:**
1. Bump `MODEL_VERSION` in `config.py`
2. Run affected pipeline steps
3. Run `setup_assets.sh` if vocab files changed
4. Run `bash upload_all.sh` to publish to S3
5. Rebuild the APK/iOS app — the new models download automatically on next app launch

---

### How the manifest controls what the app downloads

`pipeline/dist/manifest.json` is the index file the app downloads on every launch.
It lists every model's filename, SHA-256 hash, and size.
The app checks: "is this file already cached with the right hash?" — if yes, skip; if no, download.

The manifest is rebuilt automatically by `upload_all.sh` (Phase 3).
You never need to edit `manifest.json` by hand.

```json
{
  "version": 2,
  "models": {
    "whisper": {
      "file": "whisper-small-int4-v2.zip",
      "sha256": "abc123...",
      "size_mb": 183.0,
      "version": 2
    }
  }
}
```

Bump `MODEL_VERSION` in `config.py` → manifest version increments → all app installs download the new model on next launch.
