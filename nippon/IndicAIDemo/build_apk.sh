#!/bin/bash
set -e
echo "=== IndicAI Demo APK Build ==="
cd "$(dirname "$0")"

# Step 1: copy asset files
echo "[1/4] Copying tokenizer assets..."
cd ../react-native-indicai
bash setup_assets.sh
cd ../IndicAIDemo

# Step 2: install npm deps
echo "[2/4] Installing npm dependencies..."
npm install

# Step 3: build debug APK
echo "[3/4] Building debug APK..."
cd android
./gradlew assembleDebug --no-daemon 2>&1

echo "[4/4] Done!"
echo "APK: $(find . -name '*.apk' -path '*/debug/*' 2>/dev/null | head -1)"
