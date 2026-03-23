#!/bin/bash
# Copy tokenizer vocab files from pipeline to Android assets
ASSETS_DIR="$(dirname "$0")/android/src/main/assets"
MODELS_RAW="../pipeline/models/raw"
mkdir -p "$ASSETS_DIR"

# Whisper vocab
cp "$MODELS_RAW/whisper-small/vocab.json" "$ASSETS_DIR/whisper_vocab.json"
cp "$MODELS_RAW/whisper-small/merges.txt" "$ASSETS_DIR/whisper_merges.txt"

# IndicTrans2 vocab dicts
cp "$MODELS_RAW/indic-trans2-indic-en/dict.SRC.json" "$ASSETS_DIR/indic_en_src_vocab.json"
cp "$MODELS_RAW/indic-trans2-indic-en/dict.TGT.json" "$ASSETS_DIR/indic_en_tgt_vocab.json"
cp "$MODELS_RAW/indic-trans2-en-indic/dict.SRC.json" "$ASSETS_DIR/en_indic_src_vocab.json"
cp "$MODELS_RAW/indic-trans2-en-indic/dict.TGT.json" "$ASSETS_DIR/en_indic_tgt_vocab.json"

# MMS-TTS per-language vocabs
for lc in hi mr ta te kn ml bn gu pa; do
  cp "$MODELS_RAW/mms-tts-$lc/vocab.json" "$ASSETS_DIR/mms_tts_${lc}_vocab.json"
done
echo "Assets copied to $ASSETS_DIR"
