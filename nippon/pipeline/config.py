"""
config.py — Single source of truth for all models and languages.
Adding a new language = add one entry to LANGUAGES. Nothing else changes.
"""

# ── Language registry ──────────────────────────────────────────────────────────
# Each entry: lang_code → { prompt, whisper_lang, indic_flores_code }
# prompt  : steers Whisper to output correct script (permanent, never changes)
# whisper_lang : BCP-47 code Whisper understands
# flores  : language code used by IndicTrans2 tokenizer

LANGUAGES = {
    "hi": {
        "prompt":       "हिंदी में लिखें।",
        "whisper_lang": "hi",
        "flores":       "hin_Deva",
        "mms_lang":     "hin",
        "name":         "Hindi",
    },
    "mr": {
        "prompt":       "मराठी मध्ये लिहा।",
        "whisper_lang": "mr",
        "flores":       "mar_Deva",
        "mms_lang":     "mar",
        "name":         "Marathi",
    },
    "ta": {
        "prompt":       "தமிழில் எழுதுக।",
        "whisper_lang": "ta",
        "flores":       "tam_Taml",
        "mms_lang":     "tam",
        "name":         "Tamil",
    },
    "te": {
        "prompt":       "తెలుగులో రాయండి।",
        "whisper_lang": "te",
        "flores":       "tel_Telu",
        "mms_lang":     "tel",
        "name":         "Telugu",
    },
    "kn": {
        "prompt":       "ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಿರಿ।",
        "whisper_lang": "kn",
        "flores":       "kan_Knda",
        "mms_lang":     "kan",
        "name":         "Kannada",
    },
    "ml": {
        "prompt":       "മലയാളത്തിൽ എഴുതുക।",
        "whisper_lang": "ml",
        "flores":       "mal_Mlym",
        "mms_lang":     "mal",
        "name":         "Malayalam",
    },
    "bn": {
        "prompt":       "বাংলায় লিখুন।",
        "whisper_lang": "bn",
        "flores":       "ben_Beng",
        "mms_lang":     "ben",
        "name":         "Bengali",
    },
    "gu": {
        "prompt":       "ગુજરાતીમાં લખો।",
        "whisper_lang": "gu",
        "flores":       "guj_Gujr",
        "mms_lang":     "guj",
        "name":         "Gujarati",
    },
    "pa": {
        "prompt":       "ਪੰਜਾਬੀ ਵਿੱਚ ਲਿਖੋ।",
        "whisper_lang": "pa",
        "flores":       "pan_Guru",
        "mms_lang":     "pan",
        "name":         "Punjabi",
    },
}

# ── HuggingFace token ─────────────────────────────────────────────────────────
# Get yours from https://huggingface.co/settings/tokens
# Needed for gated models: IndicTrans2, MMTTS (ai4bharat org requires login)
# Whisper (openai/whisper-small) is public — token not required but doesn't hurt
#
# Option 1: set it here directly (don't commit to git — add config.py to .gitignore)
HF_TOKEN = ""   # paste your HuggingFace token here, or set env: export HF_TOKEN=hf_xxx
#
# Option 2: set as environment variable instead (safer for CI)
#   export HF_TOKEN=hf_xxxx
#   HF_TOKEN = os.environ.get("HF_TOKEN", "")
#
import os
HF_TOKEN = os.environ.get("HF_TOKEN", HF_TOKEN)  # env var takes priority over hardcoded

# ── Model sources on HuggingFace ───────────────────────────────────────────────
WHISPER_MODEL_ID    = "openai/whisper-small"
INDIC_EN_MODEL_ID   = "ai4bharat/indictrans2-indic-en-dist-200M"   # Indic → EN
INDIC_FROM_MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"   # EN → Indic (for TTS input)
# MMTTS_MODEL_PATTERN = "ai4bharat/vits_{gender}_{lang}"              # e.g. vits_female_hi
# MMTTS_GENDER        = "female"  # change to "male" if preferred
MMS_TTS_MODEL_PATTERN = "facebook/mms-tts-{mms_lang}"  # public, no token needed


# ── Quality gate thresholds ────────────────────────────────────────────────────
WER_DELTA_THRESHOLD  = 0.03   # max 3pp WER degradation vs fp32
BLEU_DELTA_THRESHOLD = 1.5    # max 1.5 BLEU drop vs fp32

# ── Versioning ─────────────────────────────────────────────────────────────────
MODEL_VERSION = 1   # bump this when you retrain/requantize

# ── S3 / CDN ───────────────────────────────────────────────────────────────────
S3_BUCKET       = "indicai-models"
CDN_BUCKET      = "indicai-cdn"
MANIFEST_KEY    = "manifest.json"
S3_MODELS_PREFIX = "models/"

# ── Local paths ────────────────────────────────────────────────────────────────
RAW_DIR       = "./models/raw"
EXPORT_DIR    = "./models/export"
QUANTIZED_DIR = "./models/quantized"
DIST_DIR      = "./dist"
EVAL_DIR      = "./eval_results"